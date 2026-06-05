"""Re-run extraction on 3 chunks and save ALL results to disk."""
import sys, json, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.novels2llm.pipeline.stage1_preprocess import preprocess_novel
from src.novels2llm.pipeline.stage2_nlp import process_chapter
from src.novels2llm.chunking.smart_chunker import chunk_novel
from src.novels2llm.extractors.character_extractor import CharacterExtractor
from src.novels2llm.extractors.world_extractor import WorldExtractor
from src.novels2llm.extractors.relationship_extractor import RelationshipExtractor
from src.novels2llm.extractors.scene_extractor import SceneExtractor
from src.novels2llm.coreference.alias_resolver import resolve_aliases
from src.novels2llm.graph.relationship_graph import RelationshipGraph
from src.novels2llm.models.output import NovelWorld, NovelMetadata, SceneEvent
from src.novels2llm.models.entities import Character, Location, Item, WorldSetting
from src.novels2llm.models.relationships import Dialogue, Relationship
from src.novels2llm.models.scene import Scene, NarrativeUnit
from src.novels2llm.pipeline.stage7_export import export_to_json, export_to_sqlite, export_character_cards
from src.novels2llm.config import config

API_KEY = config.ANTHROPIC_API_KEY
N_CHUNKS = 3
NOVEL_FILE = Path('data/jia_ting_luan_lun/ai-mu-ru-ping.md')
OUTPUT_DIR = Path('data/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_rlabels(labels: list) -> list[dict]:
    """Normalize relational_labels to list of {caller, label} dicts."""
    result = []
    for item in labels:
        if isinstance(item, dict):
            caller = item.get('caller', '').strip()
            label = item.get('label', '').strip()
            if label:
                result.append({'caller': caller or 'unknown', 'label': label})
        elif isinstance(item, str):
            item = item.strip()
            if item:
                result.append({'caller': 'unknown', 'label': item})
    return result

# ─── Stage 1 + 2 ────────────────────────────────────────────────
print("Stage 1+2: Preprocess, Chunk, NLP...")
result = preprocess_novel(NOVEL_FILE)
all_chunks = chunk_novel(
    result.raw_text, result.metadata.novel_id,
    target_size=config.TARGET_CHUNK_SIZE_CHARS,
    max_size=config.MAX_CHUNK_SIZE_CHARS,
    overlap=config.CHUNK_OVERLAP_CHARS,
)
test_chunks = all_chunks[:N_CHUNKS]

nlp_map = {}
for c in test_chunks:
    nlp = process_chapter(c.text, result.metadata.novel_id, c.chapter, use_hanlp=False)
    nlp_map[c.chunk_index] = nlp

# ─── Setup ───────────────────────────────────────────────────────
char_ext = CharacterExtractor(API_KEY)
world_ext = WorldExtractor(API_KEY)
rel_ext = RelationshipExtractor(API_KEY)
scene_ext = SceneExtractor(API_KEY)

def call_llm(prompt, max_tokens=4096):
    """Call LLM for cross-chunk merge judgments."""
    return scene_ext._call_claude(prompt, max_tokens=max_tokens)

def parse_json(text):
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    json_str = m.group(1) if m else None
    if not json_str:
        m = re.search(r'\{[\s\S]*\}', text)
        json_str = m.group(0) if m else text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return json.loads(_repair_json(json_str))


def _repair_json(s):
    """Apply common LLM JSON repairs."""
    s = s.replace('\ufeff', '').replace('\u200b', '')
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    s = s.replace('\uff08', '(').replace('\uff09', ')')
    s = s.replace('\uff1a', ':').replace('\uff0c', ',')
    # Remove trailing commas before ] or }
    s = re.sub(r',\s*([}\]])', r'\1', s)
    # Fix missing commas between JSON elements at newline boundaries:
    # "val"\n"key" -> "val",\n"key",  }\n"key" -> },\n"key",  ]\n"key" -> ],\n"key"
    s = re.sub(r'(["}\]\d])\s*\n\s*(["\[{])', r'\1,\n\2', s)
    # Same-line: "val" "key" -> "val", "key" (two unescaped quotes with space)
    s = re.sub(r'(?<!\\)"\s+(?=")', r'", ', s)
    # Fix missing closing brackets
    open_braces = s.count('{') - s.count('}')
    open_brackets = s.count('[') - s.count(']')
    s += '}' * open_braces + ']' * open_brackets
    return s

# ─── Stage 3: Extract ────────────────────────────────────────────
# Order: characters, world (chunk 0), relationships
all_chars, all_relationships = [], []
world_settings = []

# Phase 1: Extract characters, world, and relationships
for ci, c in enumerate(test_chunks):
    hint = nlp_map[c.chunk_index].to_hint_dict()
    print(f"\nChunk {c.chunk_index}/{N_CHUNKS} ({c.char_count} chars)...")

    # Character
    try:
        chars = char_ext.extract(c.text, nlp_hints=hint)
        all_chars.extend(chars)
        print(f"  Char: {len(chars)}")
    except Exception as e:
        print(f"  Char: ERROR - {e}")

    # World (chunk 0 only)
    if ci == 0:
        try:
            ws = world_ext.extract(c.text, nlp_hints=hint)
            world_settings.append(ws)
            print(f"  World: ok")
        except Exception as e:
            print(f"  World: ERROR - {e}")

    # Relationship
    for attempt in range(2):
        try:
            rels = rel_ext.extract(c.text, known_characters=all_chars)
            for r in rels: r['source_chapter'] = c.chapter
            all_relationships.extend(rels)
            print(f"  Rel: {len(rels)}" + (" (retry)" if attempt > 0 else ""))
            break
        except Exception as e:
            if attempt == 0: continue
            print(f"  Rel: ERROR - {e}")

# Prepare known locations from world settings
known_locations = []
if world_settings:
    known_locations = world_settings[0].get('locations', [])
    print(f"\nKnown locations from world: {len(known_locations)}")

# Phase 2: Scene extraction (with location hints from world)
print(f"\nStage 4: Scene + Narrative Unit extraction...")
all_scenes = []
prev_scene_summary = None

for ci, c in enumerate(test_chunks):
    print(f"\n  Scene extraction chunk {c.chunk_index}/{N_CHUNKS}...")
    for attempt in range(2):
        try:
            scenes = scene_ext.extract(
                c.text,
                chapter=c.chapter,
                chapter_title=c.chapter_title,
                known_locations=known_locations,
                known_characters=all_chars,
                previous_scene_summary=prev_scene_summary,
                chunk_index=c.chunk_index,
            )
            # Override chapter from chunk metadata (don't trust LLM-generated chapter)
            for s in scenes:
                s['chapter'] = c.chapter
            all_scenes.extend(scenes)
            # Keep last scene summary for cross-chunk continuity
            if scenes:
                prev_scene_summary = scenes[-1].get('summary', '')
            print(f"  Scene: {len(scenes)} scenes" + (" (retry)" if attempt > 0 else ""))
            break
        except Exception as e:
            if attempt == 0: continue
            print(f"  Scene: ERROR - {e}")
            import traceback; traceback.print_exc()

# ─── Stage 4.1: Cross-Chunk Scene Merging ────────────────────────
print(f"\nStage 4.1: Renumbering scene IDs globally...")

# Renumber scene IDs to be globally unique (LLM restarts from sc1 per chunk)
scene_id_counter = 1
for s in all_scenes:
    ch = s.get('chapter', 0)
    old_id = s.get('scene_id', '')
    new_id = f"ch{ch}_sc{scene_id_counter}"
    s['scene_id'] = new_id
    # Also update unit_ids to match
    for u in s.get('narrative_units', []):
        old_unit_id = u.get('unit_id', '')
        # Replace old scene prefix with new one
        u['unit_id'] = old_unit_id.replace(old_id, new_id, 1)
    scene_id_counter += 1

print(f"  Renumbered {len(all_scenes)} scenes (globally unique IDs)")

print(f"\nStage 4.2: Merging scenes across chunk boundaries...")
print(f"  Total scenes before merge: {len(all_scenes)}")

# Group scenes by chunk_index
scenes_by_chunk = {}
for s in all_scenes:
    ci = s.get('chunk_index', 0)
    scenes_by_chunk.setdefault(ci, []).append(s)

sorted_chunk_ids = sorted(scenes_by_chunk.keys())
merged_scenes = []
merge_count = 0

for i, ci in enumerate(sorted_chunk_ids):
    curr_scenes = scenes_by_chunk[ci]

    if i == 0:
        merged_scenes.extend(curr_scenes)
        continue

    prev_tail = merged_scenes[-1] if merged_scenes else None
    curr_head = curr_scenes[0] if curr_scenes else None

    # Heuristic: check if previous chunk's last scene and current chunk's first scene
    # share the same location and at least one participant
    should_llm_check = False
    if prev_tail and curr_head:
        same_location = prev_tail.get('location', '') == curr_head.get('location', '')
        prev_parts = set(prev_tail.get('participants', []))
        curr_parts = set(curr_head.get('participants', []))
        shared_parts = prev_parts & curr_parts

        if same_location and shared_parts:
            should_llm_check = True

    if should_llm_check:
        prompt = f"""判断以下两个场景是否应该合并为同一个场景（因chunk边界拆分导致）。

场景A (chunk {sorted_chunk_ids[i-1]}末尾):
- 地点: {prev_tail.get('location', '')}
- 参与者: {', '.join(prev_tail.get('participants', []))}
- 摘要: {prev_tail.get('summary', '')}

场景B (chunk {ci}开头):
- 地点: {curr_head.get('location', '')}
- 参与者: {', '.join(curr_head.get('participants', []))}
- 摘要: {curr_head.get('summary', '')}

仅输出JSON: {{"should_merge": true/false, "reason": "简述原因"}}"""
        try:
            resp = call_llm(prompt)
            data = parse_json(resp)
            if data.get('should_merge'):
                # Merge: append curr_head's narrative units to prev_tail
                prev_units = prev_tail.get('narrative_units', [])
                curr_units = curr_head.get('narrative_units', [])
                # Re-index sequence numbers
                offset = len(prev_units)
                for u in curr_units:
                    u['sequence_index'] = u.get('sequence_index', 0) + offset
                prev_tail['narrative_units'] = prev_units + curr_units
                prev_tail['participants'] = list(set(prev_tail.get('participants', []) + curr_head.get('participants', [])))
                if curr_head.get('summary'):
                    prev_tail['summary'] = (prev_tail.get('summary', '') or '') + '; ' + curr_head.get('summary', '')
                # Skip curr_head, add remaining scenes from this chunk
                merged_scenes.extend(curr_scenes[1:])
                merge_count += 1
                print(f"  Merged {curr_head.get('scene_id', '?')} -> {prev_tail.get('scene_id', '?')}")
                continue
        except Exception:
            pass  # Skip merge if LLM fails

    # No merge: add all scenes from this chunk
    merged_scenes.extend(curr_scenes)

print(f"  Scenes after merge: {len(merged_scenes)} ({merge_count} merged)")

# Extract dialogues from scene narrative units for backward compatibility
all_extracted_dialogues = []
for s in merged_scenes:
    chapter = s.get('chapter', 0)
    for u in s.get('narrative_units', []):
        if u.get('type') in ('dialogue', 'inner_thought'):
            all_extracted_dialogues.append({
                'speaker': u.get('character', 'unknown'),
                'listener': u.get('listener'),
                'content': u.get('text', ''),
                'context': s.get('summary', ''),
                'chapter': chapter,
                'scene_id': s.get('scene_id', ''),
            })

print(f"  Dialogues extracted from scenes: {len(all_extracted_dialogues)}")

# ─── Stage 5: Coreference ────────────────────────────────────────
print("\nStage 5: Coreference...")
all_chars = [c for c in all_chars if c and isinstance(c, dict)]
resolved_chars = resolve_aliases(all_chars)

name_map = {}
for c in resolved_chars:
    canonical = c['canonical_name']
    name_map[canonical] = canonical
    for alias in c.get('aliases', []):
        if alias not in name_map: name_map[alias] = canonical

def map_name(name):
    return name_map.get(name, name)

# Map relationships (normalized dedup: sorted key for lookup, prev a/b for direction detection)
mapped_rels, seen_keys = [], {}
for r in all_relationships:
    a, b = map_name(r.get('character_a','')), map_name(r.get('character_b',''))
    if not a or not b or a == b: continue
    norm_a, norm_b = sorted([a, b])
    key = (norm_a, norm_b, r.get('rel_type',''))
    if key in seen_keys:
        prev = seen_keys[key]
        # Check reversal against prev's actual a/b, not against sorted order
        if a == prev['character_a'] and b == prev['character_b']:
            # Same direction: merge directly
            for name in r.get('a_calls_b', []):
                if name not in prev.setdefault('a_calls_b', []): prev['a_calls_b'].append(name)
            for name in r.get('b_calls_a', []):
                if name not in prev.setdefault('b_calls_a', []): prev['b_calls_a'].append(name)
        else:
            # Reversed direction: swap a_calls_b ↔ b_calls_a
            for name in r.get('a_calls_b', []):
                if name not in prev.setdefault('b_calls_a', []): prev['b_calls_a'].append(name)
            for name in r.get('b_calls_a', []):
                if name not in prev.setdefault('a_calls_b', []): prev['a_calls_b'].append(name)
        # Take higher confidence
        if r.get('confidence', 0) > prev.get('confidence', 0):
            prev['confidence'] = r['confidence']
        continue
    r_copy = dict(r); r_copy['character_a'] = a; r_copy['character_b'] = b
    seen_keys[key] = r_copy
    mapped_rels.append(r_copy)

# Map dialogues (from scene narrative units)
mapped_dialogues = []
for d in all_extracted_dialogues:
    d = dict(d)
    d['speaker'] = map_name(d.get('speaker') or 'unknown')
    d['listener'] = map_name(d['listener']) if d.get('listener') else None
    mapped_dialogues.append(d)

# ─── Stage 6: Graph ──────────────────────────────────────────────
g = RelationshipGraph()
for r in mapped_rels:
    g.add_relationship(r['character_a'], r['character_b'], rel_type=r.get('rel_type','unknown'), direction=r.get('direction','bidirectional'), evidence=r.get('evidence',[]), confidence=r.get('confidence',0.5))

# ─── Build NovelWorld ───────────────────────────────────────────
characters = [Character(canonical_name=c['canonical_name'], aliases=c.get('aliases',[]), relational_labels=_normalize_rlabels(c.get('relational_labels',[])), gender=c.get('gender'), age_range=c.get('age_range'), appearance=c.get('appearance'), personality=c.get('personality'), role=c.get('role')) for c in resolved_chars]

relationships = [Relationship(character_a=r['character_a'], character_b=r['character_b'], rel_type=r['rel_type'], direction=r.get('direction','bidirectional'), intimacy_level=r.get('intimacy_level'), a_calls_b=r.get('a_calls_b',[]), b_calls_a=r.get('b_calls_a',[]), evidence=r.get('evidence',[]), confidence=r.get('confidence',0.5)) for r in mapped_rels]

dialogues = [Dialogue(speaker=d['speaker'], listener=d.get('listener'), content=d['content'], context=d.get('context'), chapter=d.get('chapter')) for d in mapped_dialogues]

ws_model = None
if world_settings:
    ws = world_settings[0]
    locations = [Location(**l) for l in ws.get('locations',[])]
    items = [Item(**it) for it in ws.get('items',[])]
    ws_model = WorldSetting(era=ws.get('era'), genre=ws.get('genre'), summary=ws.get('setting_summary'), special_rules=ws.get('special_rules',[]), key_themes=ws.get('key_themes',[]))
else:
    locations, items = [], []

# Build Scene objects with mapped names
mapped_scene_objects = []
for s in merged_scenes:
    mapped_units = []
    for u in s.get('narrative_units', []):
        mapped_units.append(NarrativeUnit(
            unit_id=u.get('unit_id', ''),
            character=map_name(u.get('character', 'narrator')),
            text=u.get('text', ''),
            type=u.get('type', 'narration'),
            listener=map_name(u['listener']) if u.get('listener') else None,
            sequence_index=u.get('sequence_index', 0),
        ))
    mapped_scene_objects.append(Scene(
        scene_id=s.get('scene_id', ''),
        chapter=s.get('chapter', 0),
        location=s.get('location', ''),
        sub_location_of=s.get('sub_location_of'),
        participants=[map_name(p) for p in s.get('participants', [])],
        narrative_units=mapped_units,
        summary=s.get('summary'),
        time_marker=s.get('time_marker'),
        chunk_index=s.get('chunk_index'),
    ))

# Build scene_events for backward compatibility (from scenes)
mapped_scene_events = []
for s in merged_scenes:
    scene_dlgs = []
    for u in s.get('narrative_units', []):
        if u.get('type') in ('dialogue', 'inner_thought'):
            scene_dlgs.append(Dialogue(
                speaker=map_name(u.get('character', 'unknown')),
                listener=map_name(u['listener']) if u.get('listener') else None,
                content=u.get('text', ''),
                context=s.get('summary', ''),
                chapter=s.get('chapter'),
            ))
    mapped_scene_events.append(SceneEvent(
        event_id=s.get('scene_id', ''),
        description=s.get('summary', ''),
        chapter=s.get('chapter', 0),
        location=s.get('location', ''),
        participants=[map_name(p) for p in s.get('participants', [])],
        dialogues=scene_dlgs,
    ))

nw = NovelWorld(
    metadata=result.metadata,
    characters=characters,
    relationships=relationships,
    dialogues=dialogues,
    world_setting=ws_model,
    locations=locations,
    items=items,
    events=[],
    scene_events=mapped_scene_events,
    scenes=mapped_scene_objects,
)

# ─── Stage 7: Export ─────────────────────────────────────────────
print("\nStage 7: Exporting...")
config.ensure_dirs()

# JSON
json_path = export_to_json(nw)
print(f"  JSON: {json_path} ({json_path.stat().st_size} bytes)")

# SQLite
db_path = export_to_sqlite(nw)
print(f"  SQLite: {db_path}")

# Character cards
card_paths = export_character_cards(nw)
print(f"  Cards: {len(card_paths)} generated")

# Also save raw intermediate data
raw_dir = OUTPUT_DIR / "raw" / result.metadata.novel_id
raw_dir.mkdir(parents=True, exist_ok=True)
(raw_dir / "all_chars.json").write_text(json.dumps(all_chars, ensure_ascii=False, indent=2))
(raw_dir / "resolved_chars.json").write_text(json.dumps(resolved_chars, ensure_ascii=False, indent=2))
(raw_dir / "relationships_raw.json").write_text(json.dumps(all_relationships, ensure_ascii=False, indent=2))
(raw_dir / "relationships_mapped.json").write_text(json.dumps(mapped_rels, ensure_ascii=False, indent=2))
(raw_dir / "dialogues.json").write_text(json.dumps(mapped_dialogues, ensure_ascii=False, indent=2))
(raw_dir / "events.json").write_text(json.dumps([s.get('summary', '') for s in merged_scenes], ensure_ascii=False, indent=2))
(raw_dir / "world.json").write_text(json.dumps(world_settings, ensure_ascii=False, indent=2))
(raw_dir / "graph.json").write_text(json.dumps(g.to_dict(), ensure_ascii=False, indent=2))
(raw_dir / "name_map.json").write_text(json.dumps(name_map, ensure_ascii=False, indent=2))
(raw_dir / "scene_events.json").write_text(json.dumps([se.model_dump() for se in mapped_scene_events], ensure_ascii=False, indent=2))
(raw_dir / "scenes.json").write_text(json.dumps([s.model_dump() for s in mapped_scene_objects], ensure_ascii=False, indent=2))
(raw_dir / "merged_scenes_raw.json").write_text(json.dumps(merged_scenes, ensure_ascii=False, indent=2))
print(f"  Raw data: {raw_dir}")

print(f"\nDone! Files in {OUTPUT_DIR}")
