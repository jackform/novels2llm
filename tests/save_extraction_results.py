"""Re-run extraction on 3 chunks and save ALL results to disk."""
import sys, json, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from anthropic.types import TextBlock
from src.novels2llm.pipeline.stage1_preprocess import preprocess_novel
from src.novels2llm.pipeline.stage2_nlp import process_chapter
from src.novels2llm.chunking.smart_chunker import chunk_novel
from src.novels2llm.extractors.character_extractor import CharacterExtractor
from src.novels2llm.extractors.world_extractor import WorldExtractor
from src.novels2llm.extractors.relationship_extractor import RelationshipExtractor
from src.novels2llm.extractors.event_extractor import EventExtractor
from src.novels2llm.coreference.alias_resolver import resolve_aliases
from src.novels2llm.graph.relationship_graph import RelationshipGraph
from src.novels2llm.models.output import NovelWorld, NovelMetadata, SceneEvent
from src.novels2llm.models.entities import Character, Location, Item, WorldSetting
from src.novels2llm.models.relationships import Dialogue, Relationship
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
all_chunks = chunk_novel(result.raw_text, result.metadata.novel_id)
test_chunks = all_chunks[:N_CHUNKS]

nlp_map = {}
for c in test_chunks:
    nlp = process_chapter(c.text, result.metadata.novel_id, c.chapter, use_hanlp=False)
    nlp_map[c.chunk_index] = nlp

# ─── Setup ───────────────────────────────────────────────────────
char_ext = CharacterExtractor(API_KEY)
world_ext = WorldExtractor(API_KEY)
rel_ext = RelationshipExtractor(API_KEY)
event_ext = EventExtractor(API_KEY)
client = Anthropic(api_key=API_KEY, base_url=config.ANTHROPIC_BASE_URL)

def call_llm(prompt):
    msg = client.messages.create(model=config.ANTHROPIC_MODEL, max_tokens=4096, messages=[{"role": "user", "content": prompt}])
    for block in msg.content:
        if isinstance(block, TextBlock): return block.text
    return ""

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
# Order: characters, world, events (with text positions first), relationships
all_chars, all_relationships, all_events = [], [], []
all_nlp_dialogues = []  # (chunk, DialogueSpan) for position-based mapping
world_settings = []

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

    # Collect NLP dialogues for later position-based mapping
    for d in nlp_map[c.chunk_index].dialogues:
        all_nlp_dialogues.append((c, d))

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

    # Event (extracted with text_start/text_end for dialogue mapping)
    for attempt in range(2):
        try:
            evts = event_ext.extract(c.text, chapter=c.chapter, chapter_title=c.chapter_title)
            for e in evts:
                e['chunk_index'] = c.chunk_index
                e['chunk_text'] = c.text
                e['chapter'] = c.chapter
            all_events.extend(evts)
            print(f"  Event: {len(evts)}" + (" (retry)" if attempt > 0 else ""))
            break
        except Exception as e:
            if attempt == 0: continue
            print(f"  Event: ERROR - {e}")

print(f"\nNLP dialogues total: {len(all_nlp_dialogues)}, Events total: {len(all_events)}")

# ─── Stage 3.4: Deduplicate Events Across Chunk Boundaries ──────
print("\nStage 3.4: Deduplicating events across chunk boundaries...")

# Step 1: Assign unique IDs incorporating chunk_index
for e in all_events:
    ci = e.get('chunk_index', 0)
    orig_id = e.get('event_id', '')
    e['event_id'] = f"{orig_id}_c{ci}"

# Step 2: Group events by chapter, then merge boundary-spanning events
by_chapter = {}
for e in all_events:
    ch = e.get('chapter', 0)
    by_chapter.setdefault(ch, []).append(e)

deduped_events = []
for ch, evts in by_chapter.items():
    by_chunk = {}
    for e in evts:
        by_chunk.setdefault(e['chunk_index'], []).append(e)

    sorted_chunks = sorted(by_chunk.keys())
    merged_ids = set()

    for ci in sorted_chunks:
        for e in sorted(by_chunk[ci], key=lambda x: x.get('chapter_order', 0)):
            if e['event_id'] in merged_ids:
                continue
            deduped_events.append(e)

    # Check adjacent chunks for boundary-crossing events
    for i in range(len(sorted_chunks) - 1):
        ci_curr = sorted_chunks[i]
        ci_next = sorted_chunks[i + 1]
        evts_curr = sorted(by_chunk[ci_curr], key=lambda x: x.get('chapter_order', 0))
        evts_next = sorted(by_chunk[ci_next], key=lambda x: x.get('chapter_order', 0))
        if not evts_curr or not evts_next:
            continue

        chunk_text_len = len(evts_curr[0].get('chunk_text', ''))

        # Events at the end of current chunk and start of next chunk
        curr_tail = [e for e in evts_curr if e.get('text_end') is not None and e['text_end'] > chunk_text_len - 500]
        next_head = [e for e in evts_next if e.get('text_start') is not None and e['text_start'] < 500]

        for ct in curr_tail:
            for nh in next_head:
                ct_parts = set(ct.get('participants', []))
                nh_parts = set(nh.get('participants', []))
                if not (ct_parts & nh_parts):
                    continue  # No shared participant, skip

                # LLM judge: are these the same event?
                prompt = f"""判断以下两个事件描述是否指向同一个事件（仅跨chunk边界时的同一个事件，因文本被拆分导致）。

事件A (chunk {ci_curr}末尾): {ct.get('description', '')}
事件B (chunk {ci_next}开头): {nh.get('description', '')}

仅输出JSON: {{"is_same_event": true/false}}"""
                try:
                    resp = call_llm(prompt)
                    data = parse_json(resp)
                    if data.get('is_same_event'):
                        ct['text_end'] = nh.get('text_end')
                        ct['participants'] = list(ct_parts | nh_parts)
                        merged_ids.add(nh['event_id'])
                        print(f"  Merged {nh['event_id']} -> {ct['event_id']} (LLM)")
                except Exception:
                    pass  # Skip merge if LLM fails

    # Remove merged events
    deduped_events = [e for e in deduped_events if e['event_id'] not in merged_ids]

print(f"  Events after dedup: {len(deduped_events)}")

# Step 3: Fill small gaps between events within each chunk
# Only fill gaps < 500 chars — don't stretch last event to chunk end
MAX_GAP = 500
for e in deduped_events:
    chunk_text = e.get('chunk_text', '')
    by_chunk_gap = {}
    for ev in deduped_events:
        by_chunk_gap.setdefault(ev['chunk_index'], []).append(ev)

for ci, evts in by_chunk_gap.items():
    evts_sorted = sorted([e for e in evts if e.get('text_start') is not None], key=lambda x: x['text_start'])
    if not evts_sorted:
        continue
    # Fill small gaps between adjacent events
    for i in range(len(evts_sorted) - 1):
        next_start = evts_sorted[i + 1]['text_start']
        current_end = evts_sorted[i].get('text_end') or 0
        gap = next_start - current_end
        if 0 < gap < MAX_GAP:
            evts_sorted[i]['text_end'] = next_start
    # Fill any remaining gaps (events without text_start get default)
    for ev in evts:
        if ev.get('text_start') is None or ev.get('text_end') is None:
            ev['text_start'] = 0
            ev['text_end'] = 0

# ─── Stage 3.5: Map NLP Dialogues to Events by Position ──────────
print("\nStage 3.5: Mapping dialogues to events by text position...")
event_dialogue_map = {}  # (chunk_index, event_id) -> list of (chunk, DialogueSpan)
unmapped_dialogues = []

for chunk, dlg in all_nlp_dialogues:
    matched = False
    for evt in deduped_events:
        if evt.get('chunk_index') != chunk.chunk_index:
            continue
        ts = evt.get('text_start')
        te = evt.get('text_end')
        if ts is not None and te is not None and ts <= dlg.start <= te:
            eid = evt.get('event_id', '')
            key = (evt.get('chunk_index'), eid)
            event_dialogue_map.setdefault(key, []).append((chunk, dlg))
            matched = True
            break
    if not matched:
        unmapped_dialogues.append((chunk, dlg))

for eid, dlgs in event_dialogue_map.items():
    print(f"  {eid}: {len(dlgs)} dialogue(s)")
print(f"  Unmapped dialogues: {len(unmapped_dialogues)}")

# ─── Stage 3.6: LLM Validate/Clean/Attribute Dialogues per Event ─
print("\nStage 3.6: LLM dialogue validation per event...")
all_validated_dialogues = []
scene_events = []
# Build full character info map: name -> rich info string
_char_map = {}
for c in all_chars:
    if not c or not isinstance(c, dict):
        continue
    name = c.get('canonical_name', '')
    if not name:
        continue
    parts = [name]
    aliases = c.get('aliases', [])
    if aliases:
        parts.append(f"别名:{'/'.join(aliases[:5])}")
    labels = c.get('relational_labels', [])
    if labels:
        caller_map = {}
        for lbl in labels:
            if isinstance(lbl, dict):
                caller = lbl.get('caller', 'unknown')
                label = lbl.get('label', '')
                if label:
                    caller_map.setdefault(caller, []).append(label)
        label_strs = [f"{caller}→{'/'.join(ls[:3])}" for caller, ls in list(caller_map.items())[:4]]
        if label_strs:
            parts.append(f"称呼:{'; '.join(label_strs)}")
    _char_map[name] = ' | '.join(parts)


def build_char_context(participants, char_map, max_other=8):
    """Build character context: participants first (full info), then other characters (name only)."""
    lines = []
    parts_set = set(participants)
    # Participants with full info
    for p in participants:
        if p in char_map:
            lines.append(char_map[p])
        else:
            lines.append(p)
    # Other characters (name only, limited)
    others = [k for k in char_map if k not in parts_set]
    if others:
        lines.append(f"其他角色: {', '.join(others[:max_other])}")
    return '\n'.join(lines)


for evt in deduped_events:
    eid = evt.get('event_id', '')
    key = (evt.get('chunk_index'), eid)
    candidates = event_dialogue_map.get(key, [])
    chapter = evt.get('chapter', evt.get('chunk_index', 0))

    if not candidates:
        scene_events.append({
            'event_id': eid,
            'description': evt.get('description', ''),
            'chapter': chapter,
            'location': evt.get('location', ''),
            'participants': evt.get('participants', []),
            'dialogues': [],
        })
        continue

    # Build candidate entries for the prompt
    candidate_entries = []
    for i, (chunk, dlg) in enumerate(candidates):
        ctx = dlg.context_before[-60:] if dlg.context_before else ""
        candidate_entries.append(f"[{i+1}] ctx: ...{ctx}\"  text: \"{dlg.text[:100]}\"")

    # Process in batches of 10 to avoid JSON parse failures on large outputs
    BATCH_SIZE = 8
    all_validated_for_event = []
    total_thoughts = 0
    total_splits = 0
    batch_errors = 0

    for batch_start in range(0, len(candidate_entries), BATCH_SIZE):
        batch_entries = candidate_entries[batch_start:batch_start + BATCH_SIZE]
        batch_candidates = candidates[batch_start:batch_start + BATCH_SIZE]

        prompt = f"""你是小说对话分析专家。以下是一个事件及其候选对话列表。

## 当前事件
- 事件ID：{eid}
- 事件描述：{evt.get('description', '')}
- 章节：第{chapter}章

## 已知角色（含别名和称呼关系，按事件参与者排序）
{build_char_context(evt.get('participants', []), _char_map)}

## 候选对话（NLP自动提取，可能有误）
{chr(10).join(batch_entries)}

## 任务
1. 判断每条候选是真实对话还是内心独白（内心独白标记 is_inner_thought: true）
2. 检查是否有合并错误（一条记录包含多人对话），有则拆分
3. 标注每条真实对话的 speaker 和 listener（speaker 必须从已知角色中选择，不要使用描述性称呼如"小混混们"、"领头的男人"等）
4. 仅输出 JSON

## 判断内心独白的线索
- 上下文有"心想"、"暗想"、"心里嘀咕"、"心中暗道"等标记
- 内容明显是角色内心活动，而非对他人说的话

## 输出格式
仅输出JSON: {{"dialogues": [{{"dialogue_index": {batch_start + 1}, "speaker": "...", "listener": null, "content": "...", "context": "...", "is_inner_thought": false, "is_split": false, "original_index": null}}]}}"""

        try:
            resp = call_llm(prompt)
            data = parse_json(resp)
            batch_validated = []
            for d in data.get('dialogues', []):
                if d.get('is_inner_thought'):
                    total_thoughts += 1
                    continue
                if d.get('is_split'):
                    total_splits += 1
                batch_validated.append({
                    'speaker': d.get('speaker') or 'unknown',
                    'listener': d.get('listener'),
                    'content': d.get('content', ''),
                    'context': d.get('context', ''),
                    'chapter': chapter,
                    'event_id': eid,
                })
            all_validated_for_event.extend(batch_validated)
            all_validated_dialogues.extend(batch_validated)
        except Exception:
            batch_errors += 1
            # Fallback: use NLP dialogues as-is for this batch
            for chunk, dlg in batch_candidates:
                d = {'speaker': dlg.speaker or 'unknown', 'listener': None,
                     'content': dlg.text, 'context': dlg.context_before[-80:],
                     'chapter': chapter, 'event_id': eid}
                all_validated_for_event.append(d)
                all_validated_dialogues.append(d)

    scene_events.append({
        'event_id': eid,
        'description': evt.get('description', ''),
        'chapter': chapter,
        'location': evt.get('location', ''),
        'participants': evt.get('participants', []),
        'dialogues': all_validated_for_event,
    })
    total_batches = len(range(0, len(candidate_entries), BATCH_SIZE))
    if batch_errors:
        print(f"  {eid}: {len(candidates)} candidates -> {len(all_validated_for_event)} dialogues ({batch_errors}/{total_batches} batches failed)")
    else:
        print(f"  {eid}: {len(candidates)} candidates -> {len(all_validated_for_event)} dialogues (filtered {total_thoughts} thoughts, {total_splits} splits)")

# Handle unmapped dialogues: attach to a synthetic "unmapped" event per chapter
if unmapped_dialogues:
    by_chapter = {}
    for chunk, dlg in unmapped_dialogues:
        ch = chunk.chapter
        if ch not in by_chapter:
            by_chapter[ch] = []
        by_chapter[ch].append(dlg)

    for ch, dlgs in by_chapter.items():
        fallback = []
        for dlg in dlgs:
            d = {'speaker': dlg.speaker or 'unknown', 'listener': None,
                 'content': dlg.text, 'context': dlg.context_before[-80:],
                 'chapter': ch, 'event_id': f'ch{ch}_unmapped'}
            fallback.append(d)
            all_validated_dialogues.append(d)
        scene_events.append({
            'event_id': f'ch{ch}_unmapped',
            'description': f'第{ch}章未归入事件的对话',
            'chapter': ch,
            'participants': [],
            'dialogues': fallback,
        })
        print(f"  ch{ch}_unmapped: {len(dlgs)} unmapped dialogues")

# ─── Stage 4: Coreference ────────────────────────────────────────
print("\nStage 4: Coreference...")
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

# Map dialogues (from validated event-linked dialogues)
mapped_dialogues = []
for d in all_validated_dialogues:
    d = dict(d)
    d['speaker'] = map_name(d.get('speaker') or 'unknown')
    d['listener'] = map_name(d['listener']) if d.get('listener') else None
    mapped_dialogues.append(d)

# ─── Stage 5: Graph ──────────────────────────────────────────────
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

# Build scene events with mapped names
mapped_scene_events = []
for se in scene_events:
    mapped_dlgs = []
    for d in se['dialogues']:
        mapped_dlgs.append(Dialogue(
            speaker=map_name(d.get('speaker') or 'unknown'),
            listener=map_name(d['listener']) if d.get('listener') else None,
            content=d.get('content', ''),
            context=d.get('context'),
            chapter=d.get('chapter'),
        ))
    mapped_scene_events.append(SceneEvent(
        event_id=se['event_id'],
        description=se['description'],
        chapter=se['chapter'],
        location=se.get('location'),
        participants=[map_name(p) for p in se.get('participants', [])],
        dialogues=mapped_dlgs,
    ))

nw = NovelWorld(metadata=result.metadata, characters=characters, relationships=relationships, dialogues=dialogues, world_setting=ws_model, locations=locations, items=items, events=[], scene_events=mapped_scene_events)

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
(raw_dir / "events.json").write_text(json.dumps(deduped_events, ensure_ascii=False, indent=2))
(raw_dir / "world.json").write_text(json.dumps(world_settings, ensure_ascii=False, indent=2))
(raw_dir / "graph.json").write_text(json.dumps(g.to_dict(), ensure_ascii=False, indent=2))
(raw_dir / "name_map.json").write_text(json.dumps(name_map, ensure_ascii=False, indent=2))
(raw_dir / "scene_events.json").write_text(json.dumps([se.model_dump() for se in mapped_scene_events], ensure_ascii=False, indent=2))
(raw_dir / "event_dialogue_map.json").write_text(json.dumps({str(eid): [(d.start, d.text[:80]) for _, d in dlgs] for eid, dlgs in event_dialogue_map.items()}, ensure_ascii=False, indent=2))
print(f"  Raw data: {raw_dir}")

print(f"\nDone! Files in {OUTPUT_DIR}")
