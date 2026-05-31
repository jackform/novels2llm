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
from src.novels2llm.models.output import NovelWorld, NovelMetadata
from src.novels2llm.models.entities import Character, Location, Item, WorldSetting
from src.novels2llm.models.relationships import Dialogue, Relationship
from src.novels2llm.pipeline.stage7_export import export_to_json, export_to_sqlite, export_character_cards
from src.novels2llm.config import config

API_KEY = config.ANTHROPIC_API_KEY
N_CHUNKS = 3
NOVEL_FILE = Path('data/jia_ting_luan_lun/qing-chun-yun-shi.md')
OUTPUT_DIR = Path('data/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    if m: return json.loads(m.group(1))
    m = re.search(r'\{[\s\S]*\}', text)
    if m: return json.loads(m.group(0))
    return json.loads(text)

# ─── Stage 3: Extract ────────────────────────────────────────────
all_chars, all_dialogues, all_relationships, all_events = [], [], [], []
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

    # Dialogue (NLP + LLM attribution)
    nlp_dlgs = nlp_map[c.chunk_index].dialogues
    batch_size = 10
    for batch_start in range(0, min(len(nlp_dlgs), 30), batch_size):
        batch = nlp_dlgs[batch_start:batch_start + batch_size]
        entries = []
        for i, d in enumerate(batch):
            ctx = d.context_before[-60:] if d.context_before else ""
            entries.append(f"[{i+1}] ctx: ...{ctx}\"  text: \"{d.text[:100]}\"")
        prompt = f"""推测以下小说对话的说话人。已知角色: {', '.join(hint.get('entities', [])[:15])}
{chr(10).join(entries)}
仅输出JSON: {{"attributions": [{{"dialogue_index": 1, "speaker": "...", "listener": "...", "confidence": 0.8}}]}}"""
        try:
            resp = call_llm(prompt)
            data = parse_json(resp)
            for attr in data.get('attributions', []):
                idx = attr.get('dialogue_index', 0) - 1
                if 0 <= idx < len(batch):
                    all_dialogues.append({'speaker': attr.get('speaker','unknown'), 'listener': attr.get('listener'), 'content': batch[idx].text, 'context': batch[idx].context_before[-80:], 'chapter': c.chapter, 'confidence': attr.get('confidence', 0.5)})
        except:
            for d in batch:
                all_dialogues.append({'speaker': d.speaker or 'unknown', 'listener': None, 'content': d.text, 'context': d.context_before[-80:], 'chapter': c.chapter})
    for d in nlp_dlgs[30:]:
        all_dialogues.append({'speaker': d.speaker or 'unknown', 'listener': None, 'content': d.text, 'context': d.context_before[-80:], 'chapter': c.chapter})
    print(f"  Dialogue: {len(all_dialogues)} total")

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

    # Event
    for attempt in range(2):
        try:
            evts = event_ext.extract(c.text, chapter=c.chapter, chapter_title=c.chapter_title)
            all_events.extend(evts)
            print(f"  Event: {len(evts)}" + (" (retry)" if attempt > 0 else ""))
            break
        except Exception as e:
            if attempt == 0: continue
            print(f"  Event: ERROR - {e}")

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

# Map relationships
mapped_rels, seen_keys = [], set()
for r in all_relationships:
    a, b = map_name(r.get('character_a','')), map_name(r.get('character_b',''))
    if not a or not b or a == b: continue
    key = (a, b, r.get('rel_type',''))
    if key in seen_keys: continue
    seen_keys.add(key)
    r_copy = dict(r); r_copy['character_a'] = a; r_copy['character_b'] = b
    mapped_rels.append(r_copy)

# Map dialogues
mapped_dialogues = []
for d in all_dialogues:
    d = dict(d)
    d['speaker'] = map_name(d.get('speaker','unknown'))
    d['listener'] = map_name(d['listener']) if d.get('listener') else None
    mapped_dialogues.append(d)

# ─── Stage 5: Graph ──────────────────────────────────────────────
g = RelationshipGraph()
for r in mapped_rels:
    g.add_relationship(r['character_a'], r['character_b'], rel_type=r.get('rel_type','unknown'), direction=r.get('direction','bidirectional'), evidence=r.get('evidence',[]), confidence=r.get('confidence',0.5))

# ─── Build NovelWorld ───────────────────────────────────────────
characters = [Character(canonical_name=c['canonical_name'], aliases=c.get('aliases',[]), gender=c.get('gender'), age_range=c.get('age_range'), appearance=c.get('appearance'), personality=c.get('personality'), role=c.get('role')) for c in resolved_chars]

relationships = [Relationship(character_a=r['character_a'], character_b=r['character_b'], rel_type=r['rel_type'], direction=r.get('direction','bidirectional'), intimacy_level=r.get('intimacy_level'), evidence=r.get('evidence',[]), confidence=r.get('confidence',0.5)) for r in mapped_rels]

dialogues = [Dialogue(speaker=d['speaker'], listener=d.get('listener'), content=d['content'], context=d.get('context'), chapter=d.get('chapter')) for d in mapped_dialogues]

ws_model = None
if world_settings:
    ws = world_settings[0]
    locations = [Location(**l) for l in ws.get('locations',[])]
    items = [Item(**it) for it in ws.get('items',[])]
    ws_model = WorldSetting(era=ws.get('era'), genre=ws.get('genre'), summary=ws.get('setting_summary'), special_rules=ws.get('special_rules',[]), key_themes=ws.get('key_themes',[]))
else:
    locations, items = [], []

nw = NovelWorld(metadata=result.metadata, characters=characters, relationships=relationships, dialogues=dialogues, world_setting=ws_model, locations=locations, items=items, events=[])

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
(raw_dir / "events.json").write_text(json.dumps(all_events, ensure_ascii=False, indent=2))
(raw_dir / "world.json").write_text(json.dumps(world_settings, ensure_ascii=False, indent=2))
(raw_dir / "graph.json").write_text(json.dumps(g.to_dict(), ensure_ascii=False, indent=2))
(raw_dir / "name_map.json").write_text(json.dumps(name_map, ensure_ascii=False, indent=2))
print(f"  Raw data: {raw_dir}")

print(f"\nDone! Files in {OUTPUT_DIR}")
