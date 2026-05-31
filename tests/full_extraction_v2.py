"""Full pipeline test v2 - robust version with all extractors."""
import sys, json, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from anthropic.types import TextBlock
from src.novels2llm.pipeline.stage1_preprocess import preprocess_novel
from src.novels2llm.pipeline.stage2_nlp import process_chapter, _extract_dialogues
from src.novels2llm.chunking.smart_chunker import chunk_novel
from src.novels2llm.extractors.character_extractor import CharacterExtractor
from src.novels2llm.extractors.world_extractor import WorldExtractor
from src.novels2llm.extractors.relationship_extractor import RelationshipExtractor
from src.novels2llm.extractors.event_extractor import EventExtractor
from src.novels2llm.coreference.alias_resolver import resolve_aliases
from src.novels2llm.coreference.entity_merger import merge_character_entries
from src.novels2llm.graph.relationship_graph import RelationshipGraph
from src.novels2llm.config import config

API_KEY = config.ANTHROPIC_API_KEY
N_CHUNKS = 3
NOVEL_FILE = Path('data/jia_ting_luan_lun/qing-chun-yun-shi.md')

# ─── Stage 1 + 2 ────────────────────────────────────────────────
print("=" * 70)
print("STAGE 1-2: Preprocess, Chunk, NLP")
print("=" * 70)

result = preprocess_novel(NOVEL_FILE)
all_chunks = chunk_novel(result.raw_text, result.metadata.novel_id)
test_chunks = all_chunks[:N_CHUNKS]

print(f"Novel: {result.metadata.title} ({result.metadata.word_count} words)")
print(f"Chunks: {len(all_chunks)} total, testing {N_CHUNKS}")

nlp_map = {}
for c in test_chunks:
    nlp = process_chapter(c.text, result.metadata.novel_id, c.chapter, use_hanlp=False)
    nlp_map[c.chunk_index] = nlp
    print(f"  Chunk {c.chunk_index}: {len(nlp.tokens)} tokens, "
          f"{len(nlp.entities)} entities, {len(nlp.dialogues)} NLP dialogues")

# ─── Helper: call DeepSeek ───────────────────────────────────────
client = Anthropic(api_key=API_KEY, base_url=config.ANTHROPIC_BASE_URL)

def call_deepseek(prompt: str) -> str:
    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if isinstance(block, TextBlock):
            return block.text
    return ""

def parse_json(text: str) -> dict:
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        return json.loads(m.group(0))
    return json.loads(text)

# ─── Stage 3: Extractions ────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 3: LLM Extraction")
print("=" * 70)

char_ext = CharacterExtractor(API_KEY)
world_ext = WorldExtractor(API_KEY)
rel_ext = RelationshipExtractor(API_KEY)
event_ext = EventExtractor(API_KEY)

all_chars = []
all_dialogues = []   # From NLP, with LLM speaker attribution
all_relationships = []
all_events = []
world_settings = []

for ci, c in enumerate(test_chunks):
    hint = nlp_map[c.chunk_index].to_hint_dict()
    print(f"\n{'─'*50}")
    print(f"Chunk {c.chunk_index} (ch{c.chapter}, {c.char_count} chars)")
    print(f"{'─'*50}")

    # ── Character ──
    try:
        chars = char_ext.extract(c.text, nlp_hints=hint)
        all_chars.extend(chars)
        print(f"  [CHAR]  {len(chars)} found")
    except Exception as e:
        print(f"  [CHAR]  ERROR: {e}")

    # ── World (chunk 0 only) ──
    if ci == 0:
        try:
            ws = world_ext.extract(c.text, nlp_hints=hint)
            world_settings.append(ws)
            print(f"  [WORLD] era={ws.get('era')}, genre={ws.get('genre')}, "
                  f"{len(ws.get('locations',[]))} locations, {len(ws.get('items',[]))} items")
        except Exception as e:
            print(f"  [WORLD] ERROR: {e}")

    # ── Dialogue (NLP-based + LLM speaker attribution) ──
    try:
        nlp_dlgs = nlp_map[c.chunk_index].dialogues
        print(f"  [DIAL]  {len(nlp_dlgs)} NLP spans, doing LLM speaker attribution...")

        if nlp_dlgs:
            # Batch dialogues: take 10 at a time with context
            batch_size = 10
            for batch_start in range(0, min(len(nlp_dlgs), 30), batch_size):
                batch = nlp_dlgs[batch_start:batch_start + batch_size]
                dlg_entries = []
                for i, d in enumerate(batch):
                    ctx = d.context_before[-60:] if d.context_before else ""
                    dlg_entries.append(f"[{i+1}] context: ...{ctx}\"\n   dialogue: \"{d.text[:100]}\"")

                attribution_prompt = f"""你是一个对话归属分析专家。以下是小说片段中检测到的对话，请推测每个对话的说话人。

已知角色: {', '.join(hint.get('entities', [])[:15])}

对话列表:
{chr(10).join(dlg_entries)}

请为每个对话推测说话人，输出JSON:
{{"attributions": [{{"dialogue_index": 1, "speaker": "...", "listener": "...", "confidence": 0.8}}]}}
只输出JSON。"""

                try:
                    resp = call_deepseek(attribution_prompt)
                    data = parse_json(resp)
                    for attr in data.get('attributions', []):
                        idx = attr.get('dialogue_index', 0) - 1
                        if 0 <= idx < len(batch):
                            all_dialogues.append({
                                'speaker': attr.get('speaker', 'unknown'),
                                'listener': attr.get('listener'),
                                'content': batch[idx].text,
                                'context': batch[idx].context_before[-80:],
                                'chapter': c.chapter,
                                'confidence': attr.get('confidence', 0.5),
                            })
                except Exception as e2:
                    # Fallback: just use rule-based speaker
                    for d in batch:
                        all_dialogues.append({
                            'speaker': d.speaker or 'unknown',
                            'listener': None,
                            'content': d.text,
                            'context': d.context_before[-80:],
                            'chapter': c.chapter,
                        })

            # Remaining beyond 30 use rule-based speaker
            for d in nlp_dlgs[30:]:
                all_dialogues.append({
                    'speaker': d.speaker or 'unknown',
                    'listener': None,
                    'content': d.text,
                    'context': d.context_before[-80:],
                    'chapter': c.chapter,
                })
            print(f"  [DIAL]  {len(all_dialogues)} total so far")
    except Exception as e:
        print(f"  [DIAL]  ERROR: {e}")

    # ── Relationship (2 attempts) ──
    try:
        rels = rel_ext.extract(c.text, known_characters=all_chars)
        for r in rels:
            r['source_chapter'] = c.chapter
        all_relationships.extend(rels)
        print(f"  [REL]   {len(rels)} found")
    except Exception as e:
        print(f"  [REL]   Attempt 1 ERROR, retrying...")
        try:
            rels = rel_ext.extract(c.text, known_characters=all_chars)
            for r in rels:
                r['source_chapter'] = c.chapter
            all_relationships.extend(rels)
            print(f"  [REL]   {len(rels)} found (retry)")
        except Exception as e2:
            print(f"  [REL]   ERROR: {e2}")

    # ── Event (2 attempts) ──
    try:
        evts = event_ext.extract(c.text, chapter=c.chapter, chapter_title=c.chapter_title)
        all_events.extend(evts)
        print(f"  [EVENT] {len(evts)} found")
    except Exception as e:
        print(f"  [EVENT] Attempt 1 ERROR, retrying...")
        try:
            evts = event_ext.extract(c.text, chapter=c.chapter, chapter_title=c.chapter_title)
            all_events.extend(evts)
            print(f"  [EVENT] {len(evts)} found (retry)")
        except Exception as e2:
            print(f"  [EVENT] ERROR: {e2}")

# ─── Stage 4: Coreference ────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 4: Coreference Resolution")
print("=" * 70)

all_chars = [c for c in all_chars if c and isinstance(c, dict)]
print(f"Raw characters: {len(all_chars)}")
resolved_chars = resolve_aliases(all_chars)
print(f"Resolved: {len(resolved_chars)} unique")

# Name map: alias → canonical
name_map = {}
for c in resolved_chars:
    canonical = c['canonical_name']
    name_map[canonical] = canonical
    for alias in c.get('aliases', []):
        if alias not in name_map:
            name_map[alias] = canonical

print("\nCanonical names:")
for c in resolved_chars:
    print(f"  {c['canonical_name']} ← {', '.join(c.get('aliases', [])[:6])}")

# ─── Map relationships ───────────────────────────────────────────
def map_name(name: str) -> str:
    return name_map.get(name, name)

mapped_rels = []
seen_keys = set()
for r in all_relationships:
    a = map_name(r.get('character_a', ''))
    b = map_name(r.get('character_b', ''))
    if not a or not b or a == b:
        continue
    key = (a, b, r.get('rel_type', ''))
    if key in seen_keys:
        continue
    seen_keys.add(key)
    r_copy = dict(r)
    r_copy['character_a'] = a
    r_copy['character_b'] = b
    mapped_rels.append(r_copy)

print(f"\nRelationships: {len(all_relationships)} raw → {len(mapped_rels)} mapped/deduped")

# ─── Stage 5: Graph ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 5: Relationship Graph")
print("=" * 70)

g = RelationshipGraph()
for r in mapped_rels:
    g.add_relationship(
        r['character_a'], r['character_b'],
        rel_type=r.get('rel_type', 'unknown'),
        direction=r.get('direction', 'bidirectional'),
        evidence=r.get('evidence', []),
        confidence=r.get('confidence', 0.5),
    )

rels_list = g.get_relationships()
print(f"Graph: {g.graph.number_of_nodes()} nodes, {len(rels_list)} edges\n")
for r in sorted(rels_list, key=lambda x: x['confidence'], reverse=True):
    ev = r.get('evidence', [])
    ev_short = f' — "{ev[0][:50]}..."' if ev else ''
    print(f"  {r['character_a']} ──[{r['rel_type']}]── {r['character_b']} "
          f"(conf={r['confidence']:.2f}){ev_short}")

# ─── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: Characters")
print("=" * 70)
for c in resolved_chars:
    look = c.get('appearance', '') or ''
    pers = c.get('personality', '') or ''
    print(f"\n  [{c.get('gender','?')}] {c['canonical_name']} ({c.get('role','?')})")
    print(f"    Age: {c.get('age_range','?')}")
    if look: print(f"    Look: {look[:100]}")
    if pers: print(f"    Personality: {pers[:100]}")

# World
if world_settings:
    ws = world_settings[0]
    print("\n" + "=" * 70)
    print("SUMMARY: World")
    print("=" * 70)
    print(f"  Era: {ws.get('era')}  Genre: {ws.get('genre')}")
    print(f"  Summary: {ws.get('setting_summary','')[:200]}")
    for loc in ws.get('locations', [])[:5]:
        print(f"  📍 {loc.get('name')} ({loc.get('type')}) - {loc.get('description','')[:80]}")
    for item in ws.get('items', [])[:5]:
        print(f"  📦 {item.get('name')} ({item.get('type')}) - {item.get('description','')[:80]}")

# Dialogue sample
print("\n" + "=" * 70)
print(f"SUMMARY: Dialogues ({len(all_dialogues)} total)")
print("=" * 70)
for d in all_dialogues[:8]:
    sp = map_name(d.get('speaker', '?'))
    li = map_name(d.get('listener', '')) if d.get('listener') else '?'
    print(f"  [{sp} → {li}] \"{d['content'][:70]}\"")

# Events
print("\n" + "=" * 70)
print(f"SUMMARY: Events ({len(all_events)} total)")
print("=" * 70)
for e in all_events:
    desc = e.get('description', '')[:80]
    parts = ', '.join(e.get('participants', [])[:4])
    print(f"  {e.get('event_id','?')}: {desc}")
    if parts: print(f"    Participants: {parts}")

print("\n✅ Pipeline complete!")
