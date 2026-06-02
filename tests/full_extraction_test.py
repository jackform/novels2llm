"""Full pipeline test on first 3 chunks of qing-chun-yun-shi.

Runs ALL extractors (character, world, dialogue, relationship, event),
then cross-chunk coreference + relationship mapping + graph construction.
"""
import sys, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.novels2llm.pipeline.stage1_preprocess import preprocess_novel
from src.novels2llm.pipeline.stage2_nlp import process_chapter
from src.novels2llm.chunking.smart_chunker import chunk_novel
from src.novels2llm.extractors.character_extractor import CharacterExtractor
from src.novels2llm.extractors.world_extractor import WorldExtractor
from src.novels2llm.extractors.dialogue_extractor import DialogueExtractor
from src.novels2llm.extractors.relationship_extractor import RelationshipExtractor
from src.novels2llm.extractors.event_extractor import EventExtractor
from src.novels2llm.coreference.alias_resolver import resolve_aliases
from src.novels2llm.config import config

API_KEY = config.ANTHROPIC_API_KEY
N_CHUNKS = 3
NOVEL_FILE = Path('data/jia_ting_luan_lun/qing-chun-yun-shi.md')

# ─── Stage 1 + 2: Preprocess & Chunk ───────────────────────────
print("=" * 70)
print("STAGE 1-2: Preprocess & Chunk")
print("=" * 70)

result = preprocess_novel(NOVEL_FILE)
chunks = chunk_novel(result.raw_text, result.metadata.novel_id)
test_chunks = chunks[:N_CHUNKS]

print(f"Novel: {result.metadata.title} ({result.metadata.word_count} words)")
print(f"Chunks: {len(chunks)} total, testing {len(test_chunks)}")

# NLP hints per chunk
nlp_map = {}
for c in test_chunks:
    nlp = process_chapter(c.text, result.metadata.novel_id, c.chapter, use_hanlp=False)
    nlp_map[c.chunk_index] = nlp

# ─── Stage 3: ALL Extractors ────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 3: LLM Extraction (character + world + dialogue + rel + event)")
print("=" * 70)

char_ext = CharacterExtractor(API_KEY)
world_ext = WorldExtractor(API_KEY)
dialogue_ext = DialogueExtractor(API_KEY)
rel_ext = RelationshipExtractor(API_KEY)
event_ext = EventExtractor(API_KEY)

all_chars = []        # Raw character dicts from all chunks
all_dialogues = []    # Raw dialogue dicts
all_relationships = []  # Raw relationship dicts
all_events = []       # Raw event dicts
world_settings = []   # World setting dicts

for c in test_chunks:
    hint = nlp_map[c.chunk_index].to_hint_dict()
    print(f"\n--- Chunk {c.chunk_index} (ch{c.chapter}, {c.char_count} chars) ---")

    # Character extraction
    try:
        chars = char_ext.extract(c.text, nlp_hints=hint)
        all_chars.extend(chars)
        print(f"  Characters: {len(chars)}")
    except Exception as e:
        print(f"  Characters: ERROR - {e}")

    # World setting (only first chunk)
    if c.chunk_index == 0:
        try:
            ws = world_ext.extract(c.text, nlp_hints=hint)
            world_settings.append(ws)
            print(f"  World: era={ws.get('era')}, genre={ws.get('genre')}, "
                  f"locations={len(ws.get('locations',[]))}, items={len(ws.get('items',[]))}")
        except Exception as e:
            print(f"  World: ERROR - {e}")

    # Dialogue extraction
    try:
        dlgs = dialogue_ext.extract(c.text, nlp_hints=hint)
        for d in dlgs:
            d['chapter'] = c.chapter
        all_dialogues.extend(dlgs)
        print(f"  Dialogues: {len(dlgs)}")
    except Exception as e:
        print(f"  Dialogues: ERROR - {e}")

    # Relationship extraction (using known chars so far)
    try:
        rels = rel_ext.extract(c.text, known_characters=all_chars)
        for r in rels:
            r['source_chapter'] = c.chapter
        all_relationships.extend(rels)
        print(f"  Relationships: {len(rels)}")
    except Exception as e:
        print(f"  Relationships: ERROR - {e}")

    # Event extraction
    try:
        evts = event_ext.extract(
            c.text, chapter=c.chapter, chapter_title=c.chapter_title
        )
        all_events.extend(evts)
        print(f"  Events: {len(evts)}")
    except Exception as e:
        print(f"  Events: ERROR - {e}")

# ─── Stage 4: Coreference + Name Mapping ────────────────────────
print("\n" + "=" * 70)
print("STAGE 4: Coreference Resolution & Cross-chunk Merging")
print("=" * 70)

print(f"Raw characters: {len(all_chars)}")
resolved_chars = resolve_aliases(all_chars)
print(f"Resolved characters: {len(resolved_chars)}")

# Build canonical name mapping: every alias → canonical name
name_map = {}
for c in resolved_chars:
    canonical = c['canonical_name']
    name_map[canonical] = canonical
    for alias in c.get('aliases', []):
        name_map[alias] = canonical

print("\nCanonical name map:")
for c in resolved_chars:
    aliases_str = ", ".join(c.get('aliases', [])[:8])
    print(f"  {c['canonical_name']} ← [{aliases_str}]")

# ─── Map relationship names to canonical ───────────────────────
print(f"\nRaw relationships: {len(all_relationships)}")

def map_name(name: str) -> str:
    """Map a character name to its canonical form."""
    return name_map.get(name, name)

mapped_rels = []
for r in all_relationships:
    a = map_name(r.get('character_a', ''))
    b = map_name(r.get('character_b', ''))
    if a == b:
        continue  # Skip self-relationships
    r = r.copy()
    r['character_a'] = a
    r['character_b'] = b
    mapped_rels.append(r)

# Deduplicate relationships (same pair + same type, sorted to catch reversed pairs)
seen_rels = {}
unique_rels = []
for r in mapped_rels:
    norm_a, norm_b = sorted([r['character_a'], r['character_b']])
    key = (norm_a, norm_b, r.get('rel_type', ''))
    if key in seen_rels:
        # Merge calling names from reversed duplicate
        prev = seen_rels[key]
        prev_entry = seen_rels[key]
        if r['character_a'] == prev_entry['character_a'] and r['character_b'] == prev_entry['character_b']:
            for name in r.get('a_calls_b', []):
                if name not in prev.get('a_calls_b', []): prev.setdefault('a_calls_b', []).append(name)
            for name in r.get('b_calls_a', []):
                if name not in prev.get('b_calls_a', []): prev.setdefault('b_calls_a', []).append(name)
        else:
            for name in r.get('a_calls_b', []):
                if name not in prev.get('b_calls_a', []): prev.setdefault('b_calls_a', []).append(name)
            for name in r.get('b_calls_a', []):
                if name not in prev.get('a_calls_b', []): prev.setdefault('a_calls_b', []).append(name)
        if r.get('confidence', 0) > prev.get('confidence', 0):
            prev['confidence'] = r['confidence']
        continue
    seen_rels[key] = r
    unique_rels.append(r)

print(f"Mapped & deduplicated relationships: {len(unique_rels)}")

# ─── Map dialogue speakers to canonical ─────────────────────────
mapped_dialogues = []
for d in all_dialogues:
    d = d.copy()
    d['speaker'] = map_name(d.get('speaker', 'unknown'))
    d['listener'] = map_name(d.get('listener', '')) if d.get('listener') else None
    mapped_dialogues.append(d)

# ─── Stage 5: Relationship Graph ────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 5: Relationship Graph")
print("=" * 70)

from src.novels2llm.graph.relationship_graph import RelationshipGraph

g = RelationshipGraph()
for r in unique_rels:
    g.add_relationship(
        r['character_a'], r['character_b'],
        rel_type=r.get('rel_type', 'unknown'),
        evidence=r.get('evidence', []),
        confidence=r.get('confidence', 0.5),
    )

all_rels = g.get_relationships()
print(f"Graph: {g.graph.number_of_nodes()} nodes, {len(all_rels)} edges")
print("\nRelationships:")
for r in sorted(all_rels, key=lambda x: x['confidence'], reverse=True):
    ev = r.get('evidence', [])
    ev_str = f'  evidence: "{ev[0][:60]}..."' if ev else ''
    print(f"  {r['character_a']} ──[{r['rel_type']}]── {r['character_b']} "
          f"(conf={r['confidence']:.1f}){ev_str}")

# ─── Summary Report ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Characters
print(f"\n{'─'*40}")
print("CHARACTERS")
print(f"{'─'*40}")
for c in resolved_chars:
    print(f"\n  [{c.get('gender','?')}] {c['canonical_name']} ({c.get('role','?')})")
    if c.get('aliases'):
        print(f"    Aliases: {', '.join(c['aliases'][:10])}")
    if c.get('age_range'):
        print(f"    Age: {c['age_range']}")
    if c.get('appearance'):
        print(f"    Look: {c['appearance'][:120]}")
    if c.get('personality'):
        print(f"    Personality: {c['personality'][:120]}")

# World
if world_settings:
    ws = world_settings[0]
    print(f"\n{'─'*40}")
    print("WORLD SETTING")
    print(f"{'─'*40}")
    print(f"  Era: {ws.get('era')}")
    print(f"  Genre: {ws.get('genre')}")
    print(f"  Summary: {ws.get('setting_summary','')[:200]}")
    for loc in ws.get('locations', [])[:5]:
        print(f"  Location: {loc.get('name')} ({loc.get('type')})")
    for item in ws.get('items', [])[:5]:
        print(f"  Item: {item.get('name')} ({item.get('type')})")

# Dialogues sample
print(f"\n{'─'*40}")
print(f"DIALOGUES ({len(mapped_dialogues)} total, showing first 10)")
print(f"{'─'*40}")
for d in mapped_dialogues[:10]:
    print(f"  [{d['speaker']} → {d.get('listener','?')}] {d['content'][:80]}")

# Events
print(f"\n{'─'*40}")
print(f"EVENTS ({len(all_events)} total)")
print(f"{'─'*40}")
for e in all_events:
    participants = ', '.join(e.get('participants', [])[:5])
    print(f"  {e.get('event_id','?')}: {e.get('description','')[:100]}")
    if participants:
        print(f"    Participants: {participants}")

print("\nDone.")
