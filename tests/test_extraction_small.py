"""Quick test: extract characters from first 3 chunks of qing-chun-yun-shi."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.novels2llm.pipeline.stage1_preprocess import preprocess_novel
from src.novels2llm.pipeline.stage2_nlp import process_chapter
from src.novels2llm.chunking.smart_chunker import chunk_novel
from src.novels2llm.extractors.character_extractor import CharacterExtractor
from src.novels2llm.extractors.world_extractor import WorldExtractor
from src.novels2llm.extractors.dialogue_extractor import DialogueExtractor
from src.novels2llm.coreference.alias_resolver import resolve_aliases
from src.novels2llm.config import config

f = Path('data/jia_ting_luan_lun/qing-chun-yun-shi.md')

# Stage 1
print('=== Stage 1: Preprocessing ===')
result = preprocess_novel(f)
print(f'Title: {result.metadata.title}')
print(f'Word count: {result.metadata.word_count}')
print(f'Chapters: {len(result.chapters)}')
print(f'Text length: {len(result.raw_text)} chars')

# Chunking
print('\n=== Chunking ===')
chunks = chunk_novel(result.raw_text, result.metadata.novel_id)
print(f'Total chunks: {len(chunks)}')
# Take first 3
test_chunks = chunks[:3]
for c in test_chunks:
    print(f'  Chunk {c.chunk_index}: ch{c.chapter} ({c.char_count} chars)')

# NLP hints for each chunk
print('\n=== Stage 2: NLP ===')
nlp_results = {}
for c in test_chunks:
    nlp = process_chapter(c.text, result.metadata.novel_id, c.chapter, use_hanlp=False)
    nlp_results[c.chunk_index] = nlp
    entities = list(set(e.text for e in nlp.entities if e.label == 'PERSON'))
    print(f'  Chunk {c.chunk_index}: {len(nlp.tokens)} tokens, {len(entities)} persons, {len(nlp.dialogues)} dialogues')
    if entities:
        print(f'    Persons: {", ".join(entities[:10])}')

# Stage 3: Character Extraction
print('\n=== Stage 3: Character Extraction (Claude) ===')
api_key = config.ANTHROPIC_API_KEY
char_extractor = CharacterExtractor(api_key)
all_chars = []

for c in test_chunks:
    nlph = nlp_results[c.chunk_index].to_hint_dict()
    print(f'\n--- Chunk {c.chunk_index} ---')
    print(f'  Text preview: {c.text[:200]}...')
    try:
        chars = char_extractor.extract(c.text, nlp_hints=nlph)
        print(f'  Found {len(chars)} characters:')
        for ch in chars:
            print(f'    {ch["canonical_name"]} ({ch.get("gender","?")}) - {ch.get("role","?")}')
            if ch.get('aliases'):
                print(f'      别名: {ch["aliases"]}')
            if ch.get('appearance'):
                print(f'      外貌: {ch["appearance"][:100]}')
            if ch.get('personality'):
                print(f'      性格: {ch["personality"][:100]}')
        all_chars.extend(chars)
    except Exception as e:
        print(f'  ERROR: {e}')

# Cross-chunk dedup
print('\n=== Coreference Resolution ===')
print(f'Raw entries: {len(all_chars)}')
resolved = resolve_aliases(all_chars)
print(f'After merge: {len(resolved)} unique characters')
print()
for c in resolved:
    print(f'  {c["canonical_name"]} ({c.get("gender","?")}) aliases={c.get("aliases",[])} role={c.get("role","?")}')
