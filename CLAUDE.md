# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install dependencies and create venv
uv venv && uv pip install -e ".[dev]"

# Run all tests
uv run pytest tests/test_pipeline.py -v

# Run a single test class or test
uv run pytest tests/test_pipeline.py::TestChapterSplitter -v
uv run pytest tests/test_pipeline.py::TestPreprocessing::test_yaml_parsing -v

# Run full pipeline on one novel (requires ANTHROPIC_API_KEY in .env)
uv run python -m src.novels2llm.cli run <novel_name_stem>

# Run preprocess only (no API calls)
uv run python -m src.novels2llm.cli preprocess <novel_name_stem>

# Quick extraction test on first 3 chunks
uv run python tests/save_extraction_results.py

# Git operations — NEVER skip hooks or use destructive commands without asking
git commit -m "message"
git push
```

## Architecture

### Pipeline Design (7 Stages)

Novel data flows through stages 1→2→3→4→5→6→7. Stages 1/2/4/5/6/7 are local Python. Stage 3 calls an external LLM API (Anthropic protocol, DeepSeek by default).

```
.md file → [1] preprocess → [2] NLP → [3] LLM extract → [4] coref → [5] graph → [6] timeline → [7] export JSON/SQLite/MD
```

### Stage 3: Five Extractors Run Per Chunk

Not a single pass — 5 independent LLM calls per chunk, each with a dedicated prompt template in `prompts/`:

| Extractor | Class | Prompt |
|-----------|-------|--------|
| Character | `CharacterExtractor` | `character_extraction.txt` |
| World | `WorldExtractor` | `world_setting.txt` |
| Dialogue | `DialogueExtractor` | `dialogue_extraction.txt` |
| Relationship | `RelationshipExtractor` | `relationship_extraction.txt` |
| Event | `EventExtractor` | `timeline_extraction.txt` |

World extraction only runs on the first 2 chapters (world-building info is front-loaded).

### Dialogue Strategy Change

The Dialogue Extractor does NOT send full text to the LLM (DeepSeek rejects explicit content). Instead:
1. NLP regex finds `""` quote spans in text
2. LLM receives batches of 10 quotes + surrounding context, only to attribute `speaker`/`listener`
3. Beyond 30 quotes per chunk, fallback to rule-based speaker inference

### Coreference Resolution

`alias_resolver.py` merges character entries across chunks in three phases:
1. **Exact match** on `canonical_name`
2. **Alias overlap** — if two groups share >50% aliases
3. **Edit distance** — >85% similarity on canonical names

Output is a `name_map` (dict[str,str]) mapping every raw name to its canonical form. This map is then applied to relationship and dialogue names.

### Prompt Template Escaping

Prompt files use Python `str.format()`. JSON examples in prompts MUST use `{{` and `}}` for literal braces. Template variables (`{text}`, `{chapter}`, etc.) remain single-braced. A missing escape will raise `KeyError` at format time.

### Model Compatibility (DeepSeek)

`base.py:_call_claude()` handles `ThinkingBlock` (DeepSeek returns reasoning blocks). The code iterates `message.content` looking for `TextBlock` instances rather than assuming `.text` on the first content block.

### Data Models

All output models in `models/` are Pydantic v2. The aggregate `NovelWorld` is the single output object per novel, containing lists of `Character`, `Relationship`, `Dialogue`, `TimelineEvent`, `Location`, `Item`, and a `WorldSetting`. `NovelMetadata` carries the YAML frontmatter fields.

### Chunking Logic

`smart_chunker.py` implements three-tier splitting:
1. Chapter boundaries first (via `chapter_splitter.py` regex)
2. Chapters >8000 chars split at sentence boundaries (`。！？`)
3. 200-char overlap between adjacent chunks

### Key Config Values (`config.py`)

- `ANTHROPIC_BASE_URL`: Default is Anthropic, set to `https://api.deepseek.com/anthropic` for DeepSeek
- `ANTHROPIC_MODEL`: `deepseek-v4-pro` or `claude-sonnet-4-6`
- `TARGET_CHUNK_SIZE_CHARS=6000`, `MAX_CHUNK_SIZE_CHARS=8000`, `CHUNK_OVERLAP_CHARS=200`
- `NOVEL_LIMIT`: Set to `N` to process only N novels for testing
- `CACHE_ENABLED`: Cache chunk-level extraction results as JSON under `data/output/cache/`

## Known Gaps

1. **No line_index on dialogues** — dialogues can't be aligned to precise text positions
2. **Event IDs duplicate across chunks** — each chunk generates `ch1_evt1`, no global dedup
3. **Dialogues ↔ Events not linked** — only shared field is `chapter`
4. **No relationship-type validation** — e.g., mother-son being labeled `sibling` passes through
5. **Cache bypassed in test scripts** — `save_extraction_results.py` calls `extract()` directly

## Output Files (.gitignored)

- `data/jia_ting_luan_lun/` — 34 novel source files (do not commit)
- `data/output/*.json` — per-novel NovelWorld export
- `data/output/novels.db` — SQLite with 7 tables (novels, characters, relationships, dialogues, timeline, locations, items)
- `data/output/character_cards/` — one .md file per character
- `data/output/raw/` — intermediate extraction data for debugging
