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

# Run Stages 1-3 (preprocess + NLP + LLM extraction)
uv run python -m src.novels2llm.cli extract <novel_name_stem>

# Export all novels to SQLite + JSON
uv run python -m src.novels2llm.cli export

# Full extraction test with all 5 extractors + event-dialogue linking (3 chunks)
uv run python tests/save_extraction_results.py

# Git operations — NEVER skip hooks or use destructive commands without asking
git commit -m "message"
git push
```

## Architecture

### Pipeline Design

The main pipeline (`cli.py cmd_run`) executes: preprocess → chunk → LLM extraction (3 extractors) → alias resolution → export.

```
.md file → [1] preprocess → [2] chunk → [3] LLM extract → [4] coref → [5] export JSON/SQLite/MD
```

The NLP stage (`stage2_nlp.py`) exists but is **not called** by the main pipeline. It is used by `save_extraction_results.py` for jieba segmentation, HanLP NER, and regex-based dialogue detection.

### Two Pipeline Paths

There are two parallel implementations:

| Aspect | `cli.py cmd_run` | `save_extraction_results.py` |
|--------|------------------|------------------------------|
| Extractors used | 3 (char, world, dialogue) | 5 (all + event-dialogue linking) |
| NLP stage | No | Yes (jieba + HanLP + regex dialogue) |
| Event extraction | No | Yes (location-based scenes) |
| Event-dialogue linking | No | Yes (via `SceneEvent` model) |
| Cross-chunk event dedup | No | Yes (LLM judgment) |
| LLM dialogue attribution | Direct via `DialogueExtractor` | Two-phase: NLP find quotes → LLM attribute speaker/listener |
| Caching | Yes (`CACHE_DIR`) | No (calls extractors directly) |

`save_extraction_results.py` is the more feature-complete path and represents the intended full pipeline. It was enhanced in `deddb2f` with location-based scene extraction, LLM dialogue validation, and event-dialogue linking.

### Stage 3: Three Extractors in Main Pipeline

The main pipeline's `Stage3Pipeline.process_chunk()` runs 3 extractors per chunk:

| Extractor | Class | Prompt | Scope |
|-----------|-------|--------|-------|
| Character | `CharacterExtractor` | `character_extraction.txt` | Every chunk |
| World | `WorldExtractor` | `world_setting.txt` | Only chapters ≤2 at chapter-start chunks |
| Dialogue | `DialogueExtractor` | `dialogue_extraction.txt` | Every chunk |

Two additional extractors (`RelationshipExtractor`, `EventExtractor`) are instantiated but **not called** by `process_chunk()`. They are used in `save_extraction_results.py`.

### `save_extraction_results.py` — Full 5-Extractor + Linking Pipeline

This script runs all 5 extractors and adds post-processing:

| Step | Description |
|------|-------------|
| 1. NLP | jieba POS + `""` regex dialogue detection |
| 2. Character extraction | Per chunk via `CharacterExtractor` |
| 3. World extraction | First chunk only via `WorldExtractor` |
| 4. Relationship extraction | `RelationshipExtractor` with caller-perspective labels |
| 5. Event extraction | Location-based scene extraction with `text_start`/`text_end` |
| 6. Dialogue attribution | Batch NLP quotes (≤8/batch) → LLM validates, filters inner thoughts, attributes speaker/listener |
| 7. Event-dialogue linking | Map dialogues to events by text position → `SceneEvent` objects |
| 8. Cross-chunk event dedup | LLM judgment for events spanning chunk boundaries |
| 9. Coref resolution | `resolve_aliases()` with alias overlap + edit distance + caller-label matching |

### Coreference Resolution

`alias_resolver.py` merges character entries across chunks:

1. **Exact match** on `canonical_name`
2. **Alias overlap** — if two groups share >50% aliases
3. **Edit distance** — >85% similarity on canonical names
4. **Caller-label overlap** — if two characters are called by the same label from the same caller, they are likely the same person (threshold 0.3)

Output is a `name_map` (dict[str,str]) mapping every raw name to its canonical form.

### Prompt Template Escaping

Prompt files use Python `str.format()`. JSON examples in prompts MUST use `{{` and `}}` for literal braces. Template variables (`{text}`, `{chapter}`, etc.) remain single-braced. A missing escape will raise `KeyError` at format time.

### Model Compatibility (DeepSeek)

`base.py:_call_claude()` handles `ThinkingBlock` (DeepSeek returns reasoning blocks). The code iterates `message.content` looking for `TextBlock` instances rather than assuming `.text` on the first content block. It also includes JSON repair for malformed LLM output (missing commas between fields).

### Data Models

All output models in `models/` are Pydantic v2. The aggregate `NovelWorld` is the single output object per novel, containing:

- `metadata: NovelMetadata` — YAML frontmatter fields
- `characters: list[Character]`, `relationships: list[Relationship]`
- `dialogues: list[Dialogue]`, `timeline: list[TimelineEvent]`
- `locations: list[Location]`, `items: list[Item]`
- `world_setting: Optional[WorldSetting]`
- `events: list[StoryEvent]` — raw extracted events
- `scene_events: list[SceneEvent]` — events linked to their dialogues by text position

`SceneEvent` bridges events and dialogues: it has `event_id`, `description`, `chapter`, `location`, `participants`, and `dialogues: list[Dialogue]`.

### Chunking Logic

`smart_chunker.py` implements three-tier splitting:
1. Chapter boundaries first (via `chapter_splitter.py` regex)
2. Chapters >8000 chars split at sentence boundaries (`。！？`)
3. 200-char overlap between adjacent chunks

### Key Config Values (`config.py`)

- `ANTHROPIC_BASE_URL`: Default `https://api.anthropic.com`, set to `https://api.deepseek.com/anthropic` for DeepSeek
- `ANTHROPIC_MODEL`: `deepseek-v4-pro` or `claude-sonnet-4-6`
- `TARGET_CHUNK_SIZE_CHARS=6000`, `MAX_CHUNK_SIZE_CHARS=8000`, `CHUNK_OVERLAP_CHARS=200`
- `NOVEL_LIMIT`: Set to `N` to process only N novels for testing
- `CACHE_ENABLED`: Cache chunk-level extraction results as JSON under `data/output/cache/`

## Known Gaps

1. **No line_index on dialogues in main pipeline** — dialogues can't be aligned to precise text positions. `save_extraction_results.py` has text positions via NLP regex but they don't flow into the final `Dialogue` model.
2. **Event IDs duplicate across chunks in main pipeline** — each chunk generates independent IDs. `save_extraction_results.py` addresses this with cross-chunk dedup.
3. **Event-dialogue linking only in test script** — `SceneEvent` and the linking logic exist only in `save_extraction_results.py`, not in the main `Stage3Pipeline`.
4. **No relationship-type validation** — e.g., mother-son being labeled `sibling` passes through.
5. **Main pipeline missing 2 extractors** — `process_chunk()` instantiates but does not call `RelationshipExtractor` or `EventExtractor`. These only run in `save_extraction_results.py`.
6. **Cache bypassed in test scripts** — `save_extraction_results.py` calls extractors directly, skipping `Stage3Pipeline` caching.

## Output Files (.gitignored)

- `data/jia_ting_luan_lun/` — 34 novel source files (do not commit)
- `data/output/*.json` — per-novel NovelWorld export
- `data/output/novels.db` — SQLite with 7 tables (novels, characters, relationships, dialogues, timeline, locations, items)
- `data/output/character_cards/` — one .md file per character
- `data/output/cache/` — per-chunk extraction cache (used by `Stage3Pipeline`)
- `data/output/raw/` — intermediate extraction data for debugging (used by `save_extraction_results.py`)
