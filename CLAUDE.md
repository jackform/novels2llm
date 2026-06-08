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

# Run full pipeline (same as `run`, preprocess + chunk + LLM extract)
uv run python -m src.novels2llm.cli extract <novel_name_stem>

# Export all novels to SQLite + JSON
uv run python -m src.novels2llm.cli export

# Full extraction test with 4 extractors + event-dialogue linking + scene merge (6 chunks)
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

The NLP stage (`stage2_nlp.py`) exists but is **not called** by the main pipeline. It is used by `save_extraction_results.py` for jieba segmentation and regex-based dialogue detection (HanLP disabled by default).

### Two Pipeline Paths

There are two parallel implementations:

| Aspect | `cli.py cmd_run` | `save_extraction_results.py` |
|--------|------------------|------------------------------|
| Extractors used | 3 (char, world, dialogue) | 4 (char, world, relationship, scene) |
| NLP stage | No | Yes (jieba + regex dialogue) |
| Scene extraction | No | Yes (location-based, via `SceneExtractor`) |
| Scene-dialogue linking | No | Yes (via `SceneEvent` model) |
| Cross-chunk scene dedup | No | Yes (heuristic + LLM judgment) |
| LLM dialogue attribution | Direct via `DialogueExtractor` | Scene-based: dialogue from narrative units |
| Caching | Yes (`CACHE_DIR`) | No (calls extractors directly) |

`save_extraction_results.py` is the more feature-complete path and represents the intended full pipeline. It was enhanced with location-based scene extraction via `SceneExtractor` and event-dialogue linking via `SceneEvent`.

### Stage 3: Three Extractors in Main Pipeline

The main pipeline's `Stage3Pipeline.process_chunk()` runs 3 extractors per chunk:

| Extractor | Class | Prompt | Scope |
|-----------|-------|--------|-------|
| Character | `CharacterExtractor` | `character_extraction.txt` | Every chunk |
| World | `WorldExtractor` | `world_setting.txt` | Only chapters ≤2 at chapter-start chunks |
| Dialogue | `DialogueExtractor` | `dialogue_extraction.txt` | Every chunk |

Two additional extractors (`RelationshipExtractor`, `EventExtractor`) are instantiated but **not called** by `process_chunk()`. `save_extraction_results.py` uses `RelationshipExtractor` and `SceneExtractor` (not `EventExtractor`).

### `save_extraction_results.py` — Full 4-Extractor + Linking Pipeline

This script uses 4 extractors (Character, World, Relationship, Scene) and adds post-processing:

| Step | Description |
|------|-------------|
| 1. NLP | jieba POS + `""` regex dialogue detection |
| 2. Character extraction | Per chunk via `CharacterExtractor` |
| 3. World extraction | All chunks via `WorldExtractor`, merged (locations/items dedup by name) |
| 4. Relationship extraction | `RelationshipExtractor` with caller-perspective labels |
| 5. Scene extraction | Location-based scene + narrative unit extraction via `SceneExtractor` |
| 6. Dialogue extraction | Dialogues pulled from scene narrative units (type `dialogue` + `inner_thought`) |
| 7. Scene-dialogue linking | Map dialogues to scenes → `SceneEvent` objects |
| 8. Cross-chunk scene merge | Heuristic check (same location + shared participants) → LLM judgment for merge |
| 9. Coref resolution | `resolve_aliases()` with alias overlap + edit distance + caller-label matching |

### Coreference Resolution

`alias_resolver.py` merges character entries across chunks:

1. **Exact match** on `canonical_name`
2. **Alias overlap** — if two groups share >50% aliases
3. **Edit distance** — >85% similarity on canonical names
4. **Caller-label overlap** — if two characters are called by the same label from the same caller, they are likely the same person (threshold 0.3)

Returns a deduplicated `list[dict]` of character entries. Callers build a `name_map` (dict[str,str] mapping raw name → canonical) from this output.

### Prompt Template Escaping

Prompt files use Python `str.format()`. JSON examples in prompts MUST use `{{` and `}}` for literal braces. Template variables (`{text}`, `{chapter}`, etc.) remain single-braced. A missing escape will raise `KeyError` at format time.

### Model Compatibility (DeepSeek)

`base.py:_call_claude()` handles `ThinkingBlock` (DeepSeek returns reasoning blocks). The code iterates `message.content` looking for `TextBlock` instances rather than assuming `.text` on the first content block. It also includes JSON repair for malformed LLM output (missing commas between fields).

### Data Models

All output models in `models/` are Pydantic v2. The aggregate `NovelWorld` is the single output object per novel, containing:

- `metadata: NovelMetadata` — YAML frontmatter fields
- `characters: list[Character]`, `relationships: list[Relationship]`
- `dialogues: list[Dialogue]`, `timeline: list[TimelineEvent]`
- `scenes: list[Scene]` — location-anchored scenes with narrative units
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

1. **No line_index on dialogues in main pipeline** — dialogues can't be aligned to precise text positions. `save_extraction_results.py` captures dialogue text via narrative units but doesn't preserve line-level position info.
2. **Scene/event IDs duplicate across chunks in main pipeline** — each chunk generates independent IDs. `save_extraction_results.py` addresses this with global renumbering + cross-chunk merge.
3. **Event-dialogue linking only in test script** — `SceneEvent` and the linking logic exist only in `save_extraction_results.py`, not in the main `Stage3Pipeline`.
4. **No relationship-type validation** — e.g., mother-son being labeled `sibling` passes through.
5. **Main pipeline missing 3 extractors** — `process_chunk()` instantiates but does not call `RelationshipExtractor` or `EventExtractor`. `SceneExtractor` also only runs in `save_extraction_results.py`.
6. **Cache bypassed in test scripts** — `save_extraction_results.py` calls extractors directly, skipping `Stage3Pipeline` caching.

## Output Files (.gitignored)

- `data/jia_ting_luan_lun/` — 34 novel source files (do not commit)
- `data/output/*.json` — per-novel NovelWorld export
- `data/output/novels.db` — SQLite with 9 tables (novels, characters, relationships, dialogues, timeline, locations, items, scenes, narrative_units)
- `data/output/character_cards/` — one .md file per character
- `data/output/cache/` — per-chunk extraction cache (used by `Stage3Pipeline`)
- `data/output/raw/` — intermediate extraction data for debugging (used by `save_extraction_results.py`)
