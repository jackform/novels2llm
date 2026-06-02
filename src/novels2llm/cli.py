"""CLI for the novels2llm pipeline."""

import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        _print_help()
        return

    command = sys.argv[1]

    if command == "preprocess":
        cmd_preprocess(sys.argv[2:])
    elif command == "extract":
        cmd_extract(sys.argv[2:])
    elif command == "run":
        cmd_run(sys.argv[2:])
    elif command == "export":
        cmd_export(sys.argv[2:])
    elif command in ("-h", "--help", "help"):
        _print_help()
    else:
        print(f"Unknown command: {command}")
        _print_help()


def _print_help():
    """Print help message."""
    print("""
novels2llm - Novel Analysis Pipeline for Role-Playing

Usage:
  novels2llm run [novel_path]       Run full pipeline on one or all novels
  novels2llm preprocess [novel_path] Run Stage 1 only (parse + split chapters)
  novels2llm extract [novel_path]    Run Stage 1-3 (preprocess + NLP + LLM extract)
  novels2llm export [novel_path]     Export all novels to SQLite + JSON

Examples:
  novels2llm run                     Process all novels
  novels2llm run shao-nian-de-fan-nao  Process single novel
  novels2llm preprocess shao-nian-de-fan-nao
  novels2llm export
""")


def cmd_run(args: list[str]):
    """Run the full pipeline."""
    from .config import config
    from .pipeline.stage1_preprocess import preprocess_novel
    from .pipeline.stage3_llm_extract import Stage3Pipeline
    from .pipeline.stage7_export import export_to_json, export_to_sqlite, export_character_cards
    from .chunking.smart_chunker import chunk_novel
    from .coreference.alias_resolver import resolve_aliases
    from .models.output import NovelWorld
    from .models.entities import Character, Location, Item, StoryEvent, WorldSetting
    from .models.relationships import Dialogue, Relationship, TimelineEvent

    novel_files = _get_novel_files(args)

    if not novel_files:
        print("No novels found!")
        return

    config.ensure_dirs()

    for filepath in novel_files:
        print(f"\n{'='*60}")
        print(f"Processing: {filepath.name}")
        print(f"{'='*60}")

        # Stage 1: Preprocess
        print("Stage 1: Preprocessing...")
        preprocessed = preprocess_novel(filepath)
        print(f"  Title: {preprocessed.metadata.title}")
        print(f"  Word count: {preprocessed.metadata.word_count}")
        print(f"  Chapters: {len(preprocessed.chapters)}")
        print(f"  Duplicates removed: {preprocessed.dedup_removed}")

        # Chunk
        print("\nStage 2: Chunking...")
        chunks = chunk_novel(
            preprocessed.raw_text,
            preprocessed.metadata.novel_id,
            target_size=config.TARGET_CHUNK_SIZE_CHARS,
            max_size=config.MAX_CHUNK_SIZE_CHARS,
            overlap=config.CHUNK_OVERLAP_CHARS,
        )
        print(f"  Created {len(chunks)} chunks from {len(preprocessed.chapters)} chapters")

        # Stage 3: LLM Extraction
        print("\nStage 3: LLM Extraction...")
        pipeline = Stage3Pipeline()
        extraction_result = pipeline.process_chapters(chunks)
        print(f"  Characters found: {len(extraction_result.characters)}")
        print(f"  Dialogues found: {len(extraction_result.dialogues)}")
        print(f"  Errors: {len(extraction_result.errors)}")

        # Stage 4: Coreference resolution
        print("\nStage 4: Coreference resolution...")
        resolved_chars = resolve_aliases(extraction_result.characters)
        print(f"  After resolution: {len(resolved_chars)} unique characters")

        # Build NovelWorld
        characters = [_dict_to_character(c) for c in resolved_chars]
        dialogues = [_dict_to_dialogue(d) for d in extraction_result.dialogues]

        # Extract world setting
        world_setting = None
        for ws in extraction_result.world_settings:
            if isinstance(ws, dict):
                world_setting = WorldSetting(
                    era=ws.get('era'),
                    genre=ws.get('genre'),
                    summary=ws.get('setting_summary'),
                    special_rules=ws.get('special_rules', []),
                    key_themes=ws.get('key_themes', []),
                )
                break

        # Extract locations and items from world settings
        locations = []
        items = []
        for ws in extraction_result.world_settings:
            if isinstance(ws, dict):
                for loc_data in ws.get('locations', []):
                    locations.append(Location(**loc_data))
                for item_data in ws.get('items', []):
                    items.append(Item(**item_data))

        novel_world = NovelWorld(
            metadata=preprocessed.metadata,
            characters=characters,
            dialogues=dialogues,
            world_setting=world_setting,
            locations=locations,
            items=items,
        )

        # Stage 7: Export
        print("\nStage 7: Exporting...")
        json_path = export_to_json(novel_world)
        print(f"  JSON: {json_path}")

        db_path = export_to_sqlite(novel_world)
        print(f"  SQLite: {db_path}")

        card_paths = export_character_cards(novel_world)
        print(f"  Character cards: {len(card_paths)} generated")

        print(f"\nDone processing: {filepath.name}")


def cmd_preprocess(args: list[str]):
    """Run Stage 1 preprocessing only."""
    from .config import config
    from .pipeline.stage1_preprocess import preprocess_novel

    novel_files = _get_novel_files(args)

    for filepath in novel_files:
        print(f"\nProcessing: {filepath.name}")
        preprocessed = preprocess_novel(filepath)
        print(f"  Title: {preprocessed.metadata.title}")
        print(f"  Word count: {preprocessed.metadata.word_count}")
        print(f"  Chapters: {len(preprocessed.chapters)}")
        print(f"  Duplicates removed: {preprocessed.dedup_removed}")

        for ch in preprocessed.chapters:
            print(f"    Chapter {ch.number}: {ch.title} ({len(ch.text)} chars)")


def cmd_extract(args: list[str]):
    """Run Stage 1-3."""
    print("Extract subcommand - runs preprocess + NLP + LLM extraction")
    cmd_run(args)  # Same as run for now, eventually will be separate


def cmd_export(args: list[str]):
    """Export all existing data to SQLite and JSON."""
    print("Export subcommand - exports all novels to SQLite + JSON")
    cmd_run(args)


def _get_novel_files(args: list[str]) -> list[Path]:
    """Get list of novel files to process."""
    from .config import config

    input_dir = config.INPUT_DIR

    if args:
        spec = args[0]
        # Could be a full path or just a stem
        if Path(spec).exists():
            return [Path(spec)]

        # Look for file in input dir by stem
        candidates = list(input_dir.glob(f"{spec}*"))
        if candidates:
            return [candidates[0]]

        print(f"Novel not found: {spec}")
        return []

    # Process all novels
    files = sorted(input_dir.glob("*.md"))
    if config.NOVEL_LIMIT:
        files = files[:config.NOVEL_LIMIT]
    return files


def _normalize_labels(labels: list) -> list[dict]:
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


def _dict_to_character(d: dict) -> Character:
    """Convert dict to Character model."""
    return Character(
        canonical_name=d.get('canonical_name', ''),
        aliases=d.get('aliases', []),
        relational_labels=_normalize_labels(d.get('relational_labels', [])),
        gender=d.get('gender'),
        age_range=d.get('age_range'),
        appearance=d.get('appearance'),
        personality=d.get('personality'),
        role=d.get('role'),
        first_chapter=d.get('first_chapter'),
        source_chunks=d.get('source_chunks', []),
    )


def _dict_to_dialogue(d: dict) -> Dialogue:
    """Convert dict to Dialogue model."""
    return Dialogue(
        speaker=d.get('speaker', 'unknown'),
        listener=d.get('listener'),
        content=d.get('content', ''),
        context=d.get('context'),
        chapter=d.get('chapter'),
        line_index=d.get('line_index'),
    )


if __name__ == "__main__":
    main()
