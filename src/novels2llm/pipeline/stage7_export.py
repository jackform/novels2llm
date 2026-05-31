"""Stage 7: Export pipeline results to various formats."""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from ..config import config
from ..models.output import NovelWorld


def export_to_json(novel_world: NovelWorld, output_path: Optional[Path] = None) -> Path:
    """Export NovelWorld to a pretty-printed JSON file."""
    if output_path is None:
        output_path = config.OUTPUT_DIR / f"{novel_world.metadata.novel_id}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        novel_world.model_dump_json_pretty(),
        encoding='utf-8',
    )
    return output_path


def export_to_sqlite(novel_world: NovelWorld, db_path: Optional[Path] = None) -> Path:
    """Export NovelWorld to SQLite database.

    Creates tables: novels, characters, relationships, dialogues, timeline, locations, items.
    """
    if db_path is None:
        db_path = config.OUTPUT_DIR / "novels.db"

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create tables
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS novels (
            novel_id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            source TEXT,
            word_count INTEGER,
            chapter_count INTEGER,
            era TEXT,
            genre TEXT,
            world_summary TEXT
        );

        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            novel_id TEXT REFERENCES novels(novel_id),
            canonical_name TEXT NOT NULL,
            aliases TEXT,  -- JSON array
            gender TEXT,
            age_range TEXT,
            appearance TEXT,
            personality TEXT,
            role TEXT,
            first_chapter INTEGER
        );

        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            novel_id TEXT REFERENCES novels(novel_id),
            character_a TEXT NOT NULL,
            character_b TEXT NOT NULL,
            rel_type TEXT NOT NULL,
            direction TEXT DEFAULT 'bidirectional',
            intimacy_level TEXT,
            evidence TEXT,  -- JSON array
            confidence REAL DEFAULT 0.5
        );

        CREATE TABLE IF NOT EXISTS dialogues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            novel_id TEXT REFERENCES novels(novel_id),
            speaker TEXT,
            listener TEXT,
            content TEXT NOT NULL,
            context TEXT,
            chapter INTEGER,
            line_index INTEGER
        );

        CREATE TABLE IF NOT EXISTS timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            novel_id TEXT REFERENCES novels(novel_id),
            event_id TEXT NOT NULL,
            description TEXT NOT NULL,
            chapter INTEGER,
            chapter_order INTEGER,
            global_order INTEGER,
            participants TEXT,  -- JSON array
            time_marker TEXT
        );

        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            novel_id TEXT REFERENCES novels(novel_id),
            name TEXT NOT NULL,
            type TEXT DEFAULT 'other',
            description TEXT,
            parent_location TEXT
        );

        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            novel_id TEXT REFERENCES novels(novel_id),
            name TEXT NOT NULL,
            type TEXT DEFAULT 'other',
            description TEXT,
            owner TEXT
        );
    """)

    novel_id = novel_world.metadata.novel_id

    # Insert/update novel metadata
    ws = novel_world.world_setting
    cursor.execute("""
        INSERT OR REPLACE INTO novels (novel_id, title, author, source, word_count,
            chapter_count, era, genre, world_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        novel_id,
        novel_world.metadata.title,
        novel_world.metadata.author,
        novel_world.metadata.source,
        novel_world.metadata.word_count,
        novel_world.metadata.chapter_count,
        ws.era if ws else None,
        ws.genre if ws else None,
        ws.summary if ws else None,
    ))

    # Insert characters
    for char in novel_world.characters:
        cursor.execute("""
            INSERT INTO characters (novel_id, canonical_name, aliases, gender,
                age_range, appearance, personality, role, first_chapter)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            novel_id,
            char.canonical_name,
            json.dumps(char.aliases, ensure_ascii=False),
            char.gender,
            char.age_range,
            char.appearance,
            char.personality,
            char.role,
            char.first_chapter,
        ))

    # Insert relationships
    for rel in novel_world.relationships:
        cursor.execute("""
            INSERT INTO relationships (novel_id, character_a, character_b,
                rel_type, direction, intimacy_level, evidence, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            novel_id,
            rel.character_a,
            rel.character_b,
            rel.rel_type,
            rel.direction,
            rel.intimacy_level,
            json.dumps(rel.evidence, ensure_ascii=False),
            rel.confidence,
        ))

    # Insert dialogues
    for dlg in novel_world.dialogues:
        cursor.execute("""
            INSERT INTO dialogues (novel_id, speaker, listener, content,
                context, chapter, line_index)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            novel_id,
            dlg.speaker,
            dlg.listener,
            dlg.content,
            dlg.context,
            dlg.chapter,
            dlg.line_index,
        ))

    # Insert timeline events
    for evt in novel_world.timeline:
        cursor.execute("""
            INSERT INTO timeline (novel_id, event_id, description, chapter,
                chapter_order, global_order, participants, time_marker)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            novel_id,
            evt.event_id,
            evt.description,
            evt.chapter,
            evt.chapter_order,
            evt.global_order,
            json.dumps(evt.participants, ensure_ascii=False),
            evt.time_marker,
        ))

    # Insert locations
    for loc in novel_world.locations:
        cursor.execute("""
            INSERT INTO locations (novel_id, name, type, description, parent_location)
            VALUES (?, ?, ?, ?, ?)
        """, (novel_id, loc.name, loc.type, loc.description, loc.parent_location))

    # Insert items
    for item in novel_world.items:
        cursor.execute("""
            INSERT INTO items (novel_id, name, type, description, owner)
            VALUES (?, ?, ?, ?, ?)
        """, (novel_id, item.name, item.type, item.description, item.owner))

    conn.commit()
    conn.close()

    return db_path


def export_character_cards(novel_world: NovelWorld, output_dir: Optional[Path] = None) -> list[Path]:
    """Generate Markdown character cards for each character."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR / "character_cards" / novel_world.metadata.novel_id

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for char in novel_world.characters:
        card = _generate_character_card(char, novel_world)
        safe_name = char.canonical_name.replace('/', '_').replace(' ', '_')
        card_path = output_dir / f"{safe_name}.md"
        card_path.write_text(card, encoding='utf-8')
        paths.append(card_path)

    return paths


def _generate_character_card(char, novel_world: NovelWorld) -> str:
    """Generate a Markdown character card."""
    # Find related characters
    related = set()
    for rel in novel_world.relationships:
        if rel.character_a == char.canonical_name:
            related.add(f"{rel.character_b} ({rel.rel_type})")
        elif rel.character_b == char.canonical_name:
            related.add(f"{rel.character_a} ({rel.rel_type})")

    lines = [
        f"# {char.canonical_name}",
        "",
        f"**小说**: {novel_world.metadata.title}",
        "",
        "## 基本信息",
        f"- **性别**: {char.gender or '未知'}",
        f"- **年龄**: {char.age_range or '未知'}",
        f"- **角色**: {char.role or '未知'}",
    ]

    if char.aliases:
        lines.append(f"- **别名**: {', '.join(char.aliases)}")

    if char.appearance:
        lines.extend(["", "## 外貌", char.appearance])

    if char.personality:
        lines.extend(["", "## 性格", char.personality])

    if related:
        lines.extend(["", "## 关系", *[f"- {r}" for r in sorted(related)]])

    # Find dialogues by this character
    char_dialogues = [
        d for d in novel_world.dialogues
        if d.speaker == char.canonical_name
    ]
    if char_dialogues:
        lines.extend(["", "## 对话（部分）", ""])
        for d in char_dialogues[:5]:
            lines.append(f"> {d.content}")
            lines.append("")

    return '\n'.join(lines)
