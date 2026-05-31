"""Stage 1: Preprocessing - YAML parsing, chapter splitting, deduplication."""

import re
import hashlib
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from ..models.output import NovelMetadata
from ..chunking.chapter_splitter import detect_chapter_boundaries, extract_chapter_number


@dataclass
class PreprocessedNovel:
    """Output of Stage 1 preprocessing."""

    metadata: NovelMetadata
    raw_text: str  # Full text without YAML frontmatter
    chapters: list['Chapter'] = field(default_factory=list)
    dedup_removed: int = 0


@dataclass
class Chapter:
    """A chapter from the novel."""

    number: int
    title: str
    text: str
    start_pos: int
    end_pos: int


def _parse_yaml_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file.

    Returns (frontmatter_dict, remaining_content).
    """
    if not content.startswith('---'):
        return {}, content

    # Find closing ---
    end = content.find('---', 3)
    if end == -1:
        return {}, content

    yaml_text = content[3:end].strip()
    body = content[end + 3:].strip()

    try:
        metadata = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        metadata = {}

    return metadata or {}, body


def _extract_metadata(frontmatter: dict, filename: str) -> NovelMetadata:
    """Extract NovelMetadata from frontmatter dict."""
    return NovelMetadata(
        novel_id=Path(filename).stem,
        title=frontmatter.get('title', Path(filename).stem),
        author=frontmatter.get('author'),
        source=frontmatter.get('source'),
        word_count=frontmatter.get('word_count'),
        chapter_count=frontmatter.get('chapter_count', 1),
    )


def _remove_markdown_formatting(text: str) -> str:
    """Remove markdown formatting that's not chapter headers.

    Specifically removes the metadata display block (lines starting with **)
    and the brief description section, keeping only chapter content.
    """
    # Remove the markdown metadata block (title/author/category lines)
    # Pattern: lines starting with #, **, or ## 简介 up to ## 章节列表
    text = re.sub(r'^# .*?\n', '', text, count=1)  # Remove main title line
    text = re.sub(r'\*\*.*?\*\*.*?\n', '', text)    # Remove **key**: value lines
    text = re.sub(r'\*\*.*?\*\*.*?$', '', text, flags=re.MULTILINE)

    # Remove the brief introduction section
    text = re.sub(r'## 简介.*?(?=## 章节列表)', '', text, flags=re.DOTALL)
    text = re.sub(r'## 章节列表', '', text, count=1)

    return text.strip()


def _deduplicate_text(text: str, chunk_size: int = 500) -> tuple[str, int]:
    """Detect and remove duplicate content blocks.

    Uses hash of first chunk_size characters + length comparison to find dupes.
    Returns (deduplicated_text, removed_count).
    """
    lines = text.split('\n')
    seen = {}  # (hash, len) -> first occurrence index
    unique_lines = []
    removed = 0

    for line in lines:
        if len(line) < 20:
            # Short lines (markdown headers, metadata) always kept
            unique_lines.append(line)
            continue

        key = (hashlib.md5(line[:chunk_size].encode()).hexdigest(), len(line))

        if key in seen:
            removed += 1
            continue

        seen[key] = len(unique_lines)
        unique_lines.append(line)

    return '\n'.join(unique_lines), removed


def _split_into_chapters(text: str) -> list[Chapter]:
    """Split text into chapters using boundary detection."""
    boundaries = detect_chapter_boundaries(text)
    chapters = []

    for i, (start, title, end) in enumerate(boundaries):
        chapter_text = text[start:end].strip()
        if not chapter_text:
            continue

        chapter_num = extract_chapter_number(title) or (i + 1)

        # Clean chapter text - remove the chapter header itself from text
        cleaned = re.sub(r'^#+\s*.*?\n', '', chapter_text, count=1)

        chapters.append(Chapter(
            number=chapter_num,
            title=title,
            text=cleaned.strip(),
            start_pos=start,
            end_pos=end,
        ))

    return chapters


def preprocess_novel(filepath: Path) -> PreprocessedNovel:
    """Preprocess a single novel file.

    Steps:
    1. Parse YAML frontmatter
    2. Clean markdown formatting
    3. Remove duplicate content blocks
    4. Split into chapters

    Returns PreprocessedNovel with metadata, raw text, and chapter list.
    """
    content = filepath.read_text(encoding='utf-8')

    # Step 1: Parse YAML frontmatter
    frontmatter, body = _parse_yaml_frontmatter(content)
    metadata = _extract_metadata(frontmatter, filepath.name)

    # Step 2: Clean markdown formatting (metadata display block)
    body = _remove_markdown_formatting(body)

    # Step 3: Deduplicate
    body, removed = _deduplicate_text(body)

    # Step 4: Split into chapters
    chapters = _split_into_chapters(body)

    return PreprocessedNovel(
        metadata=metadata,
        raw_text=body,
        chapters=chapters,
        dedup_removed=removed,
    )
