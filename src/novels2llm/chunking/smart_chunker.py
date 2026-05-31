"""Smart chunking for Chinese novels.

Chunks text at chapter boundaries first, then at sentence boundaries
for oversized chapters, with overlap for context continuity.
"""

import re
from typing import Iterator
from dataclasses import dataclass, field
from .chapter_splitter import detect_chapter_boundaries, extract_chapter_number


# Sentence boundary pattern for Chinese text
SENTENCE_BOUNDARY = re.compile(r'[。！？!?\n]')


@dataclass
class Chunk:
    """A chunk of text to be processed."""

    novel_id: str
    chunk_index: int
    chapter: int  # Chapter number (1-based)
    chapter_title: str
    text: str
    is_chapter_start: bool = False
    is_chapter_end: bool = False
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)


def _split_long_text(
    text: str,
    target_size: int = 6000,
    max_size: int = 8000,
    overlap: int = 200,
) -> list[str]:
    """Split a long text into chunks at sentence boundaries."""
    chunks = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        # Determine ideal end position
        end_pos = min(pos + target_size, text_len)

        if end_pos >= text_len:
            # Last chunk
            chunks.append(text[pos:])
            break

        # Try to find a good sentence boundary near target_size
        # Search backwards from target_size
        search_start = max(pos + target_size // 2, pos)
        search_end = min(pos + max_size, text_len)

        best_split = None
        for m in SENTENCE_BOUNDARY.finditer(text, search_start, search_end):
            best_split = m.end()
            if best_split >= pos + target_size:
                break

        if best_split is None or best_split <= pos:
            # No good boundary found, use target_size
            best_split = end_pos

        chunks.append(text[pos:best_split])

        # Move position with overlap
        if best_split >= text_len:
            break
        pos = max(best_split - overlap, pos + 1)

    return chunks


def chunk_novel(
    text: str,
    novel_id: str,
    target_size: int = 6000,
    max_size: int = 8000,
    overlap: int = 200,
) -> list[Chunk]:
    """Split a novel into processable chunks.

    Strategy:
    1. Detect chapter boundaries
    2. For chapters <= max_size, each becomes one chunk
    3. For chapters > max_size, split at sentence boundaries with overlap
    """
    boundaries = detect_chapter_boundaries(text)
    chunks: list[Chunk] = []
    chunk_index = 0

    for i, (start, title, end) in enumerate(boundaries):
        chapter_text = text[start:end].strip()
        if not chapter_text:
            continue

        chapter_num = extract_chapter_number(title) or (i + 1)

        if len(chapter_text) <= max_size:
            # Single chunk for this chapter
            chunks.append(Chunk(
                novel_id=novel_id,
                chunk_index=chunk_index,
                chapter=chapter_num,
                chapter_title=title,
                text=chapter_text,
                is_chapter_start=True,
                is_chapter_end=True,
            ))
            chunk_index += 1
        else:
            # Split long chapter
            sub_chunks = _split_long_text(
                chapter_text,
                target_size=target_size,
                max_size=max_size,
                overlap=overlap,
            )
            for j, sub_text in enumerate(sub_chunks):
                chunks.append(Chunk(
                    novel_id=novel_id,
                    chunk_index=chunk_index,
                    chapter=chapter_num,
                    chapter_title=title,
                    text=sub_text,
                    is_chapter_start=(j == 0),
                    is_chapter_end=(j == len(sub_chunks) - 1),
                ))
                chunk_index += 1

    return chunks
