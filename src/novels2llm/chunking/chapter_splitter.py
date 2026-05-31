"""Chapter boundary detection for Chinese novels."""

import re
from typing import Optional


# Patterns for chapter detection
CHAPTER_PATTERNS = [
    # "### 第1章" style
    re.compile(r'###\s*第[\d一二三四五六七八九十百千]+\s*章'),
    # "第1章" style (bare)
    re.compile(r'第[\d一二三四五六七八九十百千]+\s*章'),
    # "第一章" style
    re.compile(r'第[一二三四五六七八九十百千]+\s*章'),
    # "### 第01章" with zero-padded
    re.compile(r'###\s*第\d+\s*章'),
]


def detect_chapter_boundaries(text: str) -> list[tuple[int, str, int]]:
    """Find chapter boundaries in the text.

    Returns list of (start_position, chapter_title, end_position) tuples.
    The last tuple has end_position = len(text).
    """
    matches = []
    for pattern in CHAPTER_PATTERNS:
        for m in pattern.finditer(text):
            pos = m.start()
            # Check if we already found a match at this position
            if any(abs(pos - existing[0]) < 10 for existing in matches):
                continue
            # Get the full line containing this match
            line_start = text.rfind('\n', 0, pos) + 1
            line_end = text.find('\n', pos)
            if line_end == -1:
                line_end = len(text)
            title = text[line_start:line_end].strip()
            # Clean up markdown heading markers
            title = re.sub(r'^#+\s*', '', title)
            matches.append((pos, title, -1))

    if not matches:
        return [(0, "全文", len(text))]

    # Sort by position
    matches.sort(key=lambda x: x[0])

    # Set end positions
    boundaries = []
    for i, (start, title, _) in enumerate(matches):
        if i + 1 < len(matches):
            end = matches[i + 1][0]
        else:
            end = len(text)
        boundaries.append((start, title, end))

    # Ensure first boundary starts at 0
    if boundaries and boundaries[0][0] > 100:
        boundaries.insert(0, (0, "前言", boundaries[0][0]))

    return boundaries


def chinese_to_int(chinese_num: str) -> Optional[int]:
    """Convert Chinese numeral to integer. Returns None if parsing fails."""
    chinese_digits = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '百': 100, '千': 1000, '万': 10000,
    }

    if not chinese_num:
        return None

    # Try direct integer
    try:
        return int(chinese_num)
    except ValueError:
        pass

    result = 0
    temp = 0
    unit = 1

    for char in reversed(chinese_num):
        if char not in chinese_digits:
            return None
        value = chinese_digits[char]
        if value >= 10:
            if value > unit:
                unit = value
                if temp == 0:
                    temp = 1
            else:
                unit = unit * value
        else:
            temp = temp + value * unit

    result = temp
    return result if result > 0 else None


def extract_chapter_number(title: str) -> Optional[int]:
    """Extract chapter number from a chapter title."""
    # Try digits
    m = re.search(r'第\s*(\d+)\s*章', title)
    if m:
        return int(m.group(1))

    # Try Chinese numerals
    m = re.search(r'第([一二三四五六七八九十百千]+)\s*章', title)
    if m:
        return chinese_to_int(m.group(1))

    return None
