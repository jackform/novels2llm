"""Entity merging for coreference resolution."""

from collections import defaultdict
from typing import Optional


def merge_character_entries(
    entries: list[dict],
    canonical_name: Optional[str] = None,
) -> dict:
    """Merge multiple character entries into one canonical entry.

    Args:
        entries: List of character dicts to merge
        canonical_name: Optional canonical name override

    Returns merged character dict.
    """
    if not entries:
        return {}

    if len(entries) == 1:
        return entries[0]

    # Use the most common canonical_name, or the first one
    name_counts = defaultdict(int)
    for entry in entries:
        name = entry.get('canonical_name', '')
        if name:
            name_counts[name] += 1

    if canonical_name:
        best_name = canonical_name
    elif name_counts:
        best_name = max(name_counts, key=name_counts.get)
    else:
        best_name = entries[0].get('canonical_name', 'Unknown')

    # Collect all aliases
    all_aliases = set()
    for entry in entries:
        all_aliases.update(entry.get('aliases', []))
    all_aliases.discard(best_name)

    # Merge fields: prefer the longest/most detailed value
    merged = {
        'canonical_name': best_name,
        'aliases': sorted(all_aliases),
    }

    for field in ['gender', 'age_range', 'appearance', 'personality', 'role', 'family_role']:
        values = [e.get(field) for e in entries if e.get(field)]
        if not values:
            merged[field] = None
        else:
            # Use the most detailed (longest) description
            merged[field] = max(values, key=lambda v: len(str(v)))

    # Merge source_chunks
    all_chunks = set()
    for entry in entries:
        all_chunks.update(entry.get('source_chunks', []))
    merged['source_chunks'] = sorted(all_chunks)

    # Track first chapter
    first_chapters = [e.get('first_chapter') for e in entries if e.get('first_chapter')]
    merged['first_chapter'] = min(first_chapters) if first_chapters else None

    return merged
