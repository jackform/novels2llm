"""Stage 4: Coreference resolution - alias disambiguation + entity merging."""

from collections import defaultdict
from difflib import SequenceMatcher


def _name_similarity(a: str, b: str) -> float:
    """Compute string similarity between two names."""
    if a == b:
        return 1.0
    # Check if one contains the other
    if a in b or b in a:
        return 0.9
    return SequenceMatcher(None, a, b).ratio()


def _alias_overlap(aliases_a: list[str], aliases_b: list[str]) -> float:
    """Check if two character entries share aliases."""
    if not aliases_a or not aliases_b:
        return 0.0
    set_a = set(a.lower() for a in aliases_a)
    set_b = set(b.lower() for b in aliases_b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / min(len(set_a), len(set_b))


def resolve_aliases(
    characters: list[dict],
    sim_threshold: float = 0.85,
    alias_threshold: float = 0.5,
) -> list[dict]:
    """Merge character entries that refer to the same person.

    Strategy:
    1. Exact name match -> merge
    2. Alias overlap -> merge
    3. Name similarity -> merge (conservative threshold)

    Returns a deduplicated list of character dicts.
    """
    if not characters:
        return []

    # Filter out None entries
    characters = [c for c in characters if c and isinstance(c, dict)]

    # Phase 1: Group by exact canonical name match
    groups: dict[str, list[dict]] = defaultdict(list)
    for char in characters:
        name = char.get('canonical_name', '').strip()
        if not name:
            continue
        groups[name.lower()].append(char)

    # Phase 2: Try to merge groups by alias overlap
    merged_groups: list[list[dict]] = [[c] for c in characters]
    return _merge_groups(merged_groups, sim_threshold, alias_threshold)


def _merge_groups(
    groups: list[list[dict]],
    sim_threshold: float,
    alias_threshold: float,
) -> list[dict]:
    """Merge groups of character entries into canonical characters."""
    merged = []
    used = set()

    for i, group_i in enumerate(groups):
        if i in used:
            continue

        canonical = group_i[0].copy()
        all_aliases = set(canonical.get('aliases', []))

        for j, group_j in enumerate(groups):
            if j <= i or j in used:
                continue

            # Check if groups should be merged
            name_i = group_i[0].get('canonical_name', '')
            name_j = group_j[0].get('canonical_name', '')
            aliases_i = group_i[0].get('aliases', [])
            aliases_j = group_j[0].get('aliases', [])

            should_merge = False

            # Exact match
            if name_i.lower() == name_j.lower():
                should_merge = True
            # Alias overlap
            elif _alias_overlap(aliases_i + [name_i], aliases_j + [name_j]) >= alias_threshold:
                should_merge = True
            # Name similarity
            elif _name_similarity(name_i, name_j) >= sim_threshold:
                should_merge = True

            if should_merge:
                used.add(j)
                # Merge aliases
                all_aliases.update(group_j[0].get('aliases', []))
                all_aliases.add(name_j)
                # Merge other fields favoring non-null values
                for key in ['appearance', 'personality', 'gender', 'age_range', 'role']:
                    if not canonical.get(key) and group_j[0].get(key):
                        canonical[key] = group_j[0][key]

        # Clean up aliases: remove canonical name from alias list
        canonical_name = canonical.get('canonical_name', '')
        all_aliases.discard(canonical_name)
        canonical['aliases'] = sorted(all_aliases)
        merged.append(canonical)
        used.add(i)

    return merged


def merge_characters_across_chunks(
    all_characters: list[dict],
    use_llm: bool = False,
    api_key: str = "",
) -> list[dict]:
    """Full coreference resolution pipeline.

    Args:
        all_characters: Raw character entries from all chunks
        use_llm: Whether to use LLM for difficult disambiguation cases
        api_key: API key for LLM mode

    Returns deduplicated, merged character list.
    """
    # Stage 1: Basic alias resolution
    resolved = resolve_aliases(all_characters)

    # Stage 2: LLM-assisted disambiguation for remaining conflicts
    if use_llm and api_key:
        resolved = _llm_disambiguate(resolved, api_key)

    # Sort by role importance
    role_order = {'protagonist': 0, 'narrator': 1, 'family_member': 2,
                  'love_interest': 3, 'supporting': 4, 'antagonist': 5}
    resolved.sort(key=lambda c: role_order.get(c.get('role', ''), 10))

    return resolved


def _llm_disambiguate(characters: list[dict], api_key: str) -> list[dict]:
    """Use LLM to help disambiguate remaining character conflicts.

    Currently a placeholder - can be extended to ask Claude about tricky pairs.
    """
    # For now, return as-is. LLM disambiguation can be added later
    # for cases where the simple heuristic approach fails.
    return characters
