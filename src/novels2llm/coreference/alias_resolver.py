"""Stage 4: Coreference resolution - alias disambiguation + entity merging."""

from collections import defaultdict
from difflib import SequenceMatcher

# Generic relational labels that don't carry identity info when caller is unknown.
# These are fallback-filtered only when caller is missing/unknown.
GENERIC_LABELS = {
    # Family (vertical)
    '妈妈', '爸爸', '母亲', '父亲', '妈咪', '老爸', '老妈', '爹', '娘',
    '儿子', '女儿', '孩子', '小孩',
    # Family (siblings)
    '哥哥', '弟弟', '姐姐', '妹妹', '哥', '弟', '姐', '妹',
    '大哥', '二哥', '三哥', '大姐', '二姐', '三姐',
    # Spouse
    '老婆', '丈夫', '老公', '妻子', '太太', '先生', '夫人',
    # Professional
    '老师', '医生', '护士', '老板', '师傅', '同学', '学生',
    '教授', '经理', '主任', '局长', '校长', '院长',
    # Generic
    '她', '他', '我', '你', '您',
    '小姐', '先生', '女士', '小伙子', '姑娘',
    # Intimate
    '宝贝', '甜心', '亲爱的', '达令', '小宝贝', '好宝贝',
    '亲妹妹', '亲哥哥', '好妹妹', '好哥哥', '好儿子', '好女儿',
    '亲亲哥哥', '亲亲妹妹',
}


def _name_similarity(a: str, b: str) -> float:
    """Compute string similarity between two names."""
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _identity_aliases(char: dict) -> list[str]:
    """Extract only identity aliases (not generic relational labels) from a character entry."""
    aliases = char.get('aliases', [])
    return [a for a in aliases if a not in GENERIC_LABELS]


def _alias_overlap(aliases_a: list[str], aliases_b: list[str]) -> float:
    """Check if two character entries share IDENTITY aliases (excludes generic labels)."""
    identity_a = [a for a in aliases_a if a not in GENERIC_LABELS]
    identity_b = [a for a in aliases_b if a not in GENERIC_LABELS]
    if not identity_a or not identity_b:
        return 0.0
    set_a = set(a.lower() for a in identity_a)
    set_b = set(b.lower() for b in identity_b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / min(len(set_a), len(set_b))


def _normalize_relational_labels(labels: list) -> list[dict]:
    """Normalize relational_labels to list of {caller, label} dicts.

    Handles both old format (list[str]) and new format (list[dict]).
    """
    result = []
    for item in labels:
        if isinstance(item, dict):
            caller = item.get('caller', '').strip()
            label = item.get('label', '').strip()
            if label:
                result.append({'caller': caller or 'unknown', 'label': label})
        elif isinstance(item, str):
            # Legacy format: plain string, no caller info
            item = item.strip()
            if item:
                result.append({'caller': 'unknown', 'label': item})
    return result


def _caller_label_overlap(labels_a: list, labels_b: list, caller_label_threshold: float = 0.3) -> float:
    """Check if two characters share (caller, label) pairs.

    If character X calls both A and B by the same label, they're likely
    the same person. Only pairs with a known caller (not 'unknown') count
    as strong evidence.
    """
    norm_a = _normalize_relational_labels(labels_a)
    norm_b = _normalize_relational_labels(labels_b)

    if not norm_a or not norm_b:
        return 0.0

    # Known caller pairs (strong evidence)
    pairs_a = set()
    pairs_b = set()
    for item in norm_a:
        if item['caller'] != 'unknown':
            pairs_a.add((item['caller'].lower(), item['label']))
    for item in norm_b:
        if item['caller'] != 'unknown':
            pairs_b.add((item['caller'].lower(), item['label']))

    if pairs_a and pairs_b:
        intersection = pairs_a & pairs_b
        if intersection:
            return max(0.5, len(intersection) / min(len(pairs_a), len(pairs_b)))

    # Fallback: unknown-caller labels (weaker evidence, only generic labels are ambiguous)
    unknown_a = set(item['label'] for item in norm_a if item['caller'] == 'unknown')
    unknown_b = set(item['label'] for item in norm_b if item['caller'] == 'unknown')
    if unknown_a and unknown_b:
        # Only count non-generic labels in fallback
        specific_a = unknown_a - GENERIC_LABELS
        specific_b = unknown_b - GENERIC_LABELS
        if specific_a and specific_b:
            overlap = specific_a & specific_b
            return len(overlap) / min(len(specific_a), len(specific_b))

    return 0.0


def resolve_aliases(
    characters: list[dict],
    sim_threshold: float = 0.85,
    alias_threshold: float = 0.5,
    caller_label_threshold: float = 0.3,
) -> list[dict]:
    """Merge character entries that refer to the same person.

    Strategy:
    1. Exact name match -> merge
    2. Alias overlap -> merge
    3. Caller-label overlap (NEW: same caller uses same label) -> merge
    4. Name similarity -> merge (conservative threshold)

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

    # Phase 2: Try to merge groups
    merged_groups: list[list[dict]] = [[c] for c in characters]
    return _merge_groups(merged_groups, sim_threshold, alias_threshold, caller_label_threshold)


def _merge_groups(
    groups: list[list[dict]],
    sim_threshold: float,
    alias_threshold: float,
    caller_label_threshold: float,
) -> list[dict]:
    """Merge groups of character entries into canonical characters."""
    merged = []
    used = set()

    for i, group_i in enumerate(groups):
        if i in used:
            continue

        canonical = group_i[0].copy()
        all_aliases = set(canonical.get('aliases', []))
        all_relational_labels = _normalize_relational_labels(canonical.get('relational_labels', []))

        for j, group_j in enumerate(groups):
            if j <= i or j in used:
                continue

            # Check if groups should be merged
            name_i = group_i[0].get('canonical_name', '')
            name_j = group_j[0].get('canonical_name', '')

            should_merge = False

            # Exact name match
            if name_i.lower() == name_j.lower():
                should_merge = True
            # Alias overlap (identity aliases only)
            elif _alias_overlap(
                _identity_aliases(group_i[0]) + [name_i],
                _identity_aliases(group_j[0]) + [name_j],
            ) >= alias_threshold:
                should_merge = True
            # NEW: Caller-label overlap (same caller uses same label for both)
            elif _caller_label_overlap(
                group_i[0].get('relational_labels', []),
                group_j[0].get('relational_labels', []),
                caller_label_threshold,
            ) >= caller_label_threshold:
                should_merge = True
            # Name similarity
            elif _name_similarity(name_i, name_j) >= sim_threshold:
                should_merge = True

            if should_merge:
                used.add(j)
                # Merge identity aliases
                all_aliases.update(group_j[0].get('aliases', []))
                all_aliases.add(name_j)
                # Merge relational_labels (dedup by (caller, label) key)
                j_labels = _normalize_relational_labels(group_j[0].get('relational_labels', []))
                existing_keys = {(l['caller'], l['label']) for l in all_relational_labels}
                for label in j_labels:
                    if (label['caller'], label['label']) not in existing_keys:
                        all_relational_labels.append(label)
                        existing_keys.add((label['caller'], label['label']))
                # Merge other fields favoring non-null values
                for key in ['appearance', 'personality', 'gender', 'age_range', 'role']:
                    if not canonical.get(key) and group_j[0].get(key):
                        canonical[key] = group_j[0][key]

        # Clean up: remove canonical name from alias list
        canonical_name = canonical.get('canonical_name', '')
        all_aliases.discard(canonical_name)
        canonical['aliases'] = sorted(all_aliases)
        canonical['relational_labels'] = all_relational_labels
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
    return characters
