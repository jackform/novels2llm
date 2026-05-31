"""Character extraction using Claude API."""

from typing import Optional
from .base import BaseExtractor


class CharacterExtractor(BaseExtractor):
    """Extract characters from novel text using Claude."""

    PROMPT_FILE = "character_extraction"

    def extract(
        self,
        text: str,
        nlp_hints: Optional[dict] = None,
        known_characters: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Extract characters from text.

        Args:
            text: The text chunk to analyze
            nlp_hints: NLP pre-annotation hints (entities found)
            known_characters: Previously identified characters for context

        Returns list of character dicts.
        """
        # Build entity hints string
        entity_hints = "无"
        if nlp_hints and nlp_hints.get('entities'):
            entities = nlp_hints['entities']
            entity_hints = f"文本中检测到的命名实体: {', '.join(entities[:30])}"

        known_hint = ""
        if known_characters:
            names = [c.get('canonical_name', '') for c in known_characters[:20]]
            known_hint = f"\n已知角色: {', '.join(names)}\n请检查是否有新的角色信息需要补充。"

        prompt = self._build_prompt(
            text=text[:12000],  # Truncate to manageable size
            entity_hints=entity_hints,
        ) + known_hint

        response = self._call_claude(prompt)
        data = self._parse_json_response(response)
        return data.get('characters', [])
