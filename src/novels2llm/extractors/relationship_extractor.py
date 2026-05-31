"""Relationship extraction using Claude API."""

from typing import Optional
from .base import BaseExtractor


class RelationshipExtractor(BaseExtractor):
    """Extract character relationships from novel text using Claude."""

    PROMPT_FILE = "relationship_extraction"

    def extract(
        self,
        text: str,
        known_characters: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Extract relationships from text.

        Args:
            text: The text chunk to analyze
            known_characters: Previously identified characters

        Returns list of relationship dicts.
        """
        known_str = "无"
        if known_characters:
            names = []
            for c in known_characters[:30]:
                name = c.get('canonical_name', '')
                aliases = c.get('aliases', [])
                names.append(f"{name} (别名: {', '.join(aliases[:3])})" if aliases else name)
            known_str = '\n'.join(f"- {n}" for n in names)

        prompt = self._build_prompt(
            text=text[:12000],
            known_characters=known_str,
        )

        response = self._call_claude(prompt)
        data = self._parse_json_response(response)
        return data.get('relationships', [])
