"""World setting extraction using Claude API."""

from typing import Optional
from .base import BaseExtractor


class WorldExtractor(BaseExtractor):
    """Extract world setting information from novel text."""

    PROMPT_FILE = "world_setting"

    def extract(
        self,
        text: str,
        nlp_hints: Optional[dict] = None,
    ) -> dict:
        """Extract world setting from text.

        Args:
            text: The text chunk to analyze
            nlp_hints: NLP pre-annotation hints

        Returns world setting dict.
        """
        entity_hints = "无"
        if nlp_hints and nlp_hints.get('entities'):
            entities = nlp_hints['entities']
            # Filter for location/organization entities
            entity_hints = f"检测到的地点和实体: {', '.join(entities[:20])}"

        prompt = self._build_prompt(
            text=text[:12000],
            entity_hints=entity_hints,
        )

        response = self._call_claude(prompt)
        return self._parse_json_response(response)
