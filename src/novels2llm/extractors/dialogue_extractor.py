"""Dialogue extraction using Claude API."""

from typing import Optional
from .base import BaseExtractor


class DialogueExtractor(BaseExtractor):
    """Extract dialogues from novel text using Claude."""

    PROMPT_FILE = "dialogue_extraction"

    def extract(
        self,
        text: str,
        nlp_hints: Optional[dict] = None,
    ) -> list[dict]:
        """Extract dialogues from text.

        Args:
            text: The text chunk to analyze
            nlp_hints: NLP pre-annotation hints

        Returns list of dialogue dicts.
        """
        dialogue_count = 0
        speaker_hints = "未知"

        if nlp_hints:
            dialogue_count = nlp_hints.get('dialogue_count', 0)
            speakers = nlp_hints.get('dialogue_speakers', [])
            if speakers:
                speaker_hints = ', '.join(speakers)

        prompt = self._build_prompt(
            text=text[:12000],
            dialogue_count=dialogue_count,
            speaker_hints=speaker_hints,
        )

        response = self._call_claude(prompt)
        data = self._parse_json_response(response)
        return data.get('dialogues', [])
