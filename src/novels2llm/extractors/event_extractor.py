"""Event/timeline extraction using Claude API."""

from typing import Optional
from .base import BaseExtractor


class EventExtractor(BaseExtractor):
    """Extract story events from novel text using Claude."""

    PROMPT_FILE = "timeline_extraction"

    def extract(
        self,
        text: str,
        chapter: int = 1,
        chapter_title: str = "",
    ) -> list[dict]:
        """Extract events from text.

        Args:
            text: The text chunk to analyze
            chapter: Chapter number
            chapter_title: Chapter title

        Returns list of event dicts.
        """
        prompt = self._build_prompt(
            text=text[:12000],
            chapter=chapter,
            chapter_title=chapter_title,
        )

        response = self._call_claude(prompt)
        data = self._parse_json_response(response)
        return data.get('events', [])
