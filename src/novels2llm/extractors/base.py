"""Base class for LLM-based extractors."""

import json
import re
from typing import Optional
from anthropic import Anthropic


class BaseExtractor:
    """Base class for extractors that call the Claude API."""

    PROMPT_FILE: str = ""  # Override in subclasses

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Set it in .env file or "
                "pass api_key parameter."
            )
        from ..config import config
        base_url = base_url or config.ANTHROPIC_BASE_URL
        self.client = Anthropic(api_key=api_key, base_url=base_url)

    def _load_prompt(self) -> str:
        """Load the prompt template from file."""
        from ..config import config
        path = config.get_prompt_path(self.PROMPT_FILE)
        if path.exists():
            return path.read_text(encoding='utf-8')
        raise FileNotFoundError(f"Prompt file not found: {path}")

    def _build_prompt(self, text: str, **kwargs) -> str:
        """Build the full prompt by substituting template variables."""
        template = self._load_prompt()
        prompt = template.format(text=text, **kwargs)
        return prompt

    def _call_claude(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call the Claude API and return the response text."""
        from ..config import config
        from anthropic.types import TextBlock
        message = self.client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # DeepSeek and some models may return ThinkingBlock content;
        # extract only the TextBlock (actual response)
        for block in message.content:
            if isinstance(block, TextBlock):
                return block.text
        # Fallback: try .text attribute on first content block
        if hasattr(message.content[0], 'text'):
            return message.content[0].text
        # Last resort: string representation
        return str(message.content[0])

    def _parse_json_response(self, response: str) -> dict:
        """Extract JSON from Claude's response, with LLM JSON repair."""
        # Try to find JSON in markdown code block
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            # Try to find raw JSON
            m = re.search(r'\{[\s\S]*\}', response)
            if m:
                json_str = m.group(0)
            else:
                json_str = response

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Apply progressive JSON repairs common in LLM output
            repaired = self._repair_json(json_str)
            return json.loads(repaired)

    @staticmethod
    def _repair_json(json_str: str) -> str:
        """Apply common LLM JSON repairs."""
        s = json_str

        # 1. Remove BOM and invisible characters
        s = s.replace('\ufeff', '').replace('\u200b', '')

        # 2. Replace Chinese/smart quotes with straight quotes inside strings
        #    Only outside of already-escaped contexts
        s = s.replace('\u201c', '"').replace('\u201d', '"')  # " "
        s = s.replace('\u2018', "'").replace('\u2019', "'")  # ' '
        s = s.replace('\uff08', '(').replace('\uff09', ')')  # （ ）
        s = s.replace('\uff1a', ':').replace('\uff0c', ',')  # ： ，

        # 3. Remove trailing commas before ] or }
        s = re.sub(r',\s*([}\]])', r'\1', s)

        # 4. Fix missing commas between JSON elements at newline boundaries:
        #    "val"\n"key" -> "val",\n"key",  }\n"key" -> },\n"key"
        s = re.sub(r'(["}\]\d])\s*\n\s*(["\[{])', r'\1,\n\2', s)
        # Same-line: "val" "key" -> "val", "key" (two unescaped quotes with space)
        s = re.sub(r'(?<!\\)"\s+(?=")', r'", ', s)

        # 5. Fix missing closing brackets: count braces
        open_braces = s.count('{') - s.count('}')
        open_brackets = s.count('[') - s.count(']')
        s += '}' * open_braces + ']' * open_brackets

        # 6. Fix single-quoted JSON (LLMs sometimes use single quotes)
        if s.count('"') < 4 and s.count("'") > 4:
            # JSON should use double quotes; try replacing top-level single quotes
            pass  # Too risky for general case; skip

        return s

    def extract(self, text: str, **kwargs) -> list[dict]:
        """Extract information from text. Override in subclasses."""
        raise NotImplementedError
