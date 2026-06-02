"""Character extraction using Claude API."""

import json
import datetime
from pathlib import Path
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
        try:
            data = self._parse_json_response(response)
            return data.get('characters', [])
        except Exception:
            # Check if response looks truncated (ended mid-structure)
            stripped = response.strip().rstrip('`').strip()
            is_truncated = (
                not stripped.endswith('}') or
                len(response) < 200 or
                stripped.endswith('"') or
                stripped.endswith(':')
            )
            if is_truncated:
                print(f"  [RETRY] Response truncated ({len(response)} chars), retrying with more tokens...")
                response = self._call_claude(prompt, max_tokens=8192)
                try:
                    data = self._parse_json_response(response)
                    return data.get('characters', [])
                except Exception:
                    pass

            # Save raw response for debugging
            debug_dir = Path('data/output/debug')
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            debug_file = debug_dir / f'char_extract_fail_{ts}.txt'
            debug_file.write_text(response, encoding='utf-8')
            print(f"  [DEBUG] Raw response saved to {debug_file}")
            return []
