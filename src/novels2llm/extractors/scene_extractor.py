"""Scene and narrative unit extraction using Claude API."""

from typing import Optional
from .base import BaseExtractor


class SceneExtractor(BaseExtractor):
    """Extract location-anchored scenes with narrative units from novel text.

    Replaces the old EventExtractor + dialogue attribution pipeline with
    a single call that segments by location and extracts narrative units.
    """

    PROMPT_FILE = "scene_narrative_extraction"

    def extract(
        self,
        text: str,
        chapter: int = 1,
        chapter_title: str = "",
        known_locations: Optional[list[dict]] = None,
        known_characters: Optional[list[dict]] = None,
        previous_scene_summary: Optional[str] = None,
        chunk_index: int = 0,
    ) -> list[dict]:
        """Extract scenes with narrative units from text.

        Args:
            text: The text chunk to analyze
            chapter: Chapter number
            chapter_title: Chapter title
            known_locations: Locations from world.json (list of {name, type, description, parent_location})
            known_characters: Previously identified characters
            previous_scene_summary: Summary of the last scene from the previous chunk
            chunk_index: Current chunk index

        Returns list of scene dicts with nested narrative_units.
        """
        # Build location hints
        global_location_hint = "无"
        local_location_hint = "无"

        if known_locations:
            loc_names = []
            for loc in known_locations:
                name = loc.get('name', '') if isinstance(loc, dict) else getattr(loc, 'name', '')
                loc_type = loc.get('type', '') if isinstance(loc, dict) else getattr(loc, 'type', '')
                desc = loc.get('description', '') if isinstance(loc, dict) else getattr(loc, 'description', '')
                if name:
                    parts = [name]
                    if loc_type:
                        parts.append(f"类型:{loc_type}")
                    if desc:
                        parts.append(f"描述:{desc}")
                    loc_names.append(' | '.join(parts))
            if loc_names:
                global_location_hint = '；'.join(loc_names)

            # Local: detect which known locations appear in this chunk's text
            local_matches = []
            for loc in known_locations:
                name = loc.get('name', '') if isinstance(loc, dict) else getattr(loc, 'name', '')
                if name and name in text:
                    local_matches.append(name)
            if local_matches:
                local_location_hint = '、'.join(local_matches)
            else:
                local_location_hint = "文本中未明确提及已知地点"

        # Build character hints
        character_hint = "无"
        if known_characters:
            char_names = []
            for c in known_characters[:30]:
                name = c.get('canonical_name', '') if isinstance(c, dict) else getattr(c, 'canonical_name', '')
                if name:
                    aliases = c.get('aliases', []) if isinstance(c, dict) else getattr(c, 'aliases', [])
                    if aliases:
                        char_names.append(f"{name}({'/'.join(aliases[:3])})")
                    else:
                        char_names.append(name)
            if char_names:
                character_hint = '、'.join(char_names)

        # Build previous scene hint for cross-chunk continuity
        previous_scene_hint = ""
        if previous_scene_summary:
            previous_scene_hint = (
                f"\n## 跨Chunk连续性\n"
                f"上一个chunk的最后一个场景摘要：{previous_scene_summary}\n"
                f"如果当前文本开头是上一个场景的延续（同一地点、同一批角色），"
                f"请继续使用相同的 scene_id 前缀，不要创建新的 scene_id。"
            )

        prompt = self._build_prompt(
            text=text[:12000],
            chapter=chapter,
            chapter_title=chapter_title or f"第{chapter}章",
            global_location_hint=global_location_hint,
            local_location_hint=local_location_hint,
            character_hint=character_hint,
            previous_scene_hint=previous_scene_hint,
        )

        response = self._call_claude(prompt, max_tokens=16384)
        data = self._parse_json_response(response)
        scenes = data.get('scenes', [])

        # Annotate chunk_index on each scene
        for scene in scenes:
            scene['chunk_index'] = chunk_index

        return scenes
