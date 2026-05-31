"""Stage 3: LLM-based extraction orchestration.

Coordinates running all extractors (character, world, dialogue, relationship, event)
across all chunks, with caching.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from ..config import config
from ..models.entities import Character, Location, Item, StoryEvent, WorldSetting
from ..models.relationships import Dialogue, Relationship, TimelineEvent
from ..chunking.smart_chunker import Chunk
from ..extractors.character_extractor import CharacterExtractor
from ..extractors.world_extractor import WorldExtractor
from ..extractors.dialogue_extractor import DialogueExtractor
from ..extractors.relationship_extractor import RelationshipExtractor
from ..extractors.event_extractor import EventExtractor


@dataclass
class ExtractionResult:
    """Aggregate results from all extractors across all chunks."""

    novel_id: str
    characters: list[dict] = field(default_factory=list)
    world_settings: list[dict] = field(default_factory=list)
    dialogues: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Stage3Pipeline:
    """Orchestrates all LLM-based extraction."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        self.char_extractor = CharacterExtractor(self.api_key)
        self.world_extractor = WorldExtractor(self.api_key)
        self.dialogue_extractor = DialogueExtractor(self.api_key)
        self.rel_extractor = RelationshipExtractor(self.api_key)
        self.event_extractor = EventExtractor(self.api_key)

    def process_chunk(self, chunk: Chunk, nlp_hints: Optional[dict] = None) -> dict:
        """Process a single chunk through all extractors."""
        results = {}

        # Extract characters
        try:
            results['characters'] = self.char_extractor.extract(
                chunk.text, nlp_hints=nlp_hints,
            )
        except Exception as e:
            results.setdefault('errors', []).append(
                f"char_extract ch{chunk.chapter}/ck{chunk.chunk_index}: {e}"
            )
            results['characters'] = []

        # Extract world setting (only from first few chunks)
        if chunk.is_chapter_start and chunk.chapter <= 2:
            try:
                results['world'] = self.world_extractor.extract(
                    chunk.text, nlp_hints=nlp_hints,
                )
            except Exception as e:
                results.setdefault('errors', []).append(
                    f"world_extract ch{chunk.chapter}: {e}"
                )

        # Extract dialogues
        try:
            results['dialogues'] = self.dialogue_extractor.extract(
                chunk.text, nlp_hints=nlp_hints,
            )
        except Exception as e:
            results.setdefault('errors', []).append(
                f"dialogue_extract ch{chunk.chapter}: {e}"
            )
            results['dialogues'] = []

        return results

    def process_chapters(
        self,
        chunks: list[Chunk],
        nlp_results: Optional[dict] = None,
    ) -> ExtractionResult:
        """Process all chunks and aggregate results.

        Args:
            chunks: List of text chunks to process
            nlp_results: Optional dict mapping chapter_num -> NLPResult

        Returns aggregated ExtractionResult.
        """
        novel_id = chunks[0].novel_id if chunks else "unknown"
        result = ExtractionResult(novel_id=novel_id)
        nlp_results = nlp_results or {}

        for chunk in chunks:
            # Build NLP hints for this chunk
            nlp_hint = None
            if chunk.chapter in nlp_results:
                nlp_hint = nlp_results[chunk.chapter].to_hint_dict()

            # Try to load from cache first
            cache_key = f"{novel_id}/{chunk.chunk_index}"
            cached = self._load_cache(cache_key)
            if cached:
                result.characters.extend(cached.get('characters', []))
                result.world_settings.extend(cached.get('world_settings', []))
                result.dialogues.extend(cached.get('dialogues', []))
                continue

            # Process through all extractors
            chunk_results = self.process_chunk(chunk, nlp_hints=nlp_hint)

            result.characters.extend(chunk_results.get('characters', []))
            result.world_settings.extend(chunk_results.get('world', []))
            result.dialogues.extend(chunk_results.get('dialogues', []))
            result.errors.extend(chunk_results.get('errors', []))

            # Save to cache
            self._save_cache(cache_key, chunk_results)

        return result

    def _load_cache(self, cache_key: str) -> Optional[dict]:
        """Load cached extraction results."""
        if not config.CACHE_ENABLED:
            return None
        cache_path = config.CACHE_DIR / f"{cache_key}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_cache(self, cache_key: str, data: dict) -> None:
        """Save extraction results to cache."""
        if not config.CACHE_ENABLED:
            return
        cache_path = config.CACHE_DIR / f"{cache_key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
        except IOError:
            pass
