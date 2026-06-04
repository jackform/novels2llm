"""Aggregate output models."""

from typing import Optional
from pydantic import BaseModel, Field

from .entities import Character, Location, Item, StoryEvent, WorldSetting
from .relationships import Dialogue, Relationship, TimelineEvent
from .scene import Scene


class SceneEvent(BaseModel):
    """An event with its associated dialogues linked by text position."""

    event_id: str = Field(description="Unique event identifier")
    description: str = Field(description="Event description")
    chapter: int = Field(description="Chapter number")
    location: Optional[str] = Field(default=None, description="Location where this event/scene takes place")
    participants: list[str] = Field(default_factory=list)
    dialogues: list[Dialogue] = Field(default_factory=list, description="Dialogues occurring within this event's text span")


class NovelMetadata(BaseModel):
    """Metadata for a novel."""

    novel_id: str = Field(description="Unique identifier (filename stem)")
    title: str = Field(description="Novel title")
    author: Optional[str] = Field(default=None)
    source: Optional[str] = Field(default=None)
    word_count: Optional[int] = Field(default=None)
    chapter_count: int = Field(default=0)
    language: str = Field(default="zh")


class NovelWorld(BaseModel):
    """Aggregate output for a single novel."""

    metadata: NovelMetadata
    characters: list[Character] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    dialogues: list[Dialogue] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    world_setting: Optional[WorldSetting] = Field(default=None)
    locations: list[Location] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    events: list[StoryEvent] = Field(default_factory=list)
    scene_events: list[SceneEvent] = Field(default_factory=list, description="Events with linked dialogues")
    scenes: list[Scene] = Field(default_factory=list, description="Location-anchored scenes with narrative units")

    def model_dump_json_pretty(self, **kwargs):
        """Serialize to pretty JSON string."""
        return self.model_dump_json(indent=2, **kwargs)
