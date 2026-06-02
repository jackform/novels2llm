"""Core entity models for novel analysis."""

from typing import Optional
from pydantic import BaseModel, Field


class Character(BaseModel):
    """A character extracted from the novel."""

    canonical_name: str = Field(description="Primary/canonical name of the character")
    aliases: list[str] = Field(default_factory=list, description="Identity aliases: name variants and nicknames (NOT relational labels like '妈妈', '哥哥')")
    relational_labels: list[dict] = Field(default_factory=list, description="Who calls this character what. Each entry: {caller: str, label: str}. e.g. [{'caller': '马小伟', 'label': '妈妈'}, {'caller': '老王', 'label': '老婆'}]. These are speaker-dependent but can aid deduplication when (caller, label) pairs overlap.")
    gender: Optional[str] = Field(default=None, description="male, female, or other")
    age_range: Optional[str] = Field(default=None, description="e.g., '18-25', 'middle-aged', 'elderly'")
    appearance: Optional[str] = Field(default=None, description="Physical description")
    personality: Optional[str] = Field(default=None, description="Personality traits and characteristics")
    role: Optional[str] = Field(default=None, description="Role in the story (protagonist, antagonist, supporting, etc.)")
    first_chapter: Optional[int] = Field(default=None, description="First chapter this character appears in")
    source_chunks: list[int] = Field(default_factory=list, description="Chunk indices where this character was mentioned")


class Location(BaseModel):
    """A location/world setting element."""

    name: str = Field(description="Name of the location")
    type: str = Field(default="other", description="e.g., city, building, school, natural_feature, realm")
    description: Optional[str] = Field(default=None, description="Description of the location")
    parent_location: Optional[str] = Field(default=None, description="Parent/hierarchical location name")


class Item(BaseModel):
    """A significant item mentioned in the novel."""

    name: str = Field(description="Name of the item")
    type: str = Field(default="other", description="e.g., artifact, weapon, clothing, tool")
    description: Optional[str] = Field(default=None, description="Description of the item")
    owner: Optional[str] = Field(default=None, description="Character who owns/uses this item")


class StoryEvent(BaseModel):
    """A significant story event."""

    event_id: str = Field(description="Unique identifier for this event")
    description: str = Field(description="Description of what happened")
    participants: list[str] = Field(default_factory=list, description="Character names involved")
    chapter: Optional[int] = Field(default=None, description="Chapter number where this occurs")
    order: Optional[int] = Field(default=None, description="Relative ordering within chapter/timeline")
    timestamp: Optional[str] = Field(default=None, description="In-story time reference if available")


class WorldSetting(BaseModel):
    """Aggregate world setting information."""

    era: Optional[str] = Field(default=None, description="Time period or era of the story")
    genre: Optional[str] = Field(default=None, description="Genre classification")
    summary: Optional[str] = Field(default=None, description="Summary of the world/universe")
    special_rules: list[str] = Field(default_factory=list, description="Special rules, magic systems, or unusual mechanics")
    key_themes: list[str] = Field(default_factory=list, description="Key themes in the novel")
