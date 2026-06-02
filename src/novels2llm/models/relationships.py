"""Relationship and dialogue models."""

from typing import Optional
from pydantic import BaseModel, Field


class Dialogue(BaseModel):
    """A piece of dialogue extracted from the novel."""

    speaker: str = Field(description="Character name of the speaker")
    listener: Optional[str] = Field(default=None, description="Character being spoken to")
    content: str = Field(description="The dialogue text (without quotes)")
    context: Optional[str] = Field(default=None, description="Surrounding context/narration for this line")
    chapter: Optional[int] = Field(default=None, description="Chapter number")
    line_index: Optional[int] = Field(default=None, description="Order within chapter")


class Relationship(BaseModel):
    """A relationship between two characters."""

    character_a: str = Field(description="First character (canonical name)")
    character_b: str = Field(description="Second character (canonical name)")
    rel_type: str = Field(description="Type of relationship: spouse, parent, child, sibling, friend, rival, etc.")
    direction: str = Field(default="bidirectional", description="a_to_b, b_to_a, or bidirectional")
    intimacy_level: Optional[str] = Field(default=None, description="close, intimate, distant, hostile, etc.")
    a_calls_b: list[str] = Field(default_factory=list, description="How character_a addresses character_b in dialogue (e.g. '妈咪', '老师')")
    b_calls_a: list[str] = Field(default_factory=list, description="How character_b addresses character_a in dialogue (e.g. '小儒', '儿子')")
    evidence: list[str] = Field(default_factory=list, description="Text snippets supporting this relationship")
    source_chapter: Optional[int] = Field(default=None, description="Chapter where this relationship is established")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score for this relationship")


class TimelineEvent(BaseModel):
    """An event placed on the story timeline."""

    event_id: str
    description: str
    chapter: int
    chapter_order: int  # Order within chapter
    global_order: Optional[int] = Field(default=None, description="Global timeline position")
    participants: list[str] = Field(default_factory=list)
    causal_before: list[str] = Field(default_factory=list, description="Event IDs that must precede this one")
    causal_after: list[str] = Field(default_factory=list, description="Event IDs that must follow this one")
    time_marker: Optional[str] = Field(default=None, description="Explicit time reference if available")
