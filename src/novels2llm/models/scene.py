"""Scene and NarrativeUnit models for location-anchored narrative extraction."""

from typing import Optional
from pydantic import BaseModel, Field


class NarrativeUnit(BaseModel):
    """A single narrative unit within a scene — dialogue, action, narration, or inner thought."""

    unit_id: str = Field(description="Unique identifier, e.g. 'ch3_sc1_u5'")
    character: str = Field(description="Character name or 'narrator'")
    text: str = Field(description="Formatted text: （动作/表情）对白 for dialogue, or plain narration text")
    type: str = Field(description="Unit type: 'dialogue', 'action', 'narration', or 'inner_thought'")
    listener: Optional[str] = Field(default=None, description="Target character for dialogue units")
    sequence_index: int = Field(description="0-based order within the scene")


class Scene(BaseModel):
    """A scene segmented by location/time, containing narrative units in chronological order."""

    scene_id: str = Field(description="Unique identifier, e.g. 'ch3_sc1'")
    chapter: int = Field(description="Chapter number")
    location: str = Field(description="Location from world.json or LLM-subdivided, e.g. '家中客厅'")
    sub_location_of: Optional[str] = Field(default=None, description="Parent location from world.json locations list")
    participants: list[str] = Field(default_factory=list, description="Characters present in this scene")
    narrative_units: list[NarrativeUnit] = Field(default_factory=list, description="Narrative units in chronological order")
    summary: Optional[str] = Field(default=None, description="Brief summary of what happens in this scene")
    time_marker: Optional[str] = Field(default=None, description="Time reference, e.g. '第二天早上', '深夜'")
    chunk_index: Optional[int] = Field(default=None, description="Which chunk this scene was extracted from")
