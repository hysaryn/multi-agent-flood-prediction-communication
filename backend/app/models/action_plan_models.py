from pydantic import BaseModel
from typing import List, Literal

class Action(BaseModel):
    """A single actionable item for flood preparation/response."""
    title: str                    # e.g., "Create Emergency Kit"
    description: str              # Detailed description
    category: str                 # e.g., "evacuation", "property", "communication"
    source_doc: str | None = None  # URL of source document
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Prepare 72-hour emergency kit",
                "description": "Pack water, non-perishable food, medications, flashlight, battery radio, and important documents in waterproof container",
                "priority": "high",
                "category": "preparation",
                "source_doc": "https://example.ca/flood-guide.pdf"
            }
        }


class ActionPhase(BaseModel):
    """Actions grouped by flood phase."""
    phase: Literal["before", "during", "after"]
    actions: List[Action]
    summary: str | None = None  # Brief phase summary


class ActionPlanResponse(BaseModel):
    """Complete action plan for a location."""
    location: str
    display_name: str | None = None
    before_flood: List[Action]
    during_flood: List[Action]
    after_flood: List[Action]
    sources: List[str]  # URLs of all source documents
    generated_at: str   # ISO timestamp
    
    def total_actions(self) -> int:
        return len(self.before_flood) + len(self.during_flood) + len(self.after_flood)