from __future__ import annotations

import threading
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class IntentType(str, Enum):
    SEARCH_AND_FIND   = "search_and_find"
    INSPECT_AREA      = "inspect_area"
    TRACK_TARGET      = "track_target"
    VERIFY_EVENT      = "verify_event"
    DELIVER_OR_REPORT = "deliver_or_report"
    PATROL_OR_EXPLORE = "patrol_or_explore"
    PRIORITY_OVERRIDE = "priority_override"
    CANCEL_MISSION    = "cancel_mission"
    CANCEL_STEP       = "cancel_step"
    HIDE_OR_RELOCATE  = "hide_or_relocate"
    UNKNOWN           = "unknown"


class PriorityLevel(str, Enum):
    LOW    = "low"
    NORMAL = "normal"
    HIGH   = "high"
    URGENT = "urgent"


class StepStatus(str, Enum):
    PENDING   = "pending"
    ACTIVE    = "active"
    COMPLETED = "completed"
    FAILED    = "failed"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"


class SourceType(str, Enum):
    USER_COMMAND         = "user_command"
    USER_LIVE_OBSERVATION = "user_live_observation"
    OPERATOR_OVERRIDE    = "operator_override"
    SYSTEM_GENERATED     = "system_generated"


class ActionType(str, Enum):
    REACH_FLOOR               = "reach_floor"
    NAVIGATE_TO_AREA          = "navigate_to_area"
    SEARCH_AREA               = "search_area"
    SEARCH_ROOM               = "search_room"
    INSPECT_AREA              = "inspect_area"
    INSPECT_OBJECT_CANDIDATE  = "inspect_object_candidate"
    TRACK_ENTITY              = "track_entity"
    VERIFY_EVENT              = "verify_event"
    REPORT_RESULT             = "report_result"
    WAIT_AND_OBSERVE          = "wait_and_observe"
    REACQUIRE_TARGET          = "reacquire_target"
    ABORT_AND_REPLAN          = "abort_and_replan"
    ENUMERATE_SEARCHABLE_AREAS = "enumerate_searchable_areas"
    MARK_AREA_SEARCHED        = "mark_area_searched"
    CANCEL_ACTIVE_MISSION     = "cancel_active_mission"
    CANCEL_ACTIVE_STEP        = "cancel_active_step"
    APPROACH_TARGET           = "approach_target"
    SAFE_LAND                 = "safe_land"
    COMPLETE_MISSION          = "complete_mission"
    RESUME_PREVIOUS_PLAN      = "resume_previous_plan"
    PAUSE_CURRENT_PLAN        = "pause_current_plan"
    HIDE_OR_SEEK_COVER        = "hide_or_seek_cover"
    RETURN_TO_BASE            = "return_to_base"


# ---------------------------------------------------------------------------
# MissionContext — navigasyon durumu (node'lar arası paylaşılan)
# ---------------------------------------------------------------------------

class MissionContext:
    """Ajanın şu anki fiziksel konumunu thread-safe tutar.

    Execution Monitor her adım tamamlandığında günceller.
    Interpreter ve Planner node'ları context'i prompt'a enjekte etmek için okur.

    Kullanım:
        ctx = MissionContext()
        ctx.update(floor=3)
        ctx.update(room="207")
        d = ctx.as_dict()   # {"current_floor": 3, "current_room": "207", ...}
    """

    def __init__(self) -> None:
        self.current_floor: Optional[int] = None
        self.current_room: Optional[str] = None
        self.current_area: Optional[str] = None
        self._lock = threading.Lock()

    def update(
        self,
        floor: Optional[int] = None,
        room: Optional[str] = None,
        area: Optional[str] = None,
    ) -> None:
        with self._lock:
            if floor is not None:
                self.current_floor = floor
            if room is not None:
                self.current_room = room
            if area is not None:
                self.current_area = area

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "current_floor": self.current_floor,
                "current_room":  self.current_room,
                "current_area":  self.current_area,
            }

    def reset(self) -> None:
        with self._lock:
            self.current_floor = None
            self.current_room  = None
            self.current_area  = None


# ---------------------------------------------------------------------------
# Pydantic modeller
# ---------------------------------------------------------------------------

class ObjectTarget(BaseModel):
    type: str = Field(..., min_length=1)
    attributes: List[str] = Field(default_factory=list)

    @field_validator("attributes", mode="before")
    @classmethod
    def none_to_empty_list(cls, v):
        return [] if v is None else v


class EntityTarget(BaseModel):
    type: str = Field(..., min_length=1)
    attributes: List[str] = Field(default_factory=list)

    @field_validator("attributes", mode="before")
    @classmethod
    def none_to_empty_list(cls, v):
        return [] if v is None else v


class LocationHint(BaseModel):
    floor:    Optional[int] = None
    room:     Optional[str] = None
    corridor: Optional[str] = None
    area:     Optional[str] = None


class InterpretedCommand(BaseModel):
    command_id: str = Field(..., min_length=1)
    mission_id: str = Field(..., min_length=1)
    timestamp:  str = Field(..., min_length=1)

    intent:   IntentType
    priority: PriorityLevel = PriorityLevel.NORMAL
    source:   SourceType

    target:        Optional[ObjectTarget] = None
    target_area:   Optional[str] = None
    target_entity: Optional[EntityTarget] = None
    location_hint: Optional[LocationHint] = None

    constraints:         List[str] = Field(default_factory=list)
    success_criteria:    List[str] = Field(default_factory=list)
    secondary_objectives: List[str] = Field(default_factory=list)
    execution_modifiers:  List[str] = Field(default_factory=list)

    justification: str = Field(..., min_length=1)
    raw_text:      str = Field(..., min_length=1)

    @field_validator(
        "constraints",
        "success_criteria",
        "secondary_objectives",
        "execution_modifiers",
        mode="before",
    )
    @classmethod
    def none_to_empty_list(cls, v):
        return [] if v is None else v

    @model_validator(mode="after")
    def validate_by_intent(self) -> "InterpretedCommand":
        if self.intent == IntentType.SEARCH_AND_FIND and self.target is None:
            raise ValueError("search_and_find requires 'target'")

        if self.intent == IntentType.INSPECT_AREA:
            if self.target_area is None and self.location_hint is None:
                raise ValueError("inspect_area requires 'target_area' or 'location_hint'")

        if self.intent == IntentType.TRACK_TARGET:
            if self.target_entity is None and self.target is None:
                raise ValueError("track_target requires 'target_entity' or 'target'")

        if self.intent == IntentType.VERIFY_EVENT:
            if (
                self.target_entity is None
                and self.target_area is None
                and self.location_hint is None
            ):
                raise ValueError("verify_event requires event/entity/area information")

        if self.intent == IntentType.PATROL_OR_EXPLORE:
            if self.target_area is None and self.location_hint is None:
                raise ValueError("patrol_or_explore requires an area or location hint")

        if self.intent == IntentType.PRIORITY_OVERRIDE:
            if (
                self.target_area is None
                and self.location_hint is None
                and self.target is None
                and self.target_entity is None
                and not self.secondary_objectives
            ):
                raise ValueError(
                    "priority_override requires a new target/area/entity or secondary objective"
                )

        return self


class TaskStep(BaseModel):
    step_id: str = Field(..., min_length=1)
    action:  ActionType
    target:  Optional[str] = None

    reason:                str = Field(..., min_length=1)
    target_selection_reason: str = Field(..., min_length=1)
    safety_rationale:      str = Field(..., min_length=1)

    decision_basis:    List[str] = Field(default_factory=list)
    assumptions:       List[str] = Field(default_factory=list)
    uncertainty_notes: List[str] = Field(default_factory=list)
    evidence:          List[str] = Field(default_factory=list)

    expected_outcome:  str = Field(..., min_length=1)
    success_condition: str = Field(..., min_length=1)
    preconditions:     List[str] = Field(default_factory=list)
    fallback_if_failed: Optional[str] = None

    interruptible: bool = True
    priority:      PriorityLevel = PriorityLevel.NORMAL
    status:        StepStatus = StepStatus.PENDING

    @field_validator("target", mode="before")
    @classmethod
    def coerce_target_to_string(cls, v):
        return None if v is None else str(v)

    @field_validator(
        "decision_basis",
        "assumptions",
        "uncertainty_notes",
        "evidence",
        "preconditions",
        mode="before",
    )
    @classmethod
    def none_to_empty_list(cls, v):
        return [] if v is None else v


class TaskPlan(BaseModel):
    command_id: str = Field(..., min_length=1)
    mission_id: str = Field(..., min_length=1)
    timestamp:  str = Field(..., min_length=1)

    plan_reasoning: str = Field(..., min_length=1)
    replannable:    bool = True
    steps:          List[TaskStep] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_step_ids_unique(self) -> "TaskPlan":
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("All step_id values in TaskPlan must be unique")
        return self