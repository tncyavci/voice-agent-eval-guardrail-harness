from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class AssessmentPhase(str, Enum):
    PREPARATION = "preparation"
    CASE_PRESENTATION = "case_presentation"
    TECHNICAL_PROBING = "technical_probing"
    DIAGNOSTIC_REASONING = "diagnostic_reasoning"
    SAFETY_COMPLIANCE = "safety_compliance"
    FINAL_SYNTHESIS = "final_synthesis"
    COMPLETED = "completed"


PHASE_ORDER: List[AssessmentPhase] = [
    AssessmentPhase.PREPARATION,
    AssessmentPhase.CASE_PRESENTATION,
    AssessmentPhase.TECHNICAL_PROBING,
    AssessmentPhase.DIAGNOSTIC_REASONING,
    AssessmentPhase.SAFETY_COMPLIANCE,
    AssessmentPhase.FINAL_SYNTHESIS,
    AssessmentPhase.COMPLETED,
]


class ActionSafetyClass(str, Enum):
    SAFE = "SAFE"
    INCOMPLETE = "INCOMPLETE"
    UNSAFE = "UNSAFE"


class EchoDisposition(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED_PROMPT_ECHO = "REJECTED_PROMPT_ECHO"
    REJECTED_EMPTY_OR_NOISE = "REJECTED_EMPTY_OR_NOISE"


class EchoGuardResult(BaseModel):
    transcript: str
    disposition: EchoDisposition
    is_prompt_echo: bool
    hint_token_overlap_ratio: float = Field(ge=0.0, le=1.0)
    max_contiguous_hint_terms: int = Field(ge=0)
    token_count: int = Field(ge=0)
    reason: str


class TurnSubmissionRequest(BaseModel):
    op_id: str = Field(..., min_length=4, description="Idempotency operation key")
    expected_revision: int = Field(..., ge=0, description="Optimistic concurrency CAS revision")
    raw_transcript: str
    submitted_at_ms: int = Field(..., gt=0)
    advance_phase: bool = False
    detected_concept_tags: List[str] = Field(default_factory=list)
    safety_Classification: ActionSafetyClass = ActionSafetyClass.SAFE
    self_corrected: bool = False


class SealedTurnRecord(BaseModel):
    turn_index: int
    op_id: str
    phase: AssessmentPhase
    sealed_transcript: str
    echo_guard: EchoGuardResult
    validated_concept_tags: List[str]
    safety_class: ActionSafetyClass
    self_corrected: bool
    examiner_followup: str
    recorded_at_ms: int


class SessionSnapshot(BaseModel):
    session_id: str
    scenario_id: str
    candidate_id: str
    phase: AssessmentPhase
    revision: int = Field(ge=0)
    created_at_ms: int
    deadline_at_ms: int
    grace_period_ms: int = 15000
    hint_vocabulary: List[str]
    required_concept_tags: List[str]
    recent_op_ids: List[str] = Field(default_factory=list)
    turns: List[SealedTurnRecord] = Field(default_factory=list)


class CriterionScore(BaseModel):
    criterion_id: str
    score: float = Field(ge=0.0, le=5.0)
    max_score: float = 5.0
    is_kill_criterion: bool = False
    rationale: str


class StationEvaluationReport(BaseModel):
    session_id: str
    scenario_id: str
    raw_points: float
    final_points: float
    max_points: float
    percentage: float
    passed: bool
    kill_criterion_triggered: bool
    unsafe_action_detected: bool
    hold_for_manual_review: bool
    covered_concept_tags: List[str]
    missing_concept_tags: List[str]
    criteria: List[CriterionScore]
    summary_verdict: str

    @field_validator("percentage")
    @classmethod
    def clamp_percentage(cls, v: float) -> float:
        return round(max(0.0, min(100.0, v)), 2)
