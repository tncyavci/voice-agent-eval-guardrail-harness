from __future__ import annotations

from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from voice_eval_harness.echo_guard import PromptEchoGuard
from voice_eval_harness.judge import RubricJudgeEngine, run_calibration_suite
from voice_eval_harness.models import (
    EchoGuardResult,
    SealedTurnRecord,
    SessionSnapshot,
    StationEvaluationReport,
    TurnSubmissionRequest,
)
from voice_eval_harness.state_machine import (
    DeadlineExpiredError,
    OralSessionStore,
    RevisionConflictError,
    SessionNotFoundError,
)

app = FastAPI(
    title="Stateful Voice AI Examiner & Evaluation Guardrail Harness",
    version="1.0.0",
)

store = OralSessionStore(":memory:")
judge = RubricJudgeEngine()
echo_guard = PromptEchoGuard()


class CreateSessionRequest(BaseModel):
    session_id: str
    scenario_id: str
    candidate_id: str
    created_at_ms: int = Field(default=1_700_000_000_000, gt=0)
    duration_ms: int = Field(default=900_000, gt=0)
    hint_vocabulary: List[str] = Field(default_factory=list)
    required_concept_tags: List[str] = Field(default_factory=list)


class TurnResponse(BaseModel):
    snapshot: SessionSnapshot
    turn: SealedTurnRecord
    idempotent_replay: bool


class EchoAuditRequest(BaseModel):
    transcript: str
    hint_vocabulary: List[str]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/echo-guard/audit", response_model=EchoGuardResult)
def audit_echo(req: EchoAuditRequest) -> EchoGuardResult:
    return echo_guard.evaluate(req.transcript, req.hint_vocabulary)


@app.post("/v1/sessions", response_model=SessionSnapshot)
def create_session(req: CreateSessionRequest) -> SessionSnapshot:
    return store.create_session(
        session_id=req.session_id,
        scenario_id=req.scenario_id,
        candidate_id=req.candidate_id,
        created_at_ms=req.created_at_ms,
        duration_ms=req.duration_ms,
        hint_vocabulary=req.hint_vocabulary,
        required_concept_tags=req.required_concept_tags,
    )


@app.post("/v1/sessions/{session_id}/turns", response_model=TurnResponse)
def submit_turn(session_id: str, req: TurnSubmissionRequest) -> TurnResponse:
    try:
        snapshot, turn, replayed = store.submit_turn(session_id, req)
        return TurnResponse(
            snapshot=snapshot, turn=turn, idempotent_replay=replayed
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DeadlineExpiredError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/v1/sessions/{session_id}/evaluate", response_model=StationEvaluationReport
)
def evaluate_session(session_id: str) -> StationEvaluationReport:
    try:
        snapshot = store.get_session(session_id)
        return judge.evaluate_session(snapshot)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(
    "/v1/calibration/run", response_model=Dict[str, StationEvaluationReport]
)
def run_calibration() -> Dict[str, StationEvaluationReport]:
    return run_calibration_suite()
