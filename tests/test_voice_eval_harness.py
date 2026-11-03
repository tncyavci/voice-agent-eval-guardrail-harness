from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import app
from voice_eval_harness.echo_guard import PromptEchoGuard
from voice_eval_harness.judge import RubricJudgeEngine, run_calibration_suite
from voice_eval_harness.models import (
    ActionSafetyClass,
    AssessmentPhase,
    EchoDisposition,
    TurnSubmissionRequest,
)
from voice_eval_harness.state_machine import (
    DeadlineExpiredError,
    OralSessionStore,
    RevisionConflictError,
)
from voice_eval_harness.taxonomy import filter_valid_tags, normalize_concept_tag


HINTS = [
    "lockout tagout",
    "zero energy",
    "root cause",
    "blast radius",
    "rollback verification",
    "emergency escalation",
]


def test_prompt_echo_guard_catches_verbatim_hint_recitation() -> None:
    guard = PromptEchoGuard()
    # Whisper / gpt-4o-transcribe reciting 6 hint terms in order on silence/noise
    echo_text = "lockout tagout, zero energy, root cause, blast radius, rollback verification, emergency escalation"
    res = guard.evaluate(echo_text, HINTS)
    assert res.is_prompt_echo is True
    assert res.disposition == EchoDisposition.REJECTED_PROMPT_ECHO
    assert res.max_contiguous_hint_terms >= 5
    assert res.hint_token_overlap_ratio >= 0.70


def test_prompt_echo_guard_allows_legitimate_domain_speech() -> None:
    guard = PromptEchoGuard()
    # Candidate uses multiple domain hint terms in natural explanation and different order
    legit_text = (
        "Before we start rollback verification or isolate the root cause, "
        "we must complete lockout tagout and confirm zero energy on the panel."
    )
    res = guard.evaluate(legit_text, HINTS)
    assert res.is_prompt_echo is False
    assert res.disposition == EchoDisposition.ACCEPTED


def test_closed_vocabulary_taxonomy_normalizes_aliases_and_rejects_unknowns() -> None:
    assert normalize_concept_tag("LOTO") == "safety.lockout_tagout_verification"
    assert normalize_concept_tag("incident.root_cause_isolation") == "incident.root_cause_isolation"
    assert normalize_concept_tag("hallucinated.tag_name") is None
    assert normalize_concept_tag("invalid-format") is None

    filtered = filter_valid_tags(["loto", "safety.lockout_tagout_verification", "unknown.concept", "rca"])
    assert filtered == [
        "safety.lockout_tagout_verification",
        "incident.root_cause_isolation",
    ]


def test_state_machine_cas_revision_and_idempotent_op_replay() -> None:
    store = OralSessionStore(":memory:")
    store.create_session(
        session_id="sess-01",
        scenario_id="scen-01",
        candidate_id="cand-01",
        created_at_ms=1000,
        duration_ms=60_000,
        hint_vocabulary=HINTS,
        required_concept_tags=["loto", "rca"],
    )

    turn_req = TurnSubmissionRequest(
        op_id="op-turn-1",
        expected_revision=1,
        raw_transcript="We apply lockout tagout before isolating the root cause.",
        submitted_at_ms=5000,
        advance_phase=True,
        detected_concept_tags=["loto", "rca"],
    )
    snap1, rec1, replayed1 = store.submit_turn("sess-01", turn_req)
    assert replayed1 is False
    assert snap1.revision == 2
    assert snap1.phase == AssessmentPhase.CASE_PRESENTATION
    assert len(snap1.turns) == 1

    # Retrying with the exact same op_id returns the sealed turn idempotently
    snap_replay, rec_replay, replayed2 = store.submit_turn("sess-01", turn_req)
    assert replayed2 is True
    assert snap_replay.revision == 2
    assert rec_replay.turn_index == rec1.turn_index
    assert len(snap_replay.turns) == 1

    # Stale expected_revision with a new op_id raises RevisionConflictError (CAS failure)
    stale_req = TurnSubmissionRequest(
        op_id="op-turn-2",
        expected_revision=1,
        raw_transcript="Second turn with stale revision.",
        submitted_at_ms=6000,
    )
    with pytest.raises(RevisionConflictError):
        store.submit_turn("sess-01", stale_req)

    # Submitting past server deadline + 15s grace raises DeadlineExpiredError
    expired_req = TurnSubmissionRequest(
        op_id="op-turn-3",
        expected_revision=2,
        raw_transcript="Late submission past grace window.",
        submitted_at_ms=1000 + 60_000 + 16_000,
    )
    with pytest.raises(DeadlineExpiredError):
        store.submit_turn("sess-01", expired_req)


def test_calibration_suite_personas_and_kill_criteria() -> None:
    reports = run_calibration_suite()

    strong = reports["strong_pass"]
    assert strong.passed is True
    assert strong.kill_criterion_triggered is False
    assert strong.hold_for_manual_review is False
    assert strong.summary_verdict == "PASSED"

    self_corr = reports["borderline_self_corrected"]
    assert self_corr.passed is True
    assert self_corr.kill_criterion_triggered is False
    assert self_corr.hold_for_manual_review is True
    assert self_corr.summary_verdict == "PROVISIONAL_PASS_PENDING_SAFETY_REVIEW"

    unsafe = reports["critical_safety_fail"]
    assert unsafe.passed is False
    assert unsafe.kill_criterion_triggered is True
    assert unsafe.final_points == 0.0
    assert unsafe.hold_for_manual_review is True
    assert unsafe.summary_verdict == "FAILED_KILL_CRITERION"

    echo = reports["silence_prompt_echo_attack"]
    assert echo.passed is False
    assert echo.kill_criterion_triggered is True
    assert echo.final_points == 0.0


def test_fastapi_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}

    create_resp = client.post(
        "/v1/sessions",
        json={
            "session_id": "api-sess-1",
            "scenario_id": "scen-api",
            "candidate_id": "cand-api",
            "created_at_ms": 1000,
            "duration_ms": 120000,
            "hint_vocabulary": HINTS,
            "required_concept_tags": ["loto", "zero_energy"],
        },
    )
    assert create_resp.status_code == 200

    turn_resp = client.post(
        "/v1/sessions/api-sess-1/turns",
        json={
            "op_id": "api-op-1",
            "expected_revision": 1,
            "raw_transcript": "We verify lockout tagout and zero energy state across all breakers.",
            "submitted_at_ms": 2000,
            "advance_phase": True,
            "detected_concept_tags": ["loto", "zero_energy"],
        },
    )
    assert turn_resp.status_code == 200
    assert turn_resp.json()["idempotent_replay"] is False

    eval_resp = client.post("/v1/sessions/api-sess-1/evaluate")
    assert eval_resp.status_code == 200
    assert eval_resp.json()["passed"] is True
