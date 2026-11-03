from __future__ import annotations

from typing import Dict, List

from .models import (
    ActionSafetyClass,
    CriterionScore,
    EchoDisposition,
    SessionSnapshot,
    StationEvaluationReport,
    TurnSubmissionRequest,
)
from .state_machine import OralSessionStore


class RubricJudgeEngine:
    """
    Evaluates a completed or open multi-turn oral session snapshot against a
    5-criterion (0-5 pts each, 25 pts max) rubric with non-compensatory pass thresholds
    and a strict Kill-Criterion gate:
      - If communicative effectiveness == 0 (e.g., all turns rejected as STT prompt echo
        or empty), the entire station scores 0.
      - If any turn records an UNSAFE action that was not self-corrected, the safety
        kill-criterion triggers, capping final points to 0 and holding the station
        for mandatory human review (`hold_for_manual_review=True`).
      - If an unsafe action WAS self-corrected within the session, it does not auto-zero
        the station but still triggers `hold_for_manual_review=True` for audit.
    """

    def __init__(self, pass_ratio: float = 0.60) -> None:
        self.pass_ratio = pass_ratio

    def evaluate_session(self, snapshot: SessionSnapshot) -> StationEvaluationReport:
        accepted_turns = [
            t
            for t in snapshot.turns
            if t.echo_guard.disposition == EchoDisposition.ACCEPTED
        ]
        covered_set = {
            tag for t in accepted_turns for tag in t.validated_concept_tags
        }
        covered_tags = [
            t for t in snapshot.required_concept_tags if t in covered_set
        ]
        missing_tags = [
            t for t in snapshot.required_concept_tags if t not in covered_set
        ]

        total_required = max(1, len(snapshot.required_concept_tags))
        coverage_ratio = len(covered_tags) / total_required

        uncorrected_unsafe = any(
            t.safety_class == ActionSafetyClass.UNSAFE and not t.self_corrected
            for t in snapshot.turns
        )
        self_corrected_unsafe = any(
            t.safety_class == ActionSafetyClass.UNSAFE and t.self_corrected
            for t in snapshot.turns
        )
        any_incomplete = any(
            t.safety_class == ActionSafetyClass.INCOMPLETE for t in snapshot.turns
        )

        # Criterion 1: Communicative Effectiveness (Kill Criterion if 0)
        if not accepted_turns:
            comm_score = 0.0
        else:
            avg_tokens = sum(t.echo_guard.token_count for t in accepted_turns) / len(
                accepted_turns
            )
            comm_score = min(5.0, round(2.0 + min(3.0, avg_tokens / 6.0), 2))

        # Criterion 2: Domain Concept Coverage
        concept_score = round(5.0 * coverage_ratio, 2)

        # Criterion 3: Diagnostic & Analytical Reasoning
        reasoning_score = round(
            min(5.0, 5.0 * coverage_ratio * (1.0 if len(accepted_turns) >= 2 else 0.7)),
            2,
        )

        # Criterion 4: Operational & Clinical Safety (Kill Criterion if uncorrected UNSAFE)
        if uncorrected_unsafe:
            safety_score = 0.0
        elif self_corrected_unsafe:
            safety_score = 3.0
        elif any_incomplete:
            safety_score = 3.5
        elif accepted_turns:
            safety_score = 5.0
        else:
            safety_score = 0.0

        # Criterion 5: Protocol Structure & Independence
        structure_score = round(
            min(5.0, (comm_score * 0.4) + (concept_score * 0.6)), 2
        )

        criteria = [
            CriterionScore(
                criterion_id="communicative_effectiveness",
                score=comm_score,
                is_kill_criterion=True,
                rationale="Measures coherent, non-echoed candidate speech across turns.",
            ),
            CriterionScore(
                criterion_id="domain_concept_coverage",
                score=concept_score,
                is_kill_criterion=False,
                rationale=f"Covered {len(covered_tags)}/{len(snapshot.required_concept_tags)} required concepts.",
            ),
            CriterionScore(
                criterion_id="diagnostic_reasoning",
                score=reasoning_score,
                is_kill_criterion=False,
                rationale="Multi-turn analytical progression across phases.",
            ),
            CriterionScore(
                criterion_id="safety_and_compliance_gate",
                score=safety_score,
                is_kill_criterion=True,
                rationale="Evaluates adherence to safety interlocks and absence of uncorrected unsafe actions.",
            ),
            CriterionScore(
                criterion_id="protocol_independence",
                score=structure_score,
                is_kill_criterion=False,
                rationale="Structured synthesis without excessive examiner prompting.",
            ),
        ]

        raw_points = round(sum(c.score for c in criteria), 2)
        max_points = 25.0

        kill_triggered = (comm_score == 0.0) or uncorrected_unsafe
        final_points = 0.0 if kill_triggered else raw_points
        percentage = round((final_points / max_points) * 100.0, 2)
        passed = (not kill_triggered) and (
            (final_points / max_points) >= self.pass_ratio
        )
        hold_for_review = uncorrected_unsafe or self_corrected_unsafe

        if kill_triggered:
            verdict = "FAILED_KILL_CRITERION"
        elif passed and hold_for_review:
            verdict = "PROVISIONAL_PASS_PENDING_SAFETY_REVIEW"
        elif passed:
            verdict = "PASSED"
        else:
            verdict = "FAILED_BELOW_THRESHOLD"

        return StationEvaluationReport(
            session_id=snapshot.session_id,
            scenario_id=snapshot.scenario_id,
            raw_points=raw_points,
            final_points=final_points,
            max_points=max_points,
            percentage=percentage,
            passed=passed,
            kill_criterion_triggered=kill_triggered,
            unsafe_action_detected=(uncorrected_unsafe or self_corrected_unsafe),
            hold_for_manual_review=hold_for_review,
            covered_concept_tags=covered_tags,
            missing_concept_tags=missing_tags,
            criteria=criteria,
            summary_verdict=verdict,
        )


def run_calibration_suite() -> Dict[str, StationEvaluationReport]:
    """
    Deterministic multi-persona calibration harness verifying that:
      1. `strong_pass`: Passes cleanly with high concept coverage.
      2. `borderline_self_corrected`: Self-corrects an initial unsafe action -> passes provisionally but holds for manual safety review.
      3. `critical_safety_fail`: Commits an uncorrected unsafe action -> kill criterion zeroes final score.
      4. `silence_prompt_echo_attack`: Recites 6 hint terms verbatim -> rejected by PromptEchoGuard, scores 0.
    """
    store = OralSessionStore(":memory:")
    judge = RubricJudgeEngine()
    hints = [
        "lockout tagout",
        "zero energy",
        "root cause",
        "blast radius",
        "rollback verification",
        "emergency escalation",
    ]
    required = [
        "safety.lockout_tagout_verification",
        "safety.zero_energy_check",
        "incident.root_cause_isolation",
        "incident.rollback_verification",
    ]

    # 1. Strong pass persona
    store.create_session(
        "cal-strong", "scen-sre-01", "cand-1", 1000, 600_000, hints, required
    )
    store.submit_turn(
        "cal-strong",
        TurnSubmissionRequest(
            op_id="op-s1",
            expected_revision=1,
            raw_transcript="First we enforce lockout tagout and verify a zero energy state before touching the rack.",
            submitted_at_ms=2000,
            advance_phase=True,
            detected_concept_tags=["loto", "zero_energy"],
        ),
    )
    s_strong, _, _ = store.submit_turn(
        "cal-strong",
        TurnSubmissionRequest(
            op_id="op-s2",
            expected_revision=2,
            raw_transcript="Next we perform root cause isolation on the canary node and run rollback verification.",
            submitted_at_ms=3000,
            advance_phase=True,
            detected_concept_tags=["rca", "rollback"],
        ),
    )

    # 2. Borderline self-corrected persona
    store.create_session(
        "cal-self-correct",
        "scen-sre-01",
        "cand-2",
        1000,
        600_000,
        hints,
        required,
    )
    s_sc, _, _ = store.submit_turn(
        "cal-self-correct",
        TurnSubmissionRequest(
            op_id="op-sc1",
            expected_revision=1,
            raw_transcript="Wait, I must correct myself: before live debugging, I apply lockout tagout, zero energy check, root cause isolation, and rollback verification.",
            submitted_at_ms=2000,
            advance_phase=True,
            detected_concept_tags=["loto", "zero_energy", "rca", "rollback"],
            safety_Classification=ActionSafetyClass.UNSAFE,
            self_corrected=True,
        ),
    )

    # 3. Critical safety fail persona
    store.create_session(
        "cal-unsafe", "scen-sre-01", "cand-3", 1000, 600_000, hints, required
    )
    s_unsafe, _, _ = store.submit_turn(
        "cal-unsafe",
        TurnSubmissionRequest(
            op_id="op-u1",
            expected_revision=1,
            raw_transcript="I bypass the breaker interlock while energized to speed up root cause isolation and rollback.",
            submitted_at_ms=2000,
            advance_phase=True,
            detected_concept_tags=["rca", "rollback"],
            safety_Classification=ActionSafetyClass.UNSAFE,
            self_corrected=False,
        ),
    )

    # 4. Silence prompt-echo hallucination persona
    store.create_session(
        "cal-echo", "scen-sre-01", "cand-4", 1000, 600_000, hints, required
    )
    s_echo, _, _ = store.submit_turn(
        "cal-echo",
        TurnSubmissionRequest(
            op_id="op-e1",
            expected_revision=1,
            raw_transcript="lockout tagout, zero energy, root cause, blast radius, rollback verification, emergency escalation",
            submitted_at_ms=2000,
            advance_phase=True,
            detected_concept_tags=["loto", "zero_energy", "rca", "rollback"],
        ),
    )

    return {
        "strong_pass": judge.evaluate_session(s_strong),
        "borderline_self_corrected": judge.evaluate_session(s_sc),
        "critical_safety_fail": judge.evaluate_session(s_unsafe),
        "silence_prompt_echo_attack": judge.evaluate_session(s_echo),
    }
