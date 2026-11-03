from __future__ import annotations

import json
import sqlite3
from typing import List, Optional, Tuple

from .echo_guard import PromptEchoGuard
from .models import (
     PHASE_ORDER,
    ActionSafetyClass,
    AssessmentPhase,
    EchoDisposition,
    SealedTurnRecord,
    SessionSnapshot,
    TurnSubmissionRequest,
)
from .taxonomy import filter_valid_tags


class RevisionConflictError(Exception):
    """Raised when an optimistic-concurrency CAS check fails."""


class DeadlineExpiredError(Exception):
    """Raised when a turn is submitted past deadline_at_ms + grace_period_ms."""


class SessionNotFoundError(Exception):
    """Raised when a session ID does not exist."""


class OralSessionStore:
    """
    SQLite-backed stateful multi-turn oral examination store enforcing:
      - Compare-And-Swap (`revision` CAS) optimistic concurrency
      - Idempotent turn replay via `recent_op_ids`
      - Server-authoritative deadlines (+15s grace window)
      - Prompt-echo STT hallucination rejection
      - Append-only sealed transcript & safety classification log
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._echo_guard = PromptEchoGuard()
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS oral_sessions (
                session_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL,
                snapshot_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def create_session(
        self,
        session_id: str,
        scenario_id: str,
        candidate_id: str,
        created_at_ms: int,
        duration_ms: int,
        hint_vocabulary: List[str],
        required_concept_tags: List[str],
    ) -> SessionSnapshot:
        snapshot = SessionSnapshot(
            session_id=session_id,
            scenario_id=scenario_id,
            candidate_id=candidate_id,
            phase=AssessmentPhase.PREPARATION,
            revision=1,
            created_at_ms=created_at_ms,
            deadline_at_ms=created_at_ms + duration_ms,
            hint_vocabulary=hint_vocabulary,
            required_concept_tags=filter_valid_tags(required_concept_tags),
        )
        self._conn.execute(
            "INSERT INTO oral_sessions (session_id, revision, snapshot_json) VALUES (?, ?, ?)",
            (session_id, snapshot.revision, snapshot.model_dump_json()),
        )
        self._conn.commit()
        return snapshot

    def get_session(self, session_id: str) -> SessionSnapshot:
        row = self._conn.execute(
            "SELECT snapshot_json FROM oral_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        return SessionSnapshot.model_validate_json(row[0])

    def submit_turn(
        self,
        session_id: str,
        req: TurnSubmissionRequest,
    ) -> Tuple[SessionSnapshot, SealedTurnRecord, bool]:
        """
        Submits a candidate turn. Returns (snapshot, turn_record, was_idempotent_replay).
        """
        snapshot = self.get_session(session_id)

        # 1. Idempotent replay check (`op_id` deduplication)
        if req.op_id in snapshot.recent_op_ids:
            for existing_turn in snapshot.turns:
                if existing_turn.op_id == req.op_id:
                    return snapshot, existing_turn, True

        # 2. Optimistic Concurrency Control (CAS revision check)
        if req.expected_revision != snapshot.revision:
            raise RevisionConflictError(
                f"CAS revision mismatch for session '{session_id}': "
                f"expected {req.expected_revision}, actual {snapshot.revision}."
            )

        # 3. Server-authoritative deadline enforcement (+15s grace)
        if req.submitted_at_ms > (snapshot.deadline_at_ms + snapshot.grace_period_ms):
            raise DeadlineExpiredError(
                f"Submission at {req.submitted_at_ms} exceeds deadline "
                f"{snapshot.deadline_at_ms} + {snapshot.grace_period_ms}ms grace."
            )

        # 4. STT Prompt-Echo Hallucination Guardrail
        echo_res = self._echo_guard.evaluate(
            req.raw_transcript, snapshot.hint_vocabulary
        )

        # If prompt echo or noise is detected, zero concepts are credited
        if echo_res.disposition != EchoDisposition.ACCEPTED:
            validated_tags: List[str] = []
            followup = (
                "I could not hear a clear response. Please state your reasoning in your own words."
            )
        else:
            validated_tags = filter_valid_tags(req.detected_concept_tags)
            followup = self._build_neutral_followup(
                snapshot, validated_tags, req.advance_phase
            )

        turn_record = SealedTurnRecord(
            turn_index=len(snapshot.turns) + 1,
            op_id=req.op_id,
            phase=snapshot.phase,
            sealed_transcript=(
                req.raw_transcript
                if echo_res.disposition == EchoDisposition.ACCEPTED
                else ""
            ),
            echo_guard=echo_res,
            validated_concept_tags=validated_tags,
            safety_class=req.safety_Classification,
            self_corrected=req.self_corrected,
            examiner_followup=followup,
            recorded_at_ms=req.submitted_at_ms,
        )

        snapshot.turns.append(turn_record)
        snapshot.recent_op_ids = (snapshot.recent_op_ids + [req.op_id])[-32:]

        if req.advance_phase and echo_res.disposition == EchoDisposition.ACCEPTED:
            curr_idx = PHASE_ORDER.index(snapshot.phase)
            if curr_idx + 1 < len(PHASE_ORDER):
                snapshot.phase = PHASE_ORDER[curr_idx + 1]

        new_revision = snapshot.revision + 1
        snapshot.revision = new_revision

        cur = self._conn.execute(
            """
            UPDATE oral_sessions
               SET revision = ?, snapshot_json = ?
             WHERE session_id = ? AND revision = ?
            """,
            (
                new_revision,
                snapshot.model_dump_json(),
                session_id,
                req.expected_revision,
            ),
        )
        self._conn.commit()
        if cur.rowcount != 1:
            raise RevisionConflictError("Concurrent update failed CAS update.")

        return snapshot, turn_record, False

    @staticmethod
    def _build_neutral_followup(
        snapshot: SessionSnapshot,
        new_tags: List[str],
        advancing: bool,
    ) -> str:
        covered = {t for turn in snapshot.turns for t in turn.validated_concept_tags}
        covered.update(new_tags)
        missing = [t for t in snapshot.required_concept_tags if t not in covered]
        if advancing:
            return "Thank you. Let us proceed to the next station phase."
        if missing:
            next_domain = missing[0].split(".")[0]
            return f"Could you elaborate further on your {next_domain} protocol?"
        return "Understood. Please summarize your final verification steps."
