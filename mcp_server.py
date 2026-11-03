"""
Model Context Protocol (MCP) tool surface for Voice AI Prompt-Echo Guardrails
and Multi-Turn Oral Assessment Calibration.
"""

from __future__ import annotations

from typing import Any, Dict, List

from voice_eval_harness.echo_guard import PromptEchoGuard
from voice_eval_harness.judge import run_calibration_suite

_guard = PromptEchoGuard()


def mcp_audit_stt_prompt_echo(
    transcript: str, hint_vocabulary: List[str]
) -> Dict[str, Any]:
    """Audits an STT transcript for prompt-echo hallucination on silence/noise."""
    return _guard.evaluate(transcript, hint_vocabulary).model_dump()


def mcp_run_calibration_benchmark() -> Dict[str, Any]:
    """Executes the 4-persona oral assessment calibration suite."""
    results = run_calibration_suite()
    return {k: v.model_dump() for k, v in results.items()}
