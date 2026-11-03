from __future__ import annotations

import re
from typing import List, Sequence

from .models import EchoDisposition, EchoGuardResult

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text)]


class PromptEchoGuard:
    """
    Detects STT prompt-echo hallucinations (e.g., when gpt-4o-transcribe or Whisper
    is given a domain vocabulary hint list during silence/noise and recites the hint
    terms verbatim at high log-probability).

    A transcript is flagged as REJECTED_PROMPT_ECHO if AND ONLY IF:
      1. Token overlap with the hint vocabulary is >= `overlap_threshold` (default 0.70), AND
      2. At least `min_contiguous_hint_terms` (default 5) hint terms appear consecutively
         in the exact order they were supplied in the prompt hint.

    Requiring BOTH invariants guarantees that a legitimate candidate answer enumerating
    valid domain terms in natural prose or a different order is never falsely condemned.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.70,
        min_contiguous_hint_terms: int = 5,
        min_valid_tokens: int = 2,
    ) -> None:
        self.overlap_threshold = overlap_threshold
        self.min_contiguous_hint_terms = min_contiguous_hint_terms
        self.min_valid_tokens = min_valid_tokens

    def evaluate(
        self,
        transcript: str,
        hint_vocabulary: Sequence[str],
    ) -> EchoGuardResult:
        tokens = _tokenize(transcript)
        if len(tokens) < self.min_valid_tokens:
            return EchoGuardResult(
                transcript=transcript,
                disposition=EchoDisposition.REJECTED_EMPTY_OR_NOISE,
                is_prompt_echo=False,
                hint_token_overlap_ratio=0.0,
                max_contiguous_hint_terms=0,
                token_count=len(tokens),
                reason="Transcript contains fewer tokens than minimum threshold.",
            )

        normalized_hints: List[str] = []
        hint_token_set = set()
        for item in hint_vocabulary:
            item_tokens = _tokenize(item)
            if item_tokens:
                normalized_hints.append(" ".join(item_tokens))
                hint_token_set.update(item_tokens)

        if not normalized_hints:
            return EchoGuardResult(
                transcript=transcript,
                disposition=EchoDisposition.ACCEPTED,
                is_prompt_echo=False,
                hint_token_overlap_ratio=0.0,
                max_contiguous_hint_terms=0,
                token_count=len(tokens),
                reason="No hint vocabulary provided; transcript accepted.",
            )

        matched_tokens = sum(1 for t in tokens if t in hint_token_set)
        overlap_ratio = round(matched_tokens / len(tokens), 4)

        # Measure maximum contiguous run of hint terms appearing in the exact prompt order
        normalized_transcript = " ".join(tokens)
        max_run = 0
        n_hints = len(normalized_hints)
        for start_idx in range(n_hints):
            run_len = 0
            search_pos = 0
            for idx in range(start_idx, n_hints):
                term = normalized_hints[idx]
                found_at = normalized_transcript.find(term, search_pos)
                if found_at == -1:
                    break
                # Check that the gap between consecutive hint terms is negligible (<= 2 chars)
                if run_len > 0 and (found_at - search_pos) > 2:
                    break
                run_len += 1
                search_pos = found_at + len(term)
            if run_len > max_run:
                max_run = run_len

        is_echo = (
            overlap_ratio >= self.overlap_threshold
            and max_run >= self.min_contiguous_hint_terms
        )

        if is_echo:
            return EchoGuardResult(
                transcript=transcript,
                disposition=EchoDisposition.REJECTED_PROMPT_ECHO,
                is_prompt_echo=True,
                hint_token_overlap_ratio=overlap_ratio,
                max_contiguous_hint_terms=max_run,
                token_count=len(tokens),
                reason=(
                    f"Prompt-echo hallucination detected: {overlap_ratio:.0%} token overlap "
                    f"and {max_run} contiguous ordered hint terms."
                ),
            )

        return EchoGuardResult(
            transcript=transcript,
            disposition=EchoDisposition.ACCEPTED,
            is_prompt_echo=False,
            hint_token_overlap_ratio=overlap_ratio,
            max_contiguous_hint_terms=max_run,
            token_count=len(tokens),
            reason="Transcript passed prompt-echo guardrail.",
        )
