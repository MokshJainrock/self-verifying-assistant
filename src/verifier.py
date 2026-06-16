"""
verifier.py — the Verifier AGENT. This is the heart of Layer 2.

Its job: given the drafted answer and the SAME source chunks the Responder saw,
independently decompose the answer into atomic factual claims and decide, for each,
whether the sources actually support it. It is built on Claude (a different model
from the Responder) so the two don't share hallucination failure modes.

WHY structured output (Pydantic schema): the gate and the eval harness need a
machine-readable per-claim verdict, not prose. We ask Claude to return a typed
object so `grounded` is a real boolean we can count, not text we have to parse.
"""

from typing import Optional

import anthropic
from pydantic import BaseModel

from . import config

# The SDK reads ANTHROPIC_API_KEY from the environment. One client, reused.
_client = anthropic.Anthropic()


# --- The contract we force Claude's output into -----------------------------
# Each claim Claude extracts becomes one of these. `grounded` is the field that
# matters; `reasoning` makes every verdict auditable (this is our explainability,
# which is why we don't need to surface Claude's internal thinking trace).
class ClaimCheck(BaseModel):
    claim: str                        # one atomic factual statement from the answer
    grounded: bool                    # True only if the SOURCES support it
    supporting_source: Optional[int]  # 1-based source number, or None if unsupported
    reasoning: str                    # short justification — why grounded or not


class VerificationResult(BaseModel):
    claims: list[ClaimCheck]


# WHY this prompt is strict: the Verifier must judge ONLY against the provided
# sources (not Claude's own world knowledge), and must NOT trust the answer's own
# citations — it re-checks grounding from scratch. That independence is the point.
_SYSTEM = """You are an independent fact-checker. You are given SOURCES and an ANSWER.
Your task:
1. Break the ANSWER into its atomic factual claims (one verifiable fact each).
2. For EACH claim, decide whether the SOURCES explicitly support it.

Rules:
- Judge ONLY against the provided sources. Ignore your own world knowledge.
- Do NOT trust any citation markers in the answer; verify grounding yourself.
- A claim is "grounded" only if a source clearly states or directly entails it.
- If a claim is not supported by any source, mark grounded = false and
  supporting_source = null.
- Be strict: plausible-but-unstated is NOT grounded."""


def _format_sources(chunks) -> str:
    """Render chunks as a numbered block the Verifier can reference by number."""
    return "\n\n".join(
        f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1)
    )


# Models that support adaptive thinking. Haiku 4.5 and older models do NOT — sending
# `thinking: {type: "adaptive"}` to them returns a 400. Keep this list small and explicit
# so swapping VERIFIER_MODEL never silently breaks the verifier.
_ADAPTIVE_THINKING_MODELS = ("opus-4-6", "opus-4-7", "opus-4-8", "sonnet-4-6", "fable-5")


def _supports_adaptive_thinking(model: str) -> bool:
    return any(tag in model for tag in _ADAPTIVE_THINKING_MODELS)


def verify(answer_text: str, chunks: list) -> VerificationResult:
    """
    Run the Verifier. Returns a VerificationResult (list of per-claim checks).

    Fails SAFE: if Claude errors or refuses, we return an empty result, which the
    gate reads as "could not verify" -> it will escalate rather than auto-answer.
    A verifier that silently passes on failure would defeat its own purpose.
    """
    user = (
        f"SOURCES:\n{_format_sources(chunks)}\n\n"
        f"ANSWER TO CHECK:\n{answer_text}"
    )

    # messages.parse() validates Claude's reply against our Pydantic schema.
    # Adaptive thinking helps verification (a reasoning task) but is only supported on
    # some models — Haiku 4.5 rejects it with a 400. So we add it ONLY for models that
    # support it, and omit it otherwise (structured outputs still work without it).
    kwargs = dict(
        model=config.VERIFIER_MODEL,
        max_tokens=4000,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_format=VerificationResult,
    )
    if _supports_adaptive_thinking(config.VERIFIER_MODEL):
        kwargs["thinking"] = {"type": "adaptive"}

    try:
        resp = _client.messages.parse(**kwargs)
        # On a safety refusal, parsed_output can be None — treat as "couldn't verify".
        return resp.parsed_output or VerificationResult(claims=[])
    except anthropic.APIError as e:
        # Network/auth/rate-limit/etc. Don't crash the pipeline; escalate instead.
        print(f"[verifier] API error, failing safe to escalate: {e}")
        return VerificationResult(claims=[])


def grounding_score(result: VerificationResult) -> Optional[float]:
    """
    Fraction of claims that are grounded, in [0, 1].

    Returns None when there are zero claims (e.g., verification failed, or the
    answer had nothing checkable) — the gate treats None as "unknown" -> escalate.
    """
    if not result.claims:
        return None
    grounded = sum(1 for c in result.claims if c.grounded)
    return grounded / len(result.claims)
