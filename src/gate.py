"""
gate.py — the Confidence Gate.

Combines the available signals into one explainable decision:
  1. retrieval strength  — did we even find relevant context? (best chunk distance)
  2. responder abstention — did the drafting model decline? (answerable flag)
  3. grounding score      — did the Verifier find the answer supported?
  4. verify_status        — was the verifier run, skipped (fast path), or did it fail?

Output decision is one of:
  ANSWER   — confident & grounded; safe to return to the user
  ESCALATE — answerable but not fully grounded / unverifiable; send to a human reviewer
  REFUSE   — no answer to give (model abstained or retrieval failed)

WHY a rule-based gate (not ML): every decision must be explainable in one sentence.
That explainability IS the product. A learned classifier would hide the "why".
"""

from dataclasses import dataclass, field
from typing import Optional

from . import config


@dataclass
class GateDecision:
    decision: str               # "ANSWER" | "ESCALATE" | "REFUSE"
    confidence: float           # 0..1
    reasons: list = field(default_factory=list)  # human-readable justification(s)


def decide(*, answerable: bool, min_distance: Optional[float],
           grounding: Optional[float], verify_status: str = "verified") -> GateDecision:
    """
    Apply the gate rules in priority order. Order matters: we check the cheapest,
    most decisive failure conditions first (no answer, no context) before grounding.

    verify_status:
      "verified"      -> the Verifier ran; use `grounding`.
      "skipped_strong"-> verification skipped because retrieval was extremely strong
                         (selective-verification fast path) -> trust + answer.
      "failed"        -> the Verifier errored/returned nothing -> escalate.
    """
    reasons = []

    # Rule 1 — the Responder abstained. There is literally no answer to verify.
    if not answerable:
        reasons.append("Responder abstained (insufficient grounding in sources).")
        return GateDecision("REFUSE", 0.0, reasons)

    # Rule 2 — retrieval was too weak. Even the best chunk is far from the query,
    # so any answer would be built on irrelevant context. Don't risk it.
    if min_distance is None or min_distance > config.RETRIEVAL_MAX_DISTANCE:
        worst = min_distance if min_distance is not None else float("inf")
        reasons.append(
            f"Weak retrieval (best distance {worst:.3f} "
            f"> {config.RETRIEVAL_MAX_DISTANCE}); no relevant source found."
        )
        return GateDecision("REFUSE", 0.0, reasons)

    # Rule 3 — selective-verification FAST PATH. Retrieval was extremely strong, so we
    # skipped the costly Verifier call and trust the grounded draft. Confidence is
    # derived from retrieval strength (closer match -> higher confidence).
    if verify_status == "skipped_strong":
        conf = round(1.0 - min_distance, 2)
        reasons.append(
            f"Verifier skipped — retrieval very strong (distance {min_distance:.3f} "
            f"<= {config.STRONG_RETRIEVAL_DISTANCE}); trusting grounded draft."
        )
        return GateDecision("ANSWER", conf, reasons)

    # Rule 4 — verification could not run / produced no checkable claims.
    if verify_status == "failed" or grounding is None:
        reasons.append("Verifier could not confirm grounding; routing to human review.")
        return GateDecision("ESCALATE", 0.0, reasons)

    # Rule 5 — fully grounded AND retrieval was strong -> safe to answer.
    if grounding >= config.GROUNDING_FULL:
        reasons.append(f"All claims grounded (score {grounding:.2f}); retrieval strong.")
        return GateDecision("ANSWER", grounding, reasons)

    # Rule 6 — partially grounded. Some claim isn't supported = possible hallucination.
    severity = "low grounding — likely hallucination" if grounding < config.GROUNDING_LOW \
        else "partial grounding — some unsupported claims"
    reasons.append(f"{severity} (score {grounding:.2f}); routing to human review.")
    return GateDecision("ESCALATE", grounding, reasons)
