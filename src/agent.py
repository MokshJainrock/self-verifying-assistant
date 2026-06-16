"""
agent.py — the plan-act-observe controller (the agentic core).

It WRAPS pipeline.answer(); it does not modify it. Each loop iteration:
  OBSERVE - run the existing pipeline and read its grounding/decision signals.
  PLAN    - decide stop-or-act, HARD CAP checked before any retry.
  ACT     - reformulate the query (one LLM action) and loop.

RETRY PREDICATE (this is the part the A/B measurement tuned):
  We retry ONLY on REFUSE, never on ESCALATE. The measurement showed every recovery
  came from a REFUSE (the answer existed but retrieval missed it), and every new
  hallucination came from retrying an ESCALATE on an adversarial question — partial
  grounding means "a tempting related fact is present," and reformulation completes it
  into a confident wrong answer. So ESCALATE is left exactly as the gate decided.

The controller never overrides the gate or the verifier; it only changes WHICH evidence
they see, and only when the system flatly refused. Returns the same dict
pipeline.answer() returns, plus an "agent" trace (including first_pass for clean A/B).
"""

from . import pipeline, reformulator, config


def _ungrounded_claims(verification):
    """The claim texts the verifier could NOT ground — used to steer the rewrite."""
    if verification is None:
        return []
    return [c.claim for c in verification.claims if not c.grounded]


def _improved(curr, prev) -> bool:
    """Did grounding improve enough to justify another retry? None treated as 0.0."""
    c = curr if curr is not None else 0.0
    p = prev if prev is not None else 0.0
    return (c - p) >= config.MIN_GROUNDING_IMPROVEMENT


def answer(question, top_k=config.TOP_K, allowed_sources=None):
    """Agentic wrapper around pipeline.answer(). Bounded to MAX_AGENT_STEPS+1 passes."""
    original_question = question
    query = question
    attempt = 0
    prev_grounding = None
    best = None        # best observation so far, by grounding
    first = None       # attempt-0 result = the single-pass baseline (for a clean A/B)
    trace = []

    while True:
        # ---------------- OBSERVE ----------------
        result = pipeline.answer(query, top_k=top_k, allowed_sources=allowed_sources)
        grounding = result.get("grounding_score")
        if first is None:
            first = result

        trace.append({
            "attempt": attempt,
            "query": query,
            "decision": result["decision"],
            "grounding_score": grounding,
            "verify_status": result["verify_status"],
            "est_cost_usd": result.get("est_cost_usd", 0.0),
            "latency_ms": result.get("latency_ms", 0),
            "action": None,
        })

        if best is None or (grounding or 0.0) > (best.get("grounding_score") or 0.0):
            best = result

        # ---------------- PLAN (stop or act) ----------------
        # (A) Success.
        if result["decision"] == "ANSWER":
            return _finalize(result, first, trace, attempt,
                             recovered=(attempt > 0), reason="answered")

        # (B) HARD CAP first among give-ups — guarantees termination.
        if attempt >= config.MAX_AGENT_STEPS:
            return _finalize(best, first, trace, attempt, False, "cap_reached")

        # (C) RETRY PREDICATE: only REFUSE is retry-worthy. ESCALATE is the trap —
        # leave it exactly as the gate decided. This is what the A/B fixed.
        if result["decision"] != "REFUSE":
            return _finalize(best, first, trace, attempt, False, "escalate_not_retried")

        # (D) No improvement vs the previous attempt.
        if attempt > 0 and not _improved(grounding, prev_grounding):
            return _finalize(best, first, trace, attempt, False, "no_improvement")

        # ---------------- ACT (reformulate + loop) ----------------
        new_query = reformulator.reformulate(
            original_question, query, _ungrounded_claims(result.get("verification"))
        )
        if new_query is None:
            return _finalize(best, first, trace, attempt, False, "no_reformulation")

        trace[-1]["action"] = f"reformulate -> {new_query}"
        prev_grounding = grounding
        query = new_query
        attempt += 1


def _finalize(result, first, trace, steps, recovered, reason):
    """Return pipeline's result dict (copied) plus the agent trace. Never mutates input."""
    out = dict(result)
    out["agent"] = {
        "steps_used": steps,
        "recovered": recovered,
        "stop_reason": reason,
        "attempts": trace,
        "total_cost_usd": round(sum(t["est_cost_usd"] or 0.0 for t in trace), 6),
        "total_latency_ms": sum(t["latency_ms"] or 0 for t in trace),
        # The attempt-0 result IS the single-pass baseline. Exposing it lets the A/B
        # compare single vs agentic from ONE run — no nondeterminism, half the cost.
        "first_pass": {"decision": first["decision"], "answer": first["answer"]},
    }
    return out
