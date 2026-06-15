"""
pipeline.py — orchestration. Layer 2 inserts verify -> gate; the scaling layer adds
selective verification (cost control), access-control passthrough, and per-query logging.

Same single function the dashboard AND the eval harness call, so what we measure in
Layer 3 is exactly what we ship.
"""

import json
import os
import threading
import time

from . import retriever, responder, verifier, gate, config

# Appending to the log from many concurrent requests must not interleave lines.
_log_lock = threading.Lock()


def _estimate_cost(responder_chars, verifier_chars, verified) -> float:
    """
    Rough USD estimate for one query (~4 chars/token). Approximate on purpose — the
    point is to make per-query cost VISIBLE in the log so cost-at-scale is observable.
    Responder runs every query; Verifier only when we actually called it.
    """
    r_in = (responder_chars / 4) / 1_000_000 * config.RESPONDER_PRICE_IN
    r_out = (200 / 4) / 1_000_000 * config.RESPONDER_PRICE_OUT  # short JSON answer
    cost = r_in + r_out
    if verified:
        v_in = (verifier_chars / 4) / 1_000_000 * config.VERIFIER_PRICE_IN
        v_out = (600 / 4) / 1_000_000 * config.VERIFIER_PRICE_OUT
        cost += v_in + v_out
    return round(cost, 6)


def _log(record: dict):
    """Append one JSON line to the query log (the audit trail / metrics source)."""
    os.makedirs(os.path.dirname(config.QUERY_LOG), exist_ok=True)
    with _log_lock:
        with open(config.QUERY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def answer(question, top_k=config.TOP_K, allowed_sources=None):
    t0 = time.time()

    # --- Retrieve (optionally access-controlled to the caller's allowed sources) ---
    chunks = retriever.retrieve(question, top_k=top_k, allowed_sources=allowed_sources)
    resp = responder.respond(question, chunks)

    answer_text = resp.get("answer", "")
    answerable = resp.get("answerable", False)
    min_distance = min((c["distance"] for c in chunks), default=None)
    sources_chars = sum(len(c["text"]) for c in chunks)

    # --- Selective verification (cost lever) ---
    # Verify when there's a real answer to check, UNLESS retrieval is extremely strong
    # and we're not in force-verify mode — then take the fast path and skip the call.
    vresult = None
    gscore = None
    verify_status = "not_run"
    if answerable and chunks:
        strong = (min_distance is not None
                  and min_distance <= config.STRONG_RETRIEVAL_DISTANCE)
        if strong and not config.VERIFY_ALWAYS:
            verify_status = "skipped_strong"           # fast path: trust strong retrieval
        else:
            vresult = verifier.verify(answer_text, chunks)
            gscore = verifier.grounding_score(vresult)
            verify_status = "failed" if gscore is None else "verified"

    # --- Confidence gate: one explainable decision ---
    g = gate.decide(answerable=answerable, min_distance=min_distance,
                    grounding=gscore, verify_status=verify_status)

    latency_ms = int((time.time() - t0) * 1000)
    verified = verify_status == "verified"
    est_cost = _estimate_cost(len(question) + sources_chars,
                              sources_chars + len(answer_text), verified)

    # --- Observability: one structured line per query ---
    _log({
        "ts": time.time(),
        "question": question,
        "decision": g.decision,
        "confidence": g.confidence,
        "min_distance": min_distance,
        "grounding_score": gscore,
        "verify_status": verify_status,
        "latency_ms": latency_ms,
        "est_cost_usd": est_cost,
        "allowed_sources": allowed_sources,
    })

    return {
        "question": question,
        "chunks": chunks,
        "answer": answer_text,
        "cited_sources": resp.get("cited_sources", []),
        "answerable": answerable,
        "verification": vresult,          # VerificationResult | None
        "grounding_score": gscore,        # float | None
        "verify_status": verify_status,   # verified | skipped_strong | failed | not_run
        "decision": g.decision,           # ANSWER | ESCALATE | REFUSE
        "confidence": g.confidence,
        "reasons": g.reasons,
        "latency_ms": latency_ms,
        "est_cost_usd": est_cost,
    }
