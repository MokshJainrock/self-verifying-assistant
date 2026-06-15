"""
evaluate.py — the adversarial evaluation harness (Layer 3). The honest scoreboard.

    python -m scripts.evaluate --n 10

WHAT IT MEASURES (and why both numbers matter):
  1. ANSWER ACCURACY on answerable questions — when the system chooses to answer, is
     it right? (the usual RAG metric)
  2. CORRECT-REFUSAL RATE on UNANSWERABLE questions — when the answer isn't in the
     corpus, does the system correctly decline instead of hallucinating? (the metric
     most RAG demos quietly skip — and the whole point of this project)

THE HONEST SETUP (read this — it's an interview talking point):
  Our knowledge base was built from the first MAX_DOCS unique SQuAD contexts. If we
  evaluated on random SQuAD questions, "answerable" questions would fail simply because
  their source paragraph isn't indexed — that would conflate "retrieval coverage" with
  "system behavior." So we build the eval set from the SAME contexts we indexed:
    - answerable questions  -> the answer IS in our corpus (tests retrieve+answer+gate)
    - unanswerable questions -> the topic is in our corpus but the answer is NOT
      (tests correct refusal — these are SQuAD 2.0's adversarial "impossible" questions)
"""

import argparse
import csv
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset

from src import pipeline, config, metrics, embed_store

SEED = 42  # fixed seed -> reproducible eval set (you can defend "I report on a fixed set")


def indexed_contexts(max_docs: int) -> set:
    """
    Reconstruct the EXACT set of contexts that build_index.py indexed: the first
    `max_docs` unique contexts in SQuAD train order. We must match that logic so the
    eval set lines up with what's actually searchable.
    """
    ds = load_dataset("rajpurkar/squad_v2", split="train")
    seen = set()
    for row in ds:
        seen.add(row["context"])
        if len(seen) >= max_docs:
            break
    return seen


def build_eval_set(n_each: int):
    """
    Return (answerable, unanswerable) lists of {question, golds} drawn from questions
    whose context is in our index. In SQuAD 2.0, an UNANSWERABLE question is one whose
    `answers["text"]` list is empty.
    """
    ds = load_dataset("rajpurkar/squad_v2", split="train")
    indexed = indexed_contexts(config.MAX_DOCS)

    answerable, unanswerable = [], []
    for row in ds:
        if row["context"] not in indexed:
            continue
        golds = row["answers"]["text"]
        item = {"question": row["question"], "golds": list(golds)}
        if len(golds) == 0:
            unanswerable.append(item)
        else:
            answerable.append(item)

    # Sample a fixed, balanced subset so the run is cheap and reproducible.
    rng = random.Random(SEED)
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    return answerable[:n_each], unanswerable[:n_each]


def _process(task):
    """Run the full pipeline on one (type, item) and return its CSV row."""
    typ, item = task
    r = pipeline.answer(item["question"])
    row = {
        "type": typ,
        "question": item["question"],
        "gold": " | ".join(item["golds"]) if typ == "answerable" else "(none — should refuse)",
        "decision": r["decision"],
        "grounding": r["grounding_score"],
        "verify_status": r["verify_status"],
        "latency_ms": r["latency_ms"],
        "est_cost_usd": r["est_cost_usd"],
        "contains_gold": "",
        "f1": "",
        "answer": r["answer"].replace("\n", " ")[:300],
    }
    if typ == "answerable":
        row["contains_gold"] = metrics.best_over_golds(
            r["answer"], item["golds"], metrics.contains_gold) >= 1.0
        row["f1"] = round(metrics.best_over_golds(r["answer"], item["golds"], metrics.f1), 3)
    return row


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="number of answerable AND unanswerable questions (each)")
    parser.add_argument("--workers", type=int, default=4,
                        help="concurrent pipeline calls (throughput at scale)")
    args = parser.parse_args()

    answerable, unanswerable = build_eval_set(args.n)
    tasks = ([("answerable", it) for it in answerable]
             + [("unanswerable", it) for it in unanswerable])

    print(f"Evaluating {len(tasks)} questions "
          f"({len(answerable)} answerable + {len(unanswerable)} unanswerable) "
          f"with {args.workers} workers.")
    print("Answerable questions may make a Verifier (Claude) call. This costs a little.\n")

    # Warm the Chroma collection in the MAIN thread first, so its (non-thread-safe)
    # client is constructed once before any worker touches it.
    embed_store.get_collection()

    # Run the pipeline concurrently — the work is I/O-bound (API calls), so a thread
    # pool gives near-linear throughput. This is how you'd batch-evaluate at scale.
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_process, tasks))

    summary = summarize(rows)
    save(rows, summary)
    report(summary)


def summarize(rows: list) -> dict:
    """Aggregate the per-question records into the headline metrics."""
    ans = [r for r in rows if r["type"] == "answerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]

    # --- Answerable side ---
    # "answered" = the gate chose to deliver an answer (decision ANSWER).
    answered = [r for r in ans if r["decision"] == "ANSWER"]
    correct_delivered = [r for r in answered if r["contains_gold"] is True]

    n_ans = len(ans)
    answerable_metrics = {
        "n": n_ans,
        # Of the answers we DELIVERED, how many were correct? (precision of answers)
        "accuracy_when_answered": _safe_div(len(correct_delivered), len(answered)),
        # How often did we choose to answer at all? (coverage)
        "answer_rate": _safe_div(len(answered), n_ans),
        # Refusing a question we COULD answer is over-caution — track its cost.
        "over_refusal_rate": _safe_div(sum(r["decision"] == "REFUSE" for r in ans), n_ans),
        "escalate_rate": _safe_div(sum(r["decision"] == "ESCALATE" for r in ans), n_ans),
        # Mean token-F1 over the answers we delivered (soft accuracy).
        "mean_f1_when_answered": _safe_div(
            sum(r["f1"] for r in answered), len(answered)),
    }

    # --- Unanswerable side ---
    # The ideal behavior is to NOT confidently answer: REFUSE or ESCALATE are both
    # "correct" here (escalation hands an uncertain case to a human — a safe outcome).
    n_una = len(una)
    did_not_answer = [r for r in una if r["decision"] != "ANSWER"]
    hallucinated = [r for r in una if r["decision"] == "ANSWER"]
    unanswerable_metrics = {
        "n": n_una,
        # THE headline honesty metric.
        "correct_refusal_rate": _safe_div(len(did_not_answer), n_una),
        # The worst failure: confidently answered something with no source. Want ~0.
        "hallucination_rate": _safe_div(len(hallucinated), n_una),
        "refuse_count": sum(r["decision"] == "REFUSE" for r in una),
        "escalate_count": sum(r["decision"] == "ESCALATE" for r in una),
    }

    # --- Throughput & cost (scale signals) ---
    n_all = len(rows)
    throughput = {
        "total_questions": n_all,
        "total_est_cost_usd": round(sum(r["est_cost_usd"] for r in rows), 4),
        "mean_latency_ms": int(sum(r["latency_ms"] for r in rows) / n_all) if n_all else 0,
        "verified_calls": sum(r["verify_status"] == "verified" for r in rows),
        "fast_path_skips": sum(r["verify_status"] == "skipped_strong" for r in rows),
    }

    return {"answerable": answerable_metrics,
            "unanswerable": unanswerable_metrics,
            "throughput": throughput}


def _safe_div(a, b):
    return round(a / b, 3) if b else 0.0


def save(rows: list, summary: dict):
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "eval_results.csv")
    json_path = os.path.join("data", "eval_summary.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPer-question results -> {csv_path}")
    print(f"Summary metrics     -> {json_path}")


def report(summary: dict):
    """Pretty-print the scoreboard to the console."""
    a = summary["answerable"]
    u = summary["unanswerable"]
    print("\n" + "=" * 56)
    print("EVALUATION REPORT")
    print("=" * 56)
    print(f"\nANSWERABLE questions (n={a['n']}) — can it answer correctly?")
    print(f"  accuracy when answered : {a['accuracy_when_answered']:.1%}  "
          f"(of delivered answers, fraction containing the gold)")
    print(f"  mean F1 when answered  : {a['mean_f1_when_answered']:.3f}")
    print(f"  answer rate (coverage) : {a['answer_rate']:.1%}")
    print(f"  over-refusal rate      : {a['over_refusal_rate']:.1%}  (could answer, didn't)")
    print(f"  escalate rate          : {a['escalate_rate']:.1%}")

    print(f"\nUNANSWERABLE questions (n={u['n']}) — does it refuse instead of hallucinating?")
    print(f"  CORRECT-REFUSAL RATE   : {u['correct_refusal_rate']:.1%}  "
          f"(refused or escalated — the headline metric)")
    print(f"  hallucination rate     : {u['hallucination_rate']:.1%}  "
          f"(confidently answered with no source — want ~0%)")
    print(f"  breakdown              : {u['refuse_count']} refused, "
          f"{u['escalate_count']} escalated")

    t = summary["throughput"]
    print(f"\nTHROUGHPUT & COST (scale signals)")
    print(f"  total est. cost        : ${t['total_est_cost_usd']:.4f} "
          f"for {t['total_questions']} questions")
    print(f"  mean latency           : {t['mean_latency_ms']} ms/query")
    print(f"  verifier calls         : {t['verified_calls']} run, "
          f"{t['fast_path_skips']} skipped via fast path (cost saved)")
    print("=" * 56)


if __name__ == "__main__":
    run()
