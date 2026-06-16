"""
ab_eval.py — A/B measurement: single-pass pipeline vs the agentic controller.

Runs the SAME SQuAD eval set through BOTH systems and answers the only question that
matters: does the agentic loop earn its keep?

It reports:
  - RECOVERED   : answerable questions the single pass failed but the loop got right
                  (the upside — what the retries actually bought us).
  - NEW HALLUCINATIONS : unanswerable questions the single pass refused but the loop
                  answered (the safety regression — must stay ~0).
  - correct-refusal rate and answer accuracy for each system (the headline metrics).
  - added cost and latency per query (what the loop costs).

    python -m scripts.ab_eval --n 25

Reuses the EXACT eval set from evaluate.py so this is a clean, comparable before/after.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor

from src import pipeline, agent, config, metrics, embed_store
from scripts.evaluate import build_eval_set


def _correct_delivered(result, golds) -> bool:
    """A correct, DELIVERED answer = the gate said ANSWER and the gold text is present."""
    return (result["decision"] == "ANSWER"
            and metrics.best_over_golds(result["answer"], golds, metrics.contains_gold) >= 1.0)


def _process(task):
    """
    Run ONE agent.answer() and derive BOTH systems from it:
      - single-pass baseline = the agent's attempt-0 (first_pass)
      - agentic              = the agent's final result
    Same run => no nondeterminism between the two, and half the API cost.
    """
    typ, item = task
    q, golds = item["question"], item["golds"]

    agentic = agent.answer(q)
    ag = agentic["agent"]
    single = ag["first_pass"]                 # {decision, answer} of attempt 0
    first_attempt = ag["attempts"][0]

    return {
        "type": typ,
        "question": q,
        "single_decision": single["decision"],
        "single_correct": _correct_delivered(single, golds) if typ == "answerable" else "",
        "single_cost": first_attempt["est_cost_usd"],
        "single_latency": first_attempt["latency_ms"],
        "agent_decision": agentic["decision"],
        "agent_correct": _correct_delivered(agentic, golds) if typ == "answerable" else "",
        "agent_steps": ag["steps_used"],
        "agent_recovered": ag["recovered"],
        "agent_cost": ag["total_cost_usd"],
        "agent_latency": ag["total_latency_ms"],
    }


def summarize(rows):
    ans = [r for r in rows if r["type"] == "answerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]

    def rate(sub, pred):
        return round(sum(1 for r in sub if pred(r)) / len(sub), 3) if sub else 0.0

    def per_system(dec_key, correct_key):
        return {
            # answer accuracy = correct delivered / ALL answerable (captures coverage too)
            "answer_accuracy": rate(ans, lambda r: r[correct_key] is True),
            "correct_refusal": rate(una, lambda r: r[dec_key] != "ANSWER"),
            "hallucination": rate(una, lambda r: r[dec_key] == "ANSWER"),
        }

    single = per_system("single_decision", "single_correct")
    agentic = per_system("agent_decision", "agent_correct")

    # The two numbers the whole layer exists to produce:
    recovered = [r for r in ans if r["single_correct"] is not True and r["agent_correct"] is True]
    new_halluc = [r for r in una if r["single_decision"] != "ANSWER" and r["agent_decision"] == "ANSWER"]
    # A regression guard on the answerable side too (should be ~0 by construction).
    regressed = [r for r in ans if r["single_correct"] is True and r["agent_correct"] is not True]

    n = len(rows)
    cost = {
        "single_mean": round(sum(r["single_cost"] for r in rows) / n, 6),
        "agent_mean": round(sum(r["agent_cost"] for r in rows) / n, 6),
        "single_latency_ms": int(sum(r["single_latency"] for r in rows) / n),
        "agent_latency_ms": int(sum(r["agent_latency"] for r in rows) / n),
        "mean_steps": round(sum(r["agent_steps"] for r in rows) / n, 2),
    }

    return {
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "single": single, "agentic": agentic,
        "recovered": len(recovered), "regressed": len(regressed),
        "new_hallucinations": len(new_halluc),
        "cost": cost,
    }


def save(rows, summary):
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "ab_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join("data", "ab_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nPer-question A/B -> data/ab_results.csv")
    print("Summary          -> data/ab_summary.json")


def report(s):
    a, b, c = s["single"], s["agentic"], s["cost"]
    print("\n" + "=" * 60)
    print(f"A/B: SINGLE-PASS vs AGENTIC  (answerable={s['n_answerable']}, "
          f"unanswerable={s['n_unanswerable']})")
    print("=" * 60)
    print(f"{'metric':<24}{'single':>12}{'agentic':>12}")
    print(f"{'answer accuracy':<24}{a['answer_accuracy']:>11.0%}{b['answer_accuracy']:>12.0%}")
    print(f"{'correct-refusal rate':<24}{a['correct_refusal']:>11.0%}{b['correct_refusal']:>12.0%}")
    print(f"{'hallucination rate':<24}{a['hallucination']:>11.0%}{b['hallucination']:>12.0%}")
    print("-" * 60)
    print(f"RECOVERED (loop fixed)   : {s['recovered']} answerable case(s) "
          f"single-pass got wrong, agentic got right")
    print(f"NEW HALLUCINATIONS       : {s['new_hallucinations']}  (loop broke a correct "
          f"refusal — want 0)")
    print(f"ANSWERABLE REGRESSIONS   : {s['regressed']}  (loop broke a correct answer — want 0)")
    print("-" * 60)
    print(f"cost/query    : ${c['single_mean']:.5f}  ->  ${c['agent_mean']:.5f}  "
          f"(x{c['agent_mean']/c['single_mean']:.2f})" if c['single_mean'] else "")
    print(f"latency/query : {c['single_latency_ms']} ms  ->  {c['agent_latency_ms']} ms")
    print(f"mean retries  : {c['mean_steps']} per query")
    print("=" * 60)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25,
                        help="answerable AND unanswerable questions (each)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    answerable, unanswerable = build_eval_set(args.n)
    tasks = ([("answerable", it) for it in answerable]
             + [("unanswerable", it) for it in unanswerable])
    print(f"A/B over {len(tasks)} questions, each run through BOTH systems "
          f"(MAX_AGENT_STEPS={config.MAX_AGENT_STEPS}, verifier={config.VERIFIER_MODEL}).\n")

    embed_store.get_collection()  # warm the (non-thread-safe) client before workers start
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_process, tasks))

    summary = summarize(rows)
    save(rows, summary)
    report(summary)


if __name__ == "__main__":
    run()
