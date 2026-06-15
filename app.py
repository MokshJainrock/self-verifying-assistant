"""
app.py — interactive Streamlit dashboard.

Four tabs, so a viewer can see the whole system without touching the terminal:
  1. Ask           — live Q&A through the full pipeline (with access control)
  2. Evaluation    — the headline metrics + every per-question result (eval_summary/csv)
  3. Query log     — every query the system has served (latency, cost, decision)
  4. Knowledge base— what's actually indexed (topics + chunk counts)

    streamlit run app.py
"""

import json
import os

import pandas as pd
import streamlit as st

from src import pipeline, embed_store, config

st.set_page_config(page_title="Self-Verifying Assistant", layout="wide")
st.title("Self-Verifying Knowledge Assistant")
st.caption("Retriever → Responder (OpenAI) → Verifier (Claude) → Confidence Gate. "
           "Answers only from sources, verifies every claim, and refuses when unsure.")


# --------------------------------------------------------------------------
# Small data loaders. Index titles are cached (stable); eval/log files are read
# fresh each rerun so the dashboard always reflects the latest run.
# --------------------------------------------------------------------------
@st.cache_data
def available_sources():
    metas = embed_store.get_collection().get(include=["metadatas"])["metadatas"]
    return sorted({m["title"] for m in metas}), len(metas)


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def load_jsonl(path):
    if not os.path.exists(path):
        return None
    return pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])


tab_ask, tab_eval, tab_log, tab_kb = st.tabs(
    ["Ask", "Evaluation", "Query log", "Knowledge base"]
)

# ==========================================================================
# TAB 1 — ASK (live pipeline)
# ==========================================================================
with tab_ask:
    titles, _ = available_sources()

    # Access control: restrict retrieval to chosen documents (sidebar).
    st.sidebar.header("Access control")
    st.sidebar.caption("Restrict retrieval to specific documents — the assistant can "
                       "only answer from sources the user is allowed to see.")
    allowed = st.sidebar.multiselect("Allowed sources", titles) or None

    question = st.text_input("Ask a question about the knowledge base:")

    if question:
        with st.spinner("Retrieving → answering → verifying..."):
            r = pipeline.answer(question, allowed_sources=allowed)

        # Decision banner — the headline.
        if r["decision"] == "ANSWER":
            st.success(f"ANSWER  ·  confidence {r['confidence']:.2f}")
        elif r["decision"] == "ESCALATE":
            st.warning(f"ESCALATE TO HUMAN  ·  grounding {r['confidence']:.2f}")
        else:
            st.error("REFUSED — not enough grounded information")

        # Operational signals (what a production console shows).
        c1, c2, c3 = st.columns(3)
        c1.metric("Latency", f"{r['latency_ms']} ms")
        c2.metric("Est. cost", f"${r['est_cost_usd']:.5f}")
        c3.metric("Verifier", r["verify_status"])
        for reason in r["reasons"]:
            st.caption(f"• {reason}")

        st.subheader("Drafted answer")
        st.write(r["answer"])

        # Per-claim grounding — where hallucination is caught.
        if r["verification"] is not None:
            st.subheader("Claim-by-claim grounding (independent Verifier)")
            for c in r["verification"].claims:
                mark = "GROUNDED" if c.grounded else "UNGROUNDED"
                src = f"source [{c.supporting_source}]" if c.supporting_source else "no source"
                with st.expander(f"{mark} {c.claim}  ({src})"):
                    st.write(c.reasoning)
        elif r["verify_status"] == "skipped_strong":
            st.info("Verifier skipped — retrieval was very strong (fast path).")

        # Retrieved sources (auditability).
        st.subheader("Retrieved sources")
        for i, c in enumerate(r["chunks"], start=1):
            with st.expander(f"[{i}] {c['title']}  ·  distance {c['distance']:.3f}"):
                st.write(c["text"])

# ==========================================================================
# TAB 2 — EVALUATION (headline metrics + per-question results)
# ==========================================================================
with tab_eval:
    summary = load_json(os.path.join("data", "eval_summary.json"))
    df = load_csv(os.path.join("data", "eval_results.csv"))

    if summary is None:
        st.info("No evaluation run yet. Run:  `python -m scripts.evaluate --n 50`")
    else:
        a, u, t = summary["answerable"], summary["unanswerable"], summary["throughput"]

        st.subheader("Headline metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Correct-refusal rate", f"{u['correct_refusal_rate']:.0%}",
                  help="Unanswerable questions the system declined — the honesty metric")
        c2.metric("Hallucination rate", f"{u['hallucination_rate']:.0%}",
                  help="Unanswerable questions it answered anyway — want ~0%", delta_color="inverse")
        c3.metric("Accuracy when answered", f"{a['accuracy_when_answered']:.0%}",
                  help="Of delivered answers, fraction containing the gold answer")

        c4, c5, c6 = st.columns(3)
        c4.metric("Answer rate (coverage)", f"{a['answer_rate']:.0%}")
        c5.metric("Over-refusal", f"{a['over_refusal_rate']:.0%}")
        c6.metric("Escalate rate", f"{a['escalate_rate']:.0%}")

        st.subheader("Throughput & cost")
        c7, c8, c9 = st.columns(3)
        c7.metric("Total est. cost", f"${t['total_est_cost_usd']:.4f}",
                  help=f"for {t['total_questions']} questions")
        c8.metric("Mean latency", f"{t['mean_latency_ms']} ms")
        c9.metric("Verifier calls", f"{t['verified_calls']} run / {t['fast_path_skips']} skipped")

        if df is not None:
            st.subheader("Per-question results")
            colf1, colf2 = st.columns(2)
            type_filter = colf1.selectbox("Question type", ["all", "answerable", "unanswerable"])
            dec_filter = colf2.selectbox("Decision", ["all", "ANSWER", "ESCALATE", "REFUSE"])

            view = df.copy()
            if type_filter != "all":
                view = view[view["type"] == type_filter]
            if dec_filter != "all":
                view = view[view["decision"] == dec_filter]

            # Flag the failure cases so they're easy to spot.
            st.dataframe(view, use_container_width=True, height=380)

            # Quick view of hallucinations (the cases that matter most).
            halluc = df[(df["type"] == "unanswerable") & (df["decision"] == "ANSWER")]
            st.caption(f"{len(halluc)} hallucination(s): unanswerable questions that got an ANSWER.")

# ==========================================================================
# TAB 3 — QUERY LOG (operational audit trail)
# ==========================================================================
with tab_log:
    log = load_jsonl(config.QUERY_LOG)
    if log is None or log.empty:
        st.info("No queries logged yet. Ask a question (Ask tab) or run the eval — "
                "every query appends a line to `data/query_log.jsonl`.")
    else:
        st.subheader("Operational summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Queries served", len(log))
        c2.metric("Mean latency", f"{int(log['latency_ms'].mean())} ms")
        c3.metric("Total est. cost", f"${log['est_cost_usd'].sum():.4f}")
        c4.metric("Verifier skips", int((log["verify_status"] == "skipped_strong").sum()))

        st.subheader("Decisions")
        st.bar_chart(log["decision"].value_counts())

        st.subheader("Recent queries")
        cols = ["question", "decision", "confidence", "verify_status", "latency_ms", "est_cost_usd"]
        st.dataframe(log[cols].iloc[::-1], use_container_width=True, height=320)

# ==========================================================================
# TAB 4 — KNOWLEDGE BASE (what's indexed)
# ==========================================================================
with tab_kb:
    titles, n_chunks = available_sources()
    metas = embed_store.get_collection().get(include=["metadatas"])["metadatas"]
    counts = pd.Series([m["title"] for m in metas]).value_counts()

    c1, c2 = st.columns(2)
    c1.metric("Distinct topics (sources)", len(titles))
    c2.metric("Total indexed chunks", n_chunks)

    st.subheader("Chunks per topic (top 25)")
    st.bar_chart(counts.head(25))

    st.subheader("All indexed topics")
    st.dataframe(counts.rename_axis("topic").reset_index(name="chunks"),
                 use_container_width=True, height=320)
