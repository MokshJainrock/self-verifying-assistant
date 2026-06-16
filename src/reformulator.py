"""
reformulator.py — the agent's ONE action: rewrite a failing query so that
re-retrieval finds better-grounded passages.

WHY reformulation is the right action here:
In this system the dominant failure mode is RETRIEVAL — the answer often IS in the
corpus, but the user's phrasing didn't surface the right chunk (vague wording, missing
the key entity, different vocabulary than the source). Re-wording the query with
different terms and explicit entity names changes which vectors are nearest in the
embedding space, which can pull in the passage that actually contains the answer.

Important: we are NOT changing what is being asked. We produce a meaning-preserving
rephrase (same intent, different words). The original question is kept by the caller
for logging and scoring.

This module is deliberately tiny and has ONE public function, reformulate(). It reuses
the existing OpenAI wrapper (llm.py) — no new model, no new dependency.
"""

from . import llm

# The instruction is the whole program for an LLM action. Three things it must do:
#  1. PRESERVE meaning (don't answer, don't add facts, don't change the question).
#  2. CHANGE the wording (so re-retrieval actually searches differently).
#  3. Return ONLY the query (so the caller can use it directly, no parsing).
_SYSTEM = """You rewrite a search query so it retrieves better passages from a document
collection. Rules:
- Keep the EXACT meaning of the original question. Do not answer it, do not add facts,
  do not change what is being asked.
- Change the wording: use clearer phrasing, spell out key names/entities, and prefer
  specific nouns over vague ones, so the search looks in a different place.
- Return ONLY the rewritten query as a single line. No quotes, no explanation."""


def _normalize(q: str) -> str:
    """Lowercase + collapse whitespace, so we can compare two queries for sameness."""
    return " ".join(q.lower().split())


def reformulate(original_question: str, last_query: str, ungrounded_claims=None):
    """
    Produce a reworded retrieval query, or return None if we couldn't produce a
    usefully DIFFERENT one.

    Returning None is a real signal, not an error: it tells the controller
    "reformulation won't help here" so it stops instead of re-running the same search.
    That is one of the loop's stop conditions.

    Args:
        original_question : the user's question (anchor for meaning).
        last_query        : the query that just retrieved poorly.
        ungrounded_claims : optional list of claim strings the verifier could not
                            ground — used to aim the new query at what was missing.
    """
    # If the verifier told us which specific points couldn't be supported, pass them
    # in so the rewrite can target them rather than rephrasing blindly.
    hint = ""
    if ungrounded_claims:
        hint = ("\nThe previous attempt could not find support for these points; "
                "aim the new query at them:\n- " + "\n- ".join(ungrounded_claims[:3]))

    user = (
        f"Original question: {original_question}\n"
        f"Previous query that retrieved poorly: {last_query}{hint}\n\n"
        f"Rewrite the query (different wording, same meaning):"
    )

    try:
        # temperature > 0 ON PURPOSE: we want a genuinely DIFFERENT phrasing, not the
        # same words echoed back. At temperature 0 the model tends to return a near-copy
        # of the input, which would make re-retrieval pointless and risk looping.
        new_query = llm.chat(_SYSTEM, user, temperature=0.4).strip().strip('"').strip()
    except Exception:
        # Fail safe: if the rewrite call errors, behave as "no reformulation available"
        # so the controller stops cleanly and the gate makes the final decision.
        return None

    # Guard against the #1 wasteful outcome: the model handed back the same query.
    # If it did (or returned nothing), signal None so we don't pay for an identical search.
    if not new_query or _normalize(new_query) == _normalize(last_query):
        return None
    return new_query
