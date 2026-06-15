"""
responder.py — the Responder AGENT.

Its contract: draft an answer using ONLY the retrieved chunks, cite which chunks it
used, and ABSTAIN if the answer isn't in them. The abstention is the seed of the
Layer-2 confidence gate: refusal is built in from the start, not added later.

The prompt is the most important code in this file. Read it as code — for an LLM agent,
the prompt IS the program.
"""

from . import llm

# WHY these exact instructions:
#  - "ONLY the sources" fights the model's urge to use its training knowledge (the #1
#    hallucination source in RAG).
#  - numbered sources [1],[2]... give us machine-checkable citations.
#  - the explicit refusal clause makes "I don't know" a first-class, allowed output.
#  - JSON output gives downstream code a reliable contract.
SYSTEM_PROMPT = """You are a careful research assistant.
Answer the user's question using ONLY the numbered sources provided.
Rules:
- Use ONLY facts stated in the sources. Do not use outside knowledge.
- After each sentence, cite the source number(s) it came from, like [1] or [1][3].
- CHECK THE QUESTION'S PREMISE against the sources before answering. If the question
  states a specific date, number, name, or claim that does NOT exactly match the
  sources, do NOT answer with the nearest correct fact. Instead set "answerable" to
  false. Examples of false premises to reject:
    * question says "1988" but the source says "1968"
    * question says "7 satellites" but the source says "three"
    * question names "Compass-L1" but the source describes "Compass-M1"
    * question asks what something was "undedicated" to when the source says it WAS dedicated
- If the sources do not contain the answer, OR the question's premise contradicts the
  sources, you MUST set "answerable" to false and say you don't have enough information.
  Do NOT guess and do NOT "correct" the question by answering the related true fact.

Respond with ONLY a JSON object of this exact shape:
{
  "answer": "<your cited answer, or a brief 'not enough information' message>",
  "cited_sources": [<list of source numbers you actually used, e.g. 1, 3>],
  "answerable": <true or false>
}"""


def _format_sources(chunks):
    """Render retrieved chunks as a numbered block the LLM (and our citations) can index."""
    lines = []
    for i, c in enumerate(chunks, start=1):  # 1-based to match human-style [1] citations
        lines.append(f"[{i}] (from \"{c['title']}\")\n{c['text']}")
    return "\n\n".join(lines)


def respond(question, chunks):
    """
    Given the question and retrieved chunks, return a structured answer dict:
      {answer, cited_sources, answerable}
    """
    if not chunks:
        # Defensive: if retrieval found nothing, never call the LLM with empty context —
        # an LLM with no sources WILL fall back to memory and hallucinate. Refuse instead.
        return {"answer": "No relevant sources were found.",
                "cited_sources": [], "answerable": False}

    user_prompt = (
        f"Question: {question}\n\n"
        f"Sources:\n{_format_sources(chunks)}"
    )
    return llm.chat_json(SYSTEM_PROMPT, user_prompt)
