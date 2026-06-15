"""
llm.py — a thin, swappable LLM client (the Responder's backend).

WHY a wrapper instead of calling the SDK everywhere: every agent that uses the local
model needs to call an LLM. If those calls are scattered, swapping models or adding
retry/JSON logic means editing many files. One wrapper = one place to change.

WHY OpenAI-compatible: OpenAI's chat API and a local Ollama server speak the SAME
protocol. So this single client runs a FREE local model or a hosted one with no code
change — only the base_url/model differ. That directly satisfies the free/local constraint.
"""

import json
from openai import OpenAI
from . import config

# Build the client once at import time and reuse it.
_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    """
    Send one system+user prompt, return the raw text reply.

    temperature=0.0 by default: for a grounded QA system we want the LEAST creative,
    most deterministic output. Creativity is how hallucinations get in. (Good interview
    point: "I keep temperature at 0 because I want faithfulness, not fluency.")
    """
    resp = _client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def chat_json(system: str, user: str) -> dict:
    """
    Same as chat(), but we EXPECT a JSON object back and parse it.

    WHY structured output: downstream code (the dashboard, the verifier, the eval
    harness) needs reliable fields like `answerable`. Parsing prose is brittle; asking
    for JSON makes the contract explicit. We still defend against malformed JSON below,
    because local models don't always honor the format perfectly — that's a real failure
    mode, not a hypothetical.
    """
    raw = chat(system, user)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Models sometimes wrap JSON in ```json fences or add stray prose.
        # Salvage the substring between the first '{' and the last '}'.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        # If we still can't parse, fail SAFE: treat as "couldn't answer" rather than
        # inventing a result. Refusing is the correct behavior when we're uncertain.
        return {
            "answer": "I could not produce a reliable answer.",
            "cited_sources": [],
            "answerable": False,
            "_parse_error": True,
            "_raw": raw,
        }
