"""
config.py — every tunable knob in one place.

WHY a central config: in interviews you'll be asked "how would you tune retrieval?"
and the honest answer is "change TOP_K and CHUNK_SIZE and re-measure." Keeping them
here (not scattered as magic numbers) is what makes that an experiment instead of a hunt.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # read .env into environment variables

# --- Paths ---
# Where Chroma writes its files. Persistent so we don't re-embed on every run
# (embedding the whole corpus is the slow part; we pay it once).
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
COLLECTION_NAME = "squad_kb"

# --- Embeddings ---
# all-MiniLM-L6-v2: 384-dim, fast, runs on CPU, good enough for retrieval.
# WHY this one: it's the standard "free baseline." If retrieval quality is weak,
# the upgrade path is bge-small-en-v1.5 (better, still local) — a clean thing to mention.
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Chunking ---
# SQuAD paragraphs are short, so chunks are modest. Overlap prevents a fact from
# being split across a chunk boundary and lost. Units are characters (simple + predictable).
CHUNK_SIZE = 800        # max characters per chunk
CHUNK_OVERLAP = 120     # characters shared between consecutive chunks

# --- Retrieval ---
# TOP_K is the central accuracy/latency/cost lever:
#   higher K  → more chance the answer is present, but more tokens to the LLM (slower, pricier)
#   lower K   → cheaper/faster, but the answer may not be retrieved at all (worst failure)
TOP_K = 4

# --- LLM (the Responder) ---
# Using OpenAI's API (gpt-4o-mini). The wrapper in llm.py is OpenAI-compatible, so
# this is just three env values — base URL, key, model. (You could repoint these at a
# local Ollama server instead by changing the base URL + model; no code change needed.)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # set OPENAI key as LLM_API_KEY in .env
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# How many corpus paragraphs to index. Env-overridable so you can scale the knowledge
# base without touching code: `MAX_DOCS=19000 python -m scripts.build_index` indexes the
# whole SQuAD corpus. Default raised to 5000 (covers ~100+ Wikipedia topics, not 13).
MAX_DOCS = int(os.getenv("MAX_DOCS", "5000"))

# ----------------------------------------------------------------------------
# Layer 2 additions
# ----------------------------------------------------------------------------

# --- Verifier (Claude) ---
# A DIFFERENT model from the Responder, deliberately. Default to the strongest
# judge; downgrade only as a conscious cost decision (see study-guide notes).
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "claude-opus-4-8")

# --- Confidence Gate thresholds ---
# These are the knobs an interviewer will ask you to defend. They are EMPIRICAL:
# you set a starting value, then Layer 3's eval harness tells you if they're right.

# Retrieval gate: with normalized embeddings, Chroma cosine DISTANCE is in [0, 2]
# (0 = identical, lower = more similar). If even the BEST chunk is farther than
# this, retrieval probably missed — refuse rather than answer from weak context.
RETRIEVAL_MAX_DISTANCE = 0.6

# Grounding gate: fraction of the answer's claims that must be source-grounded.
# 1.0 = every claim must be supported to auto-answer. Anything less goes to a human.
GROUNDING_FULL = 1.0
# Below this fraction we treat the answer as mostly fabricated (used only to label
# severity in the gate's reason string; both bands still route to a human).
GROUNDING_LOW = 0.5

# ----------------------------------------------------------------------------
# Scaling controls (production behavior)
# ----------------------------------------------------------------------------

# --- Selective verification (the cost lever at scale) ---
# Verifying every answer with Claude is the dominant per-query cost. At millions of
# queries you only pay for it when you need it: if retrieval is EXTREMELY strong (best
# chunk distance below this), the grounded draft is almost certainly faithful, so we
# trust it and SKIP the verifier call (the "fast path"). Borderline/weak retrieval still
# gets verified. Set VERIFY_ALWAYS=true to force verification on every answerable query
# (max trust, max cost) — e.g. for a high-stakes deployment like claims review.
VERIFY_ALWAYS = os.getenv("VERIFY_ALWAYS", "false").lower() == "true"
STRONG_RETRIEVAL_DISTANCE = float(os.getenv("STRONG_RETRIEVAL_DISTANCE", "0.25"))

# --- Cost tracking (approx USD per 1M tokens) for the per-query log ---
# Lets the query log estimate $ per query so cost-at-scale is observable, not a mystery.
RESPONDER_PRICE_IN = 0.15    # gpt-4o-mini input  $/1M
RESPONDER_PRICE_OUT = 0.60   # gpt-4o-mini output $/1M
VERIFIER_PRICE_IN = 3.0      # claude-sonnet-4-6 input  $/1M (adjust if you change model)
VERIFIER_PRICE_OUT = 15.0    # claude-sonnet-4-6 output $/1M

# --- Observability ---
# Every query appends one JSON line here: latency, decision, signals, est. cost.
# This is the audit trail / metrics source a real deployment needs.
QUERY_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "query_log.jsonl")
