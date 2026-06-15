"""
ingest.py — load the corpus and split it into chunks.

Two responsibilities, kept separate from embedding/storage on purpose (single
responsibility = easier to test and explain):
  1. load_squad_contexts(): get unique source paragraphs from SQuAD 2.0
  2. chunk_text(): split a long string into overlapping chunks
"""

from datasets import load_dataset
from . import config


def load_squad_contexts(max_docs: int = config.MAX_DOCS):
    """
    Return a list of {"id", "title", "text"} documents.

    SQuAD pairs MANY questions with the SAME context paragraph, so the raw rows are
    full of duplicate contexts. We dedupe — indexing the same paragraph 50 times would
    waste storage and let one paragraph dominate retrieval. The `title` (Wikipedia
    article name) becomes our citation label.
    """
    # Canonical namespaced repo id ("rajpurkar/squad_v2"). Newer huggingface_hub
    # rejects the bare legacy id "squad_v2", so we use the full namespace/name form.
    ds = load_dataset("rajpurkar/squad_v2", split="train")  # free download, cached after first run

    seen = set()
    docs = []
    for row in ds:
        ctx = row["context"]
        if ctx in seen:
            continue
        seen.add(ctx)
        docs.append({
            "id": f"doc_{len(docs)}",
            "title": row["title"],   # e.g. "Beyoncé" — human-readable source name
            "text": ctx,
        })
        if len(docs) >= max_docs:
            break
    return docs


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
    """
    Split text into overlapping character windows.

    WHY chunk at all: embeddings represent a fixed-size blob of text as one vector.
    Too large a blob → the vector is a blurry average and retrieval gets imprecise.
    Too small → you lose context. Chunking is the knob that balances this.

    WHY overlap: a fact sitting on a chunk boundary ("...born in 1985." | "She won...")
    could be cut in half and lost. Overlap re-includes boundary text in both chunks so
    no single fact falls through the crack.
    """
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        # advance by (size - overlap) so consecutive chunks share `overlap` chars
        start += size - overlap
    return chunks
