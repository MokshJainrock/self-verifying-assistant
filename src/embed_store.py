"""
embed_store.py — turn text into vectors and store/query them in Chroma.

This is the boundary between "raw text" and "searchable knowledge." We keep the
embedding model and the vector DB together because they're always used as a pair:
you must query with the SAME model you indexed with (else the vectors aren't comparable).
"""

import threading

import chromadb
from sentence_transformers import SentenceTransformer
from . import config

# Load the embedding model once (loading it is slow; reuse across calls).
_embedder = SentenceTransformer(config.EMBED_MODEL)

# Chroma's PersistentClient is NOT safe to construct concurrently — building a second
# one while another is initializing corrupts its shared internal state. So we build the
# client+collection EXACTLY ONCE and hand the same object to every caller/thread.
_collection = None
_collection_lock = threading.Lock()


def embed(texts):
    """
    Convert a list of strings to a list of vectors.

    normalize_embeddings=True scales every vector to length 1. WHY: with unit vectors,
    cosine similarity == dot product, and distances become well-behaved. It makes
    "closeness" mean the same thing for every vector — important for stable retrieval.
    """
    return _embedder.encode(list(texts), normalize_embeddings=True).tolist()


def get_collection():
    """
    Open (or create) the persistent Chroma collection — memoized (built once, reused).

    PersistentClient writes to disk, so the index survives between runs — we embed the
    corpus ONCE (the expensive step) and reuse it on every query and every app restart.
    The double-checked lock makes first creation thread-safe so concurrent callers (the
    parallel eval harness) don't race to construct the client.
    """
    global _collection
    if _collection is None:
        with _collection_lock:
            if _collection is None:  # re-check inside the lock
                client = chromadb.PersistentClient(path=config.CHROMA_DIR)
                # cosine space matches our normalized embeddings above.
                _collection = client.get_or_create_collection(
                    name=config.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
    return _collection


def add_chunks(collection, ids, texts, metadatas):
    """
    Insert chunks into Chroma. We pass our OWN precomputed embeddings (rather than
    letting Chroma embed) so the embedding model is explicit and fully under our control
    — easier to reason about and to swap.
    """
    collection.add(
        ids=ids,
        documents=texts,          # Chroma stores the raw text...
        metadatas=metadatas,      # ...and metadata (title, parent doc) for citations...
        embeddings=embed(texts),  # ...alongside the vector.
    )


def query(collection, question, top_k=config.TOP_K, where=None):
    """
    Embed the question and return the top_k most similar chunks WITH their text and
    metadata — that bundling is exactly why Chroma suits a citation-first system.

    `where` is a Chroma metadata filter (e.g. {"title": {"$in": [...]}}). At scale this
    is how you enforce ACCESS CONTROL: the search only ever sees documents the current
    user is allowed to read — you filter at retrieval time, not after.
    """
    res = collection.query(
        query_embeddings=embed([question]),
        n_results=top_k,
        where=where,   # None = search everything; a filter = restricted search
    )
    # Chroma returns parallel lists wrapped in an outer list (one per query). Flatten
    # the single-query case into a clean list of dicts for the rest of the app.
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "title": res["metadatas"][0][i].get("title", "unknown"),
            "distance": res["distances"][0][i],  # smaller = more similar (cosine distance)
        })
    return hits
