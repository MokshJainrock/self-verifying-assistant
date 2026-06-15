"""
embed_store.py — turn text into vectors and store/query them in Chroma.

This is the boundary between "raw text" and "searchable knowledge." We keep the
embedding model and the vector DB together because they're always used as a pair:
you must query with the SAME model you indexed with (else the vectors aren't comparable).

It also owns BUILDING the index (build_collection / ensure_index) so the app can build
its own index on first run in a fresh deployment — a pre-built Chroma directory is a
version-specific binary that does NOT travel reliably between machines/versions, so the
safe pattern is "build where you read."
"""

import shutil
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


def _open_collection():
    """Open (or create) the persistent Chroma collection from disk."""
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    # cosine space matches our normalized embeddings above.
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_collection():
    """
    Return the Chroma collection — memoized (built once, reused), and SELF-HEALING.

    If the on-disk index can't be opened — e.g. it was built with a different ChromaDB
    version (the #1 deployment failure) — we wipe the directory and recreate an empty
    collection; ensure_index() will then rebuild it cleanly in THIS environment.

    The double-checked lock makes first creation thread-safe so concurrent callers
    (the parallel eval harness) don't race to construct the client.
    """
    global _collection
    if _collection is None:
        with _collection_lock:
            if _collection is None:  # re-check inside the lock
                try:
                    _collection = _open_collection()
                except Exception:
                    # Incompatible/corrupt persisted index -> reset and start empty.
                    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
                    _collection = _open_collection()
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


# --------------------------------------------------------------------------
# Building the index (used by scripts/build_index.py AND by app startup)
# --------------------------------------------------------------------------
def build_collection(max_docs=None, batch=256, reset=False, progress=None):
    """
    Ingest -> chunk -> embed -> store. Returns the populated collection.

    reset=True wipes any existing index first (a clean local rebuild). progress is an
    optional callable(done, total) so a UI can show a progress bar during the build.
    """
    global _collection
    from . import ingest  # local import avoids any import cycle

    if reset:
        with _collection_lock:
            _collection = None
        shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)

    docs = ingest.load_squad_contexts(max_docs or config.MAX_DOCS)
    ids, texts, metadatas = [], [], []
    for doc in docs:
        for j, chunk in enumerate(ingest.chunk_text(doc["text"])):
            ids.append(f"{doc['id']}_chunk{j}")
            texts.append(chunk)
            metadatas.append({"title": doc["title"], "parent": doc["id"]})

    col = get_collection()
    for i in range(0, len(texts), batch):
        add_chunks(col, ids[i:i + batch], texts[i:i + batch], metadatas[i:i + batch])
        if progress:
            progress(min(i + batch, len(texts)), len(texts))
    return col


def ensure_index():
    """
    Build the index if it's empty — i.e. first run or a fresh deployment. Returns the
    collection. This is what lets the app build its own index on Streamlit Cloud instead
    of relying on a committed (version-fragile) Chroma directory.
    """
    col = get_collection()
    try:
        empty = col.count() == 0
    except Exception:
        empty = True
    if empty:
        build_collection()
        col = get_collection()
    return col
