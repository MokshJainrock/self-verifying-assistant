"""
retriever.py — the Retriever AGENT.

It's small on purpose. Calling it an "agent" is about ROLE and BOUNDARY, not size:
its sole job is "given a question, return candidate sources." It does NOT answer, NOT
judge. Clean role boundaries are what make the multi-agent claim real rather than
cosmetic — each agent does one job and hands off.
"""

from . import embed_store, config


def retrieve(question, top_k=config.TOP_K, allowed_sources=None):
    """
    Return a list of source chunks (id, text, title, distance) for a question.

    `allowed_sources` is an optional list of source titles the caller is permitted to
    see. When provided, retrieval is restricted to those documents — this is per-user
    ACCESS CONTROL enforced at search time (a user never even retrieves a doc they
    aren't allowed to read, so it can't leak into an answer). When None, search is global.
    """
    collection = embed_store.get_collection()
    where = {"title": {"$in": list(allowed_sources)}} if allowed_sources else None
    return embed_store.query(collection, question, top_k=top_k, where=where)
