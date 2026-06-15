"""
build_index.py — RUN THIS ONCE before launching the app.

    python -m scripts.build_index

It loads SQuAD, chunks each paragraph, embeds the chunks, and stores them in Chroma.
Separated from the app because indexing is a slow, one-time batch job; serving queries
is a fast, repeated job. You don't want to re-embed the corpus every time Streamlit reloads.
"""

from src import ingest, embed_store


def main():
    print("Loading SQuAD contexts...")
    docs = ingest.load_squad_contexts()
    print(f"  {len(docs)} unique documents")

    print("Chunking...")
    ids, texts, metadatas = [], [], []
    for doc in docs:
        for j, chunk in enumerate(ingest.chunk_text(doc["text"])):
            ids.append(f"{doc['id']}_chunk{j}")     # stable, unique id per chunk
            texts.append(chunk)
            metadatas.append({"title": doc["title"], "parent": doc["id"]})
    print(f"  {len(texts)} chunks")

    print("Embedding + storing in Chroma (this is the slow part)...")
    collection = embed_store.get_collection()
    # Insert in batches so we don't build one giant in-memory list of embeddings.
    BATCH = 256
    for i in range(0, len(texts), BATCH):
        embed_store.add_chunks(
            collection,
            ids[i:i + BATCH],
            texts[i:i + BATCH],
            metadatas[i:i + BATCH],
        )
        print(f"  stored {min(i + BATCH, len(texts))}/{len(texts)}")

    print("Done. Index is persisted in data/chroma/.")


if __name__ == "__main__":
    main()
