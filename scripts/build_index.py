"""
build_index.py — RUN THIS ONCE to (re)build the index locally.

    python -m scripts.build_index

It loads SQuAD, chunks each paragraph, embeds the chunks, and stores them in Chroma.
The actual work lives in embed_store.build_collection() so the app can run the SAME
build on first startup in a fresh deployment.
"""

from src import embed_store, config


def main():
    print(f"Building index (MAX_DOCS={config.MAX_DOCS})...")

    def progress(done, total):
        print(f"  embedded {done}/{total} chunks")

    col = embed_store.build_collection(reset=True, progress=progress)
    print(f"Done. {col.count()} chunks indexed in data/chroma/.")


if __name__ == "__main__":
    main()
