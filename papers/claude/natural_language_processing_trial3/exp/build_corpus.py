"""Build a Wikipedia corpus for BM25 retrieval from wikimedia/wikipedia."""
import os
import sys
import json
import time
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CORPUS_PATH = os.path.join(DATA_DIR, "wiki_corpus.json")
TARGET_PASSAGES = 500_000  # 500K 100-word passages

def main():
    t0 = time.time()
    print("Loading Wikipedia Simple English (streaming)...")
    ds = load_dataset('wikimedia/wikipedia', '20231101.simple', split='train', streaming=True)

    corpus = []
    article_count = 0
    for article in ds:
        text = article["text"]
        title = article["title"]
        words = text.split()
        # Chunk into ~100 word passages
        for j in range(0, len(words), 100):
            chunk = " ".join(words[j:j+100])
            if len(chunk.split()) >= 20:
                corpus.append({
                    "id": len(corpus),
                    "title": title,
                    "text": chunk
                })
        article_count += 1
        if article_count % 10000 == 0:
            print(f"  Processed {article_count} articles, {len(corpus)} passages so far")
        if len(corpus) >= TARGET_PASSAGES:
            break

    print(f"Built corpus: {len(corpus)} passages from {article_count} articles")
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f)
    print(f"Saved to {CORPUS_PATH}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()
