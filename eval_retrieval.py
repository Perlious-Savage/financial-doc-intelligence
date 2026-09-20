"""Measure whether semantic retrieval finds near-duplicate documents.

Honest framing: this is semantic retrieval, not RAG. Nothing retrieved is injected
into a generation step.

The test is built rather than assumed. Each query document is paraphrased into a
planted near-duplicate: the same receipt with tokens reordered and amounts perturbed,
which is roughly what a resubmitted or lightly altered document looks like. The
question is whether the planted twin comes back in the top k.

A TF-IDF baseline runs alongside, because lexical matching is often better at exactly
this task, and claiming embeddings were necessary without checking would be a guess.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from src.index import embed_texts  # noqa: E402

ARTIFACTS = Path(__file__).parent / "artifacts"
K = 5


def paraphrase(text: str, rng: random.Random) -> str:
    """A near-duplicate: same content, shuffled order, perturbed numbers."""
    tokens = text.split()
    for index, token in enumerate(tokens):
        if token.replace(".", "").replace(",", "").isdigit() and rng.random() < 0.3:
            tokens[index] = str(int(token.replace(".", "").replace(",", "") or 0) + 1)
    chunk = max(1, len(tokens) // 4)
    blocks = [tokens[i : i + chunk] for i in range(0, len(tokens), chunk)]
    rng.shuffle(blocks)
    return " ".join(t for block in blocks for t in block)


def recall_at_k(similarity: np.ndarray, k: int) -> float:
    """Each query's planted twin sits at the matching index in the corpus."""
    hits = 0
    for row in range(similarity.shape[0]):
        top = np.argsort(-similarity[row])[:k]
        if row in top:
            hits += 1
    return hits / similarity.shape[0]


def main() -> None:
    rng = random.Random(0)
    dataset = load_dataset("naver-clova-ix/cord-v2", split="test", streaming=True)

    corpus = []
    for record in dataset:
        ground_truth = json.loads(record["ground_truth"])
        text = " ".join(
            w["text"]
            for line in ground_truth.get("valid_line", [])
            for w in line.get("words", [])
            if w.get("text")
        )
        if text.strip():
            corpus.append(text)

    queries = [paraphrase(text, rng) for text in corpus]
    print(f"{len(corpus)} documents, {len(queries)} paraphrased queries")

    tfidf = TfidfVectorizer().fit(corpus + queries)
    lexical = (tfidf.transform(queries) @ tfidf.transform(corpus).T).toarray()

    print("embedding ...")
    corpus_vectors = np.array(embed_texts(corpus))
    query_vectors = np.array(embed_texts(queries))
    semantic = query_vectors @ corpus_vectors.T

    result = {
        "task": "retrieve the planted near-duplicate of each query document",
        "n_documents": len(corpus),
        "k": K,
        "tfidf_recall_at_k": recall_at_k(lexical, K),
        "embedding_recall_at_k": recall_at_k(semantic, K),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "note": (
            "Near-duplicates are synthetic: token blocks shuffled and some amounts "
            "perturbed. Real resubmitted documents may differ in other ways."
        ),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "retrieval_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
