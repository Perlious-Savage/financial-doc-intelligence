"""Embed the CORD test split and load it into Postgres + pgvector.

Runs on CPU. Streams the dataset, so only the annotation text is fetched, not the
receipt images.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from datasets import Image as HFImage, load_dataset  # noqa: E402

from src.index import count_documents, embed_texts, init_schema, upsert_documents  # noqa: E402


def receipt_text(record) -> str:
    ground_truth = json.loads(record["ground_truth"])
    return " ".join(
        word["text"]
        for line in ground_truth.get("valid_line", [])
        for word in line.get("words", [])
        if word.get("text")
    )


def main() -> None:
    init_schema()

    print("streaming CORD-v2 test split ...")
    dataset = load_dataset("naver-clova-ix/cord-v2", split="test", streaming=True).cast_column(
        "image", HFImage(decode=False)
    )

    doc_ids, texts = [], []
    for index, record in enumerate(dataset):
        text = receipt_text(record)
        if text.strip():
            doc_ids.append(f"cord-test-{index:04d}")
            texts.append(text)

    print(f"embedding {len(texts)} documents ...")
    vectors = embed_texts(texts)

    upsert_documents(doc_ids, texts, vectors, [{"split": "test"} for _ in texts])
    print(f"indexed: {count_documents()} documents")


if __name__ == "__main__":
    main()
