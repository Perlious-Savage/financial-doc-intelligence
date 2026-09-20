"""Score the no-model baseline on the CORD-v2 test split.

Runs on CPU. Streams the dataset so the receipt images are never downloaded: the
baseline reads words, not pixels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datasets import Image as HFImage, load_dataset

from src.baseline import predict_tags
from src.metrics import classification_report, precision_recall_f1

ARTIFACTS = Path(__file__).parent / "artifacts"


def main() -> None:
    print("streaming CORD-v2 test split ...")
    dataset = load_dataset("naver-clova-ix/cord-v2", split="test", streaming=True).cast_column(
        "image", HFImage(decode=False)
    )

    references: list[list[str]] = []
    predictions: list[list[str]] = []

    for count, record in enumerate(dataset, 1):
        ground_truth = json.loads(record["ground_truth"])
        words, tags = [], []
        for line in ground_truth.get("valid_line", []):
            category = line.get("category", "other")
            for position, word in enumerate(line.get("words", [])):
                text = (word.get("text") or "").strip()
                if not text:
                    continue
                words.append(text)
                tags.append(f"{'B' if position == 0 else 'I'}-{category}")
        if not words:
            continue
        references.append(tags)
        predictions.append(predict_tags(words))

    precision, recall, f1 = precision_recall_f1(references, predictions)
    result = {
        "method": "keyword + position regex baseline (no model)",
        "dataset": "naver-clova-ix/cord-v2",
        "split": "test",
        "n_documents": len(references),
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "note": "Deterministic. No training, no GPU, no randomness - this number is exact.",
    }

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "baseline_metrics.json").write_text(json.dumps(result, indent=2))
    (ARTIFACTS / "baseline_per_field_report.txt").write_text(
        classification_report(references, predictions)
    )

    print(json.dumps(result, indent=2))
    print()
    print(classification_report(references, predictions))


if __name__ == "__main__":
    main()
