"""Train and evaluate the review-priority model on real labels.

The label is not synthetic. For each document, extraction is run and compared against
the CORD ground truth, and the label is whether extraction quality fell below a usable
bar. The model predicts that from validation findings, which makes this ordinary
selective prediction rather than a circular exercise in relearning a formula.

The bar is document-level F1 below 0.8, not "got everything right". That distinction
matters and was found by measurement: with a strict all-or-nothing label the rule
baseline fails on essentially every document, the positive class reaches ~99%, PR-AUC
climbs to 0.999 while meaning nothing, and no threshold can satisfy any error budget,
so coverage collapses to zero. A degenerate problem produces impressive-looking numbers
that do not survive a follow-up question. Asking whether a document is good enough to
use is both answerable and the question an operator actually has.

Split discipline: the model trains on CORD's train split, the routing threshold is
chosen on a validation slice of it, and the CORD test split is scored exactly once at
the end. Choosing the threshold on the data used to report the result would inflate
the coverage number, which is the number that matters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from datasets import Image as HFImage, load_dataset  # noqa: E402

from src.baseline import predict_tags  # noqa: E402
from src.checks import FEATURE_ORDER, findings_to_features, run_all_checks  # noqa: E402
from src.extract import from_token_predictions  # noqa: E402
from src.metrics import extract_entities  # noqa: E402
from src.risk import train_review_model  # noqa: E402

ARTIFACTS = Path(__file__).parent / "artifacts"

# A document is "needs review" when its extraction F1 falls below this.
QUALITY_BAR = 0.8


def words_and_tags(record) -> tuple[list[str], list[str]]:
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
    return words, tags


def build_dataset(split: str, limit: int | None = None):
    """Return (features, labels). Label 1 means extraction got something wrong."""
    dataset = load_dataset("naver-clova-ix/cord-v2", split=split, streaming=True).cast_column(
        # Skip image decoding entirely: only the annotation text is read here,
        # and decoding 800 receipt images to throw them away is most of the runtime.
        "image", HFImage(decode=False)
    )
    features, labels, scores = [], [], []

    for index, record in enumerate(dataset):
        if limit and index >= limit:
            break
        words, gold_tags = words_and_tags(record)
        if not words:
            continue

        predicted_tags = predict_tags(words)
        receipt = from_token_predictions(f"{split}-{index:04d}", words, predicted_tags)
        findings = run_all_checks(receipt)
        features.append(findings_to_features(receipt, findings))

        # Document-level entity F1 against ground truth.
        gold = extract_entities(gold_tags)
        got = extract_entities(predicted_tags)
        overlap = len(gold & got)
        precision = overlap / len(got) if got else 0.0
        recall = overlap / len(gold) if gold else 0.0
        doc_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        labels.append(int(doc_f1 < QUALITY_BAR))
        scores.append(doc_f1)

    return features, labels, scores


def main() -> None:
    print("building training set from CORD train split ...")
    train_features, train_labels, train_scores = build_dataset("train")
    import statistics

    print(f"  {len(train_labels)} documents")
    print(f"  below the {QUALITY_BAR} quality bar: {sum(train_labels)} "
          f"({sum(train_labels) / len(train_labels):.1%})")
    print(f"  median document F1: {statistics.median(train_scores):.3f}")

    if len(set(train_labels)) < 2:
        print("\nEvery document in the training split has the same label.")
        print("The baseline errs on all of them, so there is nothing to learn yet.")
        print("This becomes trainable once the fine-tuned model gets some documents right.")
        result = {
            "trainable": False,
            "reason": "single-class labels: baseline extraction failed on every document",
            "n_train": len(train_labels),
            "error_rate": float(np.mean(train_labels)) if train_labels else None,
        }
        ARTIFACTS.mkdir(exist_ok=True)
        (ARTIFACTS / "review_model_metrics.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    result = train_review_model(train_features, train_labels, target_error_budget=0.05)

    print("\nscoring the held-out test split once ...")
    test_features, test_labels, test_scores = build_dataset("test")

    import joblib

    from src.risk import MODEL_PATH

    model = joblib.load(MODEL_PATH)
    X = np.array([[f.get(name, 0.0) for name in FEATURE_ORDER] for f in test_features])
    probabilities = model.predict_proba(X)[:, 1]
    y = np.array(test_labels)

    auto = probabilities < result["threshold"]
    result.update(
        {
            "trainable": True,
            "n_test": int(len(y)),
            "test_auto_accept_coverage": float(auto.mean()),
            "test_realised_error_rate": float(y[auto].mean()) if auto.sum() else 0.0,
            "test_base_error_rate": float(y.mean()),
            "quality_bar": QUALITY_BAR,
            "median_test_document_f1": float(np.median(test_scores)),
            "note": (
                "Threshold frozen on validation before the test split was touched. "
                "The realised error rate is what the budget actually bought, not the "
                "budget itself."
            ),
        }
    )
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "review_model_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
