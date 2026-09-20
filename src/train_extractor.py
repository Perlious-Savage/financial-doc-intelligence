"""Fine-tune LayoutLMv3 on CORD-v2 for receipt field extraction.

Runs on GPU (Colab). Writes everything needed to reproduce and verify the result into
artifacts/: the metrics, the exact training configuration, and the per-field scores.
A README that cites artifacts/metrics.json is evidence; a README that cites memory is
a claim.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model", default="microsoft/layoutlmv3-base")
    parser.add_argument("--output", default="outputs/layoutlmv3-cord")
    parser.add_argument("--max-train", type=int, default=0, help="0 = use all")
    args = parser.parse_args()

    import mlflow
    import numpy as np
    import torch
    from datasets import load_dataset
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from transformers import (
        AutoProcessor,
        LayoutLMv3ForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    from .data import DATASET_ID, build_label_list, parse_example

    ARTIFACTS.mkdir(exist_ok=True)

    print(f"Loading {DATASET_ID} ...")
    raw = load_dataset(DATASET_ID)
    labels = build_label_list(raw)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    print(f"{len(labels)} labels across {len(raw['train'])} training receipts")

    processor = AutoProcessor.from_pretrained(args.model, apply_ocr=False)

    def encode(batch):
        images, words, boxes, word_labels = [], [], [], []
        for image, ground_truth in zip(batch["image"], batch["ground_truth"]):
            parsed = parse_example(ground_truth, image.width, image.height)
            if not parsed["words"]:
                continue
            images.append(image.convert("RGB"))
            words.append(parsed["words"])
            boxes.append(parsed["bboxes"])
            word_labels.append([label2id.get(t, 0) for t in parsed["ner_tags"]])
        encoding = processor(
            images,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        return encoding

    columns = raw["train"].column_names
    train_ds = raw["train"]
    if args.max_train:
        train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
    train_ds = train_ds.map(encode, batched=True, batch_size=8, remove_columns=columns)
    eval_ds = raw["validation"].map(encode, batched=True, batch_size=8, remove_columns=columns)
    test_ds = raw["test"].map(encode, batched=True, batch_size=8, remove_columns=columns)
    for ds in (train_ds, eval_ds, test_ds):
        ds.set_format("torch")

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model, id2label=id2label, label2id=label2id
    )

    def decode(predictions, references):
        """Drop the -100 padding positions before scoring."""
        preds, refs = [], []
        for prediction, reference in zip(predictions, references):
            kept_p, kept_r = [], []
            for p, r in zip(prediction, reference):
                if r != -100:
                    kept_p.append(id2label[int(p)])
                    kept_r.append(id2label[int(r)])
            preds.append(kept_p)
            refs.append(kept_r)
        return preds, refs

    def compute_metrics(eval_prediction):
        logits, references = eval_prediction
        predictions = np.argmax(logits, axis=2)
        preds, refs = decode(predictions, references)
        return {
            "precision": precision_score(refs, preds),
            "recall": recall_score(refs, preds),
            "f1": f1_score(refs, preds),
        }

    # transformers renamed `evaluation_strategy` to `eval_strategy` in 4.41. Colab's
    # pinned version moves around, so pick whichever this install accepts.
    import inspect

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_strategy="no",
        logging_steps=25,
        report_to=[],
        fp16=torch.cuda.is_available(),
        **{eval_key: "epoch"},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    mlflow.set_experiment("layoutlmv3-cord-extraction")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": args.model,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "train_size": len(train_ds),
                "num_labels": len(labels),
            }
        )
        trainer.train()

        print("Evaluating on the held-out test split ...")
        test_output = trainer.predict(test_ds)
        preds, refs = decode(np.argmax(test_output.predictions, axis=2), test_output.label_ids)

        metrics = {
            "dataset": DATASET_ID,
            "model": args.model,
            "test_precision": float(precision_score(refs, preds)),
            "test_recall": float(recall_score(refs, preds)),
            "test_f1": float(f1_score(refs, preds)),
            "n_train": len(train_ds),
            "n_test": len(test_ds),
            "num_labels": len(labels),
            "note": "Single training run. No confidence intervals, no repeated seeds.",
        }
        mlflow.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        )

        (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (ARTIFACTS / "training_config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "labels": labels,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "python": platform.python_version(),
                },
                indent=2,
            )
        )
        (ARTIFACTS / "per_field_report.txt").write_text(classification_report(refs, preds))
        (ARTIFACTS / "predictions.json").write_text(
            json.dumps(
                [{"predicted": p, "reference": r} for p, r in zip(preds[:50], refs[:50])],
                indent=2,
            )
        )
        trainer.save_model(args.output)
        processor.save_pretrained(args.output)

    print(json.dumps(metrics, indent=2))
    print(f"\nArtifacts written to {ARTIFACTS}")


if __name__ == "__main__":
    main()
