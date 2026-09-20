"""Validation layer: runs every check and emits structured Findings.

There is no model in this file, and that is the point. Every check is a pure function
of the extracted data, so a finding can always be traced to a computation rather than
to something a model asserted.

All checks run on every document. An earlier design had a model choose which checks to
run; with a handful of checks that is strictly worse — it adds a failure mode and saves
nothing.
"""

from __future__ import annotations

from .schemas import ExtractedReceipt, Finding, Severity
from .tools import (
    check_line_item_arithmetic,
    check_line_items_sum_to_subtotal,
    check_total_reconciles,
)


def check_required_fields(receipt: ExtractedReceipt) -> Finding:
    missing = receipt.missing_fields()
    return Finding(
        check_id="required_fields_present",
        passed=not missing,
        severity=Severity.INFO if not missing else Severity.WARNING,
        message=(
            "All core fields extracted."
            if not missing
            else f"Missing core fields: {', '.join(missing)}."
        ),
        computed={"missing_count": str(len(missing))},
    )


def check_extraction_confidence(receipt: ExtractedReceipt, threshold: float = 0.70) -> Finding:
    low = {f: c for f, c in receipt.field_confidences.items() if c < threshold}
    return Finding(
        check_id="extraction_confidence",
        passed=not low,
        severity=Severity.INFO if not low else Severity.WARNING,
        message=(
            "All field confidences above threshold."
            if not low
            else f"Low confidence on: {', '.join(sorted(low))}."
        ),
        computed={
            "threshold": str(threshold),
            "min_confidence": f"{receipt.min_confidence:.3f}",
            "low_field_count": str(len(low)),
        },
    )


def run_all_checks(receipt: ExtractedReceipt) -> list[Finding]:
    return [
        check_required_fields(receipt),
        check_extraction_confidence(receipt),
        check_line_items_sum_to_subtotal(receipt.line_items, receipt.subtotal),
        check_total_reconciles(
            receipt.subtotal,
            receipt.tax,
            receipt.service_charge,
            receipt.discount,
            receipt.total,
        ),
        check_line_item_arithmetic(receipt.line_items),
    ]


def findings_to_features(receipt: ExtractedReceipt, findings: list[Finding]) -> dict[str, float]:
    """Turn extraction and findings into the feature vector the review model scores.

    Deliberately small and interpretable: every feature is something a human reviewer
    would also look at.
    """
    failed = [f for f in findings if not f.passed]
    return {
        "mean_confidence": receipt.mean_confidence,
        "min_confidence": receipt.min_confidence,
        "missing_field_count": float(len(receipt.missing_fields())),
        "failed_check_count": float(len(failed)),
        "error_severity_count": float(
            sum(1 for f in failed if f.severity == Severity.ERROR)
        ),
        "line_item_count": float(len(receipt.line_items)),
        "has_total": 1.0 if receipt.total is not None else 0.0,
    }


FEATURE_ORDER = [
    "mean_confidence",
    "min_confidence",
    "missing_field_count",
    "failed_check_count",
    "error_severity_count",
    "line_item_count",
    "has_total",
]
