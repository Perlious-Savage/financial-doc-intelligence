# Financial Document Intelligence & Review Automation

Structured field extraction from financial documents, with deterministic validation and
confidence-based routing to human review.

Status: in progress. Metrics land in `artifacts/metrics.json` as experiments complete.

## Architecture

```
Input document
      |
      v
Extraction  (LayoutLMv3 fine-tuned on CORD-v2; regex/positional baseline for comparison)
      |
      v
Structured JSON  (Pydantic schema)
      |
      v
Deterministic validation  (Decimal arithmetic: line items -> subtotal, + tax -> total)
      |
      v
Confidence check ----- low ------> human review
      |
     high
      |
      v
Review-priority scoring  (scikit-learn)
      |
      v
FastAPI service ----+---- PostgreSQL + pgvector  (similar-document retrieval)
                    |
                    +---- GPU inference service (separate from the CPU container)

MLflow tracks training runs.  LangGraph orchestrates the flow above, including the branch.
```

## Results

### Baseline: keyword and position rules, no model

Deterministic, CPU-only, no training. Scored on the CORD-v2 test split
(100 documents, 0.509 micro F1).

Scoring is **entity-level**: a field counts as correct only if its type and its full
span match the reference exactly. A partially extracted merchant name scores zero,
because a partially extracted field is not a usable one. Implementation in
`src/metrics.py`.

The aggregate number matters less than where it comes from. The baseline solves fields
that carry a printed label next to their value, and scores **exactly zero** on every
field that requires understanding position on the page:

**Fields the rules handle**

| field | F1 | support |
|---|---|---|
| `total.cashprice` | 0.835 | 69 |
| `sub_total.subtotal_price` | 0.825 | 66 |
| `total.changeprice` | 0.804 | 56 |
| `sub_total.service_price` | 0.800 | 12 |
| `total.total_price` | 0.694 | 96 |
| `sub_total.tax_price` | 0.684 | 45 |

**Fields the rules cannot touch**

| field | F1 | support |
|---|---|---|
| `menu.cnt` | 0.000 | 220 |
| `menu.unitprice` | 0.000 | 67 |
| `menu.sub.nm` | 0.000 | 36 |
| `menu.sub.price` | 0.000 | 20 |
| `menu.sub.cnt` | 0.000 | 17 |
| `sub_total.etc` | 0.000 | 13 |

Quantities sitting in a column, unit prices positioned relative to their line item,
nested sub-items — none of these announce themselves with a keyword. That gap is the
reason to reach for a layout-aware model, and it is what the fine-tune is measured
against.

### Fine-tuned LayoutLMv3

Pending. Results land in `artifacts/metrics.json` and are written here when the run
completes.

Reference point: the published `nielsr/layoutlmv3-finetuned-cord` reports 0.964 F1 on
this dataset. That is a different evaluation setup, so it is context rather than a
like-for-like comparison.

## Design decisions

**All monetary arithmetic happens in Python, never in a model.** Language models produce
plausible numbers rather than correct ones, and the failure is silent â€” a wrong total looks
exactly like a right one. Computation lives in `src/tools.py` using `Decimal`, so that error
class is gone by construction rather than by prompting.

**The routing decision is computed from structured findings, never from generated text.**
Checks emit `Finding` objects; the decision is a function of those objects. A model cannot
introduce an unsupported claim into the decision path.

**Extraction reports three outcomes per field: correct, incorrect, abstained.** Abstaining is
safe; inventing a value is not. A metric that scores them identically measures the wrong thing.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add your database URL
pytest tests/ -q
```

Training (GPU):

```bash
pip install -r requirements-train.txt
python -m src.train_extractor --epochs 8
```

## Limitations

- CORD-v2 is a receipt dataset, not a KYC or onboarding dataset. Evaluation measures document
  extraction, not regulatory compliance.
- Results come from a limited number of training runs; no confidence intervals, no seed sweep.
- GPU inference is evaluated separately from the CPU API service.
- Semantic retrieval is evaluated as document similarity, not retrieval-augmented generation.

## Dataset

CORD-v2 (`naver-clova-ix/cord-v2`), CC-BY-4.0.
