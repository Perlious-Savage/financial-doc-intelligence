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

## Design decisions

**All monetary arithmetic happens in Python, never in a model.** Language models produce
plausible numbers rather than correct ones, and the failure is silent — a wrong total looks
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
