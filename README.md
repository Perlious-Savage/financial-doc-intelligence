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

nested sub-items - none of these announce themselves with a keyword. That gap is the

reason to reach for a layout-aware model, and it is what the fine-tune is measured

against.



### Fine-tuned LayoutLMv3



Same test split, same entity-level metric, so the comparison is like for like.



| method | precision | recall | F1 |

|---|---|---|---|

| keyword + position rules, no model | 0.557 | 0.468 | **0.509** |

| LayoutLMv3 fine-tuned | 0.942 | 0.955 | **0.948** |

| | | | **+0.439** |



8 epochs, lr 5e-05, batch size 4, 800 training

documents, 59 labels, 711s on an A100.

Validation F1 by epoch: 0.820, 0.923, 0.921, 0.950, 0.946, 0.952, 0.955, 0.955 - it plateaus

from epoch 6, so the last two epochs bought nothing.



**Caveats, stated plainly.** This is a single training run: no repeated seeds, no confidence

intervals. With 100 test documents the interval around 0.948 is not tight, and a rerun would

move the third decimal place. The published `nielsr/layoutlmv3-finetuned-cord` reports 0.964

on this dataset under a different evaluation setup - context, not a like-for-like comparison.



Artifacts: `artifacts/metrics.json`, `artifacts/training_config.json`,

`artifacts/per_field_report.txt`, `artifacts/predictions.json`.



### Where the gain actually came from

The aggregate jump from 0.509 to 0.948 is not spread evenly, and the per-field breakdown is the
part worth reading.

**Fields the rule baseline could not touch at all, now solved.** These have no printed label
beside them - a quantity sits in a column, a unit price is positioned relative to its line item -
so keyword matching scores exactly zero by construction. This is the gap a layout-aware model
exists to close, and it closed it:

| field | support | baseline F1 | LayoutLMv3 F1 |
|---|---|---|---|
| `menu.cnt` | 220 | 0.000 | **0.984** |
| `menu.unitprice` | 67 | 0.000 | **0.933** |
| `menu.sub.nm` | 36 | 0.000 | **0.892** |
| `total.menuqty_cnt` | 29 | 0.067 | **0.967** |
| `menu.sub.price` | 20 | 0.000 | **0.947** |
| `menu.sub.cnt` | 17 | 0.000 | **0.944** |
| `menu.num` | 11 | 0.000 | **0.842** |
| `total.menutype_cnt` | 7 | 0.000 | **0.833** |

**What still fails.** Every remaining weak field is a rare class, and the pattern is consistent
rather than scattered:

| field | support | baseline F1 | LayoutLMv3 F1 |
|---|---|---|---|
| `total.emoneyprice` | 2 | 0.000 | **0.400** |
| `menu.etc` | 3 | 0.000 | **0.000** |
| `total.total_etc` | 3 | 0.000 | **0.000** |
| `menu.itemsubtotal` | 6 | 0.000 | **0.000** |
| `menu.discountprice` | 10 | 0.000 | **0.571** |

None has more than ten test examples, and three have three or fewer. Two of them (`menu.etc`,
`total.total_etc`) are catch-all categories, so scarcity compounds with genuine semantic ambiguity
about what belongs in them. The honest reading is that the model has learned the frequent
structure of a receipt well and has not seen enough of the long tail - which is a data problem,
not an architecture problem, and would be addressed with more examples of those classes rather
than a different model.

## Design decisions



**All monetary arithmetic happens in Python, never in a model.** Language models produce

plausible numbers rather than correct ones, and the failure is silent - a wrong total looks

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

