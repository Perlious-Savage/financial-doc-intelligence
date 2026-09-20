"""Pipeline orchestration as an explicit state graph.

LangGraph is used here for explicit state transitions and conditional human-review
routing, not to run an LLM agent. There is no agent: no model chooses what to do next.
The branch below is taken on a measured confidence value.

That distinction is worth being precise about. A straight-line pipeline does not need
a graph framework, and dressing one up as an agent would be indefensible. What earns
the graph is the conditional edge: documents that fail confidence skip scoring entirely
and go straight to review.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .checks import run_all_checks
from .risk import ReviewScorer
from .schemas import AnalysisResult, ExtractedReceipt, Finding, ReviewDecision

CONFIDENCE_FLOOR = 0.60


class PipelineState(TypedDict, total=False):
    doc_id: str
    raw: Any
    extraction: ExtractedReceipt
    findings: list[Finding]
    decision: ReviewDecision
    route: str


def _extract_node(state: PipelineState) -> PipelineState:
    """Extraction is injected by the caller so the graph stays testable without a GPU."""
    if "extraction" not in state:
        raise ValueError("extraction must be supplied before running the graph")
    return state


def _validate_node(state: PipelineState) -> PipelineState:
    state["findings"] = run_all_checks(state["extraction"])
    return state


def _confidence_branch(state: PipelineState) -> Literal["review", "score"]:
    """The conditional edge that justifies the graph.

    Below the floor there is no point scoring priority: the document goes to a human
    regardless, so the scoring step is skipped rather than computed and discarded.
    """
    receipt = state["extraction"]
    if receipt.field_confidences and receipt.min_confidence < CONFIDENCE_FLOOR:
        return "review"
    return "score"


def _score_node(state: PipelineState) -> PipelineState:
    state["decision"] = ReviewScorer().decide(state["extraction"], state["findings"])
    state["route"] = "scored"
    return state


def _review_node(state: PipelineState) -> PipelineState:
    receipt = state["extraction"]
    state["decision"] = ReviewDecision(
        review_required=True,
        priority=1.0,
        reasons=[
            f"confidence {receipt.min_confidence:.2f} below floor {CONFIDENCE_FLOOR}"
        ],
    )
    state["route"] = "forced_review"
    return state


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("extract", _extract_node)
    graph.add_node("validate", _validate_node)
    graph.add_node("score", _score_node)
    graph.add_node("review", _review_node)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate", _confidence_branch, {"score": "score", "review": "review"}
    )
    graph.add_edge("score", END)
    graph.add_edge("review", END)
    return graph.compile()


_COMPILED = None


def analyze(extraction: ExtractedReceipt) -> AnalysisResult:
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    final = _COMPILED.invoke({"doc_id": extraction.doc_id, "extraction": extraction})
    return AnalysisResult(
        doc_id=extraction.doc_id,
        extraction=final["extraction"],
        findings=final["findings"],
        decision=final["decision"],
    )
