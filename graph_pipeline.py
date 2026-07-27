"""LangGraph-based orchestrator (non-invasive, not wired into app.py).

Side-by-side alternative to pipeline.run_pipeline() / resume_pipeline_from_result(),
per docs/design-langGraph.html. Stage modules (generate/evaluate/critique/revise/
compare/gates) are reused untouched; this file only replaces orchestration.

Design invariants:
- Nodes are thin wrappers around existing stage functions; no LangChain model wrappers.
- Nodes keep writing the same runs/TIMESTAMP/*.json artifacts, so app.py's
  _load_result_from_dir() and the history index keep working unchanged.
- Each gate is split into a compute node (deterministic scoring, artifact writes)
  and a *_wait node containing only interrupt(). The pending status is committed
  to graph state by the compute node before suspension, so the checkpoint,
  summary.json, and the PipelineResult returned to the caller all agree — and
  resuming re-executes only the trivial wait node, never the gate computation.
- on_stage / on_image_done callbacks travel in config["configurable"] under
  "__"-prefixed keys (excluded from checkpoint metadata, never serialized) and
  fire at the *top* of each node, matching legacy pre-stage notification.
- State stores Pydantic models as model_dump() dicts and paths as str, so every
  channel is JSON-serializable for the checkpointer.

Feature parity with the legacy orchestrator:
- run_pipeline_graph()    == run_pipeline() (same callbacks, same artifacts)
- resume_pipeline_graph() == HIL resume (with payload) and crash recovery
                             (without payload) from the checkpoint
- retry_pipeline_graph()  == resume_pipeline_from_result(): rebuilds state from
                             disk artifacts at the first missing stage, works
                             even when no checkpoint exists (legacy runs).
"""

from __future__ import annotations

import json
import operator
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated, Callable, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from compare import determine_winner
from errors import CheckpointMissing  # noqa: F401  (re-exported for callers)
from config import (
    CONVERGENCE_THRESHOLD,
    DEFAULT_MODEL_A,
    DEFAULT_MODEL_B,
    HIL_ENABLED_BY_DEFAULT,
    MAX_CRITIQUE_ROUNDS,
)
from critique import critique_evaluation, critique_evaluation_gemini
from evaluate import evaluate_images
from gates import evaluate_gate1, evaluate_gate2
from generate import generate_images
from pipeline import (  # reuse, don't fork: same artifacts + fallback semantics
    PipelineResult,
    _build_comparison_fallback,
    _existing_image_path,
    _final_revision,
    _find_reference_image,
    _format_exception,
    _gate1_arbitration,
    _model_label,
    _persist_reference_image,
    _save_prompt_input_metadata,
    _save_result,
    _scores_converged,
)
from revise import revise_evaluation
from schemas import (
    ComparisonResult,
    CritiqueResponse,
    GateDecision,
    HilAdjudication,
    HilArbitration,
    ImageEvaluation,
    IssueEquivalenceReport,
    PromptInputMetadata,
    RevisedEvaluation,
    RevisedImageEvaluation,
)

CHECKPOINT_DB = Path("runs") / "checkpoints.db"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    # inputs (set once at START)
    prompt: str
    model_a_key: str
    model_b_key: str
    model_a_label: str
    model_b_label: str
    run_dir: str                      # str, not Path — must be serializable
    reference_image_path: str | None
    reference_image_name: str | None  # original upload filename -> summary.json reference_image_source
    timestamp: str

    # artifacts (accumulated as model_dump() dicts)
    image_paths: dict[str, str]
    eval_a: dict | None
    eval_b: dict | None
    prompt_difficulty: str | None
    critiques: Annotated[list[dict], operator.add]    # append-only
    revisions: Annotated[list[dict], operator.add]    # append-only
    gate_decisions: dict[str, dict]   # keyed by gate name -> replace-on-rerun
    issue_equivalence: dict | None
    hil_reviews: Annotated[list[dict], operator.add]
    hil_adjudication: dict | None
    comparison: dict | None

    # control
    round_num: int                    # current critique round (1-based)
    rounds_completed: int
    stage_failed: str | None          # transient: set by a failing node, read by routers
    pipeline_status: str
    errors: Annotated[list[str], operator.add]
    requires_attention: bool


# ---------------------------------------------------------------------------
# Callback plumbing (config["configurable"], never state: not serializable)
# ---------------------------------------------------------------------------
# Keys are "__"-prefixed so LangGraph excludes them from checkpoint metadata.

def _configurable(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable") or {}


def _notify(config: RunnableConfig | None, stage_key: str | None) -> None:
    """Fire on_stage *before* the stage runs, like the legacy orchestrator."""
    on_stage = _configurable(config).get("__on_stage")
    if on_stage and stage_key:
        on_stage(stage_key)


def _gate_decision(state: PipelineState, gate_prefix: str) -> dict | None:
    for name, decision in state.get("gate_decisions", {}).items():
        if name.startswith(gate_prefix):
            return decision
    return None


# ---------------------------------------------------------------------------
# State <-> PipelineResult bridge (keeps app.py's view-model working)
# ---------------------------------------------------------------------------

def _dump(payload):
    """Inverse of _as_model(): back to a JSON-serializable dict for state."""
    return payload.model_dump() if hasattr(payload, "model_dump") else payload


def _as_model(model_cls, payload):
    """Rehydrate a HIL payload into its schema model, mirroring app.py's
    _load_result_from_dir() so the same field has the same type regardless of
    whether the result came from graph state or from disk. Payloads that don't
    validate are passed through unchanged — a hand-rolled or truncated
    arbitration must not crash the summary write mid-run.
    """
    if not isinstance(payload, dict):
        return payload
    try:
        return model_cls(**payload)
    except ValidationError:
        return payload


def state_to_result(state: PipelineState) -> PipelineResult:
    """Rehydrate the UI-facing PipelineResult from graph state."""
    result = PipelineResult()
    result.prompt = state.get("prompt", "")
    result.timestamp = state.get("timestamp", "")
    result.model_a_key = state.get("model_a_key", DEFAULT_MODEL_A)
    result.model_b_key = state.get("model_b_key", DEFAULT_MODEL_B)
    result.model_a_label = state.get("model_a_label", _model_label(result.model_a_key))
    result.model_b_label = state.get("model_b_label", _model_label(result.model_b_key))
    if state.get("run_dir"):
        result.run_dir = Path(state["run_dir"])
    if state.get("reference_image_path"):
        result.reference_image_path = Path(state["reference_image_path"])
    result.reference_image_name = state.get("reference_image_name")
    result.image_paths = {k: Path(v) for k, v in state.get("image_paths", {}).items()}
    if state.get("eval_a"):
        result.eval_a = ImageEvaluation(**state["eval_a"])
    if state.get("eval_b"):
        result.eval_b = ImageEvaluation(**state["eval_b"])
    result.critiques = [CritiqueResponse(**c) for c in state.get("critiques", [])]
    result.revisions = [RevisedEvaluation(**r) for r in state.get("revisions", [])]
    result.gate_decisions = [GateDecision(**g) for g in state.get("gate_decisions", {}).values()]
    result.hil_reviews = [_as_model(HilArbitration, r) for r in state.get("hil_reviews", [])]
    result.hil_adjudication = _as_model(HilAdjudication, state.get("hil_adjudication"))
    if state.get("issue_equivalence"):
        result.issue_equivalence = IssueEquivalenceReport(**state["issue_equivalence"])
    if state.get("comparison"):
        result.comparison = ComparisonResult(**state["comparison"])
    result.prompt_difficulty = state.get("prompt_difficulty")
    result.pipeline_status = state.get("pipeline_status", "partial")
    result.requires_attention = state.get("requires_attention", False)
    result.errors = list(state.get("errors", []))
    result.rounds_completed = state.get("rounds_completed", 0)
    return result


def _persist_summary(state: PipelineState) -> None:
    """Write summary.json / index.json via the existing pipeline helper."""
    _save_result(state_to_result(state))


# ---------------------------------------------------------------------------
# Nodes — thin wrappers around existing stage functions
# ---------------------------------------------------------------------------

def generate_node(state: PipelineState, config: RunnableConfig) -> dict:
    _notify(config, "stage_generating")
    run_dir = Path(state["run_dir"])
    ref = Path(state["reference_image_path"]) if state.get("reference_image_path") else None
    raw = generate_images(
        state["prompt"], run_dir,
        on_model_done=_configurable(config).get("__on_image_done"),
        reference_image_path=ref,
        model_a=state["model_a_key"], model_b=state["model_b_key"],
    )
    image_paths: dict[str, str] = {}
    errors: list[str] = []
    for slot, label in (("gpt_image_2", state["model_a_label"]), ("gemini_3_pro", state["model_b_label"])):
        value = raw.get(slot)
        if isinstance(value, Path):
            image_paths[slot] = str(value)
        elif isinstance(value, Exception):
            errors.append(f"{label} failed: {_format_exception(value)}")
        else:
            # Slot absent entirely: legacy appends no message here, but a bare
            # "failed" beats summary.json reading "... failed: None".
            errors.append(f"{label} failed: no image returned")
    failed = len(image_paths) < 2
    return {
        "image_paths": image_paths,
        "errors": errors,
        "stage_failed": "generate" if failed else None,
    }


def evaluate_node(state: PipelineState, config: RunnableConfig) -> dict:
    _notify(config, "stage_evaluating")
    run_dir = Path(state["run_dir"])
    try:
        eval_a, eval_b, difficulty = evaluate_images(
            state["prompt"],
            Path(state["image_paths"]["gpt_image_2"]),
            Path(state["image_paths"]["gemini_3_pro"]),
            model_a_label=state["model_a_label"],
            model_b_label=state["model_b_label"],
        )
    except Exception as e:  # router sends us to fail_and_save
        return {"errors": [f"Evaluation failed: {e}"], "stage_failed": "evaluate"}

    (run_dir / "evaluation.json").write_text(
        json.dumps({"model_a": eval_a.model_dump(), "model_b": eval_b.model_dump()}, indent=2)
    )
    (run_dir / "evaluation_v2.json").write_text(
        json.dumps(
            {"prompt_difficulty": difficulty, "model_a": eval_a.model_dump(), "model_b": eval_b.model_dump()},
            indent=2,
        )
    )
    return {
        "eval_a": eval_a.model_dump(),
        "eval_b": eval_b.model_dump(),
        "prompt_difficulty": difficulty,
        "stage_failed": None,
    }


def gate1_node(state: PipelineState, config: RunnableConfig) -> dict:
    # No "already decided" guard here (unlike gate2_node): _seed_for_retry only
    # re-enters at `evaluate` when gate1 has no decision, and enters *as* gate1
    # when it does — so this node never runs against an existing decision.
    _notify(config, "stage_gate1")
    run_dir = Path(state["run_dir"])
    decision = evaluate_gate1(
        state["prompt"],
        ImageEvaluation(**state["eval_a"]),
        ImageEvaluation(**state["eval_b"]),
        hil_enabled=HIL_ENABLED_BY_DEFAULT,
    )
    (run_dir / "gate1_decision.json").write_text(json.dumps(decision.model_dump(), indent=2))

    update = {
        "gate_decisions": {**state.get("gate_decisions", {}), decision.gate: decision.model_dump()},
        "pipeline_status": decision.pipeline_status or state.get("pipeline_status", "partial"),
        "requires_attention": state.get("requires_attention", False) or decision.requires_attention,
        "stage_failed": None,
    }
    if decision.status == "pending":
        # Committed to state *before* gate1_wait suspends, so the checkpoint,
        # summary.json, and the caller's PipelineResult all agree on the pause.
        update["pipeline_status"] = decision.pipeline_status or "pending_hil_gate1"
        update["requires_attention"] = True
        _persist_summary({**state, **update})
    return update


def gate1_wait_node(state: PipelineState, config: RunnableConfig) -> dict:
    """Interrupt-only node: re-executed on resume, so it must stay side-effect
    free above the interrupt() call. The gate decision is read from state."""
    run_dir = Path(state["run_dir"])
    decision = GateDecision(**_gate_decision(state, "gate1"))
    # Graph suspends here; state is checkpointed; the process may exit.
    # Streamlit resumes with resume_pipeline_graph(run_id, arbitration).
    arbitration = interrupt({
        "gate": decision.gate,
        "route_band": decision.route_band,
        "route_score": decision.route_score,
        "trigger_reasons": decision.trigger_reasons,
        "review_items": [item.model_dump() for item in decision.review_items],
    })
    (run_dir / "hil_review_r1.json").write_text(json.dumps(arbitration, indent=2))
    completed = decision.model_copy(update={"status": "completed", "pipeline_status": None, "requires_attention": False})
    (run_dir / "gate1_decision.json").write_text(json.dumps(completed.model_dump(), indent=2))
    return {
        "gate_decisions": {**state.get("gate_decisions", {}), completed.gate: completed.model_dump()},
        "hil_reviews": [arbitration],
        "pipeline_status": "partial",
        "requires_attention": False,
        "stage_failed": None,
    }


def critique_node(state: PipelineState, config: RunnableConfig) -> dict:
    run_dir = Path(state["run_dir"])
    round_num = len(state.get("critiques", [])) + 1
    _notify(config, "stage_critique" if round_num == 1 else "stage_critique_round2")
    image_a = Path(state["image_paths"]["gpt_image_2"])
    image_b = Path(state["image_paths"]["gemini_3_pro"])
    try:
        if round_num == 1:
            crit = critique_evaluation(
                state["prompt"],
                ImageEvaluation(**state["eval_a"]),
                ImageEvaluation(**state["eval_b"]),
                image_a, image_b,
                model_a_label=state["model_a_label"],
                model_b_label=state["model_b_label"],
            )
            metadata = PromptInputMetadata(
                step="05_critique",
                included_artifacts=["prompt", "image_a", "image_b", "evaluation.json"],
                excluded_artifacts=["revised_r1.json", "critique_r2.json", "revised_r2.json"],
            )
        else:
            prev = RevisedEvaluation(**state["revisions"][-1])
            crit = critique_evaluation_gemini(
                state["prompt"], prev.model_a, prev.model_b, image_a, image_b,
                raw_output_path=run_dir / "critique_r2_raw.txt",
                model_a_label=state["model_a_label"],
                model_b_label=state["model_b_label"],
            )
            metadata = PromptInputMetadata(
                step="07_critique",
                included_artifacts=["prompt", "image_a", "image_b", "revised_r1.json"],
                excluded_artifacts=["critique_r1.json", "evaluation.json", "revised_r2.json"],
            )
    except Exception as e:
        return {
            "errors": [f"Critique round {round_num} failed: {e}"],
            "round_num": round_num,
            "stage_failed": "critique",
        }

    (run_dir / f"critique_r{round_num}.json").write_text(json.dumps(crit.model_dump(), indent=2))
    _save_prompt_input_metadata(run_dir, metadata)
    return {"critiques": [crit.model_dump()], "round_num": round_num, "stage_failed": None}


def gate2_node(state: PipelineState, config: RunnableConfig) -> dict:
    existing = _gate_decision(state, "gate2")
    if existing and existing.get("status") != "pending":
        return {"stage_failed": None}  # seeded from a retried run; gate already decided

    _notify(config, "stage_gate2")
    run_dir = Path(state["run_dir"])
    critiques = state["critiques"]
    decision, equivalence = evaluate_gate2(
        ImageEvaluation(**state["eval_a"]),
        ImageEvaluation(**state["eval_b"]),
        RevisedEvaluation(**state["revisions"][-1]),
        CritiqueResponse(**critiques[0]),
        CritiqueResponse(**critiques[-1]),
        hil_enabled=HIL_ENABLED_BY_DEFAULT,
    )
    (run_dir / "gate2_decision.json").write_text(json.dumps(decision.model_dump(), indent=2))
    (run_dir / "issue_equivalence.json").write_text(json.dumps(equivalence.model_dump(), indent=2))

    update = {
        "gate_decisions": {**state.get("gate_decisions", {}), decision.gate: decision.model_dump()},
        "issue_equivalence": equivalence.model_dump(),
        "pipeline_status": decision.pipeline_status or state.get("pipeline_status", "partial"),
        "requires_attention": state.get("requires_attention", False) or decision.requires_attention,
        "stage_failed": None,
    }
    if decision.status == "pending":
        update["pipeline_status"] = decision.pipeline_status or "pending_hil_gate2"
        update["requires_attention"] = True
        _persist_summary({**state, **update})
    return update


def gate2_wait_node(state: PipelineState, config: RunnableConfig) -> dict:
    """Interrupt-only node; see gate1_wait_node."""
    run_dir = Path(state["run_dir"])
    decision = GateDecision(**_gate_decision(state, "gate2"))
    adjudication = interrupt({
        "gate": decision.gate,
        "trigger_reasons": decision.trigger_reasons,
        "review_dimensions": list(decision.review_dimensions),
    })
    (run_dir / "hil_adjudication.json").write_text(json.dumps(adjudication, indent=2))
    completed = decision.model_copy(update={"status": "completed", "pipeline_status": None, "requires_attention": False})
    (run_dir / "gate2_decision.json").write_text(json.dumps(completed.model_dump(), indent=2))
    return {
        "gate_decisions": {**state.get("gate_decisions", {}), completed.gate: completed.model_dump()},
        "hil_adjudication": adjudication,
        "pipeline_status": "partial",
        "requires_attention": False,
        "stage_failed": None,
    }


def revise_node(state: PipelineState, config: RunnableConfig) -> dict:
    run_dir = Path(state["run_dir"])
    round_num = state["round_num"]
    _notify(config, "stage_revising" if round_num <= 1 else "stage_revising_round2")
    if round_num == 1:
        input_a: ImageEvaluation | RevisedImageEvaluation = ImageEvaluation(**state["eval_a"])
        input_b: ImageEvaluation | RevisedImageEvaluation = ImageEvaluation(**state["eval_b"])
    else:
        prev = RevisedEvaluation(**state["revisions"][-1])
        input_a, input_b = prev.model_a, prev.model_b
    try:
        rev = revise_evaluation(
            state["prompt"], input_a, input_b,
            CritiqueResponse(**state["critiques"][-1]),
            Path(state["image_paths"]["gpt_image_2"]),
            Path(state["image_paths"]["gemini_3_pro"]),
            model_a_label=state["model_a_label"],
            model_b_label=state["model_b_label"],
        )
    except Exception as e:
        return {
            "errors": [f"Revision round {round_num} failed: {e}"],
            "stage_failed": "revise",
        }

    (run_dir / f"revised_r{round_num}.json").write_text(json.dumps(rev.model_dump(), indent=2))
    _save_prompt_input_metadata(
        run_dir,
        PromptInputMetadata(
            step="06_revision" if round_num == 1 else "08_final_revision",
            included_artifacts=(
                ["prompt", "image_a", "image_b", "evaluation.json", "critique_r1.json", "gate1_decision.json"]
                if round_num == 1
                else [
                    "prompt", "image_a", "image_b", "critique_r1.json", "revised_r1.json",
                    "critique_r2.json", "gate1_decision.json", "gate2_decision.json",
                ]
            ),
            excluded_artifacts=["critique_r2.json", "revised_r2.json"] if round_num == 1 else [],
        ),
    )
    return {"revisions": [rev.model_dump()], "rounds_completed": round_num, "stage_failed": None}


def compare_node(state: PipelineState, config: RunnableConfig) -> dict:
    """Winner determination from the final revision (r1 if round 2 degraded)."""
    _notify(config, "stage_complete")  # top-of-node, like every other stage
    run_dir = Path(state["run_dir"])
    result_view = state_to_result(state)
    revised = _final_revision(result_view)
    comparison = determine_winner(
        state["prompt"],
        ImageEvaluation(**state["eval_a"]),
        ImageEvaluation(**state["eval_b"]),
        revised,
        model_a_name=state["model_a_label"],
        model_b_name=state["model_b_label"],
        # state_to_result() already rehydrates hil_reviews with the same
        # coercion app.py applies when loading from disk, so both orchestrators
        # feed determine_winner an identically-shaped arbitration.
        arbitration=_gate1_arbitration(result_view),
    )
    (run_dir / "comparison.json").write_text(json.dumps(comparison.model_dump(), indent=2))
    # Backward-compat aliases
    if state.get("critiques"):
        (run_dir / "critique.json").write_text(json.dumps(state["critiques"][-1], indent=2))
    (run_dir / "revised.json").write_text(json.dumps(revised.model_dump(), indent=2))

    status = state.get("pipeline_status", "partial")
    update = {
        "comparison": comparison.model_dump(),
        "pipeline_status": "completed" if status == "partial" else status,
        "stage_failed": None,
    }
    _persist_summary({**state, **update})
    return update


def fallback_compare_node(state: PipelineState, config: RunnableConfig) -> dict:
    """Round-1 critique/revision failed: compare on initial evaluation scores."""
    _notify(config, "stage_complete")
    result = state_to_result(state)
    _build_comparison_fallback(result)  # reuses the exact degradation semantics
    _save_result(result)
    return {
        "revisions": [result.revised.model_dump()] if result.revised else [],
        "comparison": result.comparison.model_dump() if result.comparison else None,
        "pipeline_status": result.pipeline_status,
        "stage_failed": None,
    }


def fail_node(state: PipelineState) -> dict:
    update = {"pipeline_status": "failed", "requires_attention": True, "stage_failed": None}
    _persist_summary({**state, **update})
    return update


# ---------------------------------------------------------------------------
# Routers (conditional edges)
# ---------------------------------------------------------------------------

def after_generate(state: PipelineState) -> Literal["evaluate", "fail"]:
    return "fail" if state.get("stage_failed") else "evaluate"


def after_evaluate(state: PipelineState) -> Literal["gate1", "fail"]:
    return "fail" if state.get("stage_failed") else "gate1"


def after_gate1(state: PipelineState) -> Literal["gate1_wait", "critique"]:
    decision = _gate_decision(state, "gate1") or {}
    return "gate1_wait" if decision.get("status") == "pending" else "critique"


def after_critique(state: PipelineState) -> Literal["revise", "gate2", "fallback_compare", "compare"]:
    if state.get("stage_failed"):
        # r1 failure -> initial-score fallback; r2 failure -> keep r1 revision.
        return "fallback_compare" if state["round_num"] == 1 else "compare"
    return "gate2" if state["round_num"] >= 2 else "revise"


def after_gate2(state: PipelineState) -> Literal["gate2_wait", "revise"]:
    decision = _gate_decision(state, "gate2") or {}
    return "gate2_wait" if decision.get("status") == "pending" else "revise"


def after_revise(state: PipelineState) -> Literal["critique", "compare", "fallback_compare"]:
    if state.get("stage_failed"):
        return "fallback_compare" if state["round_num"] == 1 else "compare"
    revisions = state.get("revisions", [])
    if state["round_num"] >= MAX_CRITIQUE_ROUNDS:
        return "compare"
    if len(revisions) >= 2 and _scores_converged(
        RevisedEvaluation(**revisions[-2]), RevisedEvaluation(**revisions[-1]), CONVERGENCE_THRESHOLD
    ):
        return "compare"
    return "critique"  # loop back — the cycle LCEL can't express


# ---------------------------------------------------------------------------
# Graph builder + public API
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Compile the pipeline graph. Pass MemorySaver() in tests."""
    g = StateGraph(PipelineState)
    g.add_node("generate", generate_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("gate1", gate1_node)
    g.add_node("gate1_wait", gate1_wait_node)
    g.add_node("critique", critique_node)
    g.add_node("gate2", gate2_node)
    g.add_node("gate2_wait", gate2_wait_node)
    g.add_node("revise", revise_node)
    g.add_node("compare", compare_node)
    g.add_node("fallback_compare", fallback_compare_node)
    g.add_node("fail", fail_node)

    g.add_edge(START, "generate")
    g.add_conditional_edges("generate", after_generate)
    g.add_conditional_edges("evaluate", after_evaluate)
    g.add_conditional_edges("gate1", after_gate1)
    g.add_edge("gate1_wait", "critique")
    g.add_conditional_edges("critique", after_critique)
    g.add_conditional_edges("gate2", after_gate2)
    g.add_edge("gate2_wait", "revise")
    g.add_conditional_edges("revise", after_revise)
    g.add_edge("compare", END)
    g.add_edge("fallback_compare", END)
    g.add_edge("fail", END)

    return g.compile(checkpointer=checkpointer or default_checkpointer())


_default_saver: SqliteSaver | None = None


def default_checkpointer() -> SqliteSaver:
    """Process-wide singleton: one connection (WAL, busy timeout) shared by all
    graphs, so Streamlit reruns and background threads don't pile up
    connections or hit 'database is locked'."""
    global _default_saver
    if _default_saver is None:
        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        _default_saver = SqliteSaver(conn)
    return _default_saver


def _callback_config(thread_id: str, on_stage, on_image_done) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id,
            "__on_stage": on_stage,
            "__on_image_done": on_image_done,
        }
    }


def _run_to_result(graph, input_, config: RunnableConfig) -> PipelineResult:
    graph.invoke(input_, config)
    # Read the state back rather than trusting invoke()'s return: after an
    # interrupt this reflects the pending gate committed by the compute node.
    snapshot = graph.get_state(config)
    result = state_to_result(snapshot.values)
    # An empty `next` means the graph reached END — completed, degraded, or
    # failed. A checkpoint only exists to let an *unfinished* run continue, so
    # at that point it is dead weight (~200KB per run that SQLite never reclaims
    # on its own). A pause keeps a non-empty `next` and is left alone, as is a
    # crash, which is exactly the case recovery needs it for.
    if not snapshot.next:
        _discard_thread(graph.checkpointer, _configurable(config).get("thread_id"))
    return result


def _discard_thread(checkpointer, run_id: str | None) -> bool:
    """Delete one thread from the checkpointer. True if the delete was attempted.

    Tolerates a checkpointer that predates delete_thread; deleting an unknown
    thread is a no-op rather than an error, so callers need not check first.
    """
    if checkpointer is None or not run_id:
        return False
    try:
        checkpointer.delete_thread(run_id)
        return True
    except (AttributeError, NotImplementedError):
        return False


def discard_checkpoint(run_id: str, checkpointer=None) -> bool:
    """Drop a run's checkpoint. Call when the run itself is being deleted.

    Retries rebuild from the run's disk artifacts (retry_pipeline_graph), so no
    caller needs the thread to survive its run directory.
    """
    graph = build_graph(checkpointer)
    return _discard_thread(graph.checkpointer, run_id)


def checkpoint_thread_ids(checkpointer=None) -> set[str]:
    """Every thread id the checkpointer currently holds."""
    graph = build_graph(checkpointer)
    saver = graph.checkpointer
    if saver is None:
        return set()
    ids: set[str] = set()
    for item in saver.list(None):
        thread_id = (getattr(item, "config", None) or {}).get("configurable", {}).get("thread_id")
        if thread_id:
            ids.add(thread_id)
    return ids


def prune_stale_checkpoints(runs_dir: Path = Path("runs"), checkpointer=None) -> list[str]:
    """Drop checkpoints that can no longer serve a purpose; report what went.

    Two kinds are stale:
      - the run directory is gone (deleted outside the app);
      - the graph already reached END, so there is nothing left to resume.

    The second case is what clears the backlog: runs that finished before
    checkpoints were retired on completion keep their directory, so an
    orphan-only sweep would never touch them. A paused or crashed run has a
    non-empty `next` and is always kept — that is the case recovery needs.
    """
    graph = build_graph(checkpointer)
    saver = graph.checkpointer
    if saver is None:
        return []
    dropped = []
    for run_id in checkpoint_thread_ids(checkpointer):
        finished = not graph.get_state({"configurable": {"thread_id": run_id}}).next
        if (not (runs_dir / run_id).is_dir() or finished) and _discard_thread(saver, run_id):
            dropped.append(run_id)
    return sorted(dropped)


def _has_pending_interrupt(snapshot) -> bool:
    if getattr(snapshot, "interrupts", ()):
        return True
    return any(getattr(t, "interrupts", ()) for t in getattr(snapshot, "tasks", ()) or ())


def run_pipeline_graph(
    prompt: str,
    runs_dir: Path = Path("runs"),
    on_stage: Callable[[str], None] | None = None,
    on_image_done: Callable[[str], None] | None = None,
    reference_image: bytes | None = None,
    reference_image_name: str | None = None,
    model_a: str = DEFAULT_MODEL_A,
    model_b: str = DEFAULT_MODEL_B,
    checkpointer=None,
) -> PipelineResult:
    """Drop-in analogue of pipeline.run_pipeline().

    The run's thread_id is the run_dir timestamp — the same key used for the
    runs/ directory, so checkpointer thread, artifacts, and history index all
    share one identifier. If a gate interrupts, this returns a result with
    pipeline_status pending_hil_gate1/2; call resume_pipeline_graph() with the
    human's arbitration to continue.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = runs_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt.txt").write_text(prompt)

    initial: PipelineState = {
        "prompt": prompt,
        "model_a_key": model_a,
        "model_b_key": model_b,
        "model_a_label": _model_label(model_a),
        "model_b_label": _model_label(model_b),
        "run_dir": str(run_dir),
        "reference_image_path": None,
        "reference_image_name": reference_image_name,
        "timestamp": datetime.now().isoformat(),
        "critiques": [],
        "revisions": [],
        "gate_decisions": {},
        "hil_reviews": [],
        "errors": [],
        "round_num": 0,
        "rounds_completed": 0,
        "pipeline_status": "partial",
        "requires_attention": False,
    }

    if reference_image:
        try:
            initial["reference_image_path"] = str(_persist_reference_image(run_dir, reference_image))
        except Exception as e:
            # Legacy parity: a bad upload fails the run with a recorded summary,
            # not an unhandled exception. The graph is never entered.
            failed: PipelineState = {
                **initial,
                "errors": [f"Reference image failed: {_format_exception(e)}"],
                "pipeline_status": "failed",
                "requires_attention": True,
            }
            _persist_summary(failed)
            return state_to_result(failed)

    graph = build_graph(checkpointer)
    config = _callback_config(ts, on_stage, on_image_done)
    return _run_to_result(graph, initial, config)


def resume_pipeline_graph(
    run_id: str,
    resume_payload: dict | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_image_done: Callable[[str], None] | None = None,
    checkpointer=None,
) -> PipelineResult:
    """Resume a checkpointed run.

    Two modes, decided by the checkpoint's condition:
    - Suspended at a HIL gate: `resume_payload` (the human arbitration dict) is
      required and is delivered to the waiting interrupt().
    - Stopped mid-run without an interrupt (process crash): continues from the
      last checkpoint; no payload needed.

    `run_id` is the runs/ directory name (== thread_id). For runs with no
    checkpoint at all (legacy runs, deleted checkpoints.db), use
    retry_pipeline_graph(), which rebuilds state from disk artifacts.
    """
    graph = build_graph(checkpointer)
    config = _callback_config(run_id, on_stage, on_image_done)
    snapshot = graph.get_state(config)

    if not snapshot.values and not snapshot.next:
        raise CheckpointMissing(
            f"No checkpoint found for run {run_id!r}; "
            "use retry_pipeline_graph() to rebuild from disk artifacts."
        )
    if _has_pending_interrupt(snapshot):
        if resume_payload is None:
            raise ValueError(f"Run {run_id!r} is suspended at a HIL gate; an arbitration payload is required.")
        return _run_to_result(graph, Command(resume=resume_payload), config)
    if snapshot.next:
        return _run_to_result(graph, None, config)  # crash recovery
    return state_to_result(snapshot.values)  # already finished; nothing to do


def retry_pipeline_graph(
    result: PipelineResult,
    on_stage: Callable[[str], None] | None = None,
    on_image_done: Callable[[str], None] | None = None,
    checkpointer=None,
) -> PipelineResult:
    """Drop-in analogue of pipeline.resume_pipeline_from_result().

    Rebuilds graph state from the run's disk artifacts (via the already-loaded
    PipelineResult) and re-enters the graph at the first missing stage. Unlike
    resume_pipeline_graph() this needs no checkpoint, so it also covers runs
    started by the legacy orchestrator and runs whose checkpoints were lost.
    Any stale checkpoint thread is dropped first: disk artifacts are the source
    of truth for a retry, and stale errors/status must not leak in.
    """
    if not result.run_dir:
        raise ValueError("Cannot retry a run without a run directory")
    run_id = Path(result.run_dir).name
    graph = build_graph(checkpointer)
    saver = graph.checkpointer
    if saver is not None:
        try:
            saver.delete_thread(run_id)
        except (AttributeError, NotImplementedError):
            pass
    config = _callback_config(run_id, on_stage, on_image_done)
    seed, as_node = _seed_for_retry(result)
    graph.update_state(config, seed, as_node=as_node)
    return _run_to_result(graph, None, config)


def _seed_for_retry(result: PipelineResult) -> tuple[PipelineState, str]:
    """Mirror resume_pipeline_from_result(): re-enter at the first missing
    artifact. Returns (state seed, as_node the seed is written as)."""
    run_dir = Path(result.run_dir)
    reference = result.reference_image_path or _find_reference_image(run_dir)
    seed: PipelineState = {
        "prompt": result.prompt,
        "model_a_key": result.model_a_key,
        "model_b_key": result.model_b_key,
        "model_a_label": result.model_a_label,
        "model_b_label": result.model_b_label,
        "run_dir": str(run_dir),
        "reference_image_path": str(reference) if reference else None,
        "reference_image_name": result.reference_image_name,
        "timestamp": result.timestamp or datetime.now().isoformat(),
        "gate_decisions": {},
        "round_num": 0,
        "rounds_completed": 0,
        "pipeline_status": "partial",
        "requires_attention": False,
        "stage_failed": None,
    }

    gpt_path = _existing_image_path(result.image_paths.get("gpt_image_2"), run_dir / "gpt_image_2.png")
    gemini_path = _existing_image_path(result.image_paths.get("gemini_3_pro"), run_dir / "gemini_3_pro.png")
    if not gpt_path or not gemini_path:
        return seed, START  # regenerate; everything downstream is stale

    seed["image_paths"] = {"gpt_image_2": str(gpt_path), "gemini_3_pro": str(gemini_path)}
    if not result.eval_a or not result.eval_b:
        return seed, "generate"

    seed["eval_a"] = result.eval_a.model_dump()
    seed["eval_b"] = result.eval_b.model_dump()
    seed["prompt_difficulty"] = result.prompt_difficulty
    seed["gate_decisions"] = {g.gate: g.model_dump() for g in result.gate_decisions}
    if result.issue_equivalence:
        seed["issue_equivalence"] = result.issue_equivalence.model_dump()
    # HIL artifacts survive a retry: the human already did this work once.
    if result.hil_reviews:
        seed["hil_reviews"] = [_dump(review) for review in result.hil_reviews]
    if result.hil_adjudication is not None:
        seed["hil_adjudication"] = _dump(result.hil_adjudication)
    if not _gate_decision(seed, "gate1"):
        return seed, "evaluate"

    if not result.critiques:
        # Round 1 failed before any critique: a fallback revision may exist on
        # disk. Drop it (not seeded) before re-running, like legacy resume.
        return seed, "gate1"

    seed["critiques"] = [c.model_dump() for c in result.critiques]
    seed["revisions"] = [r.model_dump() for r in result.revisions]
    seed["rounds_completed"] = len(result.revisions)
    if len(result.revisions) < len(result.critiques):
        seed["round_num"] = len(result.critiques)
        return seed, "critique"  # router: revise (r1) or gate2 -> revise (r2+)

    seed["round_num"] = len(result.revisions)
    return seed, "revise"  # router: next critique round, or compare


def pending_gate(run_id: str, checkpointer=None) -> str | None:
    """Return the gate name ('gate1'/'gate2') the run is suspended at, or None.

    Replaces the pipeline_status == 'pending_hil_*' bookkeeping for live runs.
    Only an actual interrupt at a *_wait node counts: a run that merely stopped
    mid-flight (crash) has a non-empty snapshot.next too, and must report None
    so the UI offers crash recovery rather than an arbitration form.
    """
    graph = build_graph(checkpointer)
    snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
    if not snapshot.next or not _has_pending_interrupt(snapshot):
        return None
    node = snapshot.next[0]
    return node.removesuffix("_wait") if node.endswith("_wait") else None
