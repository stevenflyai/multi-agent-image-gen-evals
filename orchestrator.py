"""Orchestrator selection: legacy pipeline.py vs the LangGraph graph_pipeline.py.

app.py calls through here instead of importing either orchestrator directly, so
the choice lives in one place and can be tested without executing the Streamlit
script body. Both orchestrators write identical run artifacts, so everything
downstream of a run (history, dashboard, reload) is unaffected by the flag.

Selection is config.USE_GRAPH_ORCHESTRATOR, read at call time rather than import
time so tests can flip it with monkeypatch.

graph_pipeline is imported lazily, inside the functions that use it, so that an
install which never turns the flag on does not need LangGraph at all. Importing
it at module scope would make `streamlit run app.py` fail with
ModuleNotFoundError on any environment without langgraph — including a plain
system/conda Python — which would defeat the point of defaulting the flag off.
CheckpointMissing comes from errors.py for the same reason: app.py catches it
unconditionally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import config
from errors import CheckpointMissing  # noqa: F401  (re-exported for app.py)
from pipeline import PipelineResult, resume_pipeline_from_result, run_pipeline

OnStage = Callable[[str], None] | None
OnImageDone = Callable[[str], None] | None


def _graph():
    """Import graph_pipeline on demand, naming the fix if it is not installed.

    Reached only when the flag is on, so a missing dependency is a real
    misconfiguration and must fail loudly — never silently fall back to legacy,
    which would run a different orchestrator than the one that was asked for.
    """
    try:
        import graph_pipeline
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}. USE_GRAPH_ORCHESTRATOR is on, but this interpreter is missing "
            "the LangGraph dependencies. Run with `uv run streamlit run app.py`, "
            "or set USE_GRAPH_ORCHESTRATOR=false to use the legacy orchestrator."
        ) from e
    return graph_pipeline


def use_graph() -> bool:
    """Read the flag at call time so monkeypatching config takes effect."""
    return bool(config.USE_GRAPH_ORCHESTRATOR)


def thread_id(result: PipelineResult) -> str:
    """Checkpointer thread id for a run.

    Deliberately not app._run_key(): that returns the full path "runs/<ts>",
    while graph_pipeline keys checkpoint threads on the bare directory name.
    Both orchestrators name run dirs with the same timestamp format, so a run
    started by either one has a usable thread id.
    """
    return Path(result.run_dir).name if result.run_dir else ""


def start_run(
    prompt: str,
    *,
    on_stage: OnStage = None,
    on_image_done: OnImageDone = None,
    reference_image: bytes | None = None,
    reference_image_name: str | None = None,
    model_a: str,
    model_b: str,
) -> PipelineResult:
    """Start a fresh run on the configured orchestrator."""
    runner = _graph().run_pipeline_graph if use_graph() else run_pipeline
    return runner(
        prompt,
        on_stage=on_stage,
        on_image_done=on_image_done,
        reference_image=reference_image,
        reference_image_name=reference_image_name,
        model_a=model_a,
        model_b=model_b,
    )


def continue_run(
    result: PipelineResult,
    *,
    on_stage: OnStage = None,
    on_image_done: OnImageDone = None,
    resume_payload: dict | None = None,
) -> PipelineResult:
    """Continue an existing run. Two distinct semantics share this entry point:

    - `resume_payload` given: a human submitted a HIL decision. The graph must
      deliver it to the interrupt() the run is suspended at.
    - `resume_payload` omitted: "rerun from failed step" — rebuild from the run's
      disk artifacts and re-enter at the first missing stage.

    The legacy orchestrator has only the second behaviour and never read the HIL
    payload at all, so passing one changes nothing when the flag is off. That is
    what makes flipping the flag behaviour-equivalent today; wiring the payload
    into scoring is a separate, deliberate change.
    """
    if not use_graph():
        return resume_pipeline_from_result(result, on_stage=on_stage, on_image_done=on_image_done)
    graph = _graph()
    if resume_payload is not None:
        return graph.resume_pipeline_graph(
            thread_id(result), resume_payload, on_stage=on_stage, on_image_done=on_image_done
        )
    # Disk-based rebuild rather than checkpoint recovery: it matches legacy retry
    # semantics exactly and still reuses images already on disk, so it also works
    # for runs started before the flag was turned on.
    return graph.retry_pipeline_graph(result, on_stage=on_stage, on_image_done=on_image_done)


def forget_run(result: PipelineResult) -> None:
    """Release orchestrator-side state for a run that is being deleted.

    Deleting a run removes its directory; in graph mode its checkpoint thread
    would otherwise survive in runs/checkpoints.db forever (~200KB each, and
    SQLite does not hand the space back). No-op in legacy mode, which keeps no
    state outside the run directory.
    """
    if not use_graph():
        return
    run_id = thread_id(result)
    if run_id:
        _graph().discard_checkpoint(run_id)


def prune_checkpoints(runs_dir: Path = Path("runs")) -> list[str]:
    """Drop checkpoints that can no longer be resumed; return what was dropped.

    Covers runs deleted outside the app and runs that already finished — the
    latter clears the backlog left by runs that completed before checkpoints
    were retired on completion. No-op in legacy mode.
    """
    if not use_graph():
        return []
    return _graph().prune_stale_checkpoints(runs_dir)
