"""Orchestrator-level exceptions, deliberately dependency-free.

CheckpointMissing lives here rather than in graph_pipeline.py so that app.py can
catch it without importing LangGraph. The graph orchestrator is opt-in
(config.USE_GRAPH_ORCHESTRATOR, default off) and must not impose its dependency
on installs that never turn it on — otherwise "flag off" stops being a safe
fallback and a plain `streamlit run app.py` on an environment without langgraph
fails at import.
"""


class CheckpointMissing(ValueError):
    """No resumable checkpoint for this run.

    Usually not a failure: the run already finished (its checkpoint is retired
    on completion) or was started before checkpointing. Callers holding a stale
    view — a second browser tab still showing a gate form for a run someone else
    already resumed — should refresh from disk rather than report an error.
    Subclasses ValueError so existing handlers keep working.
    """
