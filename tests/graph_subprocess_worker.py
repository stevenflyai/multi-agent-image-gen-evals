"""Child-process worker for the cross-process checkpoint tests.

Launched as a script by tests/test_graph_pipeline.py. Deliberately named so
pytest does not collect it (it is not a test_*.py module).

It reuses the stage factories from the test module rather than redefining them,
so the child and parent agree on what a critique/revision looks like. Emits a
single JSON object on stdout; the parent asserts against it.

Usage:
    python tests/graph_subprocess_worker.py inspect <db> <run_id>
    python tests/graph_subprocess_worker.py resume  <db> <run_id> <payload_json>
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

import graph_pipeline as gp  # noqa: E402
from tests.test_graph_pipeline import (  # noqa: E402
    _gate2_clear,
    _make_critique,
    _make_revision,
)


def _saver(db: str) -> SqliteSaver:
    conn = sqlite3.connect(db, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return SqliteSaver(conn)


def _inspect(db: str, run_id: str) -> dict:
    """What a cold process sees for a run suspended at a HIL gate."""
    saver = _saver(db)
    graph = gp.build_graph(checkpointer=saver)
    snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
    return {
        "next": list(snapshot.next),
        "has_pending_interrupt": gp._has_pending_interrupt(snapshot),
        "pending_gate": gp.pending_gate(run_id, checkpointer=saver),
        "pipeline_status": snapshot.values.get("pipeline_status"),
        # The interrupt payload must survive SQLite serialization, not just the
        # fact that an interrupt exists.
        "interrupt_values": [
            i.value for i in (getattr(snapshot, "interrupts", ()) or ())
        ],
        "prompt": snapshot.values.get("prompt"),
        "critiques": len(snapshot.values.get("critiques", [])),
    }


def _resume(db: str, run_id: str, payload: dict) -> dict:
    """Deliver the human's arbitration from a process that never saw the run."""
    saver = _saver(db)
    stages: list[str] = []

    gp.critique_evaluation = lambda *a, **k: _make_critique(1)
    gp.critique_evaluation_gemini = lambda *a, **k: _make_critique(2)
    gp.revise_evaluation = lambda *a, **k: _make_revision([8] * 6, [6] * 6)
    gp.evaluate_gate2 = lambda *a, **k: _gate2_clear()
    # Deliberately NOT stubbed: generate_images / evaluate_images. If resume
    # re-ran them the real (network) functions would be called and the test
    # would fail loudly rather than silently redoing paid work.

    result = gp.resume_pipeline_graph(
        run_id,
        resume_payload=payload,
        on_stage=stages.append,
        checkpointer=saver,
    )
    return {
        "pipeline_status": result.pipeline_status,
        "stages": stages,
        "has_comparison": result.comparison is not None,
        "hil_reviews": len(result.hil_reviews),
        "critiques": len(result.critiques),
        "revisions": len(result.revisions),
        "errors": result.errors,
    }


def main() -> int:
    mode, db, run_id = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "inspect":
        out = _inspect(db, run_id)
    elif mode == "resume":
        out = _resume(db, run_id, json.loads(sys.argv[4]))
    else:  # pragma: no cover - guard
        raise SystemExit(f"unknown mode {mode!r}")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
