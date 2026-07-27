"""Tests for orchestrator.py — which orchestrator a call is routed to.

The dispatch is small but carries three distinctions that are easy to get wrong
and expensive to get wrong silently:
  - the flag must be read at call time, not import time;
  - "rerun from failed step" and "a human submitted a HIL decision" are
    different graph entry points despite sharing one UI code path;
  - the graph's thread id is the bare run-dir name, not app._run_key()'s path.

The orchestrators themselves are stubbed; this file is about routing only.
"""

from pathlib import Path

import pytest

import config
import orchestrator
from pipeline import PipelineResult


@pytest.fixture
def calls(monkeypatch):
    """Replace both orchestrators with recorders."""
    seen: dict[str, dict] = {}

    def _record(name):
        def _fn(*args, **kwargs):
            seen[name] = {"args": args, "kwargs": kwargs}
            return f"result-from-{name}"
        return _fn

    # Patch target follows where the name is resolved. The legacy entry points
    # are module-level in orchestrator; the graph ones are imported lazily inside
    # each function (so the flag being off means langgraph is never needed), so
    # they must be patched on graph_pipeline itself.
    for name in ("run_pipeline", "resume_pipeline_from_result"):
        monkeypatch.setattr(orchestrator, name, _record(name))
    for name in ("run_pipeline_graph", "resume_pipeline_graph", "retry_pipeline_graph"):
        monkeypatch.setattr(gp, name, _record(name))
    return seen


@pytest.fixture
def legacy(monkeypatch):
    monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", False)


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", True)


def _result(run_dir: str | None = "runs/20260726_120000_000001") -> PipelineResult:
    result = PipelineResult()
    result.run_dir = Path(run_dir) if run_dir else None
    result.prompt = "test prompt"
    return result


class TestThreadId:
    def test_is_the_bare_directory_name_not_the_path(self):
        """Regression guard: app._run_key() returns "runs/<ts>", but the graph
        keys checkpoint threads on "<ts>". Passing the path finds no checkpoint
        and the run silently looks unstarted."""
        assert orchestrator.thread_id(_result("runs/20260726_120000_000001")) == "20260726_120000_000001"

    def test_empty_when_no_run_dir(self):
        assert orchestrator.thread_id(_result(None)) == ""


class TestFlagIsReadAtCallTime:
    def test_flag_flip_takes_effect_without_reimport(self, calls, monkeypatch):
        monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", False)
        assert orchestrator.use_graph() is False
        monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", True)
        assert orchestrator.use_graph() is True


class TestStartRun:
    def test_legacy_by_default(self, calls, legacy):
        orchestrator.start_run("p", model_a="gpt-image-2", model_b="gemini-3-pro")
        assert "run_pipeline" in calls
        assert "run_pipeline_graph" not in calls

    def test_graph_when_flagged(self, calls, graph):
        orchestrator.start_run("p", model_a="gpt-image-2", model_b="gemini-3-pro")
        assert "run_pipeline_graph" in calls
        assert "run_pipeline" not in calls

    @pytest.mark.parametrize("mode", ["legacy", "graph"])
    def test_arguments_are_forwarded_identically(self, calls, monkeypatch, mode):
        """The two entry points must stay drop-in compatible; a dropped kwarg
        here means e.g. the reference image silently vanishes in one mode."""
        monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", mode == "graph")
        orchestrator.start_run(
            "a prompt",
            on_stage="STAGE",
            on_image_done="IMG",
            reference_image=b"bytes",
            reference_image_name="ref.png",
            model_a="mai-image-2.5",
            model_b="gpt-image-2",
        )
        recorded = calls["run_pipeline_graph" if mode == "graph" else "run_pipeline"]
        assert recorded["args"] == ("a prompt",)
        assert recorded["kwargs"] == {
            "on_stage": "STAGE",
            "on_image_done": "IMG",
            "reference_image": b"bytes",
            "reference_image_name": "ref.png",
            "model_a": "mai-image-2.5",
            "model_b": "gpt-image-2",
        }


class TestContinueRun:
    def test_legacy_retry(self, calls, legacy):
        orchestrator.continue_run(_result())
        assert "resume_pipeline_from_result" in calls

    def test_legacy_ignores_a_hil_payload(self, calls, legacy):
        """Legacy never read hil_reviews, so passing a payload must not change
        the call it makes — this is what keeps flipping the flag equivalent."""
        orchestrator.continue_run(_result(), resume_payload={"gate": "gate1"})
        assert "resume_pipeline_from_result" in calls
        assert "resume_pipeline_graph" not in calls
        assert "resume_payload" not in calls["resume_pipeline_from_result"]["kwargs"]

    def test_graph_retry_without_payload_rebuilds_from_disk(self, calls, graph):
        """No payload means "rerun from failed step": disk rebuild, which also
        works for runs started before the flag was on (no checkpoint)."""
        result = _result()
        orchestrator.continue_run(result)
        assert "retry_pipeline_graph" in calls
        assert "resume_pipeline_graph" not in calls
        assert calls["retry_pipeline_graph"]["args"] == (result,)

    def test_graph_with_payload_resumes_the_interrupt(self, calls, graph):
        payload = {"gate": "gate2_disagreement_detector", "labels": []}
        orchestrator.continue_run(_result("runs/20260726_120000_000001"), resume_payload=payload)
        assert "resume_pipeline_graph" in calls
        assert "retry_pipeline_graph" not in calls
        # Positional: (thread_id, payload) — and the thread id must be bare.
        assert calls["resume_pipeline_graph"]["args"] == ("20260726_120000_000001", payload)

    def test_callbacks_reach_every_branch(self, calls, monkeypatch):
        """Progress rendering breaks silently if a branch drops the callbacks."""
        cases = [
            (False, None, "resume_pipeline_from_result"),
            (True, None, "retry_pipeline_graph"),
            (True, {"gate": "gate1"}, "resume_pipeline_graph"),
        ]
        for flag, payload, expected in cases:
            calls.clear()
            monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", flag)
            orchestrator.continue_run(
                _result(), on_stage="STAGE", on_image_done="IMG", resume_payload=payload
            )
            kwargs = calls[expected]["kwargs"]
            assert kwargs["on_stage"] == "STAGE", expected
            assert kwargs["on_image_done"] == "IMG", expected


# ---------------------------------------------------------------------------
# Integration: the sequence app.py actually performs, with the flag on.
#
# The dispatch tests above stub the orchestrators, so they prove routing but not
# that the route works. This drives the real orchestrator module against the
# real graph (stage functions stubbed, no network) through the full app path:
#   start_run -> gate2 pending -> UI detects pause -> continue_run(payload) -> done
# ---------------------------------------------------------------------------

import json

from langgraph.checkpoint.memory import MemorySaver

import graph_pipeline as gp
from schemas import GateDecision, IssueEquivalenceReport, IssueEquivalenceSummary
from tests.test_graph_pipeline import (  # noqa: F401
    _adjudication_payload,
    _arbitration_payload,
    _make_revision,
    stage_calls,  # fixture, imported for use in this module
)


def _is_awaiting_hil(result: PipelineResult) -> bool:
    """Mirror of app._is_awaiting_hil (app.py can't be imported without running
    the Streamlit script body). Kept in sync by the assertions below."""
    return any(gate.status == "pending" for gate in result.gate_decisions)


class TestAppSequenceWithGraph:
    @pytest.fixture(autouse=True)
    def _graph_mode(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", True)
        # One checkpointer for the whole sequence, as the app's singleton would be.
        saver = MemorySaver()
        monkeypatch.setattr(gp, "default_checkpointer", lambda: saver)
        # start_run takes no runs_dir (neither does app.py — it relies on the
        # Path("runs") default), so isolate via cwd or the tests write into the
        # repo's real runs/ directory.
        monkeypatch.chdir(tmp_path)

    def test_gate2_pause_and_resume_through_the_app_path(self, tmp_path, stage_calls, monkeypatch):
        def pending_gate2(*args, **kwargs):
            stage_calls["gate2"] += 1
            return (
                GateDecision(
                    gate="gate2_disagreement_detector",
                    status="pending",
                    trigger_reasons=["critic_disagreement"],
                    review_dimensions=["prompt_adherence"],
                    pipeline_status="pending_hil_gate2",
                    requires_attention=True,
                ),
                IssueEquivalenceReport(equivalence_results=[], summary=IssueEquivalenceSummary()),
            )
        monkeypatch.setattr(gp, "evaluate_gate2", pending_gate2)

        # 1. The app starts a run. The flag must route it to the graph.
        paused = orchestrator.start_run(
            "test prompt", model_a="gpt-image-2", model_b="gemini-3-pro"
        )
        assert paused.pipeline_status == "pending_hil_gate2"

        # 2. The UI must classify this as "awaiting review", not as a failure.
        #    Before the fix these were indistinguishable: both lack a comparison.
        assert paused.comparison is None
        assert _is_awaiting_hil(paused) is True

        # 3. The human submits; the app hands the payload to continue_run. The
        #    form itself writes nothing in graph mode.
        adjudication = _adjudication_payload()
        resumed = orchestrator.continue_run(paused, resume_payload=adjudication)

        assert resumed.pipeline_status == "completed"
        assert resumed.comparison is not None
        assert _is_awaiting_hil(resumed) is False
        assert resumed.hil_adjudication.model_dump() == adjudication
        # Written by gate2_wait_node alone, and it matches the submitted payload.
        assert json.loads((resumed.run_dir / "hil_adjudication.json").read_text()) == adjudication
        # The gate on disk was flipped to completed by the graph, not by the form.
        gate2 = json.loads((resumed.run_dir / "gate2_decision.json").read_text())
        assert gate2["status"] == "completed"
        # Upstream work was not redone across the pause.
        assert stage_calls["generate"] == 1
        assert stage_calls["eval"] == 1

    def test_retry_button_path_does_not_need_a_payload(self, tmp_path, stage_calls, monkeypatch):
        """The other continue_run caller: "rerun from failed step"."""
        def failing_generate(prompt, run_dir, on_model_done=None, **kwargs):
            stage_calls["generate"] += 1
            if stage_calls["generate"] == 1:
                return {"gpt_image_2": RuntimeError("boom"), "gemini_3_pro": RuntimeError("boom")}
            gpt, gem = run_dir / "gpt_image_2.png", run_dir / "gemini_3_pro.png"
            gpt.write_bytes(b"a")
            gem.write_bytes(b"b")
            return {"gpt_image_2": gpt, "gemini_3_pro": gem}
        monkeypatch.setattr(gp, "generate_images", failing_generate)

        failed = orchestrator.start_run("test prompt", model_a="gpt-image-2", model_b="gemini-3-pro")
        assert failed.pipeline_status == "failed"
        assert _is_awaiting_hil(failed) is False  # a failure is not a pause

        retried = orchestrator.continue_run(failed)
        assert retried.pipeline_status == "completed"
        assert retried.errors == []

    def test_gate1_arbitration_reaches_the_final_comparison(self, tmp_path, stage_calls, monkeypatch):
        """End-to-end: a human's gate 1 call must survive the pause, the resume,
        and land in comparison.json — otherwise HIL is still write-only."""
        def pending_gate1(*args, **kwargs):
            stage_calls["gate1"] += 1
            return GateDecision(
                gate="gate1_uncertainty_risk_router",
                status="pending",
                route_score=0.8,
                route_band="required_hil",
                trigger_reasons=["narrow_margin"],
                review_dimensions=["prompt_adherence"],
                pipeline_status="pending_hil_gate1",
                requires_attention=True,
            )
        monkeypatch.setattr(gp, "evaluate_gate1", pending_gate1)

        paused = orchestrator.start_run("test prompt", model_a="gpt-image-2", model_b="gemini-3-pro")
        assert paused.pipeline_status == "pending_hil_gate1"

        # The stubbed scores give A every dimension (8s vs 6s). The human
        # overrules exactly one, against the scores.
        arbitration = _arbitration_payload()  # dimension_arbitrations: prompt_adherence -> A
        arbitration["dimension_arbitrations"] = [
            {"dimension": "prompt_adherence", "human_winner": "B"}
        ]
        resumed = orchestrator.continue_run(paused, resume_payload=arbitration)

        assert resumed.pipeline_status == "completed"
        comp = resumed.comparison
        assert comp.human_influenced is True

        dim = next(r for r in comp.dimension_results if r.dimension == "prompt_adherence")
        assert dim.winner == "model_b", "the human's call did not reach the comparison"
        assert dim.human_decided is True
        assert dim.model_winner == "model_a"
        assert dim.score_a == 8 and dim.score_b == 6, "scores must be untouched"

        # Counts follow the human; the clear mean gap still decides overall.
        assert comp.model_a_dimensions_won == 5
        assert comp.model_b_dimensions_won == 1
        assert comp.overall_winner == "model_a"

        # And it is persisted, not just in memory.
        on_disk = json.loads((resumed.run_dir / "comparison.json").read_text())
        assert on_disk["human_influenced"] is True
        assert next(
            d for d in on_disk["dimension_results"] if d["dimension"] == "prompt_adherence"
        )["winner"] == "model_b"

    def test_gate2_adjudication_reaches_the_final_comparison(self, tmp_path, stage_calls, monkeypatch):
        """The gate 2 counterpart: a human saying "round 1's critic was right"
        must move the score the run is judged on, not just get filed."""
        def diverging_revise(*args, **kwargs):
            stage_calls["revise"] += 1
            scores_a = [9, 8, 8, 8, 8, 8] if stage_calls["revise"] == 1 else [3, 8, 8, 8, 8, 8]
            return _make_revision(scores_a, [8] * 6)

        def pending_gate2(*args, **kwargs):
            stage_calls["gate2"] += 1
            return (
                GateDecision(
                    gate="gate2_disagreement_detector",
                    status="pending",
                    trigger_reasons=["critic_disagreement"],
                    review_dimensions=["prompt_adherence"],
                    pipeline_status="pending_hil_gate2",
                    requires_attention=True,
                ),
                IssueEquivalenceReport(equivalence_results=[], summary=IssueEquivalenceSummary()),
            )
        monkeypatch.setattr(gp, "revise_evaluation", diverging_revise)
        monkeypatch.setattr(gp, "evaluate_gate2", pending_gate2)

        paused = orchestrator.start_run("test prompt", model_a="gpt-image-2", model_b="gemini-3-pro")
        assert paused.pipeline_status == "pending_hil_gate2"

        # Round 2 dropped A's prompt_adherence 9 -> 3, losing it the run. The
        # human sides with round 1's critic on that dimension only.
        resumed = orchestrator.continue_run(paused, resume_payload=_adjudication_payload())

        assert resumed.pipeline_status == "completed"
        dim = next(
            r for r in resumed.comparison.dimension_results if r.dimension == "prompt_adherence"
        )
        assert dim.score_a == 9, "round 1's score did not reach the comparison"
        assert dim.winner == "model_a"
        assert resumed.comparison.overall_winner == "model_a"

        # Untouched dimensions still come from round 2.
        others = [r for r in resumed.comparison.dimension_results if r.dimension != "prompt_adherence"]
        assert all(r.score_a == 8 and r.winner == "draw" for r in others)

        # Persisted, and the per-round artifacts still hold what the model said.
        on_disk = json.loads((resumed.run_dir / "comparison.json").read_text())
        assert on_disk["overall_winner"] == "model_a"
        r2 = json.loads((resumed.run_dir / "revised_r2.json").read_text())
        assert r2["model_a"]["prompt_adherence"]["score"] == 3


class TestCheckpointCleanupDispatch:
    """Deleting a run must release orchestrator-side state — in graph mode only."""

    def test_forget_run_is_a_noop_in_legacy_mode(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", False)
        called = []
        monkeypatch.setattr(gp, "discard_checkpoint", lambda *a, **k: called.append(a))
        orchestrator.forget_run(_result())
        assert called == [], "legacy keeps no state outside the run directory"

    def test_forget_run_discards_the_thread_in_graph_mode(self, monkeypatch):
        monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", True)
        called = []
        monkeypatch.setattr(gp, "discard_checkpoint", lambda run_id: called.append(run_id))
        orchestrator.forget_run(_result("runs/20260726_120000_000001"))
        # Bare thread id, not the "runs/..." path.
        assert called == ["20260726_120000_000001"]

    def test_forget_run_without_a_run_dir_does_nothing(self, monkeypatch):
        monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", True)
        called = []
        monkeypatch.setattr(gp, "discard_checkpoint", lambda run_id: called.append(run_id))
        orchestrator.forget_run(_result(None))
        assert called == []

    def test_prune_is_a_noop_in_legacy_mode(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", False)
        monkeypatch.setattr(gp, "prune_stale_checkpoints", lambda *a, **k: ["should-not-run"])
        assert orchestrator.prune_checkpoints(tmp_path) == []

    def test_prune_delegates_in_graph_mode(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", True)
        monkeypatch.setattr(gp, "prune_stale_checkpoints", lambda d, *a, **k: [f"pruned:{d}"])
        assert orchestrator.prune_checkpoints(tmp_path) == [f"pruned:{tmp_path}"]


class TestDeletingARunReleasesItsCheckpoint:
    """End-to-end: the leak app.py._delete_run_record used to have."""

    def test_paused_run_deleted_leaves_no_checkpoint(self, tmp_path, stage_calls, monkeypatch):
        monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", True)
        saver = MemorySaver()
        monkeypatch.setattr(gp, "default_checkpointer", lambda: saver)
        monkeypatch.chdir(tmp_path)

        def pending_gate1(*args, **kwargs):
            stage_calls["gate1"] += 1
            return GateDecision(
                gate="gate1_uncertainty_risk_router", status="pending",
                route_score=0.8, route_band="required_hil", trigger_reasons=["narrow_margin"],
                review_dimensions=["prompt_adherence"],
                pipeline_status="pending_hil_gate1", requires_attention=True,
            )
        monkeypatch.setattr(gp, "evaluate_gate1", pending_gate1)

        paused = orchestrator.start_run("test prompt", model_a="gpt-image-2", model_b="gemini-3-pro")
        run_id = paused.run_dir.name
        assert gp.checkpoint_thread_ids() == {run_id}

        # What app.py._delete_run_record now does before removing the directory.
        orchestrator.forget_run(paused)
        assert gp.checkpoint_thread_ids() == set()


class TestStaleResume:
    """The second-browser-tab case: a gate form for a run someone already resumed."""

    @pytest.fixture(autouse=True)
    def _graph_mode(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "USE_GRAPH_ORCHESTRATOR", True)
        saver = MemorySaver()
        monkeypatch.setattr(gp, "default_checkpointer", lambda: saver)
        monkeypatch.chdir(tmp_path)

    def test_resuming_a_finished_run_raises_the_typed_error(self, tmp_path, stage_calls):
        """app.py branches on the type, so it must be distinguishable from a
        genuine pipeline failure — not matched on message text."""
        done = orchestrator.start_run("test prompt", model_a="gpt-image-2", model_b="gemini-3-pro")
        assert done.pipeline_status == "completed"

        with pytest.raises(orchestrator.CheckpointMissing):
            orchestrator.continue_run(done, resume_payload=_arbitration_payload())

    def test_it_is_a_valueerror_so_older_handlers_still_catch_it(self, tmp_path, stage_calls):
        done = orchestrator.start_run("test prompt", model_a="gpt-image-2", model_b="gemini-3-pro")
        with pytest.raises(ValueError):
            orchestrator.continue_run(done, resume_payload=_arbitration_payload())

    def test_a_real_failure_is_not_reported_as_stale(self, tmp_path, stage_calls, monkeypatch):
        """Guard against over-catching: a genuine error must stay a genuine error."""
        def boom(*args, **kwargs):
            raise RuntimeError("evaluate exploded")
        monkeypatch.setattr(gp, "evaluate_images", boom)

        failed = orchestrator.start_run("test prompt", model_a="gpt-image-2", model_b="gemini-3-pro")
        assert failed.pipeline_status == "failed"
        assert any("Evaluation failed" in e for e in failed.errors)

        # Retrying still fails on the same stage — it must surface as a failed
        # run carrying the real error, never as the benign "already resolved"
        # path, or a broken pipeline would look like a stale browser tab.
        retried = orchestrator.continue_run(failed)
        assert retried.pipeline_status == "failed"
        assert any("Evaluation failed" in e for e in retried.errors)

    def test_the_run_is_still_readable_from_disk_after_the_stale_submit(self, tmp_path, stage_calls):
        """What app.py falls back to: refresh the stale view from artifacts."""
        done = orchestrator.start_run("test prompt", model_a="gpt-image-2", model_b="gemini-3-pro")
        with pytest.raises(orchestrator.CheckpointMissing):
            orchestrator.continue_run(done, resume_payload=_arbitration_payload())

        # Artifacts untouched by the failed resume, so the UI can recover.
        assert (done.run_dir / "comparison.json").exists()
        assert json.loads((done.run_dir / "summary.json").read_text())["pipeline_status"] == "completed"


# ---------------------------------------------------------------------------
# app.py display sources.
#
# app.py can't be imported (importing it runs the Streamlit script body), so
# these read its source. Ugly, but the alternative is no guard at all on a
# defect that is invisible in every automated check: the card shows the score
# the human chose while the prose argues for the score they rejected.
# ---------------------------------------------------------------------------

import ast
import subprocess
import sys
import textwrap


def _function_source(name: str) -> str:
    source = Path("app.py").read_text()
    tree = ast.parse(source)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return textwrap.dedent(ast.get_source_segment(source, node) or "")


class TestPostCompletionViewsUseTheComposedRevision:
    """Scores render from comparison.dimension_results, which determine_winner()
    built from the gate-2-composed revision. Any view pairing text with those
    scores must read the same revision, or an adjudicated dimension shows round
    1's number above round 2's argument against it."""

    @pytest.mark.parametrize("func", ["render_dimension_cards", "render_dimension_comments_table"])
    def test_they_do_not_reach_for_the_last_round_directly(self, func):
        body = _function_source(func)
        assert "revisions[-1]" not in body, (
            f"{func} reads revisions[-1]; use _reasoning_source() so the text "
            f"matches the score beside it"
        )
        assert "_reasoning_source(" in body

    def test_the_helper_delegates_to_the_shared_definition(self):
        assert "_final_revision(result)" in _function_source("_reasoning_source")

    @pytest.mark.parametrize("func", ["_score_delta_text", "_revision_note_text"])
    def test_the_pending_gate2_form_still_reads_the_round_it_asks_about(self, func):
        """Counterpart: these render only while gate 2 is pending, where
        revisions[-1] IS round 1 — the "06" column under review. Composing there
        would be wrong (and a no-op), so they must NOT be converted."""
        assert "revisions[-1]" in _function_source(func)


# ---------------------------------------------------------------------------
# The flag has to be a real off-switch, dependency included.
#
# orchestrator used to import graph_pipeline at module scope, which made app.py
# require langgraph even with USE_GRAPH_ORCHESTRATOR=false — a plain
# `streamlit run app.py` on an interpreter without it died at import with
# ModuleNotFoundError. Run in a subprocess with the module blocked, because an
# in-process test would see langgraph already imported by the rest of the suite.
# ---------------------------------------------------------------------------

_BLOCKED_IMPORT = '''
import sys, builtins
_real = builtins.__import__
def _blocker(name, *a, **k):
    if name.split(".")[0] in ("langgraph", "langchain_core"):
        raise ModuleNotFoundError("No module named %r" % name)
    return _real(name, *a, **k)
builtins.__import__ = _blocker
sys.path.insert(0, {repo!r})
import config
config.USE_GRAPH_ORCHESTRATOR = {flag}
{body}
'''


def _run_without_langgraph(body: str, flag: bool = False):
    repo = str(Path(__file__).resolve().parent.parent)
    return subprocess.run(
        [sys.executable, "-c", _BLOCKED_IMPORT.format(repo=repo, flag=flag, body=body)],
        capture_output=True, text=True, cwd=repo,
    )


class TestGraphDependencyIsOptionalWhenTheFlagIsOff:
    def test_orchestrator_imports_without_langgraph(self):
        r = _run_without_langgraph("import orchestrator; print('ok')")
        assert r.returncode == 0, f"orchestrator needs langgraph at import time:\n{r.stderr}"
        assert "ok" in r.stdout

    def test_checkpoint_missing_is_catchable_without_langgraph(self):
        """app.py catches CheckpointMissing unconditionally, so the class cannot
        live behind the optional dependency."""
        r = _run_without_langgraph(
            "import orchestrator; assert issubclass(orchestrator.CheckpointMissing, ValueError); print('ok')"
        )
        assert r.returncode == 0, r.stderr
        assert "ok" in r.stdout

    def test_legacy_mode_functions_work_without_langgraph(self):
        r = _run_without_langgraph(
            "import orchestrator\n"
            "assert orchestrator.use_graph() is False\n"
            "assert orchestrator.prune_checkpoints() == []\n"
            "print('ok')"
        )
        assert r.returncode == 0, r.stderr
        assert "ok" in r.stdout

    def test_turning_the_flag_on_without_the_dependency_fails_loudly(self):
        """Lazy importing must not turn a missing dependency into silent
        legacy-mode fallback — the user asked for the graph."""
        r = _run_without_langgraph("import orchestrator; orchestrator.prune_checkpoints()", flag=True)
        assert r.returncode != 0
        assert "ModuleNotFoundError" in r.stderr
