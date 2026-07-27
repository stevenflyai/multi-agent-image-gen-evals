"""Tests for the LangGraph orchestrator (graph_pipeline.py) with mocked stages.

All LLM/image stage functions are monkeypatched on the graph_pipeline namespace,
so these tests exercise only the topology: routing, the critique-revise loop,
convergence, graceful degradation edges, gate interrupts, and the legacy-parity
behaviors (pending status, crash recovery, callbacks, retry-from-artifacts).
No network calls.
"""

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from PIL import Image

import graph_pipeline as gp
from pipeline import PipelineResult, _final_revision
from schemas import (
    DIMENSIONS,
    CritiqueResponse,
    DimensionCritique,
    DimensionScore,
    GateDecision,
    HilAdjudication,
    HilAdjudicationLabel,
    HilArbitration,
    HilDimensionArbitration,
    ImageEvaluation,
    IssueEquivalenceReport,
    IssueEquivalenceSummary,
    RevisedDimensionScore,
    RevisedEvaluation,
    RevisedImageEvaluation,
)


# --- Factories (same shapes as tests/test_pipeline.py) ---

def _make_eval(model_name: str, scores: list[int]) -> ImageEvaluation:
    dims = dict(zip(DIMENSIONS, [
        DimensionScore(score=s, reasoning=f"Score {s}") for s in scores
    ]))
    return ImageEvaluation(model_name=model_name, **dims)


def _make_revised(model_name: str, scores: list[int]) -> RevisedImageEvaluation:
    dims = dict(zip(DIMENSIONS, [
        RevisedDimensionScore(
            score=s, reasoning=f"Score {s}", critique_accepted=False, revision_note="Test"
        ) for s in scores
    ]))
    return RevisedImageEvaluation(model_name=model_name, **dims)


def _make_revision(scores_a: list[int], scores_b: list[int]) -> RevisedEvaluation:
    return RevisedEvaluation(
        model_a=_make_revised("A", scores_a),
        model_b=_make_revised("B", scores_b),
    )


def _make_critique(round_number: int) -> CritiqueResponse:
    return CritiqueResponse(
        overall_assessment=f"round {round_number}",
        dimension_critiques=[
            DimensionCritique(
                dimension="prompt_adherence",
                original_score_model_a=8,
                original_score_model_b=6,
                critique="Test critique",
                suggested_score_model_a=8,
                suggested_score_model_b=6,
            )
        ],
        bias_detection="No systematic bias detected",
        round=round_number,
        critic_model="test",
    )


def _arbitration_payload() -> dict:
    """What app.py delivers to gate1's interrupt(): HilArbitration.model_dump()."""
    now = datetime.now().isoformat()
    return HilArbitration(
        status="completed",
        route_score=0.8,
        route_band="required_hil",
        trigger_reasons=["narrow_margin"],
        review_dimensions=["prompt_adherence"],
        reviewer="tester",
        created_at=now,
        completed_at=now,
        dimension_arbitrations=[
            HilDimensionArbitration(dimension="prompt_adherence", human_winner="A")
        ],
    ).model_dump()


def _adjudication_payload() -> dict:
    """What app.py delivers to gate2's interrupt(): HilAdjudication.model_dump()."""
    now = datetime.now().isoformat()
    return HilAdjudication(
        status="completed",
        trigger_reasons=["critic_disagreement"],
        disagreement_items=[],
        reviewer="tester",
        created_at=now,
        completed_at=now,
        adjudication_labels=[
            HilAdjudicationLabel(dimension="prompt_adherence", label="agree_with_05")
        ],
    ).model_dump()


def _gate1_clear() -> GateDecision:
    return GateDecision(gate="gate1_uncertainty_risk_router", status="not_required")


def _gate2_clear() -> tuple[GateDecision, IssueEquivalenceReport]:
    return (
        GateDecision(gate="gate2_disagreement_detector", status="not_required"),
        IssueEquivalenceReport(equivalence_results=[], summary=IssueEquivalenceSummary()),
    )


def _initial_state(run_dir: Path, prompt: str = "test prompt") -> dict:
    """Mirror of run_pipeline_graph()'s initial state."""
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "prompt": prompt,
        "model_a_key": "gpt-image-2",
        "model_b_key": "gemini-3-pro",
        "model_a_label": "GPT Image 2",
        "model_b_label": "Gemini 3 Pro",
        "run_dir": str(run_dir),
        "reference_image_path": None,
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


# --- Default mocked stages (happy path); tests override as needed ---

@pytest.fixture
def stage_calls(monkeypatch):
    """Patch every stage function on the graph_pipeline namespace; count calls."""
    calls = {"generate": 0, "eval": 0, "gate1": 0, "crit1": 0, "crit2": 0, "gate2": 0, "revise": 0}

    def fake_generate_images(prompt, run_dir, on_model_done=None, reference_image_path=None,
                             model_a=None, model_b=None):
        calls["generate"] += 1
        gpt_path = run_dir / "gpt_image_2.png"
        gemini_path = run_dir / "gemini_3_pro.png"
        gpt_path.write_bytes(b"a")
        gemini_path.write_bytes(b"b")
        if on_model_done:
            on_model_done("gpt_image_2")
            on_model_done("gemini_3_pro")
        return {"gpt_image_2": gpt_path, "gemini_3_pro": gemini_path}

    def fake_evaluate_images(*args, **kwargs):
        calls["eval"] += 1
        return _make_eval("A", [8] * 6), _make_eval("B", [6] * 6), "medium"

    def fake_gate1(*args, **kwargs):
        calls["gate1"] += 1
        return _gate1_clear()

    def fake_critique(*args, **kwargs):
        calls["crit1"] += 1
        return _make_critique(1)

    def fake_critique_gemini(*args, **kwargs):
        calls["crit2"] += 1
        return _make_critique(2)

    def fake_gate2(*args, **kwargs):
        calls["gate2"] += 1
        return _gate2_clear()

    def fake_revise(*args, **kwargs):
        calls["revise"] += 1
        return _make_revision([8] * 6, [6] * 6)  # stable scores -> converges

    monkeypatch.setattr(gp, "generate_images", fake_generate_images)
    monkeypatch.setattr(gp, "evaluate_images", fake_evaluate_images)
    monkeypatch.setattr(gp, "evaluate_gate1", fake_gate1)
    monkeypatch.setattr(gp, "critique_evaluation", fake_critique)
    monkeypatch.setattr(gp, "critique_evaluation_gemini", fake_critique_gemini)
    monkeypatch.setattr(gp, "evaluate_gate2", fake_gate2)
    monkeypatch.setattr(gp, "revise_evaluation", fake_revise)
    return calls


def _run(state: dict, thread_id: str = "test-run"):
    """Compile with an in-memory checkpointer and run to completion/interrupt."""
    graph = gp.build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(state, config)
    return graph, config, graph.get_state(config).values


class TestHappyPath:
    def test_two_rounds_to_completion(self, tmp_path, stage_calls):
        _, _, final = _run(_initial_state(tmp_path / "run"))

        assert stage_calls == {"generate": 1, "eval": 1, "gate1": 1, "crit1": 1, "crit2": 1, "gate2": 1, "revise": 2}
        assert final["pipeline_status"] == "completed"
        assert len(final["critiques"]) == 2
        assert len(final["revisions"]) == 2
        assert final["rounds_completed"] == 2
        assert final["comparison"]["overall_winner"] == "model_a"
        assert final["errors"] == []

    def test_artifacts_written_same_as_legacy_pipeline(self, tmp_path, stage_calls):
        run_dir = tmp_path / "run"
        _run(_initial_state(run_dir))

        for name in (
            "evaluation.json", "evaluation_v2.json", "gate1_decision.json",
            "critique_r1.json", "revised_r1.json",
            "critique_r2.json", "gate2_decision.json", "issue_equivalence.json", "revised_r2.json",
            "comparison.json", "critique.json", "revised.json", "summary.json",
        ):
            assert (run_dir / name).exists(), f"missing artifact: {name}"

        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["pipeline_status"] == "completed"
        assert summary["winner"] == "model_a"
        assert summary["rounds_completed"] == 2

    def test_run_pipeline_graph_reports_stage_keys(self, tmp_path, stage_calls, monkeypatch):
        monkeypatch.setattr(gp, "default_checkpointer", lambda: MemorySaver())
        stages: list[str] = []

        result = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, on_stage=stages.append)

        assert stages == [
            "stage_generating", "stage_evaluating", "stage_gate1",
            "stage_critique", "stage_revising",
            "stage_critique_round2", "stage_gate2", "stage_revising_round2",
            "stage_complete",
        ]
        # state_to_result bridge rehydrates the UI view-model
        assert result.pipeline_status == "completed"
        assert result.comparison.overall_winner == "model_a"
        assert result.revised.model_a.prompt_adherence.score == 8
        assert (result.run_dir / "comparison.json").exists()


class TestLoopControl:
    def test_convergence_stops_before_max_rounds(self, tmp_path, stage_calls, monkeypatch):
        monkeypatch.setattr(gp, "MAX_CRITIQUE_ROUNDS", 3)
        # default fake_revise returns identical scores every round -> converged after r2
        _, _, final = _run(_initial_state(tmp_path / "run"))

        assert len(final["revisions"]) == 2
        assert stage_calls["crit2"] == 1  # round 3 never ran
        assert final["pipeline_status"] == "completed"

    def test_no_convergence_runs_all_rounds(self, tmp_path, stage_calls, monkeypatch):
        monkeypatch.setattr(gp, "MAX_CRITIQUE_ROUNDS", 3)
        revisions = iter([
            _make_revision([8] * 6, [6] * 6),
            _make_revision([4] * 6, [6] * 6),   # big delta vs r1
            _make_revision([8] * 6, [6] * 6),   # big delta vs r2
        ])
        monkeypatch.setattr(gp, "revise_evaluation", lambda *a, **k: next(revisions))

        _, _, final = _run(_initial_state(tmp_path / "run"))

        assert len(final["revisions"]) == 3
        assert stage_calls["crit2"] == 2  # rounds 2 and 3 both used the Gemini critic
        assert final["rounds_completed"] == 3
        assert final["pipeline_status"] == "completed"


class TestGracefulDegradation:
    def test_round1_critique_failure_falls_back_to_initial_scores(self, tmp_path, stage_calls, monkeypatch):
        def broken_critique(*args, **kwargs):
            raise ValueError("bad json")
        monkeypatch.setattr(gp, "critique_evaluation", broken_critique)

        run_dir = tmp_path / "run"
        _, _, final = _run(_initial_state(run_dir))

        assert stage_calls["revise"] == 0
        assert final["pipeline_status"] == "completed"
        assert any("Critique round 1 failed" in e for e in final["errors"])
        # fallback revision mirrors initial scores with critique_accepted=False
        assert len(final["revisions"]) == 1
        rev = RevisedEvaluation(**final["revisions"][0])
        assert rev.model_a.prompt_adherence.score == 8
        assert rev.model_a.prompt_adherence.critique_accepted is False
        assert final["comparison"]["overall_winner"] == "model_a"

    def test_round2_critique_failure_keeps_round1_scores(self, tmp_path, stage_calls, monkeypatch):
        def broken_gemini_critique(*args, **kwargs):
            raise ValueError("blocked by safety filters")
        monkeypatch.setattr(gp, "critique_evaluation_gemini", broken_gemini_critique)

        run_dir = tmp_path / "run"
        _, _, final = _run(_initial_state(run_dir))

        # round 1 completed fully; round 2 degraded to compare on r1 revision
        assert stage_calls == {"generate": 1, "eval": 1, "gate1": 1, "crit1": 1, "crit2": 0, "gate2": 0, "revise": 1}
        assert len(final["critiques"]) == 1
        assert len(final["revisions"]) == 1
        assert final["pipeline_status"] == "completed"
        assert any("Critique round 2 failed" in e for e in final["errors"])
        assert final["comparison"]["overall_winner"] == "model_a"
        assert not (run_dir / "critique_r2.json").exists()

    def test_round2_revision_failure_keeps_round1_scores(self, tmp_path, stage_calls, monkeypatch):
        revise_calls = {"n": 0}

        def revise_fails_second_time(*args, **kwargs):
            revise_calls["n"] += 1
            if revise_calls["n"] >= 2:
                raise ValueError("truncated output")
            return _make_revision([8] * 6, [6] * 6)
        monkeypatch.setattr(gp, "revise_evaluation", revise_fails_second_time)

        _, _, final = _run(_initial_state(tmp_path / "run"))

        assert len(final["critiques"]) == 2
        assert len(final["revisions"]) == 1
        assert final["pipeline_status"] == "completed"
        assert any("Revision round 2 failed" in e for e in final["errors"])
        assert final["comparison"] is not None


class TestFailures:
    def test_generation_failure_fails_run(self, tmp_path, stage_calls, monkeypatch):
        def fake_generate_images(prompt, run_dir, **kwargs):
            gemini_path = run_dir / "gemini_3_pro.png"
            gemini_path.write_bytes(b"b")
            return {
                "gpt_image_2": RuntimeError("AuthenticationTypeDisabled"),
                "gemini_3_pro": gemini_path,
            }
        monkeypatch.setattr(gp, "generate_images", fake_generate_images)

        run_dir = tmp_path / "run"
        _, _, final = _run(_initial_state(run_dir))

        assert stage_calls["eval"] == 0
        assert final["pipeline_status"] == "failed"
        assert final["requires_attention"] is True
        assert final.get("comparison") is None
        assert any("GPT Image 2 failed: AuthenticationTypeDisabled" in e for e in final["errors"])
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["pipeline_status"] == "failed"
        assert summary["has_comparison"] is False

    def test_evaluation_failure_fails_run(self, tmp_path, stage_calls, monkeypatch):
        def broken_eval(*args, **kwargs):
            raise ValueError("rate limited")
        monkeypatch.setattr(gp, "evaluate_images", broken_eval)

        run_dir = tmp_path / "run"
        _, _, final = _run(_initial_state(run_dir))

        assert stage_calls["gate1"] == 0
        assert final["pipeline_status"] == "failed"
        assert any("Evaluation failed" in e for e in final["errors"])


class TestGateInterrupts:
    def test_gate1_pending_suspends_and_resumes(self, tmp_path, stage_calls, monkeypatch):
        def pending_gate1(*args, **kwargs):
            stage_calls["gate1"] += 1
            return GateDecision(
                gate="gate1_uncertainty_risk_router",
                status="pending",
                route_score=0.8,
                route_band="required_hil",
                trigger_reasons=["narrow_margin"],
                pipeline_status="pending_hil_gate1",
                requires_attention=True,
            )
        monkeypatch.setattr(gp, "evaluate_gate1", pending_gate1)

        run_dir = tmp_path / "run"
        graph = gp.build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "hil-run"}}
        graph.invoke(_initial_state(run_dir), config)

        # Suspended in gate1_wait; no critique yet; pause visible in summary.json
        # AND in the checkpointed state (the compute node committed it).
        snapshot = graph.get_state(config)
        assert snapshot.next == ("gate1_wait",)
        assert snapshot.values["pipeline_status"] == "pending_hil_gate1"
        assert stage_calls["crit1"] == 0
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["pipeline_status"] == "pending_hil_gate1"
        assert summary["requires_attention"] is True

        # Human submits arbitration -> graph resumes inside the node and runs to completion
        arbitration = _arbitration_payload()
        graph.invoke(Command(resume=arbitration), config)

        final = graph.get_state(config).values
        assert graph.get_state(config).next == ()
        assert final["pipeline_status"] == "completed"
        assert final["comparison"] is not None
        assert final["hil_reviews"] == [arbitration]
        assert json.loads((run_dir / "hil_review_r1.json").read_text()) == arbitration
        gate1 = json.loads((run_dir / "gate1_decision.json").read_text())
        assert gate1["status"] == "completed"
        # Resume re-executed only the interrupt-only wait node, not the gate computation
        assert stage_calls["gate1"] == 1

    def test_gate2_pending_suspends_and_resumes(self, tmp_path, stage_calls, monkeypatch):
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

        run_dir = tmp_path / "run"
        graph = gp.build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "hil-run-2"}}
        graph.invoke(_initial_state(run_dir), config)

        # Round 1 done, suspended in gate2_wait before round-2 revision
        snapshot = graph.get_state(config)
        assert snapshot.next == ("gate2_wait",)
        assert snapshot.values["pipeline_status"] == "pending_hil_gate2"
        assert stage_calls["revise"] == 1
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["pipeline_status"] == "pending_hil_gate2"

        adjudication = _adjudication_payload()
        graph.invoke(Command(resume=adjudication), config)

        final = graph.get_state(config).values
        assert final["pipeline_status"] == "completed"
        assert stage_calls["revise"] == 2  # round-2 revision ran after resume
        assert len(final["revisions"]) == 2
        assert json.loads((run_dir / "hil_adjudication.json").read_text()) == adjudication
        assert stage_calls["gate2"] == 1  # gate computation did not re-run on resume


# --- Legacy-parity behaviors ---

def _pending_gate1_decision() -> GateDecision:
    return GateDecision(
        gate="gate1_uncertainty_risk_router",
        status="pending",
        route_score=0.8,
        route_band="required_hil",
        trigger_reasons=["narrow_margin"],
        pipeline_status="pending_hil_gate1",
        requires_attention=True,
    )


class TestPendingStatusParity:
    """Fix 1: the result returned across an interrupt reports the pause truthfully."""

    def test_run_pipeline_graph_returns_pending_result_and_resumes(self, tmp_path, stage_calls, monkeypatch):
        def pending_gate1(*args, **kwargs):
            stage_calls["gate1"] += 1
            return _pending_gate1_decision()
        monkeypatch.setattr(gp, "evaluate_gate1", pending_gate1)
        saver = MemorySaver()

        result = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=saver)

        assert result.pipeline_status == "pending_hil_gate1"
        assert result.requires_attention is True
        assert result.gate_decisions and result.gate_decisions[0].status == "pending"
        assert result.comparison is None
        run_id = result.run_dir.name
        assert gp.pending_gate(run_id, checkpointer=saver) == "gate1"

        arbitration = _arbitration_payload()
        resumed = gp.resume_pipeline_graph(run_id, arbitration, checkpointer=saver)

        assert resumed.pipeline_status == "completed"
        assert resumed.comparison is not None
        # Rehydrated as the same schema model app.py's disk loader produces
        assert resumed.hil_reviews == [HilArbitration(**arbitration)]
        assert resumed.hil_reviews[0].reviewer == "tester"
        assert stage_calls["gate1"] == 1  # fix 6: compute node ran exactly once
        assert gp.pending_gate(run_id, checkpointer=saver) is None

    def test_resume_at_gate_without_payload_raises(self, tmp_path, stage_calls, monkeypatch):
        monkeypatch.setattr(gp, "evaluate_gate1", lambda *a, **k: _pending_gate1_decision())
        saver = MemorySaver()

        result = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=saver)

        with pytest.raises(ValueError, match="arbitration payload"):
            gp.resume_pipeline_graph(result.run_dir.name, checkpointer=saver)


class _Crash(BaseException):
    """Simulates process death: not an Exception, so nodes don't catch it."""


class TestCrashRecovery:
    """Fix 2: a run that died mid-node continues from the checkpoint, no payload."""

    def test_crash_mid_run_resumes_from_checkpoint(self, tmp_path, stage_calls, monkeypatch):
        armed = {"crash": True}

        def eval_crashes_once(*args, **kwargs):
            if armed["crash"]:
                armed["crash"] = False
                raise _Crash()
            stage_calls["eval"] += 1
            return _make_eval("A", [8] * 6), _make_eval("B", [6] * 6), "medium"
        monkeypatch.setattr(gp, "evaluate_images", eval_crashes_once)
        saver = MemorySaver()

        with pytest.raises(_Crash):
            gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=saver)

        run_id = next(p for p in tmp_path.iterdir() if p.is_dir()).name
        resumed = gp.resume_pipeline_graph(run_id, checkpointer=saver)

        assert resumed.pipeline_status == "completed"
        assert resumed.comparison is not None
        assert stage_calls["generate"] == 1  # images were not regenerated

    def test_resume_unknown_run_raises(self):
        with pytest.raises(ValueError, match="No checkpoint"):
            gp.resume_pipeline_graph("nonexistent-run", checkpointer=MemorySaver())

    def test_pending_gate_is_none_for_a_crashed_run(self, tmp_path, stage_calls, monkeypatch):
        """A crash leaves snapshot.next non-empty at an ordinary node; that is
        not a HIL pause and must not be reported as one, or the UI would offer
        an arbitration form instead of crash recovery."""
        armed = {"crash": True}

        def eval_crashes_once(*args, **kwargs):
            if armed["crash"]:
                armed["crash"] = False
                raise _Crash()
            stage_calls["eval"] += 1
            return _make_eval("A", [8] * 6), _make_eval("B", [6] * 6), "medium"
        monkeypatch.setattr(gp, "evaluate_images", eval_crashes_once)
        saver = MemorySaver()

        with pytest.raises(_Crash):
            gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=saver)

        run_id = next(p for p in tmp_path.iterdir() if p.is_dir()).name
        assert gp.pending_gate(run_id, checkpointer=saver) is None


class TestStageCallbacks:
    """Fixes 4 + 5: on_stage fires before the stage runs; on_image_done is forwarded."""

    def test_on_stage_fires_before_stage_work(self, tmp_path, stage_calls, monkeypatch):
        events: list[str] = []
        fake_gen, fake_eval = gp.generate_images, gp.evaluate_images

        def gen_spy(*args, **kwargs):
            events.append("generate_ran")
            return fake_gen(*args, **kwargs)

        def eval_spy(*args, **kwargs):
            events.append("evaluate_ran")
            return fake_eval(*args, **kwargs)
        monkeypatch.setattr(gp, "generate_images", gen_spy)
        monkeypatch.setattr(gp, "evaluate_images", eval_spy)

        gp.run_pipeline_graph(
            "test prompt", runs_dir=tmp_path,
            on_stage=events.append, checkpointer=MemorySaver(),
        )

        assert events.index("stage_generating") < events.index("generate_ran")
        assert events.index("generate_ran") < events.index("stage_evaluating") < events.index("evaluate_ran")
        assert events[-1] == "stage_complete"

    def test_on_image_done_forwarded_to_generate(self, tmp_path, stage_calls):
        done: list[str] = []

        gp.run_pipeline_graph(
            "test prompt", runs_dir=tmp_path,
            on_image_done=done.append, checkpointer=MemorySaver(),
        )

        assert sorted(done) == ["gemini_3_pro", "gpt_image_2"]


class TestRetryFromArtifacts:
    """Fix 3: retry_pipeline_graph() == resume_pipeline_from_result() semantics."""

    def test_retry_after_generation_failure_regenerates(self, tmp_path, stage_calls, monkeypatch):
        broken = {"on": True}

        def flaky_generate(prompt, run_dir, on_model_done=None, **kwargs):
            stage_calls["generate"] += 1
            gemini_path = run_dir / "gemini_3_pro.png"
            gemini_path.write_bytes(b"b")
            if broken["on"]:
                return {"gpt_image_2": RuntimeError("quota exceeded"), "gemini_3_pro": gemini_path}
            gpt_path = run_dir / "gpt_image_2.png"
            gpt_path.write_bytes(b"a")
            return {"gpt_image_2": gpt_path, "gemini_3_pro": gemini_path}
        monkeypatch.setattr(gp, "generate_images", flaky_generate)
        saver = MemorySaver()

        failed = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=saver)
        assert failed.pipeline_status == "failed"
        assert failed.errors

        broken["on"] = False
        retried = gp.retry_pipeline_graph(failed, checkpointer=saver)

        assert retried.pipeline_status == "completed"
        assert retried.errors == []  # stale errors cleared, like legacy resume
        assert retried.comparison is not None
        assert stage_calls["generate"] == 2

    def test_retry_after_evaluation_failure_reuses_images(self, tmp_path, stage_calls, monkeypatch):
        broken = {"on": True}

        def flaky_eval(*args, **kwargs):
            stage_calls["eval"] += 1
            if broken["on"]:
                raise ValueError("rate limited")
            return _make_eval("A", [8] * 6), _make_eval("B", [6] * 6), "medium"
        monkeypatch.setattr(gp, "evaluate_images", flaky_eval)
        saver = MemorySaver()

        failed = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=saver)
        assert failed.pipeline_status == "failed"

        broken["on"] = False
        retried = gp.retry_pipeline_graph(failed, checkpointer=saver)

        assert retried.pipeline_status == "completed"
        assert stage_calls["generate"] == 1  # images reused from disk
        assert stage_calls["eval"] == 2

    def test_retry_legacy_run_without_checkpoint(self, tmp_path, stage_calls):
        """A run never touched by the graph (or whose checkpoint was lost) can
        still be retried purely from disk artifacts + the loaded result."""
        run_dir = tmp_path / "20260101_000000_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "gpt_image_2.png").write_bytes(b"a")
        (run_dir / "gemini_3_pro.png").write_bytes(b"b")

        legacy = PipelineResult()
        legacy.prompt = "test prompt"
        legacy.run_dir = run_dir
        legacy.timestamp = datetime.now().isoformat()
        legacy.eval_a = _make_eval("A", [8] * 6)
        legacy.eval_b = _make_eval("B", [6] * 6)
        legacy.prompt_difficulty = "medium"
        legacy.pipeline_status = "failed"
        legacy.errors = ["Critique round 1 failed: boom"]

        retried = gp.retry_pipeline_graph(legacy, checkpointer=MemorySaver())

        assert retried.pipeline_status == "completed"
        assert retried.errors == []
        # generation and evaluation were skipped; the loop ran from gate1 onward
        assert stage_calls == {"generate": 0, "eval": 0, "gate1": 1, "crit1": 1, "crit2": 1, "gate2": 1, "revise": 2}
        assert retried.comparison is not None
        assert (run_dir / "comparison.json").exists()


class TestLegacyParityGaps:
    """Regression tests for the three parity gaps found in the audit:
    reference_image_name propagation, hil_adjudication on the result,
    and graceful failure on a bad reference upload."""

    @staticmethod
    def _png_bytes() -> bytes:
        buf = BytesIO()
        Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
        return buf.getvalue()

    def test_reference_image_name_recorded_in_result_and_summary(self, tmp_path, stage_calls):
        result = gp.run_pipeline_graph(
            "test prompt", runs_dir=tmp_path,
            reference_image=self._png_bytes(),
            reference_image_name="my_upload.png",
            checkpointer=MemorySaver(),
        )

        assert result.pipeline_status == "completed"
        assert result.reference_image_name == "my_upload.png"
        assert result.reference_image_path is not None and result.reference_image_path.exists()
        summary = json.loads((result.run_dir / "summary.json").read_text())
        assert summary["reference_image_source"] == "my_upload.png"
        assert summary["reference_image"] == "reference_image.jpg"

    def test_reference_image_name_survives_retry(self, tmp_path, stage_calls, monkeypatch):
        broken = {"on": True}

        def flaky_eval(*args, **kwargs):
            if broken["on"]:
                raise ValueError("rate limited")
            stage_calls["eval"] += 1
            return _make_eval("A", [8] * 6), _make_eval("B", [6] * 6), "medium"
        monkeypatch.setattr(gp, "evaluate_images", flaky_eval)
        saver = MemorySaver()

        failed = gp.run_pipeline_graph(
            "test prompt", runs_dir=tmp_path,
            reference_image=self._png_bytes(),
            reference_image_name="my_upload.png",
            checkpointer=saver,
        )
        assert failed.pipeline_status == "failed"

        broken["on"] = False
        retried = gp.retry_pipeline_graph(failed, checkpointer=saver)

        assert retried.pipeline_status == "completed"
        assert retried.reference_image_name == "my_upload.png"
        summary = json.loads((retried.run_dir / "summary.json").read_text())
        assert summary["reference_image_source"] == "my_upload.png"

    def test_invalid_reference_image_fails_run_with_summary(self, tmp_path, stage_calls):
        result = gp.run_pipeline_graph(
            "test prompt", runs_dir=tmp_path,
            reference_image=b"not an image",
            reference_image_name="corrupt.png",
            checkpointer=MemorySaver(),
        )

        assert result.pipeline_status == "failed"
        assert result.requires_attention is True
        assert any("Reference image failed" in e for e in result.errors)
        assert stage_calls["generate"] == 0  # the graph was never entered
        summary = json.loads((result.run_dir / "summary.json").read_text())
        assert summary["pipeline_status"] == "failed"
        assert any("Reference image failed" in e for e in summary["errors"])

    def test_missing_generator_slot_reports_a_readable_error(self, tmp_path, stage_calls, monkeypatch):
        """A slot absent from generate_images()' dict is not an Exception; it
        must not render as '... failed: None' in summary.json."""
        def gen_missing_slot(prompt, run_dir, **kwargs):
            gemini_path = run_dir / "gemini_3_pro.png"
            gemini_path.write_bytes(b"b")
            return {"gemini_3_pro": gemini_path}
        monkeypatch.setattr(gp, "generate_images", gen_missing_slot)

        result = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=MemorySaver())

        assert result.pipeline_status == "failed"
        assert result.errors == ["GPT Image 2 failed: no image returned"]
        assert not any("None" in e for e in result.errors)

    def test_hil_review_survives_a_retry(self, tmp_path, stage_calls, monkeypatch):
        """Retrying past a completed gate1 arbitration keeps the human's work."""
        broken = {"on": True}

        def flaky_critique(*args, **kwargs):
            if broken["on"]:
                raise ValueError("rate limited")
            stage_calls["crit1"] += 1
            return _make_critique(1)
        monkeypatch.setattr(gp, "critique_evaluation", flaky_critique)
        saver = MemorySaver()

        degraded = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=saver)
        degraded.hil_reviews = [HilArbitration(**_arbitration_payload())]

        broken["on"] = False
        retried = gp.retry_pipeline_graph(degraded, checkpointer=saver)

        assert retried.pipeline_status == "completed"
        assert retried.hil_reviews == degraded.hil_reviews

    def test_gate2_adjudication_populates_result(self, tmp_path, stage_calls, monkeypatch):
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
        saver = MemorySaver()

        pending = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=saver)
        assert pending.pipeline_status == "pending_hil_gate2"
        assert pending.hil_adjudication is None

        adjudication = _adjudication_payload()
        resumed = gp.resume_pipeline_graph(pending.run_dir.name, adjudication, checkpointer=saver)

        assert resumed.pipeline_status == "completed"
        assert resumed.hil_adjudication == HilAdjudication(**adjudication)
        assert resumed.hil_adjudication.reviewer == "tester"
        assert json.loads((resumed.run_dir / "hil_adjudication.json").read_text()) == adjudication


class TestGate2ChangesTheOutcome:
    """The full loop for gate 2: pause, adjudicate, resume, and the human's
    call reaches comparison.json.

    Round 2 tanks model A's prompt_adherence from 9 to 3, handing the dimension
    (and the run, since every other dimension ties at 8) to B. A reviewer
    answering "agree_with_05" says round 2's critic did not persuade them, so
    round 1's revision stands for that one dimension. Nothing else moves.
    """

    R1_A, R2_A, B = [9, 8, 8, 8, 8, 8], [3, 8, 8, 8, 8, 8], [8] * 6

    @pytest.fixture
    def diverging_rounds(self, stage_calls, monkeypatch):
        """Two revision rounds that disagree, and a gate 2 that pauses."""
        def fake_revise(*args, **kwargs):
            stage_calls["revise"] += 1
            scores_a = self.R1_A if stage_calls["revise"] == 1 else self.R2_A
            return _make_revision(scores_a, self.B)

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

        monkeypatch.setattr(gp, "revise_evaluation", fake_revise)
        monkeypatch.setattr(gp, "evaluate_gate2", pending_gate2)
        return stage_calls

    def _run_to_completion(self, tmp_path, adjudication):
        saver = MemorySaver()
        pending = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=saver)
        assert pending.pipeline_status == "pending_hil_gate2"
        return gp.resume_pipeline_graph(pending.run_dir.name, adjudication, checkpointer=saver)

    def test_agree_with_05_flips_the_dimension_and_the_run(self, tmp_path, diverging_rounds):
        resumed = self._run_to_completion(tmp_path, _adjudication_payload())

        assert resumed.pipeline_status == "completed"
        comparison = json.loads((resumed.run_dir / "comparison.json").read_text())
        adherence = comparison["dimension_results"][0]
        assert adherence["dimension"] == "prompt_adherence"
        assert adherence["score_a"] == 9        # round 1 restored, not round 2's 3
        assert adherence["winner"] == "model_a"
        assert comparison["overall_winner"] == "model_a"

    def test_the_same_run_without_the_human_goes_the_other_way(self, tmp_path, diverging_rounds):
        """Control: identical scores, no agree_with_05 -> round 2 stands."""
        payload = _adjudication_payload()
        payload["adjudication_labels"][0]["label"] = "agree_with_07"

        resumed = self._run_to_completion(tmp_path, payload)

        comparison = json.loads((resumed.run_dir / "comparison.json").read_text())
        assert comparison["dimension_results"][0]["score_a"] == 3
        assert comparison["dimension_results"][0]["winner"] == "model_b"
        assert comparison["overall_winner"] == "model_b"

    def test_per_round_artifacts_keep_the_models_unedited(self, tmp_path, diverging_rounds):
        """revised_r*.json record what the models actually said, so reloading a
        run re-derives the same selection instead of compounding it."""
        resumed = self._run_to_completion(tmp_path, _adjudication_payload())
        run_dir = resumed.run_dir

        r1 = json.loads((run_dir / "revised_r1.json").read_text())
        r2 = json.loads((run_dir / "revised_r2.json").read_text())
        final = json.loads((run_dir / "revised.json").read_text())

        assert r1["model_a"]["prompt_adherence"]["score"] == 9
        assert r2["model_a"]["prompt_adherence"]["score"] == 3
        assert final["model_a"]["prompt_adherence"]["score"] == 9  # the composed one
        assert final["model_a"]["photorealism"]["score"] == 8      # untouched

    def test_reload_from_disk_reproduces_the_verdict(self, tmp_path, diverging_rounds):
        """A page refresh must not change the winner: app.py rebuilds the result
        from the per-round artifacts and recomposes, so the selection has to be
        idempotent."""
        resumed = self._run_to_completion(tmp_path, _adjudication_payload())

        reloaded = PipelineResult()
        reloaded.revisions = [
            RevisedEvaluation(**json.loads((resumed.run_dir / f"revised_r{r}.json").read_text()))
            for r in (1, 2)
        ]
        reloaded.hil_adjudication = HilAdjudication(
            **json.loads((resumed.run_dir / "hil_adjudication.json").read_text())
        )

        again = _final_revision(reloaded)
        assert again.model_a.prompt_adherence.score == 9
        assert again.model_a.model_dump() == json.loads(
            (resumed.run_dir / "revised.json").read_text()
        )["model_a"]


# ---------------------------------------------------------------------------
# Production checkpointer: SqliteSaver durability, the singleton, concurrency.
#
# Everything above compiles the graph with MemorySaver, which lives and dies
# with the test process. That validates the checkpoint *protocol* but never the
# path production actually uses: default_checkpointer() -> SqliteSaver on a
# file. These tests cover the gap — cross-process durability (the whole point
# of checkpointing), the WAL/busy_timeout singleton, and the Streamlit
# threading model that default_checkpointer()'s docstring justifies itself on.
# ---------------------------------------------------------------------------

import sqlite3
import subprocess
import sys
import threading
import time

from langgraph.checkpoint.sqlite import SqliteSaver

WORKER = Path(__file__).parent / "graph_subprocess_worker.py"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _sqlite_saver(db_path: Path) -> SqliteSaver:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return SqliteSaver(conn)


def _pending_gate1(stage_calls):
    """evaluate_gate1 replacement that suspends the run for human arbitration."""
    def _gate1(*args, **kwargs):
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
    return _gate1


def _run_worker(mode: str, db: Path, run_id: str, *extra: str) -> dict:
    """Run the worker in a genuinely separate interpreter and parse its JSON."""
    proc = subprocess.run(
        [sys.executable, str(WORKER), mode, str(db), run_id, *extra],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
    )
    assert proc.returncode == 0, f"worker failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestSqliteDurability:
    """A suspended run must survive the death of the process that created it."""

    def test_suspended_run_is_visible_to_a_cold_process(self, tmp_path, stage_calls, monkeypatch):
        monkeypatch.setattr(gp, "evaluate_gate1", _pending_gate1(stage_calls))
        db = tmp_path / "checkpoints.db"

        pending = gp.run_pipeline_graph(
            "cross process prompt", runs_dir=tmp_path, checkpointer=_sqlite_saver(db)
        )
        assert pending.pipeline_status == "pending_hil_gate1"
        run_id = pending.run_dir.name

        seen = _run_worker("inspect", db, run_id)

        # The cold process must reach the same conclusion the UI would.
        assert seen["next"] == ["gate1_wait"]
        assert seen["has_pending_interrupt"] is True
        assert seen["pending_gate"] == "gate1"
        assert seen["pipeline_status"] == "pending_hil_gate1"
        assert seen["prompt"] == "cross process prompt"
        # The interrupt *payload* round-trips, not merely the fact of an interrupt:
        # app.py renders the arbitration form from these fields.
        assert seen["interrupt_values"][0]["gate"] == "gate1_uncertainty_risk_router"
        assert seen["interrupt_values"][0]["route_band"] == "required_hil"
        assert seen["interrupt_values"][0]["trigger_reasons"] == ["narrow_margin"]

    def test_cold_process_resumes_without_redoing_upstream_work(self, tmp_path, stage_calls, monkeypatch):
        monkeypatch.setattr(gp, "evaluate_gate1", _pending_gate1(stage_calls))
        db = tmp_path / "checkpoints.db"

        pending = gp.run_pipeline_graph(
            "test prompt", runs_dir=tmp_path, checkpointer=_sqlite_saver(db)
        )
        run_id = pending.run_dir.name
        assert stage_calls["generate"] == 1

        out = _run_worker("resume", db, run_id, json.dumps(_arbitration_payload()))

        assert out["pipeline_status"] == "completed"
        assert out["has_comparison"] is True
        assert out["hil_reviews"] == 1
        assert out["errors"] == []
        # The expensive stages are not in the resumed stage list. The worker
        # leaves generate/evaluate unstubbed, so re-running them would have
        # errored on the network rather than passing quietly.
        assert "stage_generating" not in out["stages"]
        assert "stage_evaluating" not in out["stages"]
        assert out["stages"][0] == "stage_critique"
        # And the parent's counters confirm it from this side too.
        assert stage_calls["generate"] == 1
        assert stage_calls["eval"] == 1

    def test_completed_run_discards_its_checkpoint(self, tmp_path, stage_calls):
        """A finished run's checkpoint is retired, not kept.

        It exists only to let an unfinished run continue; once the run is done
        it is ~200KB of dead weight per run that SQLite never reclaims. The UI
        rehydrates finished runs from the disk artifacts, and a retry rebuilds
        from them too, so nothing downstream depends on the thread surviving.
        """
        db = tmp_path / "checkpoints.db"
        done = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=_sqlite_saver(db))
        assert done.pipeline_status == "completed"
        assert done.comparison is not None

        seen = _run_worker("inspect", db, done.run_dir.name)
        assert seen["next"] == []
        assert seen["has_pending_interrupt"] is False
        assert seen["pending_gate"] is None, "a finished run must not look like it needs arbitration"
        assert seen["critiques"] == 0, "the checkpoint should have been discarded on completion"
        assert gp.checkpoint_thread_ids(checkpointer=_sqlite_saver(db)) == set()
        # The run itself is fully intact on disk — only the checkpoint went.
        assert (done.run_dir / "comparison.json").exists()
        assert (done.run_dir / "revised_r2.json").exists()

    def test_paused_run_keeps_its_checkpoint(self, tmp_path, stage_calls, monkeypatch):
        """The mirror image: a pause must survive, or the human can never resume."""
        monkeypatch.setattr(gp, "evaluate_gate1", _pending_gate1(stage_calls))
        db = tmp_path / "checkpoints.db"
        paused = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=_sqlite_saver(db))
        assert paused.pipeline_status == "pending_hil_gate1"
        assert gp.checkpoint_thread_ids(checkpointer=_sqlite_saver(db)) == {paused.run_dir.name}


class TestCheckpointLifecycle:
    """runs/checkpoints.db must not grow without bound."""

    def test_discard_checkpoint_removes_one_thread(self, tmp_path, stage_calls, monkeypatch):
        monkeypatch.setattr(gp, "evaluate_gate1", _pending_gate1(stage_calls))
        db = tmp_path / "checkpoints.db"
        paused = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=_sqlite_saver(db))
        run_id = paused.run_dir.name
        assert gp.checkpoint_thread_ids(checkpointer=_sqlite_saver(db)) == {run_id}

        assert gp.discard_checkpoint(run_id, checkpointer=_sqlite_saver(db)) is True
        assert gp.checkpoint_thread_ids(checkpointer=_sqlite_saver(db)) == set()

    def test_discarding_an_unknown_thread_is_harmless(self, tmp_path):
        """Callers should not have to check first — deletion is idempotent."""
        db = tmp_path / "checkpoints.db"
        assert gp.discard_checkpoint("never-existed", checkpointer=_sqlite_saver(db)) is True
        assert gp.checkpoint_thread_ids(checkpointer=_sqlite_saver(db)) == set()

    def test_prune_drops_only_threads_whose_run_dir_is_gone(self, tmp_path, stage_calls, monkeypatch):
        monkeypatch.setattr(gp, "evaluate_gate1", _pending_gate1(stage_calls))
        db = tmp_path / "checkpoints.db"
        keep = gp.run_pipeline_graph("keep me", runs_dir=tmp_path, checkpointer=_sqlite_saver(db))
        drop = gp.run_pipeline_graph("delete me", runs_dir=tmp_path, checkpointer=_sqlite_saver(db))

        import shutil
        shutil.rmtree(drop.run_dir)  # deleted outside the app

        pruned = gp.prune_stale_checkpoints(tmp_path, checkpointer=_sqlite_saver(db))
        assert pruned == [drop.run_dir.name]
        assert gp.checkpoint_thread_ids(checkpointer=_sqlite_saver(db)) == {keep.run_dir.name}

    def test_prune_clears_the_backlog_of_finished_runs(self, tmp_path, stage_calls, monkeypatch):
        """A run that finished *before* checkpoints were retired on completion
        keeps its directory, so an orphan-only sweep would never reach it."""
        db = tmp_path / "checkpoints.db"
        # Reproduce the old behaviour: complete a run without retiring its thread.
        monkeypatch.setattr(gp, "_discard_thread", lambda *a, **k: False)
        done = gp.run_pipeline_graph("finished earlier", runs_dir=tmp_path, checkpointer=_sqlite_saver(db))
        monkeypatch.undo()

        assert done.pipeline_status == "completed"
        assert gp.checkpoint_thread_ids(checkpointer=_sqlite_saver(db)) == {done.run_dir.name}
        assert done.run_dir.is_dir(), "the directory survives; only the checkpoint is stale"

        pruned = gp.prune_stale_checkpoints(tmp_path, checkpointer=_sqlite_saver(db))
        assert pruned == [done.run_dir.name]
        assert gp.checkpoint_thread_ids(checkpointer=_sqlite_saver(db)) == set()
        assert done.run_dir.is_dir(), "pruning must not touch the run itself"
        assert (done.run_dir / "comparison.json").exists()

    def test_prune_is_a_noop_when_everything_is_live(self, tmp_path, stage_calls, monkeypatch):
        monkeypatch.setattr(gp, "evaluate_gate1", _pending_gate1(stage_calls))
        db = tmp_path / "checkpoints.db"
        paused = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=_sqlite_saver(db))
        assert gp.prune_stale_checkpoints(tmp_path, checkpointer=_sqlite_saver(db)) == []
        assert gp.checkpoint_thread_ids(checkpointer=_sqlite_saver(db)) == {paused.run_dir.name}

    def test_pruning_never_breaks_a_paused_resume(self, tmp_path, stage_calls, monkeypatch):
        """Retirement must not touch a run someone still has to come back to."""
        monkeypatch.setattr(gp, "evaluate_gate1", _pending_gate1(stage_calls))
        db = tmp_path / "checkpoints.db"
        paused = gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=_sqlite_saver(db))
        run_id = paused.run_dir.name

        assert gp.prune_stale_checkpoints(tmp_path, checkpointer=_sqlite_saver(db)) == []
        assert gp.pending_gate(run_id, checkpointer=_sqlite_saver(db)) == "gate1"

        monkeypatch.setattr(gp, "evaluate_gate1", lambda *a, **k: _gate1_clear())
        resumed = gp.resume_pipeline_graph(
            run_id, _arbitration_payload(), checkpointer=_sqlite_saver(db)
        )
        assert resumed.pipeline_status == "completed"
        assert resumed.comparison is not None

    def test_pruning_never_breaks_crash_recovery(self, tmp_path, stage_calls, monkeypatch):
        """A crash leaves a non-empty `next`, which must read as "unfinished",
        not "finished" — otherwise recovery is deleted out from under the user."""
        db = tmp_path / "checkpoints.db"
        crashed = {"yet": False}

        def eval_crashes_once(*args, **kwargs):
            stage_calls["eval"] += 1
            if not crashed["yet"]:
                crashed["yet"] = True
                raise _Crash("process died")
            return _make_eval("A", [8] * 6), _make_eval("B", [6] * 6), "medium"

        monkeypatch.setattr(gp, "evaluate_images", eval_crashes_once)
        with pytest.raises(_Crash):
            gp.run_pipeline_graph("test prompt", runs_dir=tmp_path, checkpointer=_sqlite_saver(db))
        run_id = next(p.name for p in tmp_path.iterdir() if p.is_dir())

        assert gp.prune_stale_checkpoints(tmp_path, checkpointer=_sqlite_saver(db)) == []
        # Not an interrupt: the UI should offer recovery, not an arbitration form.
        assert gp.pending_gate(run_id, checkpointer=_sqlite_saver(db)) is None

        recovered = gp.resume_pipeline_graph(run_id, checkpointer=_sqlite_saver(db))
        assert recovered.pipeline_status == "completed"
        assert stage_calls["generate"] == 1, "images must not be regenerated"

    def test_db_does_not_grow_across_repeated_completed_runs(self, tmp_path, stage_calls):
        """The leak this whole class exists to prevent: ~200KB per finished run."""
        db = tmp_path / "checkpoints.db"
        for i in range(5):
            gp.run_pipeline_graph(f"run {i}", runs_dir=tmp_path, checkpointer=_sqlite_saver(db))
        assert gp.checkpoint_thread_ids(checkpointer=_sqlite_saver(db)) == set()
        rows = sqlite3.connect(db).execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        assert rows == 0, f"{rows} checkpoint rows survived 5 finished runs"


class TestDefaultCheckpointer:
    """default_checkpointer() is the production factory; nothing tested it."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        """Never touch the repo's real runs/checkpoints.db."""
        monkeypatch.setattr(gp, "CHECKPOINT_DB", tmp_path / "runs" / "checkpoints.db")
        monkeypatch.setattr(gp, "_default_saver", None)
        yield
        gp._default_saver = None

    def test_is_a_process_wide_singleton(self):
        first = gp.default_checkpointer()
        second = gp.default_checkpointer()
        assert first is second, "a new connection per call is what the singleton exists to prevent"

    def test_creates_parent_directory(self, tmp_path):
        assert not (tmp_path / "runs").exists()
        gp.default_checkpointer()
        assert (tmp_path / "runs" / "checkpoints.db").exists()

    def test_applies_wal_and_busy_timeout(self):
        saver = gp.default_checkpointer()
        conn = saver.conn
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000

    def test_connection_is_usable_from_another_thread(self):
        """check_same_thread=False is load-bearing: app.py runs the graph in a
        daemon thread while the main thread polls."""
        saver = gp.default_checkpointer()
        errors: list[Exception] = []

        def _touch():
            try:
                saver.conn.execute("SELECT 1").fetchone()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        thread = threading.Thread(target=_touch)
        thread.start()
        thread.join()
        assert errors == []

    def test_build_graph_uses_the_singleton_when_none_passed(self):
        graph = gp.build_graph()
        assert graph.checkpointer is gp.default_checkpointer()


class TestConcurrency:
    """The Streamlit model: graph work on a daemon thread, UI polling the state.

    default_checkpointer()'s docstring justifies the shared WAL connection on
    exactly this workload ("Streamlit reruns and background threads don't ...
    hit 'database is locked'"). These tests hold it to that.
    """

    def test_parallel_runs_share_one_connection_without_locking(self, tmp_path, stage_calls):
        saver = _sqlite_saver(tmp_path / "checkpoints.db")
        results: dict[int, PipelineResult] = {}
        errors: list[BaseException] = []
        barrier = threading.Barrier(6)  # maximise overlap on the shared connection

        def _worker(index: int) -> None:
            try:
                barrier.wait(timeout=30)
                results[index] = gp.run_pipeline_graph(
                    f"prompt {index}", runs_dir=tmp_path, checkpointer=saver
                )
            except BaseException as exc:  # noqa: BLE001 - surfaced in the assert
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not any(t.is_alive() for t in threads), "a run deadlocked on the shared connection"
        assert errors == [], f"concurrent runs raised: {errors}"
        assert len(results) == 6
        assert all(r.pipeline_status == "completed" for r in results.values())
        # Threads must not bleed into each other: distinct runs, distinct prompts.
        assert len({r.run_dir for r in results.values()}) == 6
        assert {r.prompt for r in results.values()} == {f"prompt {i}" for i in range(6)}

    def test_state_is_readable_while_the_run_is_in_flight(self, tmp_path, stage_calls, monkeypatch):
        """app.py polls get_state() every 500ms from the main thread while the
        graph writes checkpoints from a daemon thread."""
        released = threading.Event()

        def _slow_evaluate(*args, **kwargs):
            stage_calls["eval"] += 1
            released.wait(timeout=30)  # hold the run open so polling overlaps writes
            return _make_eval("A", [8] * 6), _make_eval("B", [6] * 6), "medium"

        monkeypatch.setattr(gp, "evaluate_images", _slow_evaluate)

        saver = _sqlite_saver(tmp_path / "checkpoints.db")
        graph = gp.build_graph(checkpointer=saver)
        run_dir = tmp_path / "run"
        config = {"configurable": {"thread_id": "concurrent-read"}}
        run_errors: list[BaseException] = []

        def _run():
            try:
                graph.invoke(_initial_state(run_dir), config)
            except BaseException as exc:  # noqa: BLE001
                run_errors.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        poll_errors: list[Exception] = []
        polls = 0
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and stage_calls["eval"] == 0:
            time.sleep(0.02)
        for _ in range(20):  # poll concurrently with the in-flight write
            try:
                graph.get_state(config)
                polls += 1
            except Exception as exc:  # pragma: no cover - failure path
                poll_errors.append(exc)
            time.sleep(0.02)

        released.set()
        thread.join(timeout=60)

        assert poll_errors == [], f"reading state during a run raised: {poll_errors}"
        assert polls == 20
        assert run_errors == []
        assert graph.get_state(config).values["pipeline_status"] == "completed"

    def test_concurrent_resume_of_distinct_suspended_runs(self, tmp_path, stage_calls, monkeypatch):
        """Two humans arbitrate two paused runs at the same time."""
        monkeypatch.setattr(gp, "evaluate_gate1", _pending_gate1(stage_calls))
        saver = _sqlite_saver(tmp_path / "checkpoints.db")

        paused = [
            gp.run_pipeline_graph(f"prompt {i}", runs_dir=tmp_path, checkpointer=saver)
            for i in range(2)
        ]
        assert all(p.pipeline_status == "pending_hil_gate1" for p in paused)

        monkeypatch.setattr(gp, "evaluate_gate1", lambda *a, **k: _gate1_clear())
        resumed: dict[str, PipelineResult] = {}
        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def _resume(run_id: str) -> None:
            try:
                barrier.wait(timeout=30)
                resumed[run_id] = gp.resume_pipeline_graph(
                    run_id, _arbitration_payload(), checkpointer=saver
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_resume, args=(p.run_dir.name,)) for p in paused]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert errors == [], f"concurrent resume raised: {errors}"
        assert len(resumed) == 2
        assert all(r.pipeline_status == "completed" for r in resumed.values())
        assert all(len(r.hil_reviews) == 1 for r in resumed.values())
        # Each thread resumed its own run, not the other's.
        for run_id, result in resumed.items():
            assert result.run_dir.name == run_id
