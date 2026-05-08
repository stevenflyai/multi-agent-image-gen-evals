"""Tests for pipeline convergence and fallback logic."""

import json
from io import BytesIO

import pytest
from PIL import Image
from pipeline import _scores_converged, _build_comparison_fallback, PipelineResult, resume_pipeline_from_result, run_pipeline
from schemas import (
    CritiqueResponse,
    DIMENSIONS,
    DimensionCritique,
    DimensionScore,
    GateDecision,
    ImageEvaluation,
    RevisedDimensionScore,
    RevisedEvaluation,
    RevisedImageEvaluation,
)


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
                original_score_model_b=7,
                critique="Test critique",
                suggested_score_model_a=8,
                suggested_score_model_b=7,
            )
        ],
        bias_detection="No systematic bias detected",
        round=round_number,
        critic_model="test",
    )


class TestScoresConverged:
    def test_identical_scores_converge(self):
        rev = _make_revision([7, 7, 7, 7, 7, 7], [6, 6, 6, 6, 6, 6])
        assert _scores_converged(rev, rev, threshold=1) is True

    def test_small_delta_converges(self):
        rev1 = _make_revision([7, 7, 7, 7, 7, 7], [6, 6, 6, 6, 6, 6])
        rev2 = _make_revision([7, 7, 7, 7, 7, 7], [6, 6, 6, 6, 6, 6])
        assert _scores_converged(rev1, rev2, threshold=1) is True

    def test_large_delta_does_not_converge(self):
        rev1 = _make_revision([7, 7, 7, 7, 7, 7], [6, 6, 6, 6, 6, 6])
        rev2 = _make_revision([9, 7, 7, 7, 7, 7], [6, 6, 6, 6, 6, 6])
        assert _scores_converged(rev1, rev2, threshold=1) is False

    def test_threshold_boundary(self):
        rev1 = _make_revision([7, 7, 7, 7, 7, 7], [6, 6, 6, 6, 6, 6])
        rev2 = _make_revision([8, 7, 7, 7, 7, 7], [6, 6, 6, 6, 6, 6])
        # Delta of exactly 1 should NOT converge (>= threshold)
        assert _scores_converged(rev1, rev2, threshold=1) is False
        # But with threshold 2, it should converge
        assert _scores_converged(rev1, rev2, threshold=2) is True


class TestBuildComparisonFallback:
    def test_builds_fallback_from_initial_scores(self):
        result = PipelineResult()
        result.prompt = "test"
        result.eval_a = _make_eval("A", [8, 8, 8, 8, 8, 8])
        result.eval_b = _make_eval("B", [6, 6, 6, 6, 6, 6])

        _build_comparison_fallback(result)

        assert result.comparison is not None
        assert result.comparison.overall_winner == "model_a"
        assert len(result.revisions) == 1
        # All critique_accepted should be False
        for d in DIMENSIONS:
            assert getattr(result.revisions[0].model_a, d).critique_accepted is False

    def test_no_fallback_without_evals(self):
        result = PipelineResult()
        result.prompt = "test"
        _build_comparison_fallback(result)
        assert result.comparison is None


class TestImageGenerationFailure:
    def test_image_generation_failure_marks_pipeline_failed(self, tmp_path, monkeypatch):
        def fake_generate_images(*args, **kwargs):
            gemini_path = tmp_path / "gemini_3_pro.png"
            gemini_path.write_bytes(b"image")
            return {
                "gpt_image_2": RuntimeError("AuthenticationTypeDisabled"),
                "gemini_3_pro": gemini_path,
            }

        monkeypatch.setattr("pipeline.generate_images", fake_generate_images)

        result = run_pipeline("test prompt", runs_dir=tmp_path)

        assert result.pipeline_status == "failed"
        assert result.requires_attention is True
        assert result.comparison is None
        assert any("GPT Image-2 failed" in error for error in result.errors)
        summary = json.loads((result.run_dir / "summary.json").read_text())
        assert summary["pipeline_status"] == "failed"
        assert summary["has_comparison"] is False

    def test_image_generation_failure_with_empty_exception_has_message(self, tmp_path, monkeypatch):
        def fake_generate_images(*args, **kwargs):
            gemini_path = tmp_path / "gemini_3_pro.png"
            gemini_path.write_bytes(b"image")
            return {
                "gpt_image_2": TimeoutError(),
                "gemini_3_pro": gemini_path,
            }

        monkeypatch.setattr("pipeline.generate_images", fake_generate_images)

        result = run_pipeline("test prompt", runs_dir=tmp_path)

        assert result.errors == ["GPT Image-2 failed: TimeoutError"]
        summary = json.loads((result.run_dir / "summary.json").read_text())
        assert summary["errors"] == ["GPT Image-2 failed: TimeoutError"]

    def test_reference_image_is_persisted_for_generation(self, tmp_path, monkeypatch):
        image_buffer = BytesIO()
        Image.new("RGB", (2000, 1000), color="red").save(image_buffer, format="JPEG")
        seen = {}

        def fake_generate_images(prompt, run_dir, on_model_done=None, reference_image_path=None):
            seen["reference_image_path"] = reference_image_path
            return {
                "gpt_image_2": RuntimeError("stop before eval"),
                "gemini_3_pro": RuntimeError("stop before eval"),
            }

        monkeypatch.setattr("pipeline.generate_images", fake_generate_images)

        result = run_pipeline(
            "turn this into a product hero image",
            runs_dir=tmp_path,
            reference_image=image_buffer.getvalue(),
            reference_image_name="customer-upload.jpg",
        )

        assert result.pipeline_status == "failed"
        assert result.reference_image_path == result.run_dir / "reference_image.jpg"
        assert seen["reference_image_path"] == result.run_dir / "reference_image.jpg"
        assert result.reference_image_path.exists()
        with Image.open(result.reference_image_path) as persisted_image:
            assert persisted_image.format == "JPEG"
            assert max(persisted_image.size) <= 768
        summary = json.loads((result.run_dir / "summary.json").read_text())
        assert summary["reference_image"] == "reference_image.jpg"
        assert summary["reference_image_source"] == "customer-upload.jpg"


class TestPipelineResultProperties:
    def test_critique_property_returns_last(self):
        from schemas import CritiqueResponse
        result = PipelineResult()
        c1 = CritiqueResponse(overall_assessment="r1", dimension_critiques=[], bias_detection="n", round=1, critic_model="gpt-5.4")
        c2 = CritiqueResponse(overall_assessment="r2", dimension_critiques=[], bias_detection="n", round=2, critic_model="gemini-3.1-pro")
        result.critiques = [c1, c2]
        assert result.critique.round == 2

    def test_revised_property_returns_last(self):
        rev1 = _make_revision([7, 7, 7, 7, 7, 7], [6, 6, 6, 6, 6, 6])
        rev2 = _make_revision([8, 8, 8, 8, 8, 8], [5, 5, 5, 5, 5, 5])
        result = PipelineResult()
        result.revisions = [rev1, rev2]
        assert result.revised.model_a.prompt_adherence.score == 8

    def test_empty_properties_return_none(self):
        result = PipelineResult()
        assert result.critique is None
        assert result.revised is None


class TestResumePipeline:
    def test_resumes_by_regenerating_failed_images(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        reference_image = run_dir / "reference_image.png"
        reference_image.write_bytes(b"reference")
        gemini_old = run_dir / "gemini_3_pro.png"
        gemini_old.write_bytes(b"old gemini")

        result = PipelineResult()
        result.prompt = "test prompt"
        result.run_dir = run_dir
        result.reference_image_path = reference_image
        result.image_paths = {
            "gpt_image_2": RuntimeError("AuthenticationTypeDisabled"),
            "gemini_3_pro": gemini_old,
        }
        result.errors = ["GPT Image-2 failed: AuthenticationTypeDisabled"]
        image_done = []
        calls = {"generate": 0, "eval": 0, "gate1": 0, "critique": 0, "revise": 0}

        def fake_generate_images(prompt, run_dir_arg, on_model_done=None, reference_image_path=None):
            calls["generate"] += 1
            assert prompt == "test prompt"
            assert run_dir_arg == run_dir
            assert reference_image_path == reference_image
            gpt_path = run_dir / "gpt_image_2.png"
            gemini_path = run_dir / "gemini_3_pro.png"
            gpt_path.write_bytes(b"new gpt")
            gemini_path.write_bytes(b"new gemini")
            if on_model_done:
                on_model_done("gpt_image_2")
                on_model_done("gemini_3_pro")
            return {"gpt_image_2": gpt_path, "gemini_3_pro": gemini_path}

        def fake_evaluate_images(*args, **kwargs):
            calls["eval"] += 1
            return _make_eval("A", [8, 8, 8, 8, 8, 8]), _make_eval("B", [6, 6, 6, 6, 6, 6]), "medium"

        def fake_gate1(*args, **kwargs):
            calls["gate1"] += 1
            return GateDecision(gate="gate1_uncertainty_risk_router", status="not_required")

        def fake_critique(*args, **kwargs):
            calls["critique"] += 1
            return _make_critique(1)

        def fake_revise(*args, **kwargs):
            calls["revise"] += 1
            return _make_revision([8, 8, 8, 8, 8, 8], [6, 6, 6, 6, 6, 6])

        monkeypatch.setattr("pipeline.MAX_CRITIQUE_ROUNDS", 1)
        monkeypatch.setattr("pipeline.generate_images", fake_generate_images)
        monkeypatch.setattr("pipeline.evaluate_images", fake_evaluate_images)
        monkeypatch.setattr("pipeline.evaluate_gate1", fake_gate1)
        monkeypatch.setattr("pipeline.critique_evaluation", fake_critique)
        monkeypatch.setattr("pipeline.revise_evaluation", fake_revise)

        resumed = resume_pipeline_from_result(result, on_image_done=image_done.append)

        assert calls == {"generate": 1, "eval": 1, "gate1": 1, "critique": 1, "revise": 1}
        assert image_done == ["gpt_image_2", "gemini_3_pro"]
        assert resumed.errors == []
        assert resumed.pipeline_status == "completed"
        assert resumed.comparison is not None
        assert resumed.image_paths["gpt_image_2"] == run_dir / "gpt_image_2.png"
        assert resumed.image_paths["gemini_3_pro"] == run_dir / "gemini_3_pro.png"

    def test_resumes_from_round2_critique_failure(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        image_a = run_dir / "gpt_image_2.png"
        image_b = run_dir / "gemini_3_pro.png"
        image_a.write_bytes(b"a")
        image_b.write_bytes(b"b")

        result = PipelineResult()
        result.prompt = "test prompt"
        result.run_dir = run_dir
        result.image_paths = {"gpt_image_2": image_a, "gemini_3_pro": image_b}
        result.eval_a = _make_eval("A", [8, 8, 8, 8, 8, 8])
        result.eval_b = _make_eval("B", [6, 6, 6, 6, 6, 6])
        result.critiques = [_make_critique(1)]
        result.revisions = [_make_revision([8, 8, 8, 8, 8, 8], [6, 6, 6, 6, 6, 6])]
        result.errors = ["Critique round 2 failed: bad json"]

        calls = {"eval": 0, "crit2": 0, "revise": 0}

        def fail_eval(*args, **kwargs):
            calls["eval"] += 1
            raise AssertionError("resume should not re-run evaluation")

        def fake_critique_round2(*args, **kwargs):
            calls["crit2"] += 1
            return _make_critique(2)

        def fake_revise(*args, **kwargs):
            calls["revise"] += 1
            return _make_revision([9, 8, 8, 8, 8, 8], [6, 6, 6, 6, 6, 6])

        monkeypatch.setattr("pipeline.evaluate_images", fail_eval)
        monkeypatch.setattr("pipeline.critique_evaluation_gemini", fake_critique_round2)
        monkeypatch.setattr("pipeline.revise_evaluation", fake_revise)

        resumed = resume_pipeline_from_result(result)

        assert calls == {"eval": 0, "crit2": 1, "revise": 1}
        assert resumed.errors == []
        assert len(resumed.critiques) == 2
        assert len(resumed.revisions) == 2
        assert resumed.comparison is not None
        assert (run_dir / "critique_r2.json").exists()
        assert (run_dir / "revised_r2.json").exists()
