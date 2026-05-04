"""Tests for pipeline convergence and fallback logic."""

import json

import pytest
from pipeline import _scores_converged, _build_comparison_fallback, PipelineResult, resume_pipeline_from_result, run_pipeline
from schemas import (
    CritiqueResponse,
    DIMENSIONS,
    DimensionCritique,
    DimensionScore,
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
