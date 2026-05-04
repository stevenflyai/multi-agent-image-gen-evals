"""Schema validation tests for all Pydantic models."""

import pytest
from schemas import (
    ComparisonResult,
    CritiqueResponse,
    DimensionCritique,
    DimensionResult,
    DimensionScore,
    ImageEvaluation,
    RevisedDimensionScore,
    RevisedEvaluation,
    RevisedImageEvaluation,
)


def _make_eval(model_name: str = "TestModel", base_score: int = 7) -> ImageEvaluation:
    dims = {
        d: DimensionScore(score=base_score, reasoning=f"Test reasoning for {d}")
        for d in [
            "prompt_adherence", "photorealism", "aesthetic_quality",
            "composition", "color_accuracy", "creativity",
        ]
    }
    return ImageEvaluation(model_name=model_name, **dims)


def test_image_evaluation_valid():
    ev = _make_eval()
    assert ev.model_name == "TestModel"
    assert ev.prompt_adherence.score == 7
    assert ev.mean_score() == 7.0
    assert len(ev.scores_dict()) == 6


def test_image_evaluation_rejects_missing_dimensions():
    with pytest.raises(Exception):
        ImageEvaluation(
            model_name="Test",
            prompt_adherence=DimensionScore(score=5, reasoning="ok"),
            # Missing other dimensions
        )


def test_image_evaluation_rejects_out_of_range():
    with pytest.raises(Exception):
        DimensionScore(score=11, reasoning="too high")
    with pytest.raises(Exception):
        DimensionScore(score=0, reasoning="too low")


def test_critique_response_valid():
    cr = CritiqueResponse(
        overall_assessment="Good evaluation overall",
        dimension_critiques=[
            DimensionCritique(
                dimension="prompt_adherence",
                original_score_model_a=8,
                original_score_model_b=6,
                critique="Score seems high for model A",
                suggested_score_model_a=7,
                suggested_score_model_b=6,
            )
        ],
        bias_detection="No systematic bias detected",
    )
    assert len(cr.dimension_critiques) == 1
    assert cr.dimension_critiques[0].suggested_score_model_a == 7
    # Default round/model values
    assert cr.round == 1
    assert cr.critic_model == ""


def test_critique_response_with_round_and_model():
    cr = CritiqueResponse(
        overall_assessment="Round 2 review",
        dimension_critiques=[],
        bias_detection="None",
        round=2,
        critic_model="gemini-3.1-pro",
    )
    assert cr.round == 2
    assert cr.critic_model == "gemini-3.1-pro"


def test_revised_evaluation_valid():
    dims = {
        d: RevisedDimensionScore(
            score=8, reasoning="Revised", critique_accepted=True, revision_note="Accepted"
        )
        for d in [
            "prompt_adherence", "photorealism", "aesthetic_quality",
            "composition", "color_accuracy", "creativity",
        ]
    }
    rev = RevisedImageEvaluation(model_name="Test", **dims)
    assert rev.mean_score() == 8.0

    full = RevisedEvaluation(model_a=rev, model_b=rev)
    assert full.model_a.model_name == "Test"


def test_comparison_result_valid():
    cr = ComparisonResult(
        prompt="test prompt",
        model_a_name="GPT Image-2",
        model_b_name="Gemini 3 Pro",
        dimension_results=[
            DimensionResult(
                dimension="prompt_adherence",
                score_a=8, score_b=7,
                pre_critique_score_a=9, pre_critique_score_b=7,
                winner="model_a",
            ),
        ],
        overall_winner="model_a",
        model_a_mean=8.0,
        model_b_mean=7.0,
        model_a_dimensions_won=1,
        model_b_dimensions_won=0,
    )
    assert cr.overall_winner == "model_a"
