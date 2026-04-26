"""Winner determination logic tests."""

import pytest
from compare import determine_winner
from schemas import (
    DimensionScore,
    ImageEvaluation,
    RevisedDimensionScore,
    RevisedEvaluation,
    RevisedImageEvaluation,
)


def _make_eval(model_name: str, scores: list[int]) -> ImageEvaluation:
    dims = ["prompt_adherence", "photorealism", "aesthetic_quality",
            "composition", "color_accuracy", "creativity"]
    return ImageEvaluation(
        model_name=model_name,
        **{d: DimensionScore(score=s, reasoning=f"Score {s}") for d, s in zip(dims, scores)},
    )


def _make_revised(model_name: str, scores: list[int]) -> RevisedImageEvaluation:
    dims = ["prompt_adherence", "photorealism", "aesthetic_quality",
            "composition", "color_accuracy", "creativity"]
    return RevisedImageEvaluation(
        model_name=model_name,
        **{d: RevisedDimensionScore(
            score=s, reasoning=f"Score {s}", critique_accepted=False, revision_note="Test"
        ) for d, s in zip(dims, scores)},
    )


def test_model_a_wins_higher_mean():
    eval_a = _make_eval("A", [8, 8, 8, 8, 8, 8])
    eval_b = _make_eval("B", [6, 6, 6, 6, 6, 6])
    revised = RevisedEvaluation(
        model_a=_make_revised("A", [8, 8, 8, 8, 8, 8]),
        model_b=_make_revised("B", [6, 6, 6, 6, 6, 6]),
    )
    result = determine_winner("test", eval_a, eval_b, revised)
    assert result.overall_winner == "model_a"
    assert result.model_a_dimensions_won == 6
    assert result.model_b_dimensions_won == 0


def test_tiebreak_by_largest_dimension_lead():
    # Same mean (7.0 each) but A has a bigger single-dimension lead
    eval_a = _make_eval("A", [9, 7, 7, 7, 7, 5])
    eval_b = _make_eval("B", [7, 7, 7, 7, 5, 9])
    revised = RevisedEvaluation(
        model_a=_make_revised("A", [9, 7, 7, 7, 7, 5]),  # mean=7, max lead=+2 (dim 0: 9-7)
        model_b=_make_revised("B", [7, 7, 7, 7, 5, 9]),  # mean=7, max lead=+4 (dim 5: 9-5)
    )
    result = determine_winner("test", eval_a, eval_b, revised)
    # B has larger single-dimension lead (9-5=4 vs 9-7=2)
    assert result.overall_winner == "model_b"


def test_complete_tie_is_draw():
    eval_a = _make_eval("A", [7, 7, 7, 7, 7, 7])
    eval_b = _make_eval("B", [7, 7, 7, 7, 7, 7])
    revised = RevisedEvaluation(
        model_a=_make_revised("A", [7, 7, 7, 7, 7, 7]),
        model_b=_make_revised("B", [7, 7, 7, 7, 7, 7]),
    )
    result = determine_winner("test", eval_a, eval_b, revised)
    assert result.overall_winner == "draw"
    assert result.model_a_dimensions_won == 0
    assert result.model_b_dimensions_won == 0
