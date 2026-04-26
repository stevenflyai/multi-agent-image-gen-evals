"""Winner determination logic.

Per-dimension winner by higher post-critique score.
Overall winner by mean score, tiebreak by largest single-dimension lead.
"""

from schemas import (
    DIMENSIONS,
    ComparisonResult,
    DimensionResult,
    RevisedEvaluation,
    ImageEvaluation,
)


def determine_winner(
    prompt: str,
    initial_eval_a: ImageEvaluation,
    initial_eval_b: ImageEvaluation,
    revised: RevisedEvaluation,
) -> ComparisonResult:
    """Determine per-dimension and overall winners."""
    dimension_results: list[DimensionResult] = []

    for dim in DIMENSIONS:
        score_a = getattr(revised.model_a, dim).score
        score_b = getattr(revised.model_b, dim).score
        pre_a = getattr(initial_eval_a, dim).score
        pre_b = getattr(initial_eval_b, dim).score

        if score_a > score_b:
            winner = "model_a"
        elif score_b > score_a:
            winner = "model_b"
        else:
            winner = "draw"

        dimension_results.append(
            DimensionResult(
                dimension=dim,
                score_a=score_a,
                score_b=score_b,
                pre_critique_score_a=pre_a,
                pre_critique_score_b=pre_b,
                winner=winner,
            )
        )

    mean_a = revised.model_a.mean_score()
    mean_b = revised.model_b.mean_score()

    a_won = sum(1 for r in dimension_results if r.winner == "model_a")
    b_won = sum(1 for r in dimension_results if r.winner == "model_b")

    if mean_a > mean_b:
        overall = "model_a"
    elif mean_b > mean_a:
        overall = "model_b"
    else:
        # Tiebreak: largest single-dimension lead
        max_a_lead = max((r.score_a - r.score_b) for r in dimension_results)
        max_b_lead = max((r.score_b - r.score_a) for r in dimension_results)
        if max_a_lead > max_b_lead:
            overall = "model_a"
        elif max_b_lead > max_a_lead:
            overall = "model_b"
        else:
            overall = "draw"

    return ComparisonResult(
        prompt=prompt,
        model_a_name=revised.model_a.model_name,
        model_b_name=revised.model_b.model_name,
        dimension_results=dimension_results,
        overall_winner=overall,
        model_a_mean=round(mean_a, 2),
        model_b_mean=round(mean_b, 2),
        model_a_dimensions_won=a_won,
        model_b_dimensions_won=b_won,
    )
