"""Winner determination logic.

Per-dimension winner by higher post-critique score, unless a human arbitrated
that dimension at gate 1 — their call wins and the scores are left untouched.
Overall winner by mean score, tiebreak by human-decided dimensions first and
largest single-dimension lead otherwise.

Gate 2's adjudication is applied earlier, by apply_gate2_adjudication(): it
selects, per dimension, which round's revision stands before any of this runs.
Both human touchpoints live here so the ways a reviewer can move a result are
in one file.
"""

from schemas import (
    DIMENSIONS,
    ComparisonResult,
    DimensionResult,
    HilAdjudication,
    HilArbitration,
    RevisedEvaluation,
    RevisedImageEvaluation,
    ImageEvaluation,
    MarginType,
    WinnerType,
)

# gate 1 records a human's per-dimension call as A / B / TIE.
_HUMAN_WINNER_TO_MODEL: dict[str, WinnerType] = {
    "A": "model_a",
    "B": "model_b",
    "TIE": "draw",
}


def apply_gate2_adjudication(
    revised_r1: RevisedEvaluation,
    revised_r2: RevisedEvaluation,
    adjudication: HilAdjudication | None,
) -> RevisedEvaluation:
    """Compose the final revision from the two rounds, per the human's call.

    Gate 2 asks, per dimension, which critic was more convincing: the round-1
    critic ("05") or the round-2 critic ("07"). `agree_with_05` means round 2's
    change to that dimension did not persuade, so round 1's revision stands.
    Everything else — `agree_with_07`, `both_partially_right`, and any dimension
    nobody adjudicated — keeps round 2, which is the existing behaviour.

    No score is invented: each dimension is taken whole (score, reasoning,
    revision note) from one of the two revisions the model already produced, so
    the human is selecting between model outputs rather than grading.

    Returns `revised_r2` unchanged when there is nothing to apply.
    """
    picks = {
        label.dimension
        for label in (getattr(adjudication, "adjudication_labels", None) or [])
        if getattr(label, "label", None) == "agree_with_05"
    }
    if not picks:
        return revised_r2

    def _compose(side: str) -> RevisedImageEvaluation:
        from_r1, from_r2 = getattr(revised_r1, side), getattr(revised_r2, side)
        dims = {
            dim: getattr(from_r1 if dim in picks else from_r2, dim)
            for dim in DIMENSIONS
        }
        return RevisedImageEvaluation(model_name=from_r2.model_name, **dims)

    return RevisedEvaluation(model_a=_compose("model_a"), model_b=_compose("model_b"))


def _human_calls(arbitration: HilArbitration | None) -> dict[str, WinnerType]:
    """Per-dimension winners a human decided, keyed by dimension.

    Anything unrecognised is skipped rather than raising: a malformed or
    hand-edited arbitration must not take down winner determination.
    """
    if arbitration is None:
        return {}
    calls: dict[str, WinnerType] = {}
    for item in getattr(arbitration, "dimension_arbitrations", []) or []:
        mapped = _HUMAN_WINNER_TO_MODEL.get(getattr(item, "human_winner", None))
        if mapped is not None:
            calls[item.dimension] = mapped
    return calls


def determine_winner(
    prompt: str,
    initial_eval_a: ImageEvaluation,
    initial_eval_b: ImageEvaluation,
    revised: RevisedEvaluation,
    model_a_name: str | None = None,
    model_b_name: str | None = None,
    arbitration: HilArbitration | None = None,
) -> ComparisonResult:
    """Determine per-dimension and overall winners.

    `model_a_name` / `model_b_name` override the displayed model names (from the
    user's model selection); they fall back to the evaluator-provided names.

    `arbitration` is the gate 1 human review, when one happened. It overrides the
    winner of each dimension it covers **without touching any score** — scores
    stay the model's, the verdict becomes the human's. Omitting it reproduces the
    original score-only behaviour exactly.
    """
    human_calls = _human_calls(arbitration)
    dimension_results: list[DimensionResult] = []

    for dim in DIMENSIONS:
        score_a = getattr(revised.model_a, dim).score
        score_b = getattr(revised.model_b, dim).score
        pre_a = getattr(initial_eval_a, dim).score
        pre_b = getattr(initial_eval_b, dim).score

        if score_a > score_b:
            model_winner: WinnerType = "model_a"
        elif score_b > score_a:
            model_winner = "model_b"
        else:
            model_winner = "draw"

        human_winner = human_calls.get(dim)
        decided_by_human = human_winner is not None

        dimension_results.append(
            DimensionResult(
                dimension=dim,
                score_a=score_a,
                score_b=score_b,
                pre_critique_score_a=pre_a,
                pre_critique_score_b=pre_b,
                winner=human_winner if decided_by_human else model_winner,
                human_decided=decided_by_human,
                model_winner=model_winner if decided_by_human else None,
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
        # Means are tied — the case human review exists for. Prefer the human's
        # dimension verdicts; fall back to the original largest-lead rule when
        # nobody arbitrated or their calls are also even.
        overall = _break_tie(dimension_results, a_won, b_won, bool(human_calls))

    largest_gap = max(abs(r.score_a - r.score_b) for r in dimension_results)
    margin = _classify_margin(overall, a_won, b_won, largest_gap)
    conflict_notes = _build_conflict_notes(dimension_results, overall)

    return ComparisonResult(
        prompt=prompt,
        model_a_name=model_a_name or revised.model_a.model_name,
        model_b_name=model_b_name or revised.model_b.model_name,
        dimension_results=dimension_results,
        overall_winner=overall,
        model_a_mean=round(mean_a, 2),
        model_b_mean=round(mean_b, 2),
        model_a_dimensions_won=a_won,
        model_b_dimensions_won=b_won,
        margin=margin,
        conflict_notes=conflict_notes,
        human_influenced=bool(human_calls),
    )


def _break_tie(
    dimension_results: list[DimensionResult],
    a_won: int,
    b_won: int,
    has_human_calls: bool,
) -> WinnerType:
    """Resolve an equal-means result."""
    if has_human_calls and a_won != b_won:
        return "model_a" if a_won > b_won else "model_b"
    max_a_lead = max((r.score_a - r.score_b) for r in dimension_results)
    max_b_lead = max((r.score_b - r.score_a) for r in dimension_results)
    if max_a_lead > max_b_lead:
        return "model_a"
    if max_b_lead > max_a_lead:
        return "model_b"
    return "draw"


def _classify_margin(overall: str, a_won: int, b_won: int, largest_gap: int) -> MarginType:
    if overall == "draw":
        return "tie"
    winner_dims = max(a_won, b_won)
    loser_dims = min(a_won, b_won)
    if winner_dims >= 4 and largest_gap >= 3:
        return "decisive"
    if winner_dims > loser_dims and largest_gap >= 2:
        return "clear"
    return "narrow"


def _build_conflict_notes(dimension_results: list[DimensionResult], overall: str) -> str | None:
    if overall == "draw":
        return "Overall result is tied after score and largest-lead comparison."
    split_winners = {result.winner for result in dimension_results if result.winner != "draw"}
    if len(split_winners) > 1:
        return "Dimensions split between both models; final winner depends on aggregate score and margin."
    return None
