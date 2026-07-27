"""Winner determination logic tests."""

import pytest
from compare import determine_winner
from schemas import (
    DIMENSIONS,
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


# ---------------------------------------------------------------------------
# Gate 1 human arbitration.
#
# The rule: a human's per-dimension A/B/TIE overrides that dimension's winner
# and never touches a score. Overall winner stays mean-based, except that an
# equal-means result — the case human review exists for — is broken by the
# human's dimension verdicts before falling back to largest-lead.
# ---------------------------------------------------------------------------

from datetime import datetime

from schemas import HilArbitration, HilDimensionArbitration


def _arbitration(**calls: str) -> HilArbitration:
    """Build a gate 1 arbitration from dimension=winner kwargs."""
    now = datetime.now().isoformat()
    return HilArbitration(
        status="completed",
        route_score=0.8,
        route_band="required_hil",
        trigger_reasons=["narrow_margin"],
        review_dimensions=list(calls),
        reviewer="tester",
        created_at=now,
        completed_at=now,
        dimension_arbitrations=[
            HilDimensionArbitration(dimension=d, human_winner=w) for d, w in calls.items()
        ],
    )


def _pair(scores_a: list[int], scores_b: list[int]):
    return (
        _make_eval("A", scores_a),
        _make_eval("B", scores_b),
        RevisedEvaluation(model_a=_make_revised("A", scores_a), model_b=_make_revised("B", scores_b)),
    )


class TestArbitrationOverridesDimensionWinner:
    def test_human_can_award_the_dimension_to_the_lower_score(self):
        """The whole point: the verdict becomes the human's, the scores stay the
        model's — so a dimension can show 7 beating 8."""
        eval_a, eval_b, revised = _pair([7] * 6, [8] * 6)
        result = determine_winner(
            "t", eval_a, eval_b, revised, arbitration=_arbitration(prompt_adherence="A")
        )
        dim = next(r for r in result.dimension_results if r.dimension == "prompt_adherence")
        assert dim.winner == "model_a"
        assert dim.score_a == 7 and dim.score_b == 8  # untouched
        assert dim.human_decided is True
        assert dim.model_winner == "model_b"  # what the scores alone said

    def test_tie_call_makes_the_dimension_a_draw(self):
        eval_a, eval_b, revised = _pair([9] * 6, [5] * 6)
        result = determine_winner(
            "t", eval_a, eval_b, revised, arbitration=_arbitration(photorealism="TIE")
        )
        dim = next(r for r in result.dimension_results if r.dimension == "photorealism")
        assert dim.winner == "draw"
        assert dim.human_decided is True

    def test_unarbitrated_dimensions_keep_the_model_verdict(self):
        eval_a, eval_b, revised = _pair([7] * 6, [8] * 6)
        result = determine_winner(
            "t", eval_a, eval_b, revised, arbitration=_arbitration(prompt_adherence="A")
        )
        others = [r for r in result.dimension_results if r.dimension != "prompt_adherence"]
        assert all(r.winner == "model_b" for r in others)
        assert all(r.human_decided is False for r in others)
        assert all(r.model_winner is None for r in others)

    def test_dimension_counts_follow_the_human(self):
        eval_a, eval_b, revised = _pair([7] * 6, [8] * 6)
        result = determine_winner(
            "t", eval_a, eval_b, revised,
            arbitration=_arbitration(prompt_adherence="A", photorealism="A"),
        )
        assert result.model_a_dimensions_won == 2
        assert result.model_b_dimensions_won == 4

    def test_means_are_never_changed_by_arbitration(self):
        eval_a, eval_b, revised = _pair([7] * 6, [8] * 6)
        plain = determine_winner("t", eval_a, eval_b, revised)
        arbitrated = determine_winner(
            "t", eval_a, eval_b, revised,
            arbitration=_arbitration(**{d: "A" for d in
                ["prompt_adherence", "photorealism", "aesthetic_quality",
                 "composition", "color_accuracy", "creativity"]}),
        )
        assert arbitrated.model_a_mean == plain.model_a_mean == 7.0
        assert arbitrated.model_b_mean == plain.model_b_mean == 8.0
        # Even sweeping every dimension does not flip a clear score gap.
        assert arbitrated.overall_winner == "model_b"

    def test_human_influenced_flag(self):
        eval_a, eval_b, revised = _pair([7] * 6, [8] * 6)
        assert determine_winner("t", eval_a, eval_b, revised).human_influenced is False
        assert determine_winner(
            "t", eval_a, eval_b, revised, arbitration=_arbitration(creativity="A")
        ).human_influenced is True


class TestArbitrationTiebreak:
    def test_human_breaks_an_equal_means_tie(self):
        """Means equal (7.5 each); scores alone would tie on largest lead too."""
        eval_a, eval_b, revised = _pair([8, 7, 8, 7, 8, 7], [7, 8, 7, 8, 7, 8])
        plain = determine_winner("t", eval_a, eval_b, revised)
        assert plain.model_a_mean == plain.model_b_mean
        assert plain.overall_winner == "draw"  # equal leads -> draw without a human

        arbitrated = determine_winner(
            "t", eval_a, eval_b, revised,
            arbitration=_arbitration(photorealism="A", composition="A"),
        )
        assert arbitrated.model_a_mean == arbitrated.model_b_mean  # scores untouched
        assert arbitrated.overall_winner == "model_a"

    def test_falls_back_to_largest_lead_when_human_calls_are_even(self):
        eval_a, eval_b, revised = _pair([8, 7, 8, 7, 8, 7], [7, 8, 7, 8, 7, 8])
        result = determine_winner(
            "t", eval_a, eval_b, revised,
            arbitration=_arbitration(prompt_adherence="A", photorealism="B"),
        )
        # A and B each hold 3 dimensions -> no human majority -> original rule.
        assert result.model_a_dimensions_won == result.model_b_dimensions_won
        assert result.overall_winner == "draw"


class TestArbitrationRobustness:
    def test_no_arbitration_is_byte_identical_to_before(self):
        """Regression guard for every historical run and the legacy path."""
        eval_a, eval_b, revised = _pair([8, 6, 7, 9, 5, 8], [7, 7, 7, 7, 7, 7])
        explicit_none = determine_winner("t", eval_a, eval_b, revised, arbitration=None)
        omitted = determine_winner("t", eval_a, eval_b, revised)
        assert explicit_none.model_dump() == omitted.model_dump()
        assert omitted.human_influenced is False
        assert all(r.human_decided is False for r in omitted.dimension_results)

    def test_empty_arbitration_changes_nothing(self):
        eval_a, eval_b, revised = _pair([7] * 6, [8] * 6)
        baseline = determine_winner("t", eval_a, eval_b, revised)
        empty = determine_winner("t", eval_a, eval_b, revised, arbitration=_arbitration())
        assert empty.model_dump() == baseline.model_dump()

    def test_unknown_winner_token_is_ignored_not_fatal(self):
        """A hand-edited arbitration must not take down the whole run."""
        eval_a, eval_b, revised = _pair([7] * 6, [8] * 6)
        arb = _arbitration(prompt_adherence="A")
        arb.dimension_arbitrations[0].human_winner = "MAYBE"  # bypass validation
        result = determine_winner("t", eval_a, eval_b, revised, arbitration=arb)
        assert result.overall_winner == "model_b"
        assert result.human_influenced is False


# ---------------------------------------------------------------------------
# Gate 2 adjudication: per-dimension selection between the two revisions.
#
# Gate 2 asks which critic was more convincing. The pipeline only ever fed the
# revision model round 2's critique, so an "agree_with_05" answer had nowhere
# to go — these cover the selection that gives it one.
# ---------------------------------------------------------------------------

from compare import apply_gate2_adjudication
from pipeline import PipelineResult, _final_revision
from schemas import HilAdjudication, HilAdjudicationLabel


def _adjudication(**labels: str) -> HilAdjudication:
    """Build a gate 2 adjudication from dimension=label kwargs."""
    now = datetime.now().isoformat()
    return HilAdjudication(
        status="completed",
        trigger_reasons=["critic_disagreement"],
        disagreement_items=[],
        reviewer="tester",
        created_at=now,
        completed_at=now,
        adjudication_labels=[
            HilAdjudicationLabel(dimension=d, label=lab) for d, lab in labels.items()
        ],
    )


def _revision(scores_a: list[int], scores_b: list[int]) -> RevisedEvaluation:
    return RevisedEvaluation(
        model_a=_make_revised("A", scores_a), model_b=_make_revised("B", scores_b)
    )


class TestGate2Selection:
    def test_agree_with_05_restores_that_dimension_from_round_1(self):
        r1 = _revision([9, 8, 8, 8, 8, 8], [6] * 6)
        r2 = _revision([3, 2, 8, 8, 8, 8], [6] * 6)

        final = apply_gate2_adjudication(r1, r2, _adjudication(prompt_adherence="agree_with_05"))

        assert final.model_a.prompt_adherence.score == 9   # from round 1
        assert final.model_a.photorealism.score == 2       # untouched dimension stays round 2
        assert final.model_b.prompt_adherence.score == 6   # both sides selected together

    def test_selection_carries_the_whole_dimension_not_just_the_score(self):
        """No score is invented: reasoning and revision note travel with it, so
        the card text always describes the number beside it."""
        r1, r2 = _revision([9] * 6, [6] * 6), _revision([3] * 6, [6] * 6)
        r1.model_a.prompt_adherence.revision_note = "round 1 note"

        final = apply_gate2_adjudication(r1, r2, _adjudication(prompt_adherence="agree_with_05"))

        assert final.model_a.prompt_adherence.revision_note == "round 1 note"
        assert final.model_a.prompt_adherence.reasoning == "Score 9"

    @pytest.mark.parametrize("label", ["agree_with_07", "both_partially_right"])
    def test_other_labels_keep_round_2(self, label):
        r1, r2 = _revision([9] * 6, [6] * 6), _revision([3] * 6, [6] * 6)
        final = apply_gate2_adjudication(r1, r2, _adjudication(prompt_adherence=label))
        assert final is r2

    def test_no_adjudication_is_a_no_op(self):
        r1, r2 = _revision([9] * 6, [6] * 6), _revision([3] * 6, [6] * 6)
        assert apply_gate2_adjudication(r1, r2, None) is r2

    def test_multiple_dimensions_select_independently(self):
        r1 = _revision([9, 9, 9, 9, 9, 9], [6] * 6)
        r2 = _revision([1, 2, 3, 4, 5, 6], [6] * 6)
        final = apply_gate2_adjudication(
            r1, r2, _adjudication(prompt_adherence="agree_with_05", composition="agree_with_05")
        )
        assert [getattr(final.model_a, d).score for d in DIMENSIONS] == [9, 2, 3, 9, 5, 6]

    def test_selection_can_flip_the_overall_winner(self):
        """End of the chain: the human's call has to reach the verdict, not just
        the stored scores."""
        eval_a, eval_b = _make_eval("A", [8] * 6), _make_eval("B", [6] * 6)
        r1 = _revision([9, 9, 9, 9, 9, 9], [7] * 6)   # A ahead
        r2 = _revision([1, 1, 1, 1, 1, 1], [7] * 6)   # round 2 tanks A

        without = determine_winner("t", eval_a, eval_b, apply_gate2_adjudication(r1, r2, None))
        with_human = determine_winner(
            "t", eval_a, eval_b,
            apply_gate2_adjudication(r1, r2, _adjudication(**{d: "agree_with_05" for d in DIMENSIONS})),
        )

        assert without.overall_winner == "model_b"
        assert with_human.overall_winner == "model_a"


class TestFinalRevisionSelection:
    """_final_revision() is the single definition both orchestrators call."""

    def _result(self, revisions, adjudication=None):
        result = PipelineResult()
        result.revisions = revisions
        result.hil_adjudication = adjudication
        return result

    def test_single_round_has_nothing_to_select_between(self):
        only = _revision([9] * 6, [6] * 6)
        assert _final_revision(self._result([only], _adjudication(prompt_adherence="agree_with_05"))) is only

    def test_without_adjudication_the_last_round_stands(self):
        r1, r2 = _revision([9] * 6, [6] * 6), _revision([3] * 6, [6] * 6)
        assert _final_revision(self._result([r1, r2])) is r2

    def test_adjudication_is_applied(self):
        r1, r2 = _revision([9] * 6, [6] * 6), _revision([3] * 6, [6] * 6)
        final = _final_revision(self._result([r1, r2], _adjudication(prompt_adherence="agree_with_05")))
        assert final.model_a.prompt_adherence.score == 9

    def test_adjudication_stored_as_a_raw_dict_is_coerced(self):
        """app.py hands back model_dump()s; disk reload hands back models. Both
        have to work or the verdict changes on page refresh."""
        r1, r2 = _revision([9] * 6, [6] * 6), _revision([3] * 6, [6] * 6)
        payload = _adjudication(prompt_adherence="agree_with_05").model_dump()
        final = _final_revision(self._result([r1, r2], payload))
        assert final.model_a.prompt_adherence.score == 9

    def test_malformed_adjudication_falls_back_to_round_2(self):
        r1, r2 = _revision([9] * 6, [6] * 6), _revision([3] * 6, [6] * 6)
        assert _final_revision(self._result([r1, r2], {"status": "nonsense"})) is r2
