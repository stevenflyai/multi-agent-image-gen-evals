"""Tests for HIL V2 gate decisions."""

from gates import PHASE1_NO_VARIANCE_WEIGHTS, PHASE2_WITH_VARIANCE_WEIGHTS, build_issue_equivalence_report, evaluate_gate1, evaluate_gate2
from prompts import CRITIQUE_ROUND2_SYSTEM_PROMPT
from schemas import CritiqueResponse, DIMENSIONS, DimensionCritique, DimensionScore, ImageEvaluation, RevisedDimensionScore, RevisedEvaluation, RevisedImageEvaluation


def _make_eval(model_name: str, scores: list[int], confidence: int | None = 4, evidence: str = "Clear visual evidence with concrete objects.") -> ImageEvaluation:
    values = {
        dimension: DimensionScore(
            score=score,
            reasoning=evidence or "Generic reasoning without concrete evidence.",
            evidence=evidence,
            confidence=confidence,
        )
        for dimension, score in zip(DIMENSIONS, scores)
    }
    return ImageEvaluation(model_name=model_name, **values)


def _make_revised(model_name: str, scores: list[int]) -> RevisedImageEvaluation:
    values = {
        dimension: RevisedDimensionScore(
            score=score,
            reasoning="Revised evidence",
            evidence="Concrete revised visual evidence.",
            confidence=4,
            critique_accepted=True,
            revision_note="Updated",
        )
        for dimension, score in zip(DIMENSIONS, scores)
    }
    return RevisedImageEvaluation(model_name=model_name, **values)


def _critique(dimension: str, text: str, score_a: int, score_b: int, round_number: int) -> CritiqueResponse:
    return CritiqueResponse(
        overall_assessment="assessment",
        dimension_critiques=[
            DimensionCritique(
                dimension=dimension,
                original_score_model_a=score_a,
                original_score_model_b=score_b,
                critique=text,
                suggested_score_model_a=score_a,
                suggested_score_model_b=score_b,
            )
        ],
        bias_detection="No systematic bias detected",
        round=round_number,
        critic_model="test",
    )


def test_phase_weight_profiles_sum_to_one():
    assert sum(PHASE1_NO_VARIANCE_WEIGHTS.values()) == 1.0
    assert sum(PHASE2_WITH_VARIANCE_WEIGHTS.values()) == 1.0


def test_gate1_phase1_no_variance_profile():
    eval_a = _make_eval("A", [9, 9, 9, 9, 9, 9], confidence=5)
    eval_b = _make_eval("B", [3, 3, 3, 3, 3, 3], confidence=5)

    decision = evaluate_gate1("simple portrait", eval_a, eval_b, hil_enabled=False)

    assert decision.route_weight_profile == "phase1_no_variance"
    assert decision.route_features.normalized_variance is None


def test_gate1_hil_off_critical_risk_is_flagged():
    eval_a = _make_eval("A", [8, 8, 8, 8, 8, 8], confidence=1, evidence="")
    eval_b = _make_eval("B", [8, 8, 8, 8, 8, 8], confidence=1, evidence="")

    decision = evaluate_gate1("text logo chart with exactly five objects", eval_a, eval_b, hil_enabled=False)

    assert decision.route_band == "required_hil"
    assert decision.pipeline_status == "auto_continued_critical_risk"
    assert decision.requires_attention is True


def test_gate1_hil_enabled_critical_risk_pends():
    eval_a = _make_eval("A", [8, 8, 8, 8, 8, 8], confidence=1, evidence="")
    eval_b = _make_eval("B", [8, 8, 8, 8, 8, 8], confidence=1, evidence="")

    decision = evaluate_gate1("text logo chart with exactly five objects", eval_a, eval_b, hil_enabled=True)

    assert decision.status == "pending"
    assert decision.route_band == "required_hil"
    assert decision.pipeline_status == "pending_hil_gate1"
    assert decision.requires_attention is True


def test_issue_equivalence_checker_classifies_equivalent_issue():
    critique_r1 = _critique("prompt_adherence", "The text requirement is missing and should reduce the prompt adherence score.", 8, 6, 1)
    critique_r2 = _critique("prompt_adherence", "The revised evaluation still ignores missing text in the prompt adherence judgment.", 8, 6, 2)

    report = build_issue_equivalence_report(critique_r1, critique_r2)

    assert report.summary.equivalent_count == 1
    assert report.equivalence_results[0].classification == "equivalent"


def test_gate2_detects_winner_flip_and_equivalent_issue():
    initial_a = _make_eval("A", [8, 8, 8, 8, 8, 8])
    initial_b = _make_eval("B", [6, 6, 6, 6, 6, 6])
    revised = RevisedEvaluation(
        model_a=_make_revised("A", [5, 5, 5, 5, 5, 5]),
        model_b=_make_revised("B", [8, 8, 8, 8, 8, 8]),
    )
    critique_r1 = _critique("prompt_adherence", "The text requirement is missing and should reduce the score.", 8, 6, 1)
    critique_r2 = _critique("prompt_adherence", "The revised evaluation still ignores the missing text requirement.", 8, 6, 2)

    decision, report = evaluate_gate2(initial_a, initial_b, revised, critique_r1, critique_r2, hil_enabled=False)

    assert decision.status == "skipped"
    assert "winner_flip" in decision.trigger_reasons
    assert report.summary.equivalent_count == 1


def test_gate2_hil_enabled_ignores_new_issue_without_major_change():
    initial_a = _make_eval("A", [8, 8, 8, 8, 8, 8])
    initial_b = _make_eval("B", [6, 6, 6, 6, 6, 6])
    revised = RevisedEvaluation(
        model_a=_make_revised("A", [8, 8, 8, 8, 8, 8]),
        model_b=_make_revised("B", [6, 6, 6, 6, 6, 6]),
    )
    critique_r1 = _critique("prompt_adherence", "The text requirement is handled well.", 8, 6, 1)
    critique_r2 = _critique("composition", "The layout has a minor visual balance issue.", 8, 6, 2)

    decision, report = evaluate_gate2(initial_a, initial_b, revised, critique_r1, critique_r2, hil_enabled=True)

    assert report.summary.new_count == 1
    assert decision.status == "not_required"
    assert decision.trigger_reasons == []


def test_gate2_hil_enabled_pends_on_major_change():
    initial_a = _make_eval("A", [8, 8, 8, 8, 8, 8])
    initial_b = _make_eval("B", [6, 6, 6, 6, 6, 6])
    revised = RevisedEvaluation(
        model_a=_make_revised("A", [5, 5, 5, 5, 5, 5]),
        model_b=_make_revised("B", [8, 8, 8, 8, 8, 8]),
    )
    critique_r1 = _critique("prompt_adherence", "The text requirement strongly favors A.", 8, 6, 1)
    critique_r2 = _critique("prompt_adherence", "The text requirement strongly favors B.", 6, 8, 2)

    decision, _report = evaluate_gate2(initial_a, initial_b, revised, critique_r1, critique_r2, hil_enabled=True)

    assert decision.status == "pending"
    assert decision.pipeline_status == "pending_hil_gate2"
    assert "winner_flip" in decision.trigger_reasons
    assert "large_score_delta" in decision.trigger_reasons


def test_gate2_review_dimensions_exclude_unchanged_new_issue():
    initial_a = _make_eval("A", [8, 8, 8, 8, 8, 8])
    initial_b = _make_eval("B", [6, 7, 6, 6, 6, 6])
    revised = RevisedEvaluation(
        model_a=_make_revised("A", [5, 8, 8, 8, 8, 8]),
        model_b=_make_revised("B", [8, 7, 6, 6, 6, 6]),
    )
    critique_r1 = CritiqueResponse(
        overall_assessment="assessment",
        dimension_critiques=[
            DimensionCritique(
                dimension="prompt_adherence",
                original_score_model_a=8,
                original_score_model_b=6,
                critique="Prompt adherence score should flip toward B.",
                suggested_score_model_a=5,
                suggested_score_model_b=8,
            ),
            DimensionCritique(
                dimension="photorealism",
                original_score_model_a=8,
                original_score_model_b=7,
                critique="Photorealism scores are defensible.",
                suggested_score_model_a=8,
                suggested_score_model_b=7,
            ),
        ],
        bias_detection="No systematic bias detected",
    )
    critique_r2 = CritiqueResponse(
        overall_assessment="assessment",
        dimension_critiques=[
            DimensionCritique(
                dimension="prompt_adherence",
                original_score_model_a=5,
                original_score_model_b=8,
                critique="Prompt adherence still materially favors B.",
                suggested_score_model_a=5,
                suggested_score_model_b=8,
            ),
            DimensionCritique(
                dimension="photorealism",
                original_score_model_a=8,
                original_score_model_b=7,
                critique="This is a new note, but both scores remain reasonable.",
                suggested_score_model_a=8,
                suggested_score_model_b=7,
            ),
        ],
        bias_detection="No systematic bias detected",
        round=2,
    )

    decision, report = evaluate_gate2(initial_a, initial_b, revised, critique_r1, critique_r2, hil_enabled=True)

    assert decision.status == "pending"
    assert "prompt_adherence" in decision.review_dimensions
    assert "photorealism" not in decision.review_dimensions
    assert any(item.dimension == "photorealism" and item.classification == "new" for item in report.equivalence_results)


def test_round2_prompt_does_not_reference_round1_critique():
    forbidden = ["previous reviewer", "round 1 critique", "GPT-5.4", "first review"]
    prompt = CRITIQUE_ROUND2_SYSTEM_PROMPT.lower()
    for phrase in forbidden:
        assert phrase.lower() not in prompt