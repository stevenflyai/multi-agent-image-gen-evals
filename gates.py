"""Gate decision logic for HIL V2.

The gates are deterministic by default so Phase 2 can run with HIL disabled
and produce auditable artifacts without adding another network dependency.
"""

from __future__ import annotations

import re
from statistics import median

from schemas import (
    ABWinnerType,
    CritiqueResponse,
    DIMENSIONS,
    DimensionName,
    Gate1ReviewItem,
    GateDecision,
    ImageEvaluation,
    IssueEquivalenceReport,
    IssueEquivalenceResult,
    IssueEquivalenceSummary,
    RevisedEvaluation,
    RouteBand,
    RouteFeatures,
)

PHASE1_NO_VARIANCE_WEIGHTS = {
    "self_confidence_risk": 0.20,
    "margin_risk": 0.27,
    "cross_dim_conflict": 0.20,
    "difficulty_prior": 0.20,
    "evidence_quality_risk": 0.13,
}

PHASE2_WITH_VARIANCE_WEIGHTS = {
    "self_confidence_risk": 0.15,
    "variance": 0.25,
    "margin_risk": 0.20,
    "cross_dim_conflict": 0.15,
    "difficulty_prior": 0.15,
    "evidence_quality_risk": 0.10,
}

DIFFICULTY_KEYWORDS = {
    "text",
    "logo",
    "sign",
    "chart",
    "diagram",
    "ui",
    "interface",
    "exactly",
    "count",
    "five",
    "six",
    "seven",
    "negative",
    "without",
    "behind",
    "between",
    "left",
    "right",
    "multi-panel",
    "occlusion",
}


def evaluate_gate1(
    prompt: str,
    eval_a: ImageEvaluation,
    eval_b: ImageEvaluation,
    *,
    hil_enabled: bool = False,
    variance: float | None = None,
) -> GateDecision:
    """Compute the Gate 1 uncertainty/risk decision."""
    route_features = _build_route_features(prompt, eval_a, eval_b, variance)
    route_score, weight_profile = _route_score(route_features)
    route_band = _route_band(route_score)
    trigger_reasons = _trigger_reasons(route_features, route_band)
    review_items = _review_items(eval_a, eval_b, route_features)

    if route_band == "none":
        status = "not_required"
        pipeline_status = None
        requires_attention = False
    elif route_band == "soft_hil":
        status = "recommended"
        pipeline_status = None
        requires_attention = False
    elif hil_enabled:
        status = "pending"
        pipeline_status = "pending_hil_gate1"
        requires_attention = True
    elif route_band == "strong_hil":
        status = "skipped"
        pipeline_status = "auto_continued_high_risk"
        requires_attention = True
    else:
        status = "skipped"
        pipeline_status = "auto_continued_critical_risk"
        requires_attention = True

    return GateDecision(
        gate="gate1_uncertainty_risk_router",
        status=status,
        route_score=route_score,
        route_band=route_band,
        trigger_reasons=trigger_reasons,
        route_features=route_features,
        route_weight_profile=weight_profile,
        review_dimensions=[item.dimension for item in review_items],
        review_items=review_items,
        pipeline_status=pipeline_status,
        requires_attention=requires_attention,
    )


def evaluate_gate2(
    initial_a: ImageEvaluation,
    initial_b: ImageEvaluation,
    revised_r1: RevisedEvaluation,
    critique_r1: CritiqueResponse,
    critique_r2: CritiqueResponse,
    *,
    hil_enabled: bool = False,
) -> tuple[GateDecision, IssueEquivalenceReport]:
    """Compute Gate 2 disagreement status after 05/06/07."""
    equivalence = build_issue_equivalence_report(critique_r1, critique_r2)
    trigger_reasons: list[str] = []
    review_dimensions: set[DimensionName] = set()

    if equivalence.summary.contradicting_count:
        trigger_reasons.append("critic_disagreement")
        review_dimensions.update(item.dimension for item in equivalence.equivalence_results if item.classification == "contradicting")
    if _overall_winner(initial_a, initial_b) != _overall_winner(revised_r1.model_a, revised_r1.model_b):
        trigger_reasons.append("winner_flip")
        review_dimensions.update(_winner_changed_dimensions(initial_a, initial_b, revised_r1))
    if _has_large_score_delta(initial_a, initial_b, revised_r1):
        trigger_reasons.append("large_score_delta")
        review_dimensions.update(_large_score_delta_dimensions(initial_a, initial_b, revised_r1))

    review_dimensions.update(_critic_score_disagreement_dimensions(critique_r1, critique_r2))

    status = "pending" if trigger_reasons and review_dimensions and hil_enabled else "skipped" if trigger_reasons else "not_required"
    pipeline_status = "pending_hil_gate2" if status == "pending" else None

    decision = GateDecision(
        gate="gate2_disagreement_detector",
        status=status,
        trigger_reasons=trigger_reasons,
        review_dimensions=sorted(review_dimensions),
        pipeline_status=pipeline_status,
        requires_attention=status == "pending",
    )
    return decision, equivalence


def build_issue_equivalence_report(
    critique_r1: CritiqueResponse,
    critique_r2: CritiqueResponse,
) -> IssueEquivalenceReport:
    """Classify 07 critique issues relative to 05 using a local fallback checker."""
    results: list[IssueEquivalenceResult] = []
    equivalent_count = 0
    new_count = 0
    contradicting_count = 0

    for index_07, issue_07 in enumerate(critique_r2.dimension_critiques, start=1):
        issue_id_07 = f"07-{issue_07.dimension}-{index_07}"
        matches = [
            (index_05, issue_05)
            for index_05, issue_05 in enumerate(critique_r1.dimension_critiques, start=1)
            if issue_05.dimension == issue_07.dimension
        ]
        classification = "new"
        matched_issue_id_05 = None
        rationale = "No semantically similar 05 issue found in the same dimension."

        for index_05, issue_05 in matches:
            similarity = _text_similarity(issue_05.critique, issue_07.critique)
            score_direction_05 = _score_direction(issue_05.suggested_score_model_a, issue_05.suggested_score_model_b)
            score_direction_07 = _score_direction(issue_07.suggested_score_model_a, issue_07.suggested_score_model_b)
            if score_direction_05 != score_direction_07 and similarity >= 0.18:
                classification = "contradicting"
                matched_issue_id_05 = f"05-{issue_05.dimension}-{index_05}"
                rationale = "Same dimension with similar critique terms but opposite suggested winner direction."
                break
            if similarity >= 0.22:
                classification = "equivalent"
                matched_issue_id_05 = f"05-{issue_05.dimension}-{index_05}"
                rationale = "Same dimension with overlapping critique semantics."
                break

        if classification == "equivalent":
            equivalent_count += 1
        elif classification == "contradicting":
            contradicting_count += 1
        else:
            new_count += 1

        results.append(
            IssueEquivalenceResult(
                issue_id_07=issue_id_07,
                classification=classification,
                matched_issue_id_05=matched_issue_id_05,
                dimension=issue_07.dimension,
                rationale=rationale,
                gate2_trigger=True,
            )
        )

    summary = IssueEquivalenceSummary(
        equivalent_count=equivalent_count,
        new_count=new_count,
        contradicting_count=contradicting_count,
        should_trigger_gate2=bool(results),
    )
    return IssueEquivalenceReport(equivalence_results=results, summary=summary)


def _build_route_features(
    prompt: str,
    eval_a: ImageEvaluation,
    eval_b: ImageEvaluation,
    variance: float | None,
) -> RouteFeatures:
    return RouteFeatures(
        normalized_self_confidence_risk=_confidence_risk(eval_a, eval_b),
        normalized_variance=variance,
        normalized_margin_risk=_margin_risk(eval_a, eval_b),
        cross_dim_conflict_score=_conflict_score(eval_a, eval_b),
        difficulty_prior=_difficulty_prior(prompt),
        evidence_quality_risk=_evidence_quality_risk(eval_a, eval_b),
    )


def _route_score(features: RouteFeatures) -> tuple[float, str]:
    if features.normalized_variance is None:
        weights = PHASE1_NO_VARIANCE_WEIGHTS
        score = (
            weights["self_confidence_risk"] * features.normalized_self_confidence_risk
            + weights["margin_risk"] * features.normalized_margin_risk
            + weights["cross_dim_conflict"] * features.cross_dim_conflict_score
            + weights["difficulty_prior"] * features.difficulty_prior
            + weights["evidence_quality_risk"] * features.evidence_quality_risk
        )
        return round(score, 4), "phase1_no_variance"

    weights = PHASE2_WITH_VARIANCE_WEIGHTS
    score = (
        weights["self_confidence_risk"] * features.normalized_self_confidence_risk
        + weights["variance"] * features.normalized_variance
        + weights["margin_risk"] * features.normalized_margin_risk
        + weights["cross_dim_conflict"] * features.cross_dim_conflict_score
        + weights["difficulty_prior"] * features.difficulty_prior
        + weights["evidence_quality_risk"] * features.evidence_quality_risk
    )
    return round(score, 4), "phase2_with_variance"


def _route_band(score: float) -> RouteBand:
    if score < 0.35:
        return "none"
    if score < 0.55:
        return "soft_hil"
    if score < 0.70:
        return "strong_hil"
    return "required_hil"


def _trigger_reasons(features: RouteFeatures, route_band: RouteBand) -> list[str]:
    if route_band == "none":
        return []
    reasons = []
    if features.normalized_self_confidence_risk >= 0.45:
        reasons.append("low_self_confidence")
    if features.normalized_variance is not None and features.normalized_variance >= 0.45:
        reasons.append("high_sample_variance")
    if features.normalized_margin_risk >= 0.55:
        reasons.append("narrow_margin")
    if features.cross_dim_conflict_score >= 0.45:
        reasons.append("cross_dimension_conflict")
    if features.difficulty_prior >= 0.45:
        reasons.append("hard_prompt")
    if features.evidence_quality_risk >= 0.45:
        reasons.append("missing_or_weak_evidence")
    return reasons or ["composite_route_score"]


def _review_items(
    eval_a: ImageEvaluation,
    eval_b: ImageEvaluation,
    features: RouteFeatures,
) -> list[Gate1ReviewItem]:
    items: list[Gate1ReviewItem] = []
    for dim in DIMENSIONS:
        score_a = getattr(eval_a, dim).score
        score_b = getattr(eval_b, dim).score
        conf_a = getattr(eval_a, dim).confidence
        conf_b = getattr(eval_b, dim).confidence
        evidence_a = _raw_evidence_text(getattr(eval_a, dim))
        evidence_b = _raw_evidence_text(getattr(eval_b, dim))

        reasons = []
        if abs(score_a - score_b) <= 1:
            reasons.append("narrow_margin")
        if (conf_a is not None and conf_a <= 2) or (conf_b is not None and conf_b <= 2):
            reasons.append("low_self_confidence")
        if _is_weak_evidence(evidence_a) or _is_weak_evidence(evidence_b):
            reasons.append("missing_or_weak_evidence")
        if features.difficulty_prior >= 0.65 and dim == "prompt_adherence":
            reasons.append("hard_prompt")
        if reasons:
            items.append(
                Gate1ReviewItem(
                    dimension=dim,
                    uncertainty_reasons=sorted(set(reasons)),
                    initial_winner=_ab_winner(score_a, score_b),
                )
            )
    return items


def _confidence_risk(eval_a: ImageEvaluation, eval_b: ImageEvaluation) -> float:
    confidences = [
        getattr(ev, dim).confidence
        for ev in (eval_a, eval_b)
        for dim in DIMENSIONS
        if getattr(ev, dim).confidence is not None
    ]
    if not confidences:
        return 0.5
    average = sum(confidences) / len(confidences)
    return round(1 - (average / 5), 4)


def _margin_risk(eval_a: ImageEvaluation, eval_b: ImageEvaluation) -> float:
    diffs = [abs(getattr(eval_a, dim).score - getattr(eval_b, dim).score) for dim in DIMENSIONS]
    mean_margin = abs(eval_a.mean_score() - eval_b.mean_score())
    prompt_margin = abs(eval_a.prompt_adherence.score - eval_b.prompt_adherence.score)
    normalized_margin = min(1.0, min(mean_margin, prompt_margin, median(diffs)) / 3)
    return round(1 - normalized_margin, 4)


def _conflict_score(eval_a: ImageEvaluation, eval_b: ImageEvaluation) -> float:
    conflicts = 0
    total = 0
    for ev in (eval_a, eval_b):
        for dim in DIMENSIONS:
            total += 1
            score = getattr(ev, dim).score
            raw_evidence = _raw_evidence_text(getattr(ev, dim))
            text = _evidence_text(getattr(ev, dim)).lower()
            if score >= 8 and (_is_weak_evidence(raw_evidence) or any(word in text for word in ("artifact", "missing", "distorted", "wrong", "fails"))):
                conflicts += 1
    split_wins = {_ab_winner(getattr(eval_a, dim).score, getattr(eval_b, dim).score) for dim in DIMENSIONS}
    if "A" in split_wins and "B" in split_wins:
        conflicts += 1
        total += 1
    return round(conflicts / total if total else 0, 4)


def _difficulty_prior(prompt: str) -> float:
    lowered = prompt.lower()
    hits = sum(1 for keyword in DIFFICULTY_KEYWORDS if keyword in lowered)
    comma_complexity = min(lowered.count(",") / 8, 0.3)
    return round(min(1.0, hits / 8 + comma_complexity), 4)


def _evidence_quality_risk(eval_a: ImageEvaluation, eval_b: ImageEvaluation) -> float:
    weak = 0
    total = 0
    for ev in (eval_a, eval_b):
        for dim in DIMENSIONS:
            total += 1
            if _is_weak_evidence(_raw_evidence_text(getattr(ev, dim))):
                weak += 1
    return round(weak / total if total else 0, 4)


def _is_weak_evidence(text: str) -> bool:
    stripped = text.strip().lower()
    return not stripped or len(stripped) < 24 or stripped in {"n/a", "none", "good", "bad", "ok"}


def _evidence_text(score_obj: object) -> str:
    evidence = getattr(score_obj, "evidence", "") or ""
    reasoning = getattr(score_obj, "reasoning", "") or ""
    return evidence or reasoning


def _raw_evidence_text(score_obj: object) -> str:
    return getattr(score_obj, "evidence", "") or ""


def _ab_winner(score_a: int, score_b: int) -> ABWinnerType:
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return "TIE"


def _overall_winner(eval_a: ImageEvaluation | object, eval_b: ImageEvaluation | object) -> ABWinnerType:
    mean_a = eval_a.mean_score()
    mean_b = eval_b.mean_score()
    return _ab_winner(mean_a, mean_b)


def _has_large_score_delta(
    initial_a: ImageEvaluation,
    initial_b: ImageEvaluation,
    revised: RevisedEvaluation,
) -> bool:
    return bool(_large_score_delta_dimensions(initial_a, initial_b, revised))


def _large_score_delta_dimensions(
    initial_a: ImageEvaluation,
    initial_b: ImageEvaluation,
    revised: RevisedEvaluation,
) -> list[DimensionName]:
    dimensions: list[DimensionName] = []
    for dim in DIMENSIONS:
        if abs(getattr(revised.model_a, dim).score - getattr(initial_a, dim).score) >= 3:
            dimensions.append(dim)
            continue
        if abs(getattr(revised.model_b, dim).score - getattr(initial_b, dim).score) >= 3:
            dimensions.append(dim)
    return dimensions


def _winner_changed_dimensions(
    initial_a: ImageEvaluation,
    initial_b: ImageEvaluation,
    revised: RevisedEvaluation,
) -> list[DimensionName]:
    dimensions: list[DimensionName] = []
    for dim in DIMENSIONS:
        initial_winner = _ab_winner(getattr(initial_a, dim).score, getattr(initial_b, dim).score)
        revised_winner = _ab_winner(getattr(revised.model_a, dim).score, getattr(revised.model_b, dim).score)
        if initial_winner != revised_winner:
            dimensions.append(dim)
    return dimensions


def _critic_score_disagreement_dimensions(
    critique_r1: CritiqueResponse,
    critique_r2: CritiqueResponse,
) -> list[DimensionName]:
    dimensions: set[DimensionName] = set()
    round1_by_dim = {item.dimension: item for item in critique_r1.dimension_critiques}
    for issue_07 in critique_r2.dimension_critiques:
        issue_05 = round1_by_dim.get(issue_07.dimension)
        if not issue_05:
            if _critique_changes_score(issue_07):
                dimensions.add(issue_07.dimension)
            continue
        if _score_direction(issue_05.suggested_score_model_a, issue_05.suggested_score_model_b) != _score_direction(issue_07.suggested_score_model_a, issue_07.suggested_score_model_b):
            dimensions.add(issue_07.dimension)
            continue
        if abs(issue_05.suggested_score_model_a - issue_07.suggested_score_model_a) >= 2:
            dimensions.add(issue_07.dimension)
            continue
        if abs(issue_05.suggested_score_model_b - issue_07.suggested_score_model_b) >= 2:
            dimensions.add(issue_07.dimension)
    return sorted(dimensions)


def _critique_changes_score(issue: DimensionCritique) -> bool:
    return (
        issue.original_score_model_a != issue.suggested_score_model_a
        or issue.original_score_model_b != issue.suggested_score_model_b
    )


def _score_direction(score_a: int, score_b: int) -> ABWinnerType:
    return _ab_winner(score_a, score_b)


def _text_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(re.findall(r"[a-zA-Z][a-zA-Z_]{2,}", text_a.lower()))
    tokens_b = set(re.findall(r"[a-zA-Z][a-zA-Z_]{2,}", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)