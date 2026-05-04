"""Pydantic schemas for the image evaluation pipeline.

All LLM calls return structured JSON validated against these models.
Six evaluation dimensions with 1-10 scores and anchored rubrics.
"""

from typing import Literal

from pydantic import BaseModel, Field


DIMENSIONS = [
    "prompt_adherence",
    "photorealism",
    "aesthetic_quality",
    "composition",
    "color_accuracy",
    "creativity",
]

DimensionName = Literal[
    "prompt_adherence", "photorealism", "aesthetic_quality",
    "composition", "color_accuracy", "creativity",
]

WinnerType = Literal["model_a", "model_b", "draw"]
ABWinnerType = Literal["A", "B", "TIE"]
PromptDifficulty = Literal["easy", "medium", "hard"]
MarginType = Literal["decisive", "clear", "narrow", "tie"]
PipelineStatus = Literal[
    "completed",
    "pending_hil_gate1",
    "pending_hil_gate2",
    "auto_continued_high_risk",
    "auto_continued_critical_risk",
    "failed",
    "partial",
]
GateStatus = Literal["not_required", "recommended", "pending", "completed", "skipped"]
RouteBand = Literal["none", "soft_hil", "strong_hil", "required_hil"]
IssueClassification = Literal["equivalent", "new", "contradicting"]
AdjudicationLabel = Literal["agree_with_05", "agree_with_07", "both_partially_right"]


class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=10)
    reasoning: str = Field(min_length=1)
    reasoning_zh: str = ""
    evidence: str = ""
    confidence: int | None = Field(default=None, ge=1, le=5)


class ImageEvaluation(BaseModel):
    """Evaluation of a single image across all 6 dimensions."""

    model_name: str
    prompt_adherence: DimensionScore
    photorealism: DimensionScore
    aesthetic_quality: DimensionScore
    composition: DimensionScore
    color_accuracy: DimensionScore
    creativity: DimensionScore

    def mean_score(self) -> float:
        scores = [getattr(self, d).score for d in DIMENSIONS]
        return sum(scores) / len(scores)

    def scores_dict(self) -> dict[str, int]:
        return {d: getattr(self, d).score for d in DIMENSIONS}


class DimensionCritique(BaseModel):
    dimension: DimensionName
    original_score_model_a: int = Field(ge=1, le=10)
    original_score_model_b: int = Field(ge=1, le=10)
    critique: str = Field(min_length=1)
    suggested_score_model_a: int = Field(ge=1, le=10)
    suggested_score_model_b: int = Field(ge=1, le=10)


class CritiqueResponse(BaseModel):
    """A reviewer's critique of an evaluation (GPT-5.4 round 1, Gemini 3.1 Pro round 2)."""

    overall_assessment: str
    dimension_critiques: list[DimensionCritique]
    bias_detection: str
    round: int = 1
    critic_model: str = ""


class RevisedDimensionScore(BaseModel):
    score: int = Field(ge=1, le=10)
    reasoning: str = Field(min_length=1)
    reasoning_zh: str = ""
    evidence: str = ""
    confidence: int | None = Field(default=None, ge=1, le=5)
    critique_accepted: bool
    revision_note: str


class RevisedImageEvaluation(BaseModel):
    """Revised evaluation after considering GPT-5.4's critique."""

    model_name: str
    prompt_adherence: RevisedDimensionScore
    photorealism: RevisedDimensionScore
    aesthetic_quality: RevisedDimensionScore
    composition: RevisedDimensionScore
    color_accuracy: RevisedDimensionScore
    creativity: RevisedDimensionScore

    def mean_score(self) -> float:
        scores = [getattr(self, d).score for d in DIMENSIONS]
        return sum(scores) / len(scores)

    def scores_dict(self) -> dict[str, int]:
        return {d: getattr(self, d).score for d in DIMENSIONS}


class RevisedEvaluation(BaseModel):
    """Full revised evaluation for both models."""

    model_a: RevisedImageEvaluation
    model_b: RevisedImageEvaluation


class DimensionResult(BaseModel):
    dimension: DimensionName
    score_a: int
    score_b: int
    pre_critique_score_a: int
    pre_critique_score_b: int
    winner: WinnerType


class ComparisonResult(BaseModel):
    """Final comparison result with winner determination."""

    prompt: str
    model_a_name: str
    model_b_name: str
    dimension_results: list[DimensionResult]
    overall_winner: WinnerType
    model_a_mean: float
    model_b_mean: float
    model_a_dimensions_won: int
    model_b_dimensions_won: int
    margin: MarginType | None = None
    conflict_notes: str | None = None
    human_influenced: bool = False


class RouteFeatures(BaseModel):
    normalized_self_confidence_risk: float = Field(ge=0, le=1)
    normalized_variance: float | None = Field(default=None, ge=0, le=1)
    normalized_margin_risk: float = Field(ge=0, le=1)
    cross_dim_conflict_score: float = Field(ge=0, le=1)
    difficulty_prior: float = Field(ge=0, le=1)
    evidence_quality_risk: float = Field(ge=0, le=1)


class Gate1ReviewItem(BaseModel):
    dimension: DimensionName
    uncertainty_reasons: list[str]
    model_marked_by_step: str = "04_evaluation_agent"
    initial_winner: ABWinnerType


class GateDecision(BaseModel):
    gate: Literal["gate1_uncertainty_risk_router", "gate2_disagreement_detector"]
    status: GateStatus
    route_score: float | None = Field(default=None, ge=0, le=1)
    route_band: RouteBand | None = None
    trigger_reasons: list[str] = Field(default_factory=list)
    route_features: RouteFeatures | None = None
    route_weight_profile: str | None = None
    review_dimensions: list[DimensionName] = Field(default_factory=list)
    review_items: list[Gate1ReviewItem] = Field(default_factory=list)
    pipeline_status: PipelineStatus | None = None
    requires_attention: bool = False


class HilDimensionArbitration(BaseModel):
    dimension: DimensionName
    human_winner: ABWinnerType


class HilArbitration(BaseModel):
    gate: Literal["gate1_uncertainty_risk_router"] = "gate1_uncertainty_risk_router"
    status: GateStatus
    route_score: float = Field(ge=0, le=1)
    route_band: RouteBand
    trigger_reasons: list[str]
    review_dimensions: list[DimensionName]
    reviewer: str
    created_at: str
    completed_at: str | None = None
    dimension_arbitrations: list[HilDimensionArbitration] = Field(default_factory=list)


class DisagreementItem(BaseModel):
    dimension: DimensionName
    issue_from_05: str = ""
    change_from_06: str = ""
    issue_from_07: str = ""


class HilAdjudicationLabel(BaseModel):
    dimension: DimensionName
    label: AdjudicationLabel


class HilAdjudication(BaseModel):
    gate: Literal["gate2_disagreement_detector"] = "gate2_disagreement_detector"
    status: GateStatus
    trigger_reasons: list[str]
    disagreement_items: list[DisagreementItem]
    reviewer: str
    created_at: str
    completed_at: str | None = None
    adjudication_labels: list[HilAdjudicationLabel] = Field(default_factory=list)


class IssueEquivalenceResult(BaseModel):
    issue_id_07: str
    classification: IssueClassification
    matched_issue_id_05: str | None = None
    dimension: DimensionName
    rationale: str
    gate2_trigger: bool


class IssueEquivalenceSummary(BaseModel):
    equivalent_count: int = 0
    new_count: int = 0
    contradicting_count: int = 0
    should_trigger_gate2: bool = False


class IssueEquivalenceReport(BaseModel):
    equivalence_results: list[IssueEquivalenceResult]
    summary: IssueEquivalenceSummary


class PromptInputMetadata(BaseModel):
    step: str
    included_artifacts: list[str]
    excluded_artifacts: list[str]
