"""Pydantic schemas for the image evaluation pipeline.

All LLM calls return structured JSON validated against these models.
Six evaluation dimensions with 1-10 scores and anchored rubrics.
"""

from pydantic import BaseModel, Field


DIMENSIONS = [
    "prompt_adherence",
    "photorealism",
    "aesthetic_quality",
    "composition",
    "color_accuracy",
    "creativity",
]


class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=10)
    reasoning: str = Field(min_length=1)


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
    dimension: str
    original_score_model_a: int = Field(ge=1, le=10)
    original_score_model_b: int = Field(ge=1, le=10)
    critique: str = Field(min_length=1)
    suggested_score_model_a: int = Field(ge=1, le=10)
    suggested_score_model_b: int = Field(ge=1, le=10)


class CritiqueResponse(BaseModel):
    """GPT-5.4's review of Claude Opus's initial evaluation."""

    overall_assessment: str
    dimension_critiques: list[DimensionCritique]
    bias_detection: str


class RevisedDimensionScore(BaseModel):
    score: int = Field(ge=1, le=10)
    reasoning: str = Field(min_length=1)
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
    dimension: str
    score_a: int
    score_b: int
    pre_critique_score_a: int
    pre_critique_score_b: int
    winner: str  # "model_a", "model_b", or "draw"


class ComparisonResult(BaseModel):
    """Final comparison result with winner determination."""

    prompt: str
    model_a_name: str
    model_b_name: str
    dimension_results: list[DimensionResult]
    overall_winner: str  # "model_a", "model_b", or "draw"
    model_a_mean: float
    model_b_mean: float
    model_a_dimensions_won: int
    model_b_dimensions_won: int
