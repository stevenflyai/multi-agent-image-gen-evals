"""Pipeline orchestrator wiring all steps together.

Runs the full evaluation pipeline and persists all intermediate
JSON to the runs/ directory for inspection and caching.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from compare import determine_winner
from critique import critique_evaluation
from evaluate import evaluate_images
from generate import generate_images
from i18n import t
from revise import revise_evaluation
from schemas import (
    ComparisonResult,
    CritiqueResponse,
    ImageEvaluation,
    RevisedEvaluation,
)


class PipelineResult:
    """Container for all pipeline outputs."""

    def __init__(self) -> None:
        self.prompt: str = ""
        self.run_dir: Path | None = None
        self.image_paths: dict[str, Path | Exception] = {}
        self.eval_a: ImageEvaluation | None = None
        self.eval_b: ImageEvaluation | None = None
        self.critique: CritiqueResponse | None = None
        self.revised: RevisedEvaluation | None = None
        self.comparison: ComparisonResult | None = None
        self.errors: list[str] = []
        self.timestamp: str = ""


def run_pipeline(
    prompt: str,
    runs_dir: Path = Path("runs"),
    on_stage: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run the full evaluation pipeline.

    Args:
        prompt: The image generation prompt.
        runs_dir: Base directory for run outputs.
        on_stage: Optional callback for stage progress updates.
    """
    result = PipelineResult()
    result.prompt = prompt
    result.timestamp = datetime.now().isoformat()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    result.run_dir = run_dir

    # Save prompt
    (run_dir / "prompt.txt").write_text(prompt)

    def notify(stage: str) -> None:
        if on_stage:
            on_stage(stage)

    # Step 1: Generate images
    notify(t("stage_generating"))
    result.image_paths = generate_images(prompt, run_dir)

    gpt_path = result.image_paths.get("gpt_image_2")
    gemini_path = result.image_paths.get("gemini_3_pro")

    if isinstance(gpt_path, Exception):
        result.errors.append(f"GPT Image-2 failed: {gpt_path}")
        gpt_path = None
    if isinstance(gemini_path, Exception):
        result.errors.append(f"Gemini 3 Pro failed: {gemini_path}")
        gemini_path = None

    if not gpt_path or not gemini_path:
        _save_result(result)
        return result

    # Step 2: Evaluate
    notify(t("stage_evaluating"))
    try:
        result.eval_a, result.eval_b = evaluate_images(prompt, gpt_path, gemini_path)
        (run_dir / "evaluation.json").write_text(
            json.dumps({"model_a": result.eval_a.model_dump(), "model_b": result.eval_b.model_dump()}, indent=2)
        )
    except Exception as e:
        result.errors.append(f"Evaluation failed: {e}")
        _save_result(result)
        return result

    # Step 3: Critique
    notify(t("stage_critique"))
    try:
        result.critique = critique_evaluation(
            prompt, result.eval_a, result.eval_b, gpt_path, gemini_path
        )
        (run_dir / "critique.json").write_text(json.dumps(result.critique.model_dump(), indent=2))
    except Exception as e:
        result.errors.append(f"Critique failed: {e}")
        # Degrade gracefully: use initial scores
        result.revised = None
        _build_comparison_from_initial(result)
        _save_result(result)
        return result

    # Step 4: Revise
    notify(t("stage_revising"))
    try:
        result.revised = revise_evaluation(
            prompt, result.eval_a, result.eval_b, result.critique, gpt_path, gemini_path
        )
        (run_dir / "revised.json").write_text(json.dumps(result.revised.model_dump(), indent=2))
    except Exception as e:
        result.errors.append(f"Revision failed: {e}")
        _build_comparison_from_initial(result)
        _save_result(result)
        return result

    # Step 5: Compare
    notify(t("stage_complete"))
    result.comparison = determine_winner(prompt, result.eval_a, result.eval_b, result.revised)
    (run_dir / "comparison.json").write_text(json.dumps(result.comparison.model_dump(), indent=2))

    _save_result(result)
    return result


def _build_comparison_from_initial(result: PipelineResult) -> None:
    """Build comparison using initial evaluation scores when critique/revision fails."""
    if result.eval_a and result.eval_b:
        from schemas import RevisedEvaluation, RevisedImageEvaluation, RevisedDimensionScore, DIMENSIONS

        def _to_revised(ev: ImageEvaluation) -> RevisedImageEvaluation:
            dims = {}
            for d in DIMENSIONS:
                orig = getattr(ev, d)
                dims[d] = RevisedDimensionScore(
                    score=orig.score,
                    reasoning=orig.reasoning,
                    critique_accepted=False,
                    revision_note="Critique/revision unavailable. Using initial scores.",
                )
            return RevisedImageEvaluation(model_name=ev.model_name, **dims)

        fallback_revised = RevisedEvaluation(
            model_a=_to_revised(result.eval_a),
            model_b=_to_revised(result.eval_b),
        )
        result.revised = fallback_revised
        result.comparison = determine_winner(
            result.prompt, result.eval_a, result.eval_b, fallback_revised
        )


def _save_result(result: PipelineResult) -> None:
    """Save final pipeline result summary."""
    if result.run_dir:
        summary = {
            "prompt": result.prompt,
            "timestamp": result.timestamp,
            "errors": result.errors,
            "has_eval": result.eval_a is not None,
            "has_critique": result.critique is not None,
            "has_revised": result.revised is not None,
            "has_comparison": result.comparison is not None,
        }
        if result.comparison:
            summary["winner"] = result.comparison.overall_winner
            summary["model_a_mean"] = result.comparison.model_a_mean
            summary["model_b_mean"] = result.comparison.model_b_mean
        (result.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
