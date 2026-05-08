"""Pipeline orchestrator wiring all steps together.

Runs the full evaluation pipeline with multi-round critique loop
and persists all intermediate JSON to the runs/ directory.
"""

import json
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from compare import determine_winner
from config import CONVERGENCE_THRESHOLD, HIL_ENABLED_BY_DEFAULT, IMAGE_MAX_SIZE, IMAGE_QUALITY, MAX_CRITIQUE_ROUNDS
from critique import critique_evaluation, critique_evaluation_gemini
from evaluate import evaluate_images
from gates import evaluate_gate1, evaluate_gate2
from generate import generate_images
from revise import revise_evaluation
from schemas import (
    DIMENSIONS,
    ComparisonResult,
    CritiqueResponse,
    GateDecision,
    ImageEvaluation,
    IssueEquivalenceReport,
    PipelineStatus,
    PromptDifficulty,
    PromptInputMetadata,
    RevisedDimensionScore,
    RevisedEvaluation,
    RevisedImageEvaluation,
)


class PipelineResult:
    """Container for all pipeline outputs."""

    def __init__(self) -> None:
        self.prompt: str = ""
        self.run_dir: Path | None = None
        self.image_paths: dict[str, Path | Exception] = {}
        self.reference_image_path: Path | None = None
        self.reference_image_name: str | None = None
        self.eval_a: ImageEvaluation | None = None
        self.eval_b: ImageEvaluation | None = None
        self.critiques: list[CritiqueResponse] = []
        self.revisions: list[RevisedEvaluation] = []
        self.hil_reviews: list = []
        self.hil_adjudication = None
        self.gate_decisions: list[GateDecision] = []
        self.issue_equivalence: IssueEquivalenceReport | None = None
        self.prompt_difficulty: PromptDifficulty | None = None
        self.pipeline_status: PipelineStatus = "partial"
        self.requires_attention: bool = False
        self.comparison: ComparisonResult | None = None
        self.errors: list[str] = []
        self.timestamp: str = ""
        self.rounds_completed: int = 0

    # Backward-compat properties: return last critique/revision
    @property
    def critique(self) -> CritiqueResponse | None:
        return self.critiques[-1] if self.critiques else None

    @property
    def revised(self) -> RevisedEvaluation | None:
        return self.revisions[-1] if self.revisions else None


def _scores_converged(prev: RevisedEvaluation, curr: RevisedEvaluation, threshold: int) -> bool:
    """Check if all score deltas between two revisions are below the threshold."""
    for d in DIMENSIONS:
        prev_a = getattr(prev.model_a, d).score
        curr_a = getattr(curr.model_a, d).score
        prev_b = getattr(prev.model_b, d).score
        curr_b = getattr(curr.model_b, d).score
        if abs(curr_a - prev_a) >= threshold or abs(curr_b - prev_b) >= threshold:
            return False
    return True


def run_pipeline(
    prompt: str,
    runs_dir: Path = Path("runs"),
    on_stage: Callable[[str], None] | None = None,
    on_image_done: Callable[[str], None] | None = None,
    reference_image: bytes | None = None,
    reference_image_name: str | None = None,
) -> PipelineResult:
    """Run the full evaluation pipeline with multi-round critique.

    Args:
        prompt: The image generation prompt.
        runs_dir: Base directory for run outputs.
        on_stage: Optional callback for stage progress updates.
    """
    result = PipelineResult()
    result.prompt = prompt
    result.timestamp = datetime.now().isoformat()
    result.reference_image_name = reference_image_name

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = runs_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    result.run_dir = run_dir

    # Save prompt
    (run_dir / "prompt.txt").write_text(prompt)
    if reference_image:
        try:
            result.reference_image_path = _persist_reference_image(run_dir, reference_image)
        except Exception as e:
            result.errors.append(f"Reference image failed: {_format_exception(e)}")
            result.pipeline_status = "failed"
            result.requires_attention = True
            _save_result(result)
            return result

    def notify(stage: str) -> None:
        if on_stage:
            on_stage(stage)

    # Step 1: Generate images
    notify("stage_generating")
    result.image_paths = generate_images(
        prompt,
        run_dir,
        on_model_done=on_image_done,
        reference_image_path=result.reference_image_path,
    )

    gpt_path = result.image_paths.get("gpt_image_2")
    gemini_path = result.image_paths.get("gemini_3_pro")

    if isinstance(gpt_path, Exception):
        result.errors.append(f"GPT Image-2 failed: {_format_exception(gpt_path)}")
        gpt_path = None
    if isinstance(gemini_path, Exception):
        result.errors.append(f"Gemini 3 Pro failed: {_format_exception(gemini_path)}")
        gemini_path = None

    if not gpt_path or not gemini_path:
        result.pipeline_status = "failed"
        result.requires_attention = True
        _save_result(result)
        return result

    # Step 2: Evaluate
    notify("stage_evaluating")
    try:
        result.eval_a, result.eval_b, result.prompt_difficulty = evaluate_images(prompt, gpt_path, gemini_path)
        (run_dir / "evaluation.json").write_text(
            json.dumps({"model_a": result.eval_a.model_dump(), "model_b": result.eval_b.model_dump()}, indent=2)
        )
        (run_dir / "evaluation_v2.json").write_text(
            json.dumps(
                {
                    "prompt_difficulty": result.prompt_difficulty,
                    "model_a": result.eval_a.model_dump(),
                    "model_b": result.eval_b.model_dump(),
                },
                indent=2,
            )
        )
    except Exception as e:
        result.errors.append(f"Evaluation failed: {e}")
        result.pipeline_status = "failed"
        _save_result(result)
        return result

    notify("stage_gate1")
    gate1_decision = evaluate_gate1(prompt, result.eval_a, result.eval_b, hil_enabled=HIL_ENABLED_BY_DEFAULT)
    result.gate_decisions.append(gate1_decision)
    if gate1_decision.pipeline_status:
        result.pipeline_status = gate1_decision.pipeline_status
    result.requires_attention = result.requires_attention or gate1_decision.requires_attention
    (run_dir / "gate1_decision.json").write_text(json.dumps(gate1_decision.model_dump(), indent=2))
    if gate1_decision.status == "pending":
        _save_result(result)
        return result

    # Step 3-4: Multi-round critique-revision loop
    for round_num in range(1, MAX_CRITIQUE_ROUNDS + 1):
        # Critique
        if round_num == 1:
            notify("stage_critique")
            try:
                crit = critique_evaluation(
                    prompt, result.eval_a, result.eval_b, gpt_path, gemini_path
                )
                result.critiques.append(crit)
                (run_dir / f"critique_r{round_num}.json").write_text(
                    json.dumps(crit.model_dump(), indent=2)
                )
                _save_prompt_input_metadata(
                    run_dir,
                    PromptInputMetadata(
                        step="05_critique",
                        included_artifacts=["prompt", "image_a", "image_b", "evaluation.json"],
                        excluded_artifacts=["revised_r1.json", "critique_r2.json", "revised_r2.json"],
                    ),
                )
            except Exception as e:
                result.errors.append(f"Critique round {round_num} failed: {e}")
                _build_comparison_fallback(result)
                _save_result(result)
                return result
        else:
            notify("stage_critique_round2")
            prev_revised = result.revisions[-1]
            try:
                crit = critique_evaluation_gemini(
                    prompt, prev_revised.model_a, prev_revised.model_b,
                    gpt_path, gemini_path,
                    raw_output_path=run_dir / "critique_r2_raw.txt",
                )
                result.critiques.append(crit)
                (run_dir / f"critique_r{round_num}.json").write_text(
                    json.dumps(crit.model_dump(), indent=2)
                )
                _save_prompt_input_metadata(
                    run_dir,
                    PromptInputMetadata(
                        step="07_critique",
                        included_artifacts=["prompt", "image_a", "image_b", "revised_r1.json"],
                        excluded_artifacts=["critique_r1.json", "evaluation.json", "revised_r2.json"],
                    ),
                )
            except Exception as e:
                result.errors.append(f"Critique round {round_num} failed: {e}")
                # Graceful degradation: use previous round's scores
                break

        if round_num == 2 and result.revisions and len(result.critiques) >= 2:
            notify("stage_gate2")
            gate2_decision, issue_equivalence = evaluate_gate2(
                result.eval_a,
                result.eval_b,
                result.revisions[-1],
                result.critiques[0],
                result.critiques[-1],
                hil_enabled=HIL_ENABLED_BY_DEFAULT,
            )
            result.gate_decisions.append(gate2_decision)
            result.issue_equivalence = issue_equivalence
            if gate2_decision.pipeline_status:
                result.pipeline_status = gate2_decision.pipeline_status
            result.requires_attention = result.requires_attention or gate2_decision.requires_attention
            (run_dir / "gate2_decision.json").write_text(json.dumps(gate2_decision.model_dump(), indent=2))
            (run_dir / "issue_equivalence.json").write_text(json.dumps(issue_equivalence.model_dump(), indent=2))
            if gate2_decision.status == "pending":
                _save_result(result)
                return result

        # Revise
        stage_key = "stage_revising" if round_num == 1 else "stage_revising_round2"
        notify(stage_key)
        try:
            revision_input_a = result.eval_a if round_num == 1 else result.revisions[-1].model_a
            revision_input_b = result.eval_b if round_num == 1 else result.revisions[-1].model_b
            rev = revise_evaluation(
                prompt, revision_input_a, revision_input_b,
                result.critiques[-1], gpt_path, gemini_path,
            )
            result.revisions.append(rev)
            (run_dir / f"revised_r{round_num}.json").write_text(
                json.dumps(rev.model_dump(), indent=2)
            )
            _save_prompt_input_metadata(
                run_dir,
                PromptInputMetadata(
                    step="06_revision" if round_num == 1 else "08_final_revision",
                    included_artifacts=(
                        ["prompt", "image_a", "image_b", "evaluation.json", "critique_r1.json", "gate1_decision.json"]
                        if round_num == 1
                        else [
                            "prompt", "image_a", "image_b", "critique_r1.json", "revised_r1.json",
                            "critique_r2.json", "gate1_decision.json", "gate2_decision.json",
                        ]
                    ),
                    excluded_artifacts=["critique_r2.json", "revised_r2.json"] if round_num == 1 else [],
                ),
            )
            result.rounds_completed = round_num
        except Exception as e:
            result.errors.append(f"Revision round {round_num} failed: {e}")
            if round_num == 1:
                _build_comparison_fallback(result)
                _save_result(result)
                return result
            # Round 2 revision failed — use round 1 scores
            break

        # Convergence check (only after round 2+)
        if len(result.revisions) >= 2:
            if _scores_converged(result.revisions[-2], result.revisions[-1], CONVERGENCE_THRESHOLD):
                break

    # If no revision succeeded, fall back to initial scores
    if not result.revisions:
        _build_comparison_fallback(result)
        _save_result(result)
        return result

    # Step 5: Compare (uses final revised scores)
    notify("stage_complete")
    result.comparison = determine_winner(prompt, result.eval_a, result.eval_b, result.revised)
    if result.pipeline_status == "partial":
        result.pipeline_status = "completed"
    (run_dir / "comparison.json").write_text(json.dumps(result.comparison.model_dump(), indent=2))

    # Save backward-compat aliases
    if result.critique:
        (run_dir / "critique.json").write_text(json.dumps(result.critique.model_dump(), indent=2))
    if result.revised:
        (run_dir / "revised.json").write_text(json.dumps(result.revised.model_dump(), indent=2))

    _save_result(result)
    return result


def resume_pipeline_from_result(
    result: PipelineResult,
    on_stage: Callable[[str], None] | None = None,
    on_image_done: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Resume a partially failed run from the first missing pipeline artifact."""
    if not result.run_dir:
        raise ValueError("Cannot resume a run without a run directory")

    run_dir = result.run_dir
    result.errors = []
    result.pipeline_status = "partial"

    def notify(stage: str) -> None:
        if on_stage:
            on_stage(stage)

    if not result.reference_image_path:
        result.reference_image_path = _find_reference_image(run_dir)

    gpt_path = _existing_image_path(result.image_paths.get("gpt_image_2"), run_dir / "gpt_image_2.png")
    gemini_path = _existing_image_path(result.image_paths.get("gemini_3_pro"), run_dir / "gemini_3_pro.png")

    if not gpt_path or not gemini_path:
        notify("stage_generating")
        result.eval_a = None
        result.eval_b = None
        result.prompt_difficulty = None
        result.critiques = []
        result.revisions = []
        result.gate_decisions = []
        result.issue_equivalence = None
        result.comparison = None
        result.requires_attention = False
        result.rounds_completed = 0
        result.image_paths = generate_images(
            result.prompt,
            run_dir,
            on_model_done=on_image_done,
            reference_image_path=result.reference_image_path,
        )
        gpt_path = _existing_image_path(result.image_paths.get("gpt_image_2"), run_dir / "gpt_image_2.png")
        gemini_path = _existing_image_path(result.image_paths.get("gemini_3_pro"), run_dir / "gemini_3_pro.png")
        if isinstance(result.image_paths.get("gpt_image_2"), Exception):
            result.errors.append(f"GPT Image-2 failed: {_format_exception(result.image_paths['gpt_image_2'])}")
        if isinstance(result.image_paths.get("gemini_3_pro"), Exception):
            result.errors.append(f"Gemini 3 Pro failed: {_format_exception(result.image_paths['gemini_3_pro'])}")
        if not gpt_path or not gemini_path:
            result.pipeline_status = "failed"
            result.requires_attention = True
            _save_result(result)
            return result

    result.image_paths["gpt_image_2"] = gpt_path
    result.image_paths["gemini_3_pro"] = gemini_path

    if not result.eval_a or not result.eval_b:
        notify("stage_evaluating")
        try:
            result.eval_a, result.eval_b, result.prompt_difficulty = evaluate_images(result.prompt, gpt_path, gemini_path)
            _persist_evaluation(run_dir, result)
        except Exception as e:
            result.errors.append(f"Evaluation failed: {e}")
            result.pipeline_status = "failed"
            _save_result(result)
            return result

    if not _gate_by_name(result, "gate1_uncertainty_risk_router"):
        notify("stage_gate1")
        gate1_decision = evaluate_gate1(result.prompt, result.eval_a, result.eval_b, hil_enabled=HIL_ENABLED_BY_DEFAULT)
        _replace_gate_decision(result, gate1_decision)
        if gate1_decision.pipeline_status:
            result.pipeline_status = gate1_decision.pipeline_status
        result.requires_attention = result.requires_attention or gate1_decision.requires_attention
        (run_dir / "gate1_decision.json").write_text(json.dumps(gate1_decision.model_dump(), indent=2))
        if gate1_decision.status == "pending":
            _save_result(result)
            return result

    # If round 1 failed before any critique, an old fallback revision may exist. Drop it before resuming.
    if not result.critiques and result.revisions:
        result.revisions = []

    for round_num in range(1, MAX_CRITIQUE_ROUNDS + 1):
        if len(result.critiques) < round_num:
            if round_num == 1:
                notify("stage_critique")
                try:
                    crit = critique_evaluation(result.prompt, result.eval_a, result.eval_b, gpt_path, gemini_path)
                    result.critiques.append(crit)
                    (run_dir / "critique_r1.json").write_text(json.dumps(crit.model_dump(), indent=2))
                    _save_prompt_input_metadata(
                        run_dir,
                        PromptInputMetadata(
                            step="05_critique",
                            included_artifacts=["prompt", "image_a", "image_b", "evaluation.json"],
                            excluded_artifacts=["revised_r1.json", "critique_r2.json", "revised_r2.json"],
                        ),
                    )
                except Exception as e:
                    result.errors.append(f"Critique round {round_num} failed: {e}")
                    _build_comparison_fallback(result)
                    _save_result(result)
                    return result
            else:
                if len(result.revisions) < 1:
                    break
                notify("stage_critique_round2")
                prev_revised = result.revisions[-1]
                try:
                    crit = critique_evaluation_gemini(
                        result.prompt, prev_revised.model_a, prev_revised.model_b, gpt_path, gemini_path,
                        raw_output_path=run_dir / "critique_r2_raw.txt",
                    )
                    result.critiques.append(crit)
                    (run_dir / "critique_r2.json").write_text(json.dumps(crit.model_dump(), indent=2))
                    _save_prompt_input_metadata(
                        run_dir,
                        PromptInputMetadata(
                            step="07_critique",
                            included_artifacts=["prompt", "image_a", "image_b", "revised_r1.json"],
                            excluded_artifacts=["critique_r1.json", "evaluation.json", "revised_r2.json"],
                        ),
                    )
                except Exception as e:
                    result.errors.append(f"Critique round {round_num} failed: {e}")
                    _save_result(result)
                    return result

        if round_num == 2 and result.revisions and len(result.critiques) >= 2 and not _gate_by_name(result, "gate2_disagreement_detector"):
            notify("stage_gate2")
            gate2_decision, issue_equivalence = evaluate_gate2(
                result.eval_a,
                result.eval_b,
                result.revisions[-1],
                result.critiques[0],
                result.critiques[-1],
                hil_enabled=HIL_ENABLED_BY_DEFAULT,
            )
            _replace_gate_decision(result, gate2_decision)
            result.issue_equivalence = issue_equivalence
            if gate2_decision.pipeline_status:
                result.pipeline_status = gate2_decision.pipeline_status
            result.requires_attention = result.requires_attention or gate2_decision.requires_attention
            (run_dir / "gate2_decision.json").write_text(json.dumps(gate2_decision.model_dump(), indent=2))
            (run_dir / "issue_equivalence.json").write_text(json.dumps(issue_equivalence.model_dump(), indent=2))
            if gate2_decision.status == "pending":
                _save_result(result)
                return result

        if len(result.revisions) < round_num:
            stage_key = "stage_revising" if round_num == 1 else "stage_revising_round2"
            notify(stage_key)
            try:
                revision_input_a = result.eval_a if round_num == 1 else result.revisions[-1].model_a
                revision_input_b = result.eval_b if round_num == 1 else result.revisions[-1].model_b
                rev = revise_evaluation(
                    result.prompt, revision_input_a, revision_input_b, result.critiques[-1], gpt_path, gemini_path,
                )
                result.revisions.append(rev)
                (run_dir / f"revised_r{round_num}.json").write_text(json.dumps(rev.model_dump(), indent=2))
                _save_prompt_input_metadata(
                    run_dir,
                    PromptInputMetadata(
                        step="06_revision" if round_num == 1 else "08_final_revision",
                        included_artifacts=(
                            ["prompt", "image_a", "image_b", "evaluation.json", "critique_r1.json", "gate1_decision.json"]
                            if round_num == 1
                            else [
                                "prompt", "image_a", "image_b", "critique_r1.json", "revised_r1.json",
                                "critique_r2.json", "gate1_decision.json", "gate2_decision.json",
                            ]
                        ),
                        excluded_artifacts=["critique_r2.json", "revised_r2.json"] if round_num == 1 else [],
                    ),
                )
                result.rounds_completed = round_num
            except Exception as e:
                result.errors.append(f"Revision round {round_num} failed: {e}")
                _save_result(result)
                return result

        if len(result.revisions) >= 2 and _scores_converged(result.revisions[-2], result.revisions[-1], CONVERGENCE_THRESHOLD):
            break

    if not result.revisions:
        _build_comparison_fallback(result)
        _save_result(result)
        return result

    notify("stage_complete")
    result.comparison = determine_winner(result.prompt, result.eval_a, result.eval_b, result.revised)
    if result.pipeline_status == "partial":
        result.pipeline_status = "completed"
    (run_dir / "comparison.json").write_text(json.dumps(result.comparison.model_dump(), indent=2))
    if result.critique:
        (run_dir / "critique.json").write_text(json.dumps(result.critique.model_dump(), indent=2))
    if result.revised:
        (run_dir / "revised.json").write_text(json.dumps(result.revised.model_dump(), indent=2))
    _save_result(result)
    return result


def _build_comparison_fallback(result: PipelineResult) -> None:
    """Build comparison using initial evaluation scores when critique/revision fails."""
    if result.eval_a and result.eval_b:
        def _to_revised(ev: ImageEvaluation) -> RevisedImageEvaluation:
            dims = {}
            for d in DIMENSIONS:
                orig = getattr(ev, d)
                dims[d] = RevisedDimensionScore(
                    score=orig.score,
                    reasoning=orig.reasoning,
                    evidence=orig.evidence,
                    confidence=orig.confidence,
                    critique_accepted=False,
                    revision_note="Critique/revision unavailable. Using initial scores.",
                )
            return RevisedImageEvaluation(model_name=ev.model_name, **dims)

        fallback_revised = RevisedEvaluation(
            model_a=_to_revised(result.eval_a),
            model_b=_to_revised(result.eval_b),
        )
        result.revisions = [fallback_revised]
        result.comparison = determine_winner(
            result.prompt, result.eval_a, result.eval_b, fallback_revised
        )
        if result.pipeline_status == "partial":
            result.pipeline_status = "completed"


def _persist_evaluation(run_dir: Path, result: PipelineResult) -> None:
    """Persist both backward-compatible and V2 evaluation artifacts."""
    (run_dir / "evaluation.json").write_text(
        json.dumps({"model_a": result.eval_a.model_dump(), "model_b": result.eval_b.model_dump()}, indent=2)
    )
    (run_dir / "evaluation_v2.json").write_text(
        json.dumps(
            {
                "prompt_difficulty": result.prompt_difficulty,
                "model_a": result.eval_a.model_dump(),
                "model_b": result.eval_b.model_dump(),
            },
            indent=2,
        )
    )


def _existing_image_path(candidate: Path | Exception | None, fallback: Path) -> Path | None:
    if isinstance(candidate, Path) and candidate.exists():
        return candidate
    if fallback.exists():
        return fallback
    return None


def _find_reference_image(run_dir: Path) -> Path | None:
    for filename in ("reference_image.jpg", "reference_image.jpeg", "reference_image.png", "reference_image.webp"):
        path = run_dir / filename
        if path.exists():
            return path
    return None


def _format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    fallback = repr(exc).strip()
    if fallback and fallback != f"{type(exc).__name__}()":
        return fallback
    return type(exc).__name__


def _persist_reference_image(run_dir: Path, image_bytes: bytes) -> Path:
    output_path = run_dir / "reference_image.jpg"
    with Image.open(BytesIO(image_bytes)) as image:
        image.thumbnail((IMAGE_MAX_SIZE, IMAGE_MAX_SIZE), Image.LANCZOS)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(output_path, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
    return output_path


def _gate_by_name(result: PipelineResult, gate_name: str) -> GateDecision | None:
    for gate in result.gate_decisions:
        if gate.gate == gate_name:
            return gate
    return None


def _replace_gate_decision(result: PipelineResult, decision: GateDecision) -> None:
    result.gate_decisions = [gate for gate in result.gate_decisions if gate.gate != decision.gate]
    result.gate_decisions.append(decision)


def _save_result(result: PipelineResult) -> None:
    """Save final pipeline result summary."""
    if result.run_dir:
        summary = {
            "prompt": result.prompt,
            "timestamp": result.timestamp,
            "errors": result.errors,
            "reference_image": result.reference_image_path.name if result.reference_image_path else None,
            "reference_image_source": result.reference_image_name,
            "pipeline_status": result.pipeline_status,
            "requires_attention": result.requires_attention,
            "prompt_difficulty": result.prompt_difficulty,
            "rounds_completed": result.rounds_completed,
            "has_eval": result.eval_a is not None,
            "has_critique": len(result.critiques) > 0,
            "has_revised": len(result.revisions) > 0,
            "has_comparison": result.comparison is not None,
            "gate_decisions": [decision.model_dump() for decision in result.gate_decisions],
        }
        if result.comparison:
            summary["winner"] = result.comparison.overall_winner
            summary["model_a_mean"] = result.comparison.model_a_mean
            summary["model_b_mean"] = result.comparison.model_b_mean
        (result.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if result.comparison:
            _update_history_index(result)


def _update_history_index(result: PipelineResult) -> None:
    """Record successful comparisons in the runs history index."""
    if not result.run_dir or not result.comparison:
        return

    index_path = result.run_dir.parent / "index.json"
    try:
        raw = json.loads(index_path.read_text()) if index_path.exists() else {}
    except json.JSONDecodeError:
        raw = {}

    entries = raw.get("runs", []) if isinstance(raw, dict) else []
    run_dir = str(result.run_dir)
    entry = {
        "run_dir": run_dir,
        "prompt": result.prompt,
        "timestamp": result.timestamp,
        "winner": result.comparison.overall_winner,
        "model_a_mean": result.comparison.model_a_mean,
        "model_b_mean": result.comparison.model_b_mean,
        "pipeline_status": result.pipeline_status,
        "requires_attention": result.requires_attention,
    }
    entries = [item for item in entries if item.get("run_dir") != run_dir]
    entries.append(entry)
    entries.sort(key=lambda item: item.get("timestamp", ""), reverse=True)

    index_path.write_text(json.dumps({"version": 1, "runs": entries}, indent=2))


def _save_prompt_input_metadata(run_dir: Path, metadata: PromptInputMetadata) -> None:
    safe_step = metadata.step.lower().replace(" ", "_")
    (run_dir / f"prompt_inputs_{safe_step}.json").write_text(json.dumps(metadata.model_dump(), indent=2))
