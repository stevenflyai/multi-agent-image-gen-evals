# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the Streamlit app
streamlit run app.py

# Run all tests
pytest

# Run a single test
pytest tests/test_winner.py::test_tiebreak_by_largest_dimension_lead -v

# Install dependencies (uses uv; requires Python >=3.12)
uv sync
```

Dependencies are managed by `uv` — do not hand-edit `uv.lock`.

## Architecture

Cross-model adversarial image evaluation pipeline with multi-round critique: two AI models generate images from a prompt, a third evaluates them, a fourth critiques the evaluation, the evaluator revises, then a fifth model performs a second critique round, and the evaluator revises again (if scores haven't converged).

### Pipeline stages (pipeline.py orchestrates)

1. **generate.py** — GPT Image-2 + Gemini 3 Pro generate images in parallel (ThreadPoolExecutor)
2. **evaluate.py** — Claude Opus scores both images across 6 dimensions (1-10 anchored rubric)
3. **Multi-round critique-revision loop** (max 2 rounds, configurable in config.py):
   - Round 1: **critique.py** `critique_evaluation()` — GPT-5.4 reviews Claude's evaluation
   - Round 1: **revise.py** — Claude Opus re-evaluates with GPT-5.4's critique
   - Round 2: **critique.py** `critique_evaluation_gemini()` — Gemini 3.1 Pro reviews revised scores
   - Round 2: **revise.py** — Claude Opus re-evaluates with Gemini's critique
   - Loop stops early if all score deltas < `CONVERGENCE_THRESHOLD` (default: 1)
4. **compare.py** — Determines per-dimension and overall winner (mean score, tiebreak by largest single-dimension lead)
5. **gates.py** — Deterministic HIL (human-in-the-loop) routing. Computes risk features (self-confidence, margin, cross-dim conflict, difficulty prior, evidence quality, variance) and weights them per phase (`PHASE1_NO_VARIANCE_WEIGHTS`, `PHASE2_WITH_VARIANCE_WEIGHTS`) to produce a `GateDecision` / `RouteBand`. Gated by `HIL_ENABLED_BY_DEFAULT` in config.py. Designed so phase 2 can run with HIL disabled and still emit auditable artifacts.

### Key modules

- **config.py** — Centralized configuration: model names, retry settings, image sizes, pipeline parameters.
- **utils.py** — Shared utilities: `image_to_b64()`, `retry_llm_call()`, `parse_llm_json()`, `strip_markdown_fences()`.
- **prompts.py** — Single source of truth for rubric text and all system prompts (evaluation, critique, critique round 2, revision).
- **schemas.py** — All Pydantic models. `ImageEvaluation`, `CritiqueResponse` (with `round` and `critic_model` fields), `RevisedEvaluation`, `ComparisonResult`, plus gate-side models (`GateDecision`, `RouteFeatures`, `RouteBand`, `Gate1ReviewItem`, `IssueEquivalenceReport`). The 6 dimensions are defined in `DIMENSIONS` list.
- **pipeline.py** — `PipelineResult` container (stores `critiques: list` and `revisions: list` for multi-round), `run_pipeline()` with `on_stage` callback and convergence check. Degrades gracefully: if any round's critique/revision fails, uses scores from previous round.
- **app.py** — Streamlit dark-theme dashboard. CSS loaded from `static/style.css`. Renders side-by-side images, 6-step animated progress tracker, radar chart, dimension cards, multi-round critique transcript, raw JSON. Loads pre-baked results from `runs/` directory.
- **i18n.py** — Dict-based translations (en/zh). Use `t(key, **kwargs)` for UI strings, `dim_label(dimension)` for dimension names. Default language is `"zh"`.

### Data flow

All LLM calls are wrapped in `retry_llm_call()` (3 attempts, exponential backoff). JSON responses are parsed via `parse_llm_json()` which strips markdown fences. Images are resized to 768x768 JPEG before sending as base64. Each pipeline run persists all intermediate JSON to `runs/TIMESTAMP/` (with per-round files: `critique_r1.json`, `revised_r1.json`, etc.).

## Environment

Requires `.env` with three API keys (see `.env.example`):
- `OPENAI_API_KEY` — GPT Image-2 generation + GPT-5.4 critique (round 1)
- `ANTHROPIC_API_KEY` — Claude Opus 4.7 evaluation + revision
- `GOOGLE_API_KEY` — Gemini 3 Pro generation + Gemini 3.1 Pro critique (round 2)

Optional `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL` for Azure/Databricks proxies.

## Conventions

- All user-facing strings go through `t()` from i18n.py — never hardcode UI text
- HTML injected via `unsafe_allow_html=True` must escape dynamic values with `html.escape()`
- Pipeline stage notifications use i18n keys: `stage_generating`, `stage_evaluating`, `stage_critique`, `stage_revising`, `stage_critique_round2`, `stage_revising_round2`, `stage_complete`
- All model names and pipeline parameters are defined in `config.py` — never hardcode in module files
- Rubric text and system prompts live in `prompts.py` — single source of truth
