# AGENTS.md

Guidance for AI coding agents working in this repository. For richer detail, see [CLAUDE.md](CLAUDE.md) and [README.md](README.md).

## What this is

Multi-agent image generation evals pipeline: two models generate images from one prompt, an evaluator scores them, two critic models challenge the evaluation across two rounds, the evaluator revises, and deterministic gates can route uncertain cases to human-in-the-loop review. UI is a bilingual Streamlit dashboard.

## Commands

```bash
uv sync                         # install deps
streamlit run app.py            # run the dashboard
pytest                          # full test suite
pytest tests/test_winner.py -v  # single file
```

Python `>=3.12` required. Dependencies are managed with `uv` — do not edit `uv.lock` by hand.

## Architecture map

Pipeline is orchestrated by [pipeline.py](pipeline.py). Each stage is its own module:

| Module | Role |
| --- | --- |
| [generate.py](generate.py) | Parallel image generation (GPT Image-2 + Gemini 3 Pro) |
| [evaluate.py](evaluate.py) | Initial scoring by Claude Opus across 6 rubric dimensions |
| [critique.py](critique.py) | Round 1 (GPT-5.4) and round 2 (Gemini 3.1 Pro) critiques |
| [revise.py](revise.py) | Claude re-scores after each critique round |
| [gates.py](gates.py) | Deterministic HIL routing on uncertainty / disagreement |
| [compare.py](compare.py) | Per-dimension and overall winner; tiebreak by largest single-dimension lead |
| [config.py](config.py) | All model names, retry counts, image sizes, pipeline params |
| [prompts.py](prompts.py) | Single source of truth for the rubric and every system prompt |
| [schemas.py](schemas.py) | Pydantic models — `ImageEvaluation`, `CritiqueResponse`, `RevisedEvaluation`, `ComparisonResult`; `DIMENSIONS` list |
| [utils.py](utils.py) | `image_to_b64`, `retry_llm_call`, `parse_llm_json`, `strip_markdown_fences` |
| [i18n.py](i18n.py) | en/zh translations; `t(key, **kwargs)` and `dim_label(dim)` |
| [app.py](app.py) | Streamlit UI; loads pre-baked runs from `runs/` |

## Conventions to follow

- **No hardcoded UI strings** — route every user-facing string through `t()` in [i18n.py](i18n.py). Default language is `zh`.
- **No hardcoded model names or pipeline params** — they live in [config.py](config.py).
- **No hardcoded prompts or rubric text** — they live in [prompts.py](prompts.py).
- **All LLM calls go through `retry_llm_call()`** (3 attempts, exponential backoff) and JSON responses through `parse_llm_json()`.
- **Escape dynamic HTML** with `html.escape()` whenever using `unsafe_allow_html=True`.
- **Stage notifications** use the i18n keys listed in [CLAUDE.md](CLAUDE.md) (`stage_generating`, `stage_evaluating`, `stage_critique`, `stage_revising`, `stage_critique_round2`, `stage_revising_round2`, `stage_complete`).
- **Run artifacts** persist under `runs/TIMESTAMP/` (per-round critique/revision JSON files). Don't change this layout without updating the loader in `app.py`.

## Environment

Requires `.env` with `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`. See [.env.example](.env.example). Optional `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` for proxy routing.

## Pitfalls

- Pipeline degrades gracefully — if a critique or revision round fails, the previous round's scores are reused. Don't add hard failures inside the loop.
- Convergence check stops the loop early when all per-dimension deltas fall below `CONVERGENCE_THRESHOLD` in `config.py`.
- Image inputs are resized to 768x768 JPEG before being base64-encoded; preserve this when adding new image-handling code.
