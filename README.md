# Multi Agents Image Generation Evals Pipeline

Languages: [English](README.md) | [简体中文](README.zh-CN.md)

Multi Agents Image Generation Evals Pipeline is a multiple agent system for evaluating and comparing AI image generation models. The app sends one prompt to two image-generation agents, uses a dedicated evaluation agent to score both outputs with a rubric, runs independent critique and revision agents to challenge the scores, optionally pauses for human-in-the-loop (HIL) review when risk is high, and archives every run for later inspection.

The primary UI is a bilingual Streamlit dashboard for operating this multiple agent workflow, with side-by-side images, progress/activity logs, score visualizations, critique transcripts, HIL review controls, historical run loading, delete, and rerun actions.

## What It Does

- Coordinates generation, evaluation, critique, revision, gate, and comparison agents in one auditable workflow.
- Generates two images from the same prompt in parallel.
- Scores both images across six dimensions on a calibrated 1-10 rubric.
- Uses independent critique agents to find evaluator mistakes, bias, unsupported reasoning, and score inconsistencies.
- Revises scores after critique while preserving the original evaluation artifacts.
- Routes uncertain or internally conflicted cases through deterministic HIL gates.
- Persists images, prompts, JSON artifacts, raw model outputs, gate decisions, activity snapshots, and final summaries under `runs/`.
- Loads pre-baked or historical examples from prior runs, with controls to delete only the selected run or rerun its prompt.

## Models

| Role | Model | Provider |
| --- | --- | --- |
| Image generation A | GPT Image-2 | OpenAI |
| Image generation B | Gemini 3 Pro | Google |
| Initial evaluation | Claude Opus 4.7 | Anthropic |
| Revision | Claude Opus 4.7 | Anthropic |
| Critique round 1 | GPT-5.4 | OpenAI |
| Critique round 2 | Gemini 3.1 Pro | Google |

Model names and runtime settings live in [config.py](config.py).

## Pipeline

```text
Prompt
  |
  +--> GPT Image-2 --------+
  |                        |
  +--> Gemini 3 Pro -------+--> Claude evaluates both images
           |
           v
         Gate 1: uncertainty/risk router
           |
           v
         GPT-5.4 critique (round 1)
           |
           v
         Claude revision (round 1)
           |
           v
         Gemini critique (round 2)
           |
           v
         Gate 2: disagreement detector
           |
           v
         Claude final revision
           |
           v
         Deterministic comparison
```

The orchestrator is [pipeline.py](pipeline.py). Provider adapters are split across [generate.py](generate.py), [evaluate.py](evaluate.py), [critique.py](critique.py), and [revise.py](revise.py). Rubrics and prompt contracts are centralized in [prompts.py](prompts.py), and all output contracts are defined in [schemas.py](schemas.py).

### Core Stages

1. **Generation**: GPT Image-2 and Gemini 3 Pro run in parallel with retry and timeout handling.
2. **Evaluation**: Claude scores both images across six rubric dimensions and records evidence/confidence fields.
3. **Gate 1**: A deterministic uncertainty/risk router checks margin risk, difficulty, evidence quality, confidence, and cross-dimension conflicts.
4. **Critique round 1**: GPT-5.4 independently critiques the initial evaluation.
5. **Revision round 1**: Claude revises the evaluation with critique context.
6. **Critique round 2**: Gemini independently critiques the revised evaluation. Raw Gemini output is archived to aid debugging malformed or truncated JSON.
7. **Gate 2**: A deterministic disagreement detector compares critique signals and score movement.
8. **Final revision**: Claude reconciles critique signals and any HIL labels.
9. **Comparison**: [compare.py](compare.py) computes the final winner by mean score, then largest dimension lead, then draw.

If a generation step fails, the run is marked failed instead of pretending the pipeline succeeded. If a later critique/revision step fails, the pipeline records the error and falls back to the best available completed scores where possible.

## Human-in-the-loop Gates

HIL logic is implemented in [gates.py](gates.py) and surfaced in [app.py](app.py).

| Gate | Purpose | Possible effect |
| --- | --- | --- |
| Gate 1: uncertainty/risk router | Flags narrow, hard, low-confidence, or evidence-poor evaluations | May pause before critique when HIL is enabled |
| Gate 2: disagreement detector | Flags critique disagreement, winner flips, large score movement, or new material issues | May pause before final revision when HIL is enabled |

Human labels are narrow and auditable. They guide later revision and decision context, but they do not directly mutate raw model scores.

## Evaluation Dimensions

Scores use the full 1-10 scale:

| Score band | Meaning |
| --- | --- |
| 1-2 | Unusable or almost no evidence |
| 3-4 | Poor with major failures |
| 5-6 | Mixed or acceptable with clear flaws |
| 7-8 | Strong with minor or moderate flaws |
| 9 | Excellent with tiny issues |
| 10 | Essentially flawless and rare |

| Dimension | What it measures |
| --- | --- |
| Prompt adherence | Whether explicit and implied prompt requirements are satisfied |
| Photorealism | Physical plausibility, artifact control, and medium consistency |
| Aesthetic quality | Visual craft, polish, appeal, and genre fit |
| Composition | Framing, hierarchy, depth, cropping, and visual flow |
| Color accuracy | Prompt-specified colors, lighting, materials, and palette harmony |
| Creativity | Useful, surprising interpretation that improves the prompt without violating it |

The dimension list is defined once in [schemas.py](schemas.py).

## Dashboard

Run the Streamlit app to use the main workflow:

```bash
streamlit run app.py
```

Common local command used during development:

```bash
/opt/anaconda3/bin/python -m streamlit run app.py --server.port 8504
```

The dashboard includes:

- Prompt entry and run controls.
- Live activity log with model/stage timing for new runs.
- Historical activity reconstruction for older runs without fake generated timings.
- Side-by-side image comparison.
- Winner banner and dimension-level score cards.
- Radar chart for initial vs revised scores.
- Dashboard table with adaptive category column layout.
- Multi-round critique transcript.
- HIL Gate 1 and Gate 2 review panels when required.
- Raw JSON expanders for evaluation, critiques, revisions, gate decisions, and comparison.
- Pre-baked example dropdown backed by `runs/index.json` and `runs/*/summary.json`.
- Delete selected run and rerun selected prompt actions.
- English and Chinese UI strings via [i18n.py](i18n.py).

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) or an equivalent Python environment

### Install

```bash
git clone https://github.com/stevenflyai/evalsimagegen.git
cd evalsimagegen
uv sync
```

If you are using the local Conda Python environment from this workspace, run commands with:

```bash
/opt/anaconda3/bin/python -m pytest -q
```

### Environment Variables

Copy [.env.example](.env.example) to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Used for |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | GPT Image-2 generation and GPT-5.4 critique |
| `ANTHROPIC_API_KEY` | Yes | Claude Opus evaluation and revision |
| `GOOGLE_API_KEY` | Yes | Gemini image generation and critique |
| `OPENAI_BASE_URL` | Optional | Azure OpenAI or compatible proxy endpoint |
| `ANTHROPIC_BASE_URL` | Optional | Anthropic-compatible proxy endpoint |

Do not commit `.env`; it is ignored by git.

## Testing

Run the full test suite:

```bash
pytest -q
```

Or with the workspace Conda Python:

```bash
/opt/anaconda3/bin/python -m pytest -q
```

Focused examples:

```bash
pytest tests/test_utils.py -q
pytest tests/test_gates.py tests/test_pipeline.py -q
pytest tests/test_winner.py::test_tiebreak_by_largest_dimension_lead -v
```

Current suite coverage includes schemas, JSON repair/truncation handling, generation failure handling, HIL gate decisions, pipeline resume/fallback behavior, and winner selection.

## Run Artifacts

Each run is stored under `runs/YYYYMMDD_HHMMSS_microseconds/`. The directory may include:

```text
prompt.txt
gpt_image_2.png
gemini_3_pro.png
evaluation.json
evaluation_v2.json
gate1_decision.json
critique_r1.json
revised_r1.json
critique_r2_raw.txt
critique_r2.json
issue_equivalence.json
gate2_decision.json
revised_r2.json
comparison.json
summary.json
activity_snapshot.json
prompt_inputs_05_critique.json
prompt_inputs_06_revision.json
prompt_inputs_07_critique.json
prompt_inputs_08_final_revision.json
```

`runs/` is gitignored because it contains generated images, raw model outputs, and local history.

## Project Structure

```text
evalsimagegen/
|-- app.py                 # Streamlit dashboard and run-management UI
|-- pipeline.py            # Pipeline orchestration, persistence, resume/fallback logic
|-- generate.py            # Parallel GPT Image-2 and Gemini 3 Pro generation
|-- evaluate.py            # Claude initial scoring
|-- critique.py            # GPT round 1 and Gemini round 2 critique adapters
|-- revise.py              # Claude revision adapter
|-- compare.py             # Final deterministic winner selection
|-- gates.py               # HIL Gate 1 and Gate 2 decision logic
|-- schemas.py             # Pydantic models and shared dimensions
|-- prompts.py             # Rubric and prompt contracts
|-- utils.py               # Retry, image encoding, JSON parsing/repair helpers
|-- i18n.py                # English/Chinese UI text helpers
|-- config.py              # Model names, timeouts, retries, HIL defaults
|-- static/style.css       # Streamlit UI styling
|-- tests/                 # Pytest suite
|-- docs/superpowers/      # Design notes and Architecture V2 docs
|-- runs/                  # Local generated runs, ignored by git
|-- pyproject.toml
|-- .env.example
|-- .gitignore
`-- CLAUDE.md
```

## Configuration

Key settings in [config.py](config.py):

| Setting | Default | Description |
| --- | --- | --- |
| `MAX_CRITIQUE_ROUNDS` | `2` | Maximum critique/revision rounds |
| `CONVERGENCE_THRESHOLD` | `1` | Stop when all score deltas are below this threshold |
| `MAX_RETRIES` | `3` | Retry attempts for LLM/image calls |
| `RETRY_BACKOFF` | `2.0` | Exponential retry backoff base |
| `IMAGE_MAX_SIZE` | `768` | Max image dimension sent to evaluator/critics |
| `IMAGE_QUALITY` | `85` | JPEG quality for LLM image inputs |
| `LLM_MAX_TOKENS` | `4096` | Default text model output budget |
| `CRITIQUE_ROUND2_MAX_TOKENS` | `8192` | Larger Gemini critique budget to reduce truncation |
| `IMAGE_GEN_TIMEOUT` | `600` | Parallel image generation timeout in seconds |
| `HIL_ENABLED_BY_DEFAULT` | `True` | Whether gate-triggered cases can pause for HIL review |

## Notes for Contributors

- Keep user-facing UI strings in [i18n.py](i18n.py) and use `t(...)`/`dim_label(...)` from the app.
- Keep rubric text and system prompts in [prompts.py](prompts.py).
- Keep model names and runtime knobs in [config.py](config.py).
- Escape dynamic HTML values before rendering with `unsafe_allow_html=True`.
- Preserve prompt-input metadata boundaries so critique independence remains auditable.
- Do not commit local `runs/`, `.env`, screenshots, caches, or generated secrets.

## Author

Steven Lian (stevenlian2981@gmail.com)

Repository: https://github.com/stevenflyai/evalsimagegen
