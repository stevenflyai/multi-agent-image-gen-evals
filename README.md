# Image Eval Pipeline

Cross-model adversarial image evaluation with multi-round critique: two AI models generate images from the same prompt, a third scores them, a fourth critiques the scores, the evaluator revises, then a fifth model performs a second review. Results are displayed in a bilingual (EN/ZH) Streamlit dashboard.

## Models

| Role | Model | Provider |
|------|-------|----------|
| Image generation | GPT Image-2 | OpenAI |
| Image generation | Gemini 3 Pro | Google |
| Evaluation + Revision | Claude Opus 4.7 | Anthropic |
| Critique (Round 1) | GPT-5.4 | OpenAI |
| Critique (Round 2) | Gemini 3.1 Pro | Google |

## Pipeline

```
Prompt
  │
  ├──► GPT Image-2 ──┐
  │                   ├──► Claude Opus evaluates both (6 dimensions, 1-10)
  └──► Gemini 3 Pro ──┘           │
                                  ▼
                      ┌─── Round 1 ────────────────────────┐
                      │ GPT-5.4 critiques evaluation       │
                      │           │                        │
                      │           ▼                        │
                      │ Claude Opus revises scores          │
                      │   (accept/reject per point)        │
                      └────────────────────────────────────┘
                                  │
                          convergence check
                         (all deltas < 1?)
                                  │ no
                                  ▼
                      ┌─── Round 2 ────────────────────────┐
                      │ Gemini 3.1 Pro critiques revision   │
                      │           │                        │
                      │           ▼                        │
                      │ Claude Opus revises again            │
                      └────────────────────────────────────┘
                                  │
                                  ▼
                        Winner determined
                          (mean → tiebreak by largest dimension lead → draw)
```

### Stages

1. **Generate** — Both models generate 1024x1024 images in parallel (ThreadPoolExecutor, 600s timeout)
2. **Evaluate** — Claude Opus receives both images (resized to 768x768 JPEG, base64) and scores each across 6 dimensions with an anchored rubric
3. **Multi-round critique-revision loop** (max 2 rounds):
   - **Round 1**: GPT-5.4 reviews the evaluation for inconsistencies, bias, and unsupported reasoning. Claude Opus revises all 6 dimensions, tracking accept/reject per critique point.
   - **Convergence check**: If all score deltas between rounds are < 1, the loop stops early.
   - **Round 2**: Gemini 3.1 Pro reviews the revised evaluation. Claude Opus revises again.
4. **Compare** — Determines per-dimension and overall winner

All LLM calls have retry with exponential backoff (3 attempts). If any round's critique/revision fails, the pipeline degrades gracefully using scores from the previous round.

### Evaluation Dimensions

Scores use the full 1-10 scale: 1-2 = unusable or almost no evidence, 3-4 = poor with major failures, 5-6 = mixed or acceptable with clear flaws, 7-8 = strong with minor/moderate flaws, 9 = excellent with tiny issues, and 10 = essentially flawless and rare.

| Dimension | 1 (low) | 5 (mid) | 10 (high) |
|-----------|---------|---------|-----------|
| **Prompt Adherence** | Ignores requested subject or medium | Main subject captured, but important details, text/logos, layout, style, or constraints are missed | Explicit and strongly implied requirements are faithfully rendered |
| **Photorealism** | Artificial, distorted, or physically incoherent for the requested medium | Plausible at first glance, but obvious AI tells or material/medium errors remain | Photo-real when requested; otherwise materially believable, artifact-free, and medium-consistent |
| **Aesthetic Quality** | Visually unpleasant, messy, or low-craft | Acceptable but unremarkable, with limited polish or impact | Exceptional visual appeal, polish, craft, and genre-appropriate impact |
| **Composition** | Chaotic, unbalanced, confusing, or badly cropped | Competent framing/layout, but weak hierarchy, depth, or flow | Masterful framing, balance, hierarchy, depth, cropping, and visual flow |
| **Color Accuracy** | Colors contradict the prompt or look physically implausible | Adequate palette, but noticeable color, lighting, white-balance, or material issues remain | Prompt-specified colors, material colors, lighting, and palette harmony are rich and consistent |
| **Creativity** | Generic, literal, or unimaginative | Competent interpretation with some appropriate choices | Surprising, delightful choices that enhance the prompt without violating requirements |

## Dashboard

The Streamlit app provides:

- **Side-by-side images** — GPT Image-2 (left) vs Gemini 3 Pro (right)
- **Animated pipeline progress** — 6-step tracker with pulse/shimmer/check animations
- **Winner banner** — Model name, dimensions won count, overall mean comparison
- **Radar chart** — Plotly polar plot overlaying pre-critique (faded) and post-critique (solid) scores
- **Dimension cards** — 3x2 grid with per-dimension scores, deltas, and winner indicators
- **Critique transcript** — Expandable multi-round view (initial eval → GPT-5.4 review → revision → Gemini 3.1 Pro review → final revision)
- **Raw JSON inspectors** — Expandable panels for each pipeline stage (including round 2 data)
- **Pre-baked results** — Dropdown to load previous runs from `runs/` directory
- **Language switcher** — English / 中文 (sidebar, default: Chinese)

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone <repo-url>
cd evalsimagegen
uv sync
```

### API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | GPT Image-2 generation + GPT-5.4 critique |
| `ANTHROPIC_API_KEY` | Yes | Claude Opus evaluation + revision |
| `GOOGLE_API_KEY` | Yes | Gemini 3 Pro generation + Gemini 3.1 Pro critique |
| `OPENAI_BASE_URL` | No | Azure OpenAI endpoint |
| `ANTHROPIC_BASE_URL` | No | Databricks proxy endpoint |

## Usage

### Run the app

```bash
streamlit run app.py
```

### Run tests

```bash
pytest
pytest tests/test_winner.py::test_tiebreak_by_largest_dimension_lead -v
```

## Project Structure

```
evalsimagegen/
├── app.py              # Streamlit dashboard (6-step progress, multi-round transcript)
├── pipeline.py         # Orchestrator (multi-round loop + convergence + graceful degradation)
├── generate.py         # Parallel image generation (GPT Image-2 + Gemini 3 Pro)
├── evaluate.py         # Claude Opus initial evaluation (6 dimensions)
├── critique.py         # GPT-5.4 critique (R1) + Gemini 3.1 Pro critique (R2)
├── revise.py           # Claude Opus revision with critique context
├── compare.py          # Winner determination (mean → tiebreak → draw)
├── schemas.py          # Pydantic models (ImageEvaluation, CritiqueResponse, etc.)
├── config.py           # Centralized model names, retry settings, pipeline params
├── prompts.py          # Shared rubric text + system prompts (single source of truth)
├── utils.py            # image_to_b64(), retry_llm_call(), parse_llm_json()
├── i18n.py             # Translations (EN/ZH), t() and dim_label() helpers
├── static/
│   └── style.css       # Dark theme CSS
├── tests/
│   ├── conftest.py     # Adds project root to sys.path
│   ├── test_schemas.py # Pydantic model validation tests
│   ├── test_winner.py  # Winner determination logic tests
│   ├── test_utils.py   # Shared utility tests (retry, JSON parse, image encoding)
│   └── test_pipeline.py# Convergence check, fallback, PipelineResult property tests
├── runs/               # Persisted pipeline outputs (gitignored)
│   └── YYYYMMDD_HHMMSS/
│       ├── prompt.txt
│       ├── gpt_image_2.png
│       ├── gemini_3_pro.png
│       ├── evaluation.json
│       ├── critique_r1.json
│       ├── revised_r1.json
│       ├── critique_r2.json
│       ├── revised_r2.json
│       ├── comparison.json
│       └── summary.json
├── pyproject.toml
├── .env.example
└── CLAUDE.md
```

## Configuration

All pipeline parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_CRITIQUE_ROUNDS` | 2 | Maximum critique-revision rounds |
| `CONVERGENCE_THRESHOLD` | 1 | Stop loop when all score deltas below this |
| `MAX_RETRIES` | 3 | LLM call retry attempts |
| `RETRY_BACKOFF` | 2.0 | Exponential backoff base (seconds) |
| `IMAGE_MAX_SIZE` | 768 | Max image dimension for LLM input |
| `LLM_MAX_TOKENS` | 4096 | Max tokens for LLM responses |

## Internationalization

All UI strings go through `t(key, **kwargs)` from `i18n.py`. Dimension names use `dim_label(dimension)`. The default language is Chinese (`"zh"`), switchable at runtime via the sidebar.

Supported languages: **English**, **中文 (Chinese Mandarin)**
