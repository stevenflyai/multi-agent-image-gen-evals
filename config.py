"""Centralized configuration for models, retries, and pipeline settings."""

import os

from dotenv import load_dotenv

# Load .env so model names below can be overridden via environment variables.
load_dotenv()

# --- Model configuration ---
# All model names default to the values below but can be overridden in .env,
# e.g. set CRITIQUE_MODEL=gpt-5.4-pro to use a different OpenAI critic, or
# EVAL_MODEL=claude-opus-4-8 for a different Anthropic evaluator.
EVAL_MODEL = os.environ.get("EVAL_MODEL", "claude-opus-4-7")
CRITIQUE_MODEL = os.environ.get("CRITIQUE_MODEL", "gpt-5.4")
CRITIQUE_ROUND2_MODEL = os.environ.get("CRITIQUE_ROUND2_MODEL", "gemini-3.1-pro-preview")
IMAGE_GEN_GPT_MODEL = os.environ.get("IMAGE_GEN_GPT_MODEL", "gpt-image-2")
IMAGE_GEN_GEMINI_MODEL = os.environ.get("IMAGE_GEN_GEMINI_MODEL", "gemini-3-pro-image-preview")

# --- Selectable image generation models ---
# Each entry maps a stable key to its display label, provider routing, and the
# concrete API model id sent to the provider:
#   "openai" -> OpenAI images API on OPENAI_BASE_URL
#   "gemini" -> google-genai client
#   "mai"    -> Azure AI Foundry images endpoint (MAI_BASE_URL / MAI_API_KEY)
IMAGE_GEN_MODELS: dict[str, dict[str, str]] = {
	"gpt-image-2": {"label": "GPT Image 2", "provider": "openai", "api_model": "gpt-image-2"},
	"gemini-3-pro": {"label": "Gemini 3 Pro", "provider": "gemini", "api_model": "gemini-3-pro-image-preview"},
	"mai-image-2": {"label": "MAI-Image 2", "provider": "mai", "api_model": "MAI-Image-2"},
	"mai-image-2.5": {"label": "MAI-Image 2.5", "provider": "mai", "api_model": "MAI-Image-2.5"},
	"gpt-image-1.5": {"label": "GPT Image 1.5", "provider": "openai", "api_model": "gpt-image-1.5"},
}
DEFAULT_MODEL_A = "gpt-image-2"
DEFAULT_MODEL_B = "gemini-3-pro"


def model_label(model_key: str, fallback: str = "") -> str:
    """Display label for a model key.

    Aggregations key off the registry key rather than the recorded label, so a
    model whose label changed between runs (e.g. "GPT Image-2" -> "GPT Image 2")
    still resolves to one identity. `fallback` covers runs whose key is no longer
    in the registry — there the recorded label is the best name available.
    """
    info = IMAGE_GEN_MODELS.get(model_key)
    if info:
        return info["label"]
    return fallback or model_key

# MAI (Azure AI Foundry) images endpoint. Uses Bearer auth and the OpenAI-compatible
# /images/generations route. Falls back to the OpenAI creds if MAI-specific ones are unset.
def _normalize_base_url(url: str | None) -> str | None:
    """The OpenAI SDK appends /images/generations itself; strip it (and any
    trailing slash) so a full generations URL in env still resolves correctly."""
    if not url:
        return url
    url = url.rstrip("/")
    if url.endswith("/images/generations"):
        url = url[: -len("/images/generations")]
    return url


MAI_BASE_URL = _normalize_base_url(os.environ.get("MAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
MAI_API_KEY = os.environ.get("MAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
MAI_OUTPUT_FORMAT = os.environ.get("MAI_OUTPUT_FORMAT", "png")
MAI_OUTPUT_COMPRESSION = int(os.environ.get("MAI_OUTPUT_COMPRESSION", "100"))

# --- Image processing ---
IMAGE_MAX_SIZE = 768
IMAGE_QUALITY = 85

# --- LLM call settings ---
# Evaluation/revision budget. Typical evaluations land around 1.5k output tokens, but
# length varies a lot with prompt density — a detailed prompt (many constraints, plus
# per-dimension reasoning in both English and Chinese) can run several times longer and
# truncate mid-JSON. Revision output is larger still, since it adds critique_accepted
# and revision_note per dimension. Both call sites stream, so a large cap carries no
# HTTP-timeout risk, and unused headroom is not billed.
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "32768"))
# Critique budgets are separate from LLM_MAX_TOKENS because the critics are reasoning
# models: for them max_output_tokens caps reasoning + visible output together, so a
# budget sized for visible JSON alone gets fully consumed by reasoning and the call
# returns status="incomplete" with no text. Measured for gpt-5.4-pro on a round-1
# critique: ~5.3-5.9k reasoning + ~1k JSON. These are caps, not targets — unused
# headroom is not billed, so they are set well above the observed ceiling.
CRITIQUE_MAX_TOKENS = int(os.environ.get("CRITIQUE_MAX_TOKENS", "16384"))
CRITIQUE_ROUND2_MAX_TOKENS = int(os.environ.get("CRITIQUE_ROUND2_MAX_TOKENS", "16384"))
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# --- Gemini endpoint ---
# The GOOGLE_API_KEY can address Gemini two ways:
#   Developer API (default) - generativelanguage.googleapis.com, no regions.
#   Vertex Express Mode     - vertexai=True with an API key. Google pins the
#       project and region for you, and the SDK rejects an explicit `location`
#       alongside `api_key`, so an unavailable model in that region cannot be
#       routed around. gemini-3-pro-image-preview is not served in the region
#       Express Mode assigns, which surfaces as a 404 NOT_FOUND on generation.
# Set GEMINI_USE_VERTEX=true only with credentials that can actually reach the
# models in the assigned region.
GEMINI_USE_VERTEX = os.environ.get("GEMINI_USE_VERTEX", "false").strip().lower() in {"1", "true", "yes"}

# --- Gemini safety settings ---
# Low blocking strength: only block high-probability safety hits.
GEMINI_SAFETY_THRESHOLD = "BLOCK_ONLY_HIGH"
GEMINI_SAFETY_CATEGORIES = (
	"HARM_CATEGORY_HATE_SPEECH",
	"HARM_CATEGORY_DANGEROUS_CONTENT",
	"HARM_CATEGORY_HARASSMENT",
	"HARM_CATEGORY_SEXUALLY_EXPLICIT",
)

# --- Pipeline settings ---
MAX_CRITIQUE_ROUNDS = 2
CONVERGENCE_THRESHOLD = 1  # Stop multi-round loop when all score deltas < this
IMAGE_GEN_TIMEOUT = 600
IMAGE_GEN_GPT_TIMEOUT = 300
IMAGE_GEN_GEMINI_TIMEOUT = 600
OPENAI_IMAGE_REQUEST_TIMEOUT = 240
OPENAI_IMAGE_PARTIAL_IMAGES = 1
OPENAI_IMAGE_QUALITY = "low"
HIL_ENABLED_BY_DEFAULT = True

# --- Orchestrator selection ---
# false (default): pipeline.run_pipeline() / resume_pipeline_from_result().
# true:            the LangGraph orchestrator in graph_pipeline.py, which adds a
#   SQLite checkpoint (runs/checkpoints.db) so a HIL pause is a real suspend and
#   a crashed run resumes without redoing upstream stages. Both write identical
#   run artifacts, so history, the dashboard, and reloading are unaffected either
#   way and the flag can be flipped back at any time.
USE_GRAPH_ORCHESTRATOR = os.environ.get("USE_GRAPH_ORCHESTRATOR", "false").strip().lower() in {"1", "true", "yes"}
