"""Centralized configuration for models, retries, and pipeline settings."""

# --- Model configuration ---
EVAL_MODEL = "claude-opus-4-7"
CRITIQUE_MODEL = "gpt-5.4"
CRITIQUE_ROUND2_MODEL = "gemini-3.1-pro-preview"
IMAGE_GEN_GPT_MODEL = "gpt-image-2"
IMAGE_GEN_GEMINI_MODEL = "gemini-3-pro-image-preview"

# --- Image processing ---
IMAGE_MAX_SIZE = 768
IMAGE_QUALITY = 85

# --- LLM call settings ---
LLM_MAX_TOKENS = 4096
CRITIQUE_ROUND2_MAX_TOKENS = 8192
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

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
HIL_ENABLED_BY_DEFAULT = True
