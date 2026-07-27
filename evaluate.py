"""Claude Opus evaluation of both generated images.

Sends both images as base64 content blocks with the anchored rubric.
Returns structured ImageEvaluation for each model.
"""

import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from config import EVAL_MODEL, LLM_MAX_TOKENS
from prompts import eval_system_prompt, image_caption
from schemas import ImageEvaluation, PromptDifficulty
from utils import image_to_b64, parse_llm_json, retry_llm_call

load_dotenv()


def evaluate_images(
    prompt: str,
    image_a_path: Path,
    image_b_path: Path,
    *,
    model_a_label: str,
    model_b_label: str,
) -> tuple[ImageEvaluation, ImageEvaluation, PromptDifficulty | None]:
    """Evaluate both images using Claude Opus.

    Args:
        model_a_label: Display label of the generator that produced image A.
        model_b_label: Display label of the generator that produced image B.

    Returns (eval_model_a, eval_model_b, prompt_difficulty).
    """
    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )

    image_a_b64, media_a = image_to_b64(image_a_path)
    image_b_b64, media_b = image_to_b64(image_b_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f'The prompt used to generate both images: "{prompt}"'},
                {"type": "text", "text": image_caption("A", model_a_label)},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_a,
                        "data": image_a_b64,
                    },
                },
                {"type": "text", "text": image_caption("B", model_b_label)},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_b,
                        "data": image_b_b64,
                    },
                },
                {"type": "text", "text": "Evaluate both images. Return JSON only."},
            ],
        }
    ]

    def _call() -> str:
        with client.messages.stream(
            model=EVAL_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            system=eval_system_prompt(model_a_label, model_b_label),
            messages=messages,
        ) as stream:
            return stream.get_final_text()

    data = retry_llm_call(lambda: parse_llm_json(_call()))
    eval_a = ImageEvaluation(**data["model_a"])
    eval_b = ImageEvaluation(**data["model_b"])
    prompt_difficulty = data.get("prompt_difficulty")
    if prompt_difficulty not in {"easy", "medium", "hard"}:
        prompt_difficulty = None
    return eval_a, eval_b, prompt_difficulty
