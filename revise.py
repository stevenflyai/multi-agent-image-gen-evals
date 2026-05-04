"""Claude Opus full re-evaluation with critique context.

Re-evaluates ALL 6 dimensions (not targeted adjustment) with
a reviewer's critique as additional context. Both images included
so Opus can genuinely verify critique claims.
"""

import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from config import EVAL_MODEL, LLM_MAX_TOKENS
from prompts import REVISE_SYSTEM_PROMPT
from schemas import CritiqueResponse, ImageEvaluation, RevisedEvaluation, RevisedImageEvaluation
from utils import build_eval_summary, image_to_b64, parse_llm_json, retry_llm_call

load_dotenv()


def revise_evaluation(
    prompt: str,
    eval_a: ImageEvaluation | RevisedImageEvaluation,
    eval_b: ImageEvaluation | RevisedImageEvaluation,
    critique: CritiqueResponse,
    image_a_path: Path,
    image_b_path: Path,
) -> RevisedEvaluation:
    """Claude Opus revises evaluation considering a reviewer's critique."""
    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )

    image_a_b64, media_a = image_to_b64(image_a_path)
    image_b_b64, media_b = image_to_b64(image_b_path)

    original_eval = build_eval_summary(eval_a, eval_b)

    critique_summary = json.dumps(critique.model_dump(), indent=2)
    reviewer_name = critique.critic_model or "Reviewer"

    def _call() -> str:
        with client.messages.stream(
            model=EVAL_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            system=REVISE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f'Prompt: "{prompt}"\n\nYour original evaluation:\n{original_eval}\n\n{reviewer_name} reviewer critique:\n{critique_summary}',
                        },
                        {"type": "text", "text": "Image A (GPT Image-2):"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_a,
                                "data": image_a_b64,
                            },
                        },
                        {"type": "text", "text": "Image B (Gemini 3 Pro):"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_b,
                                "data": image_b_b64,
                            },
                        },
                        {"type": "text", "text": "Re-evaluate both images considering the critique. Return JSON only."},
                    ],
                }
            ],
        ) as stream:
            return stream.get_final_text()

    data = retry_llm_call(lambda: parse_llm_json(_call()))
    return RevisedEvaluation(**data)
