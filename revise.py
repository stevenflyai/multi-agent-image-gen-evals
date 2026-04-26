"""Claude Opus full re-evaluation with critique context.

Re-evaluates ALL 6 dimensions (not targeted adjustment) with
GPT-5.4's critique as additional context. Both images included
so Opus can genuinely verify critique claims.
"""

import base64
import io
import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image

from schemas import CritiqueResponse, ImageEvaluation, RevisedEvaluation

load_dotenv()


def _image_to_b64(path: Path, max_size: int = 768) -> tuple[str, str]:
    """Resize image and return (base64_data, media_type)."""
    img = Image.open(path)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

RUBRIC = """Score each dimension 1-10 using these anchors:

**Prompt adherence**: 1=unrelated, 5=main subject but misses details, 10=every element faithfully rendered
**Photorealism**: 1=clearly artificial, 5=plausible at first glance, 10=indistinguishable from photograph
**Aesthetic quality**: 1=visually unpleasant, 5=acceptable, 10=gallery-worthy
**Composition**: 1=chaotic/unbalanced, 5=competent framing, 10=masterful use of space
**Color accuracy**: 1=colors contradict prompt, 5=adequate palette, 10=rich and well-harmonized
**Creativity**: 1=generic/literal, 5=competent, 10=surprising and delightful"""

SYSTEM_PROMPT = f"""You are an expert image quality evaluator performing a REVISED evaluation. You previously evaluated two images, and a reviewer (GPT-5.4) has challenged some of your scores.

{RUBRIC}

Re-evaluate ALL 6 dimensions for BOTH images. For each dimension:
- Consider the reviewer's critique carefully
- Look at the images again to verify their claims
- Accept or reject each critique point based on what you actually see
- Provide your revised score and reasoning

Return JSON with this exact schema:

{{
  "model_a": {{
    "model_name": "GPT Image-2",
    "prompt_adherence": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}},
    "photorealism": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}},
    "aesthetic_quality": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}},
    "composition": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}},
    "color_accuracy": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}},
    "creativity": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}}
  }},
  "model_b": {{
    "model_name": "Gemini 3 Pro",
    "prompt_adherence": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}},
    "photorealism": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}},
    "aesthetic_quality": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}},
    "composition": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}},
    "color_accuracy": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}},
    "creativity": {{"score": N, "reasoning": "...", "critique_accepted": true/false, "revision_note": "..."}}
  }}
}}

Return ONLY valid JSON, no other text."""


def revise_evaluation(
    prompt: str,
    eval_a: ImageEvaluation,
    eval_b: ImageEvaluation,
    critique: CritiqueResponse,
    image_a_path: Path,
    image_b_path: Path,
) -> RevisedEvaluation:
    """Claude Opus revises evaluation considering GPT-5.4's critique."""
    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )

    image_a_b64, media_a = _image_to_b64(image_a_path)
    image_b_b64, media_b = _image_to_b64(image_b_path)

    original_eval = json.dumps(
        {
            "model_a": {
                "model_name": eval_a.model_name,
                **{d: {"score": getattr(eval_a, d).score, "reasoning": getattr(eval_a, d).reasoning} for d in [
                    "prompt_adherence", "photorealism", "aesthetic_quality",
                    "composition", "color_accuracy", "creativity"
                ]},
            },
            "model_b": {
                "model_name": eval_b.model_name,
                **{d: {"score": getattr(eval_b, d).score, "reasoning": getattr(eval_b, d).reasoning} for d in [
                    "prompt_adherence", "photorealism", "aesthetic_quality",
                    "composition", "color_accuracy", "creativity"
                ]},
            },
        },
        indent=2,
    )

    critique_summary = json.dumps(critique.model_dump(), indent=2)

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f'Prompt: "{prompt}"\n\nYour original evaluation:\n{original_eval}\n\nGPT-5.4 reviewer critique:\n{critique_summary}',
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
        response_text = stream.get_final_text()
    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines)

    return RevisedEvaluation(**json.loads(response_text))
