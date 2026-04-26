"""Claude Opus evaluation of both generated images.

Sends both images as base64 content blocks with the anchored rubric.
Returns structured ImageEvaluation for each model.
"""

import base64
import io
import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image

from schemas import ImageEvaluation

load_dotenv()


def _image_to_b64(path: Path, max_size: int = 768) -> tuple[str, str]:
    """Resize image and return (base64_data, media_type)."""
    img = Image.open(path)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

RUBRIC = """Score each dimension 1-10 using these anchors:

**Prompt adherence**
- 1 = unrelated to prompt
- 5 = captures main subject but misses details
- 10 = every element of the prompt is faithfully rendered

**Photorealism**
- 1 = clearly artificial/distorted
- 5 = plausible at first glance but obvious tells
- 10 = indistinguishable from a photograph

**Aesthetic quality**
- 1 = visually unpleasant
- 5 = acceptable but unremarkable
- 10 = gallery-worthy composition and visual impact

**Composition**
- 1 = chaotic/unbalanced layout
- 5 = competent framing
- 10 = masterful use of space, depth, and visual flow

**Color accuracy**
- 1 = colors contradict the prompt or are unrealistic
- 5 = adequate color palette
- 10 = rich, accurate, well-harmonized colors

**Creativity**
- 1 = generic/literal interpretation
- 5 = competent interpretation
- 10 = surprising and delightful artistic choices while respecting the prompt"""

SYSTEM_PROMPT = f"""You are an expert image quality evaluator. You will be shown two AI-generated images (Image A from GPT Image-2, Image B from Gemini 3 Pro) created from the same prompt.

Evaluate EACH image independently across 6 dimensions. Be precise and rigorous.

{RUBRIC}

For each dimension, provide a score (1-10) and detailed reasoning explaining your score with specific visual evidence from the image.

Return TWO evaluations as JSON — one for each model. Use this exact schema:

{{
  "model_a": {{
    "model_name": "GPT Image-2",
    "prompt_adherence": {{"score": N, "reasoning": "..."}},
    "photorealism": {{"score": N, "reasoning": "..."}},
    "aesthetic_quality": {{"score": N, "reasoning": "..."}},
    "composition": {{"score": N, "reasoning": "..."}},
    "color_accuracy": {{"score": N, "reasoning": "..."}},
    "creativity": {{"score": N, "reasoning": "..."}}
  }},
  "model_b": {{
    "model_name": "Gemini 3 Pro",
    "prompt_adherence": {{"score": N, "reasoning": "..."}},
    "photorealism": {{"score": N, "reasoning": "..."}},
    "aesthetic_quality": {{"score": N, "reasoning": "..."}},
    "composition": {{"score": N, "reasoning": "..."}},
    "color_accuracy": {{"score": N, "reasoning": "..."}},
    "creativity": {{"score": N, "reasoning": "..."}}
  }}
}}

Return ONLY valid JSON, no other text."""


def evaluate_images(
    prompt: str,
    image_a_path: Path,
    image_b_path: Path,
) -> tuple[ImageEvaluation, ImageEvaluation]:
    """Evaluate both images using Claude Opus.

    Returns (eval_model_a, eval_model_b).
    """
    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )

    image_a_b64, media_a = _image_to_b64(image_a_path)
    image_b_b64, media_b = _image_to_b64(image_b_path)

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'The prompt used to generate both images: "{prompt}"'},
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
                    {"type": "text", "text": "Evaluate both images. Return JSON only."},
                ],
            }
        ],
    ) as stream:
        response_text = stream.get_final_text()
    # Strip markdown code fences if present
    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines)

    data = json.loads(response_text)
    eval_a = ImageEvaluation(**data["model_a"])
    eval_b = ImageEvaluation(**data["model_b"])
    return eval_a, eval_b
