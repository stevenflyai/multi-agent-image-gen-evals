"""GPT-5.4 critique of Claude Opus's initial evaluation.

Reviews the evaluation for inconsistencies, unsupported reasoning,
and potential bias. Returns dimension-level counterarguments.
"""

import base64
import io
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from schemas import CritiqueResponse, ImageEvaluation

load_dotenv()


def _image_to_b64(path: Path, max_size: int = 768) -> tuple[str, str]:
    """Resize image and return (base64_data, media_type)."""
    img = Image.open(path)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

RUBRIC = """The evaluation used these anchored rubrics (1/5/10):

**Prompt adherence**: 1=unrelated, 5=main subject but misses details, 10=every element faithfully rendered
**Photorealism**: 1=clearly artificial, 5=plausible at first glance, 10=indistinguishable from photograph
**Aesthetic quality**: 1=visually unpleasant, 5=acceptable, 10=gallery-worthy
**Composition**: 1=chaotic/unbalanced, 5=competent framing, 10=masterful use of space
**Color accuracy**: 1=colors contradict prompt, 5=adequate palette, 10=rich and well-harmonized
**Creativity**: 1=generic/literal, 5=competent, 10=surprising and delightful"""

SYSTEM_PROMPT = f"""You are a critical reviewer of AI image evaluations. Another model (Claude Opus) has evaluated two AI-generated images. Your job is to review that evaluation for:

1. Scoring inconsistencies (e.g., high photorealism score but reasoning mentions artifacts)
2. Unsupported reasoning (claims about image features that aren't visible)
3. Potential bias toward either model (systematically higher scores for one)
4. Whether scores match the rubric anchor definitions

{RUBRIC}

Both images are included so you can verify claims made in the evaluation.

Return your critique as JSON with this exact schema:

{{
  "overall_assessment": "Your overall assessment of the evaluation quality",
  "dimension_critiques": [
    {{
      "dimension": "prompt_adherence",
      "original_score_model_a": N,
      "original_score_model_b": N,
      "critique": "What's wrong with this scoring and why",
      "suggested_score_model_a": N,
      "suggested_score_model_b": N
    }}
  ],
  "bias_detection": "Any systematic bias detected or 'No systematic bias detected'"
}}

Include ALL 6 dimensions in dimension_critiques, even if you agree with the original scores.
Return ONLY valid JSON, no other text."""


def critique_evaluation(
    prompt: str,
    eval_a: ImageEvaluation,
    eval_b: ImageEvaluation,
    image_a_path: Path,
    image_b_path: Path,
) -> CritiqueResponse:
    """GPT-5.4 reviews Claude Opus's evaluation."""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    image_a_b64, media_a = _image_to_b64(image_a_path)
    image_b_b64, media_b = _image_to_b64(image_b_path)

    eval_summary = json.dumps(
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

    response = client.chat.completions.create(
        model="gpt-5.4",
        max_completion_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'Prompt: "{prompt}"\n\nClaude Opus evaluation:\n{eval_summary}'},
                    {"type": "text", "text": "Image A (GPT Image-2):"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_a};base64,{image_a_b64}"},
                    },
                    {"type": "text", "text": "Image B (Gemini 3 Pro):"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_b};base64,{image_b_b64}"},
                    },
                    {"type": "text", "text": "Review this evaluation. Return JSON only."},
                ],
            },
        ],
    )

    response_text = response.choices[0].message.content
    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines)

    return CritiqueResponse(**json.loads(response_text))
