"""Critique of evaluation scores by GPT-5.4 (round 1) and Gemini 3.1 Pro (round 2).

Reviews evaluation for inconsistencies, unsupported reasoning,
and potential bias. Returns dimension-level counterarguments.
"""

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

from config import CRITIQUE_MODEL, CRITIQUE_ROUND2_MAX_TOKENS, CRITIQUE_ROUND2_MODEL, GEMINI_SAFETY_CATEGORIES, GEMINI_SAFETY_THRESHOLD, LLM_MAX_TOKENS
from prompts import CRITIQUE_ROUND2_SYSTEM_PROMPT, CRITIQUE_SYSTEM_PROMPT
from schemas import CritiqueResponse, ImageEvaluation, RevisedImageEvaluation
from utils import build_eval_summary, image_to_b64, parse_llm_json, retry_llm_call

load_dotenv()


def critique_evaluation(
    prompt: str,
    eval_a: ImageEvaluation,
    eval_b: ImageEvaluation,
    image_a_path: Path,
    image_b_path: Path,
) -> CritiqueResponse:
    """GPT-5.4 reviews Claude Opus's evaluation (round 1)."""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    image_a_b64, media_a = image_to_b64(image_a_path)
    image_b_b64, media_b = image_to_b64(image_b_path)
    eval_summary = build_eval_summary(eval_a, eval_b)

    def _call() -> str:
        response = client.chat.completions.create(
            model=CRITIQUE_MODEL,
            max_completion_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
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
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"OpenAI returned empty content (finish_reason: {response.choices[0].finish_reason})")
        return content

    data = retry_llm_call(lambda: parse_llm_json(_call()))
    return CritiqueResponse(round=1, critic_model=CRITIQUE_MODEL, **data)


def critique_evaluation_gemini(
    prompt: str,
    revised_a: RevisedImageEvaluation,
    revised_b: RevisedImageEvaluation,
    image_a_path: Path,
    image_b_path: Path,
    raw_output_path: Path | None = None,
) -> CritiqueResponse:
    """Gemini 3.1 Pro reviews revised evaluation (round 2) via Vertex AI."""
    client = genai.Client(vertexai=True, api_key=os.environ["GOOGLE_API_KEY"])

    image_a_b64, media_a = image_to_b64(image_a_path)
    image_b_b64, media_b = image_to_b64(image_b_path)
    eval_summary = build_eval_summary(revised_a, revised_b)

    user_text = (
        f'Prompt: "{prompt}"\n\n'
        f"Claude Opus revised evaluation:\n{eval_summary}\n\n"
        "Review this revised evaluation. Return JSON only."
    )

    image_a_part = genai.types.Part.from_bytes(
        data=base64.b64decode(image_a_b64),
        mime_type=media_a,
    )
    image_b_part = genai.types.Part.from_bytes(
        data=base64.b64decode(image_b_b64),
        mime_type=media_b,
    )

    def _call() -> str:
        response = client.models.generate_content(
            model=CRITIQUE_ROUND2_MODEL,
            contents=[
                user_text,
                "Image A (GPT Image-2):",
                image_a_part,
                "Image B (Gemini 3 Pro):",
                image_b_part,
            ],
            config=genai.types.GenerateContentConfig(
                system_instruction=CRITIQUE_ROUND2_SYSTEM_PROMPT,
                max_output_tokens=CRITIQUE_ROUND2_MAX_TOKENS,
                response_mime_type="application/json",
                safety_settings=_gemini_safety_settings(),
            ),
        )
        try:
            text = response.text
        except Exception as exc:
            details = _gemini_response_details(response)
            raise ValueError(f"Gemini returned no text content ({details}; text_error={exc})") from exc
        if not text:
            details = _gemini_response_details(response)
            raise ValueError(f"Gemini returned no text content ({details})")
        return text

    attempt_number = 0

    def _call_and_parse() -> dict:
        nonlocal attempt_number
        attempt_number += 1
        text = _call()
        if raw_output_path:
            raw_output_path.write_text(text)
            raw_output_path.with_name(f"{raw_output_path.stem}_attempt_{attempt_number}{raw_output_path.suffix}").write_text(text)
        return parse_llm_json(text)

    data = retry_llm_call(_call_and_parse)
    return CritiqueResponse(round=2, critic_model=CRITIQUE_ROUND2_MODEL, **data)


def _gemini_safety_settings() -> list[genai.types.SafetySetting]:
    """Build Gemini safety settings from centralized config."""
    threshold = getattr(genai.types.HarmBlockThreshold, GEMINI_SAFETY_THRESHOLD)
    return [
        genai.types.SafetySetting(
            category=getattr(genai.types.HarmCategory, category),
            threshold=threshold,
        )
        for category in GEMINI_SAFETY_CATEGORIES
    ]


def _gemini_response_details(response: object) -> str:
    """Extract useful diagnostics from an empty Gemini response."""
    details: list[str] = []
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback:
        details.append(f"prompt_feedback={prompt_feedback}")

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        candidate = candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason:
            details.append(f"finish_reason={finish_reason}")
        safety_ratings = getattr(candidate, "safety_ratings", None)
        if safety_ratings:
            details.append(f"safety_ratings={safety_ratings}")

    return "; ".join(details) if details else "no candidate text returned; possibly blocked by safety filters"
