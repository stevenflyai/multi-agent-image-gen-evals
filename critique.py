"""Critique of evaluation scores by the round-1 OpenAI critic (CRITIQUE_MODEL) and
the round-2 Gemini critic (CRITIQUE_ROUND2_MODEL).

Reviews evaluation for inconsistencies, unsupported reasoning,
and potential bias. Returns dimension-level counterarguments.
"""

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

from config import CRITIQUE_MAX_TOKENS, CRITIQUE_MODEL, CRITIQUE_ROUND2_MAX_TOKENS, CRITIQUE_ROUND2_MODEL, GEMINI_SAFETY_CATEGORIES, GEMINI_SAFETY_THRESHOLD
from prompts import CRITIQUE_ROUND2_SYSTEM_PROMPT, CRITIQUE_SYSTEM_PROMPT, image_caption
from schemas import CritiqueResponse, ImageEvaluation, RevisedImageEvaluation
from utils import build_eval_summary, gemini_client, image_to_b64, parse_llm_json, retry_llm_call

load_dotenv()


def critique_evaluation(
    prompt: str,
    eval_a: ImageEvaluation,
    eval_b: ImageEvaluation,
    image_a_path: Path,
    image_b_path: Path,
    *,
    model_a_label: str,
    model_b_label: str,
) -> CritiqueResponse:
    """The round-1 OpenAI critic reviews Claude Opus's evaluation."""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    image_a_b64, media_a = image_to_b64(image_a_path)
    image_b_b64, media_b = image_to_b64(image_b_path)
    eval_summary = build_eval_summary(eval_a, eval_b)

    def _call() -> str:
        # Uses the Responses API so reasoning/"pro" critics (e.g. gpt-5.4-pro,
        # which is not served via chat.completions) work alongside standard models.
        response = client.responses.create(
            model=CRITIQUE_MODEL,
            instructions=CRITIQUE_SYSTEM_PROMPT,
            max_output_tokens=CRITIQUE_MAX_TOKENS,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f'Prompt: "{prompt}"\n\nClaude Opus evaluation:\n{eval_summary}'},
                        {"type": "input_text", "text": image_caption("A", model_a_label)},
                        {"type": "input_image", "image_url": f"data:{media_a};base64,{image_a_b64}"},
                        {"type": "input_text", "text": image_caption("B", model_b_label)},
                        {"type": "input_image", "image_url": f"data:{media_b};base64,{image_b_b64}"},
                        {"type": "input_text", "text": "Review this evaluation. Return JSON only."},
                    ],
                },
            ],
        )
        content = response.output_text
        if not content:
            raise ValueError(f"OpenAI returned empty content ({_openai_response_details(response)})")
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
    *,
    model_a_label: str,
    model_b_label: str,
) -> CritiqueResponse:
    """The round-2 Gemini critic reviews the revised evaluation."""
    client = gemini_client()

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
                image_caption("A", model_a_label),
                image_a_part,
                image_caption("B", model_b_label),
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


def _openai_response_details(response: object) -> str:
    """Extract useful diagnostics from an empty OpenAI Responses API result.

    A reasoning critic that exhausts max_output_tokens before emitting its message
    returns status="incomplete" with empty output_text, so surface the token
    breakdown that explains it rather than just the bare status.
    """
    details: list[str] = [f"status={getattr(response, 'status', 'unknown')}"]

    incomplete = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None)
    if reason:
        details.append(f"incomplete_reason={reason}")

    usage = getattr(response, "usage", None)
    if usage:
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is not None:
            details.append(f"output_tokens={output_tokens}")
        reasoning_tokens = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None)
        if reasoning_tokens is not None:
            details.append(f"reasoning_tokens={reasoning_tokens}")

    if reason == "max_output_tokens":
        details.append(f"budget exhausted before any text was emitted; raise CRITIQUE_MAX_TOKENS (currently {CRITIQUE_MAX_TOKENS})")

    return "; ".join(details)


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
