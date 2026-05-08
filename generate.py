"""Parallel image generation using GPT Image-2 and Gemini 3 Pro.

Each generator is a self-contained function. Both run in parallel
via ThreadPoolExecutor. Images saved as PNGs to runs/ directory.
"""

import base64
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI
from google import genai

from config import (
    GEMINI_SAFETY_CATEGORIES,
    GEMINI_SAFETY_THRESHOLD,
    IMAGE_GEN_GEMINI_MODEL,
    IMAGE_GEN_GEMINI_TIMEOUT,
    IMAGE_GEN_GPT_MODEL,
    IMAGE_GEN_GPT_TIMEOUT,
    IMAGE_GEN_TIMEOUT,
    OPENAI_IMAGE_PARTIAL_IMAGES,
    OPENAI_IMAGE_QUALITY,
    OPENAI_IMAGE_REQUEST_TIMEOUT,
)
from utils import retry_llm_call

load_dotenv()


def generate_gpt_image(prompt: str, output_path: Path, reference_image_path: Path | None = None) -> Path:
    """Generate an image with GPT Image-2 and save to output_path."""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=OPENAI_IMAGE_REQUEST_TIMEOUT,
        max_retries=0,
    )
    if reference_image_path:
        with reference_image_path.open("rb") as image_file:
            response = client.images.edit(
                model=IMAGE_GEN_GPT_MODEL,
                image=[image_file],
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality=OPENAI_IMAGE_QUALITY,
                stream=True,
                partial_images=OPENAI_IMAGE_PARTIAL_IMAGES,
            )
    else:
        response = client.images.generate(
            model=IMAGE_GEN_GPT_MODEL,
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality=OPENAI_IMAGE_QUALITY,
            stream=True,
            partial_images=OPENAI_IMAGE_PARTIAL_IMAGES,
        )
    img_bytes = _openai_image_response_bytes(response)

    output_path.write_bytes(img_bytes)
    return output_path


def _openai_image_response_bytes(response) -> bytes:
    if hasattr(response, "data"):
        image_data = response.data[0]
        if image_data.b64_json:
            return base64.b64decode(image_data.b64_json)
        if image_data.url:
            import httpx
            url = image_data.url
            if not url.startswith("https://"):
                raise ValueError(f"Unexpected non-https URL from OpenAI: {url[:80]}")
            return httpx.get(url, timeout=30, follow_redirects=False).content
        raise ValueError("GPT Image-2 returned neither b64_json nor url")

    last_partial_b64 = None
    for event in response:
        image_b64 = getattr(event, "b64_json", None)
        if not image_b64:
            continue
        event_type = getattr(event, "type", "")
        if event_type.endswith("completed"):
            return base64.b64decode(image_b64)
        last_partial_b64 = image_b64

    if last_partial_b64:
        return base64.b64decode(last_partial_b64)
    raise ValueError("GPT Image-2 stream returned no image data")


def generate_gemini_image(prompt: str, output_path: Path, reference_image_path: Path | None = None) -> Path:
    """Generate an image with Gemini 3 Pro via Vertex AI and save to output_path."""
    client = genai.Client(vertexai=True, api_key=os.environ["GOOGLE_API_KEY"])
    contents = prompt
    if reference_image_path:
        contents = [
            prompt,
            genai.types.Part.from_bytes(
                data=reference_image_path.read_bytes(),
                mime_type=_mime_type_for_path(reference_image_path),
            ),
        ]
    response = client.models.generate_content(
        model=IMAGE_GEN_GEMINI_MODEL,
        contents=contents,
        config=genai.types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=_gemini_safety_settings(),
        ),
    )
    if not response.candidates or not response.candidates[0].content.parts:
        raise ValueError("Gemini 3 Pro returned no image")

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            output_path.write_bytes(part.inline_data.data)
            return output_path

    raise ValueError("Gemini 3 Pro response contained no image data")


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


def _mime_type_for_path(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "image/png"


def generate_images(
    prompt: str,
    run_dir: Path,
    on_model_done: Callable[[str], None] | None = None,
    reference_image_path: Path | None = None,
) -> dict[str, Path | Exception]:
    """Generate images from both models in parallel.

    Returns dict with 'gpt_image_2' and 'gemini_3_pro' keys.
    Values are either Path (success) or Exception (failure).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    gpt_path = run_dir / "gpt_image_2.png"
    gemini_path = run_dir / "gemini_3_pro.png"

    results: dict[str, Path | Exception] = {}

    def _run(model_key: str, fn, path: Path):
        image_max_attempts = 1 if model_key == "gpt_image_2" and reference_image_path else None
        if image_max_attempts is None:
            result = retry_llm_call(lambda: fn(prompt, path, reference_image_path))
        else:
            result = retry_llm_call(lambda: fn(prompt, path, reference_image_path), max_attempts=image_max_attempts)
        if on_model_done:
            on_model_done(model_key)
        return result

    executor = ThreadPoolExecutor(max_workers=2)
    try:
        futures: dict[str, Future] = {
            "gpt_image_2": executor.submit(_run, "gpt_image_2", generate_gpt_image, gpt_path),
            "gemini_3_pro": executor.submit(_run, "gemini_3_pro", generate_gemini_image, gemini_path),
        }
        timeouts = {
            "gpt_image_2": IMAGE_GEN_GPT_TIMEOUT,
            "gemini_3_pro": IMAGE_GEN_GEMINI_TIMEOUT,
        }
        for model, future in futures.items():
            timeout = timeouts.get(model, IMAGE_GEN_TIMEOUT)
            try:
                results[model] = future.result(timeout=timeout)
            except FutureTimeoutError:
                future.cancel()
                results[model] = TimeoutError(f"timed out after {timeout}s")
            except Exception as e:
                results[model] = e
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return results
