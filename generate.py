"""Parallel image generation using GPT Image-2 and Gemini 3 Pro.

Each generator is a self-contained function. Both run in parallel
via ThreadPoolExecutor. Images saved as PNGs to runs/ directory.
"""

import base64
import os
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI
from google import genai

from config import GEMINI_SAFETY_CATEGORIES, GEMINI_SAFETY_THRESHOLD, IMAGE_GEN_GEMINI_MODEL, IMAGE_GEN_GPT_MODEL, IMAGE_GEN_TIMEOUT
from utils import retry_llm_call

load_dotenv()


def generate_gpt_image(prompt: str, output_path: Path) -> Path:
    """Generate an image with GPT Image-2 and save to output_path."""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    response = client.images.generate(
        model=IMAGE_GEN_GPT_MODEL,
        prompt=prompt,
        n=1,
        size="1024x1024",
    )
    image_data = response.data[0]
    if image_data.b64_json:
        img_bytes = base64.b64decode(image_data.b64_json)
    elif image_data.url:
        import httpx
        url = image_data.url
        if not url.startswith("https://"):
            raise ValueError(f"Unexpected non-https URL from OpenAI: {url[:80]}")
        img_bytes = httpx.get(url, timeout=30, follow_redirects=False).content
    else:
        raise ValueError("GPT Image-2 returned neither b64_json nor url")

    output_path.write_bytes(img_bytes)
    return output_path


def generate_gemini_image(prompt: str, output_path: Path) -> Path:
    """Generate an image with Gemini 3 Pro via Vertex AI and save to output_path."""
    client = genai.Client(vertexai=True, api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model=IMAGE_GEN_GEMINI_MODEL,
        contents=prompt,
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


def generate_images(
    prompt: str,
    run_dir: Path,
    on_model_done: Callable[[str], None] | None = None,
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
        result = retry_llm_call(lambda: fn(prompt, path))
        if on_model_done:
            on_model_done(model_key)
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures: dict[str, Future] = {
            "gpt_image_2": executor.submit(_run, "gpt_image_2", generate_gpt_image, gpt_path),
            "gemini_3_pro": executor.submit(_run, "gemini_3_pro", generate_gemini_image, gemini_path),
        }
        for model, future in futures.items():
            try:
                results[model] = future.result(timeout=IMAGE_GEN_TIMEOUT)
            except Exception as e:
                results[model] = e

    return results
