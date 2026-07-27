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

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

from config import (
    DEFAULT_MODEL_A,
    DEFAULT_MODEL_B,
    GEMINI_SAFETY_CATEGORIES,
    GEMINI_SAFETY_THRESHOLD,
    IMAGE_GEN_GEMINI_MODEL,
    IMAGE_GEN_GEMINI_TIMEOUT,
    IMAGE_GEN_GPT_MODEL,
    IMAGE_GEN_GPT_TIMEOUT,
    IMAGE_GEN_MODELS,
    IMAGE_GEN_TIMEOUT,
    MAI_API_KEY,
    MAI_BASE_URL,
    MAI_OUTPUT_COMPRESSION,
    MAI_OUTPUT_FORMAT,
    OPENAI_IMAGE_PARTIAL_IMAGES,
    OPENAI_IMAGE_QUALITY,
    OPENAI_IMAGE_REQUEST_TIMEOUT,
)
from utils import gemini_client, retry_llm_call

load_dotenv()


def generate_gpt_image(
    prompt: str,
    output_path: Path,
    reference_image_path: Path | None = None,
    api_model: str = IMAGE_GEN_GPT_MODEL,
) -> Path:
    """Generate an image with an OpenAI image model and save to output_path."""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=OPENAI_IMAGE_REQUEST_TIMEOUT,
        max_retries=0,
    )
    if reference_image_path:
        with reference_image_path.open("rb") as image_file:
            response = client.images.edit(
                model=api_model,
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
            model=api_model,
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


def generate_mai_image(
    prompt: str,
    output_path: Path,
    reference_image_path: Path | None = None,
    api_model: str = "MAI-Image-2.5",
) -> Path:
    """Generate an image with a MAI model via the Azure AI Foundry images endpoint.

    Mirrors the validated sample: a direct POST to {base}/mai/v1/images/generations
    with Bearer auth and `model` = the deployment name, returning base64 PNG in
    data[0].b64_json. Reference images use the analogous /images/edits route.
    """
    base = (MAI_BASE_URL or "").rstrip("/")
    auth = {"Authorization": f"Bearer {MAI_API_KEY}"}
    if reference_image_path:
        with reference_image_path.open("rb") as image_file:
            response = httpx.post(
                f"{base}/images/edits",
                headers=auth,
                data={"model": api_model, "prompt": prompt, "n": "1", "size": "1024x1024"},
                files={"image": (reference_image_path.name, image_file, _mime_type_for_path(reference_image_path))},
                timeout=OPENAI_IMAGE_REQUEST_TIMEOUT,
            )
    else:
        response = httpx.post(
            f"{base}/images/generations",
            headers={**auth, "Content-Type": "application/json"},
            json={
                "prompt": prompt,
                "model": api_model,
                "size": "1024x1024",
                "n": 1,
                "output_format": MAI_OUTPUT_FORMAT,
                "output_compression": MAI_OUTPUT_COMPRESSION,
            },
            timeout=OPENAI_IMAGE_REQUEST_TIMEOUT,
        )
    if response.status_code != 200:
        raise ValueError(f"{api_model} generation failed (HTTP {response.status_code}): {response.text[:200]}")
    data = response.json().get("data") or []
    if not data or not data[0].get("b64_json"):
        raise ValueError(f"{api_model} returned no image data")
    output_path.write_bytes(base64.b64decode(data[0]["b64_json"]))
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


def generate_gemini_image(
    prompt: str,
    output_path: Path,
    reference_image_path: Path | None = None,
    api_model: str = IMAGE_GEN_GEMINI_MODEL,
) -> Path:
    """Generate an image with a Gemini image model and save to output_path."""
    client = gemini_client()
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
        model=api_model,
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
    model_a: str = DEFAULT_MODEL_A,
    model_b: str = DEFAULT_MODEL_B,
) -> dict[str, Path | Exception]:
    """Generate images from the two selected models in parallel.

    `model_a` / `model_b` are keys into config.IMAGE_GEN_MODELS. The result
    dict always uses the stable slot keys 'gpt_image_2' (slot A) and
    'gemini_3_pro' (slot B) so downstream stages stay decoupled from the
    concrete model choice. Values are either Path (success) or Exception.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    gpt_path = run_dir / "gpt_image_2.png"
    gemini_path = run_dir / "gemini_3_pro.png"

    # Per-slot model selection -> (api_model, generator function by provider).
    provider_fns = {
        "openai": generate_gpt_image,
        "gemini": generate_gemini_image,
        "mai": generate_mai_image,
    }
    slot_models = {"gpt_image_2": model_a, "gemini_3_pro": model_b}

    def _spec(model_key: str):
        info = IMAGE_GEN_MODELS.get(model_key, IMAGE_GEN_MODELS[DEFAULT_MODEL_A])
        return provider_fns.get(info["provider"], generate_gpt_image), info["api_model"]

    results: dict[str, Path | Exception] = {}

    def _run(slot_key: str, fn, path: Path):
        api_model = _spec(slot_models[slot_key])[1]
        image_max_attempts = 1 if slot_key == "gpt_image_2" and reference_image_path else None
        call = lambda: fn(prompt, path, reference_image_path, api_model)
        if image_max_attempts is None:
            result = retry_llm_call(call)
        else:
            result = retry_llm_call(call, max_attempts=image_max_attempts)
        if on_model_done:
            on_model_done(slot_key)
        return result

    executor = ThreadPoolExecutor(max_workers=2)
    try:
        futures: dict[str, Future] = {
            "gpt_image_2": executor.submit(_run, "gpt_image_2", _spec(model_a)[0], gpt_path),
            "gemini_3_pro": executor.submit(_run, "gemini_3_pro", _spec(model_b)[0], gemini_path),
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
