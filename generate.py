"""Parallel image generation using GPT Image-2 and Gemini 3 Pro.

Each generator is a self-contained function. Both run in parallel
via ThreadPoolExecutor. Images saved as PNGs to runs/ directory.
"""

import base64
import os
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from google import genai

load_dotenv()


def generate_gpt_image(prompt: str, output_path: Path) -> Path:
    """Generate an image with GPT Image-2 and save to output_path."""
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    response = client.images.generate(
        model="gpt-image-2",
        prompt=prompt,
        n=1,
        size="1024x1024",
    )
    image_data = response.data[0]
    if image_data.b64_json:
        img_bytes = base64.b64decode(image_data.b64_json)
    elif image_data.url:
        import httpx
        img_bytes = httpx.get(image_data.url).content
    else:
        raise ValueError("GPT Image-2 returned neither b64_json nor url")

    output_path.write_bytes(img_bytes)
    return output_path


def generate_gemini_image(prompt: str, output_path: Path) -> Path:
    """Generate an image with Gemini 3 Pro and save to output_path."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=prompt,
        config=genai.types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    if not response.candidates or not response.candidates[0].content.parts:
        raise ValueError("Gemini 3 Pro returned no image")

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            output_path.write_bytes(part.inline_data.data)
            return output_path

    raise ValueError("Gemini 3 Pro response contained no image data")


def generate_images(prompt: str, run_dir: Path) -> dict[str, Path | Exception]:
    """Generate images from both models in parallel.

    Returns dict with 'gpt_image_2' and 'gemini_3_pro' keys.
    Values are either Path (success) or Exception (failure).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    gpt_path = run_dir / "gpt_image_2.png"
    gemini_path = run_dir / "gemini_3_pro.png"

    results: dict[str, Path | Exception] = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures: dict[str, Future] = {
            "gpt_image_2": executor.submit(generate_gpt_image, prompt, gpt_path),
            "gemini_3_pro": executor.submit(generate_gemini_image, prompt, gemini_path),
        }
        for model, future in futures.items():
            try:
                results[model] = future.result(timeout=600)
            except Exception as e:
                results[model] = e

    return results
