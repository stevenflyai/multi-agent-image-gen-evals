"""Tests for parallel image generation behavior."""

import base64
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from types import SimpleNamespace

from generate import generate_gpt_image, generate_images


def test_generate_images_uses_retry_wrapper(tmp_path, monkeypatch):
    calls = {"retry": 0, "gpt": 0, "gemini": 0}

    def fake_retry(fn):
        calls["retry"] += 1
        last_error = None
        for _ in range(3):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
        raise last_error

    def fake_gpt(prompt: str, output_path: Path, reference_image_path: Path | None = None) -> Path:
        calls["gpt"] += 1
        if calls["gpt"] == 1:
            raise RuntimeError("transient GPT failure")
        output_path.write_bytes(b"gpt")
        return output_path

    def fake_gemini(prompt: str, output_path: Path, reference_image_path: Path | None = None) -> Path:
        calls["gemini"] += 1
        output_path.write_bytes(b"gemini")
        return output_path

    monkeypatch.setattr("generate.retry_llm_call", fake_retry)
    monkeypatch.setattr("generate.generate_gpt_image", fake_gpt)
    monkeypatch.setattr("generate.generate_gemini_image", fake_gemini)

    results = generate_images("prompt", tmp_path)

    assert calls == {"retry": 2, "gpt": 2, "gemini": 1}
    assert results["gpt_image_2"] == tmp_path / "gpt_image_2.png"
    assert results["gemini_3_pro"] == tmp_path / "gemini_3_pro.png"


def test_generate_images_passes_reference_image(tmp_path, monkeypatch):
    reference_path = tmp_path / "reference_image.png"
    reference_path.write_bytes(b"reference")
    seen = []

    retry_attempts = []

    def fake_retry(fn, max_attempts=3, backoff=2.0):
        retry_attempts.append(max_attempts)
        return fn()

    def fake_gpt(prompt: str, output_path: Path, reference_image_path: Path | None = None) -> Path:
        seen.append(("gpt", reference_image_path))
        output_path.write_bytes(b"gpt")
        return output_path

    def fake_gemini(prompt: str, output_path: Path, reference_image_path: Path | None = None) -> Path:
        seen.append(("gemini", reference_image_path))
        output_path.write_bytes(b"gemini")
        return output_path

    monkeypatch.setattr("generate.retry_llm_call", fake_retry)
    monkeypatch.setattr("generate.generate_gpt_image", fake_gpt)
    monkeypatch.setattr("generate.generate_gemini_image", fake_gemini)

    results = generate_images("prompt", tmp_path, reference_image_path=reference_path)

    assert results["gpt_image_2"] == tmp_path / "gpt_image_2.png"
    assert results["gemini_3_pro"] == tmp_path / "gemini_3_pro.png"
    assert seen == [("gpt", reference_path), ("gemini", reference_path)]
    assert sorted(retry_attempts) == [1, 3]


def test_generate_images_reports_model_specific_timeout(tmp_path, monkeypatch):
    class FakeFuture:
        def __init__(self, result=None, should_timeout=False):
            self._result = result
            self._should_timeout = should_timeout
            self.cancelled = False

        def result(self, timeout=None):
            if self._should_timeout:
                raise FutureTimeoutError()
            return self._result

        def cancel(self):
            self.cancelled = True

    submitted = []
    shutdown_calls = []

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, fn, model_key, generator, output_path):
            submitted.append(model_key)
            if model_key == "gpt_image_2":
                return FakeFuture(should_timeout=True)
            output_path.write_bytes(b"gemini")
            return FakeFuture(output_path)

        def shutdown(self, wait=True, cancel_futures=False):
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr("generate.ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr("generate.IMAGE_GEN_GPT_TIMEOUT", 123)

    results = generate_images("prompt", tmp_path)

    assert submitted == ["gpt_image_2", "gemini_3_pro"]
    assert isinstance(results["gpt_image_2"], TimeoutError)
    assert str(results["gpt_image_2"]) == "timed out after 123s"
    assert results["gemini_3_pro"] == tmp_path / "gemini_3_pro.png"
    assert shutdown_calls == [(False, True)]


def test_generate_gpt_image_streams_reference_edit(tmp_path, monkeypatch):
    reference_path = tmp_path / "reference_image.jpg"
    reference_path.write_bytes(b"reference")
    output_path = tmp_path / "gpt_image_2.png"
    final_image = b"final image bytes"
    partial_image = b"partial image bytes"
    seen = {}

    class FakeImages:
        def edit(self, **kwargs):
            seen.update(kwargs)
            return iter(
                [
                    SimpleNamespace(
                        type="image_edit.partial_image",
                        b64_json=base64.b64encode(partial_image).decode(),
                    ),
                    SimpleNamespace(
                        type="image_edit.completed",
                        b64_json=base64.b64encode(final_image).decode(),
                    ),
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen["client_kwargs"] = kwargs
            self.images = FakeImages()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("generate.OpenAI", FakeOpenAI)
    monkeypatch.setattr("generate.OPENAI_IMAGE_PARTIAL_IMAGES", 1)

    result = generate_gpt_image("prompt", output_path, reference_path)

    assert result == output_path
    assert output_path.read_bytes() == final_image
    assert seen["client_kwargs"]["max_retries"] == 0
    assert seen["model"] == "gpt-image-2"
    assert isinstance(seen["image"], list)
    assert seen["image"][0].closed
    assert seen["quality"] == "low"
    assert seen["stream"] is True
    assert seen["partial_images"] == 1
