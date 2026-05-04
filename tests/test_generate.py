"""Tests for parallel image generation behavior."""

from pathlib import Path

from generate import generate_images


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

    def fake_gpt(prompt: str, output_path: Path) -> Path:
        calls["gpt"] += 1
        if calls["gpt"] == 1:
            raise RuntimeError("transient GPT failure")
        output_path.write_bytes(b"gpt")
        return output_path

    def fake_gemini(prompt: str, output_path: Path) -> Path:
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
