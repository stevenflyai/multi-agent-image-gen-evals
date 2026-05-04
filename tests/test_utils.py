"""Tests for shared utilities: image encoding, retry, and JSON parsing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from utils import image_to_b64, parse_llm_json, retry_llm_call, strip_markdown_fences


class TestImageToB64:
    def test_returns_base64_and_media_type(self, tmp_path):
        img = Image.new("RGB", (100, 100), color="red")
        path = tmp_path / "test.png"
        img.save(path)

        b64, media = image_to_b64(path)
        assert isinstance(b64, str)
        assert len(b64) > 0
        assert media == "image/jpeg"

    def test_resizes_large_image(self, tmp_path):
        img = Image.new("RGB", (2000, 2000), color="blue")
        path = tmp_path / "big.png"
        img.save(path)

        b64, _ = image_to_b64(path, max_size=100)
        # Decode and check size
        import base64, io
        decoded = base64.b64decode(b64)
        result_img = Image.open(io.BytesIO(decoded))
        assert max(result_img.size) <= 100


class TestStripMarkdownFences:
    def test_strips_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_strips_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_no_fences_unchanged(self):
        text = '{"key": "value"}'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_strips_whitespace(self):
        text = '  \n{"key": "value"}\n  '
        assert strip_markdown_fences(text) == '{"key": "value"}'


class TestParseLlmJson:
    def test_parses_clean_json(self):
        result = parse_llm_json('{"a": 1}')
        assert result == {"a": 1}

    def test_parses_fenced_json(self):
        result = parse_llm_json('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_parses_json_with_extra_text(self):
        result = parse_llm_json('Here is the JSON:\n{"a": 1}\nThanks')
        assert result == {"a": 1}

    def test_repairs_literal_newline_inside_string(self):
        result = parse_llm_json('{"critique": "line one\nline two"}')
        assert result == {"critique": "line one\nline two"}

    def test_repairs_missing_comma_after_nested_object(self):
        result = parse_llm_json('''{
    "overall_assessment": "ok",
    "dimension_critiques": [
        {
            "dimension": "prompt_adherence",
            "original_score_model_a": 8,
            "original_score_model_b": 7,
            "critique": "reasonable",
            "suggested_score_model_a": 8,
            "suggested_score_model_b": 7
        }
    ]
    "bias_detection": "none"
}''')
        assert result["bias_detection"] == "none"
        assert result["dimension_critiques"][0]["dimension"] == "prompt_adherence"

    def test_repairs_missing_comma_between_array_items(self):
        result = parse_llm_json('''{
    "dimension_critiques": [
        {
            "dimension": "prompt_adherence",
            "original_score_model_a": 8,
            "original_score_model_b": 7,
            "critique": "first",
            "suggested_score_model_a": 8,
            "suggested_score_model_b": 7
        }
        {
            "dimension": "photorealism",
            "original_score_model_a": 8,
            "original_score_model_b": 7,
            "critique": "second",
            "suggested_score_model_a": 8,
            "suggested_score_model_b": 7
        }
    ]
}''')
        assert len(result["dimension_critiques"]) == 2
        assert result["dimension_critiques"][1]["dimension"] == "photorealism"

    def test_repairs_missing_comma_between_fields(self):
        result = parse_llm_json('''{
    "overall_assessment": "ok"
    "dimension_critiques": [
        {
            "dimension": "prompt_adherence"
            "original_score_model_a": 8
            "original_score_model_b": 7
        }
    ],
    "bias_detection": "none"
}''')
        assert result["overall_assessment"] == "ok"
        assert result["dimension_critiques"][0]["original_score_model_a"] == 8

    def test_repairs_unquoted_property_names_and_trailing_commas(self):
        result = parse_llm_json('''{
    "overall_assessment": "ok",
    "dimension_critiques": [
        {
            dimension: "prompt_adherence",
            original_score_model_a: 8,
            original_score_model_b: 7,
        },
    ],
    "bias_detection": "none",
}''')
        assert result["dimension_critiques"][0]["dimension"] == "prompt_adherence"
        assert result["dimension_critiques"][0]["original_score_model_b"] == 7

    def test_repairs_missing_comma_before_unquoted_field(self):
        result = parse_llm_json('''{
    "overall_assessment": "ok"
    dimension_critiques: [
        {
            dimension: "prompt_adherence"
            original_score_model_a: 8
            original_score_model_b: 7
            critique: "score is reasonable"
            suggested_score_model_a: 8
            suggested_score_model_b: 7
        }
    ]
    bias_detection: "none"
}''')
        assert result["overall_assessment"] == "ok"
        assert result["dimension_critiques"][0]["dimension"] == "prompt_adherence"
        assert result["dimension_critiques"][0]["suggested_score_model_b"] == 7
        assert result["bias_detection"] == "none"

    def test_repairs_same_line_missing_commas_before_unquoted_fields(self):
        result = parse_llm_json('''{
    "overall_assessment": "ok",
    "dimension_critiques": [
        {
            "dimension": "prompt_adherence" original_score_model_a: 8 original_score_model_b: 7 critique: "score is reasonable" suggested_score_model_a: 8 suggested_score_model_b: 7
        }
    ],
    "bias_detection": "none"
}''')
        item = result["dimension_critiques"][0]
        assert item["dimension"] == "prompt_adherence"
        assert item["original_score_model_a"] == 8
        assert item["original_score_model_b"] == 7
        assert item["suggested_score_model_b"] == 7

    def test_reports_truncated_json_clearly(self):
        with pytest.raises(ValueError, match="truncated"):
            parse_llm_json('''{
    "overall_assessment": "ok",
    "dimension_critiques": [
        {
            "dimension": "prompt_adherence",
            "original_score_model_a": 7''')

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("not json at all")


class TestRetryLlmCall:
    def test_succeeds_on_first_try(self):
        fn = MagicMock(return_value="ok")
        assert retry_llm_call(fn, max_attempts=3, backoff=0.01) == "ok"
        assert fn.call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        fn = MagicMock(side_effect=[ValueError("fail"), "ok"])
        assert retry_llm_call(fn, max_attempts=3, backoff=0.01) == "ok"
        assert fn.call_count == 2

    def test_retries_parse_failure_then_succeeds(self):
        fn = MagicMock(side_effect=['{"a": "unterminated', '{"a": 1}'])
        result = retry_llm_call(lambda: parse_llm_json(fn()), max_attempts=2, backoff=0.01)
        assert result == {"a": 1}
        assert fn.call_count == 2

    def test_raises_after_max_attempts(self):
        fn = MagicMock(side_effect=ValueError("always fail"))
        with pytest.raises(ValueError, match="always fail"):
            retry_llm_call(fn, max_attempts=2, backoff=0.01)
        assert fn.call_count == 2

    def test_keyboard_interrupt_not_retried(self):
        fn = MagicMock(side_effect=KeyboardInterrupt)
        with pytest.raises(KeyboardInterrupt):
            retry_llm_call(fn, max_attempts=3, backoff=0.01)
        assert fn.call_count == 1
