"""Shared utilities: image encoding, LLM retry, and JSON parsing."""

import base64
import functools
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, TypeVar, Union

from PIL import Image

from config import IMAGE_MAX_SIZE, IMAGE_QUALITY, MAX_RETRIES, RETRY_BACKOFF
from schemas import DIMENSIONS, ImageEvaluation, RevisedImageEvaluation

logger = logging.getLogger(__name__)

T = TypeVar("T")


@functools.lru_cache(maxsize=32)
def image_to_b64(path: Path, max_size: int = IMAGE_MAX_SIZE) -> tuple[str, str]:
    """Resize image and return (base64_data, media_type)."""
    with Image.open(path) as img:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=IMAGE_QUALITY)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


EvalModel = Union[ImageEvaluation, RevisedImageEvaluation]


def build_eval_summary(eval_a: EvalModel, eval_b: EvalModel) -> str:
    """Build JSON summary of evaluation scores for critique/revision input."""
    def _dimension_payload(eval_model: EvalModel, dimension: str) -> dict:
        score = getattr(eval_model, dimension)
        payload = {
            "score": score.score,
            "reasoning": score.reasoning,
        }
        if getattr(score, "evidence", ""):
            payload["evidence"] = score.evidence
        if getattr(score, "confidence", None) is not None:
            payload["confidence"] = score.confidence
        return payload

    return json.dumps(
        {
            "model_a": {
                "model_name": eval_a.model_name,
                **{d: _dimension_payload(eval_a, d) for d in DIMENSIONS},
            },
            "model_b": {
                "model_name": eval_b.model_name,
                **{d: _dimension_payload(eval_b, d) for d in DIMENSIONS},
            },
        },
        indent=2,
    )


def retry_llm_call(
    fn: Callable[[], T],
    max_attempts: int = MAX_RETRIES,
    backoff: float = RETRY_BACKOFF,
) -> T:
    """Retry an LLM call with exponential backoff.

    Retries on any exception except KeyboardInterrupt.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception:
            if attempt == max_attempts - 1:
                raise
            wait = backoff ** attempt
            logger.warning("LLM call failed (attempt %d/%d), retrying in %.1fs", attempt + 1, max_attempts, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")  # Satisfies type checker


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # Remove closing fence
        text = "\n".join(lines)
    return text


def parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM output, stripping fences and repairing common LLM issues."""
    cleaned = strip_markdown_fences(text)
    candidates = [cleaned]
    extracted = _extract_json_object(cleaned)
    if extracted and extracted != cleaned:
        candidates.append(extracted)

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        for attempt in _json_repair_attempts(candidate):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError as exc:
                last_error = exc
    if last_error:
        if any(_has_unclosed_json_container(candidate) for candidate in candidates):
            raise ValueError(f"LLM output appears truncated before a complete JSON object ({last_error})") from last_error
        raise last_error
    return json.loads(cleaned)


def _json_repair_attempts(text: str) -> tuple[str, ...]:
    """Return progressively repaired JSON candidates."""
    escaped = _escape_unescaped_control_chars(text)
    comma_repaired = _insert_missing_commas(escaped)
    key_repaired = _quote_unquoted_property_names(comma_repaired)
    trailing_comma_repaired = _remove_trailing_commas(key_repaired)
    guided_repaired = _repair_missing_commas_from_errors(trailing_comma_repaired)
    guided_key_repaired = _quote_unquoted_property_names(guided_repaired)
    guided_trailing_comma_repaired = _remove_trailing_commas(guided_key_repaired)
    final_guided_repaired = _repair_missing_commas_from_errors(guided_trailing_comma_repaired)
    return tuple(
        dict.fromkeys(
            (
                text,
                escaped,
                comma_repaired,
                key_repaired,
                trailing_comma_repaired,
                guided_repaired,
                guided_key_repaired,
                guided_trailing_comma_repaired,
                final_guided_repaired,
            )
        )
    )


def _insert_missing_commas(text: str) -> str:
    """Insert commas when LLMs omit separators between JSON lines."""
    value_end = r'(?:"|\d|true|false|null|\]|\})'
    next_field = r'(?=\s*(?:"[A-Za-z_][A-Za-z0-9_]*"|[A-Za-z_][A-Za-z0-9_]*)\s*:)'
    repaired = re.sub(rf'({value_end})\s*\n\s*{next_field}', r'\1,\n', text)
    repaired = re.sub(r'(\]|\})\s*\n\s*(?=\{)', r'\1,\n', repaired)
    return repaired


def _quote_unquoted_property_names(text: str) -> str:
    """Quote bare object keys while ignoring string contents."""
    chars: list[str] = []
    in_string = False
    escaped = False
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if escaped:
            chars.append(char)
            escaped = False
            index += 1
            continue
        if char == "\\" and in_string:
            chars.append(char)
            escaped = True
            index += 1
            continue
        if char == '"':
            in_string = not in_string
            chars.append(char)
            index += 1
            continue

        if not in_string and char in "{,":
            chars.append(char)
            index += 1
            while index < length and text[index].isspace():
                chars.append(text[index])
                index += 1
            if index < length and (text[index].isalpha() or text[index] == "_"):
                start = index
                while index < length and (text[index].isalnum() or text[index] == "_"):
                    index += 1
                key = text[start:index]
                lookahead = index
                while lookahead < length and text[lookahead].isspace():
                    lookahead += 1
                if lookahead < length and text[lookahead] == ":":
                    chars.append(f'"{key}"')
                    continue
                chars.append(key)
                continue
            continue

        chars.append(char)
        index += 1

    return "".join(chars)


def _remove_trailing_commas(text: str) -> str:
    """Remove commas immediately before object/array closers outside strings."""
    chars: list[str] = []
    in_string = False
    escaped = False
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if escaped:
            chars.append(char)
            escaped = False
            index += 1
            continue
        if char == "\\" and in_string:
            chars.append(char)
            escaped = True
            index += 1
            continue
        if char == '"':
            in_string = not in_string
            chars.append(char)
            index += 1
            continue
        if not in_string and char == ",":
            lookahead = index + 1
            while lookahead < length and text[lookahead].isspace():
                lookahead += 1
            if lookahead < length and text[lookahead] in "}]":
                index += 1
                continue
        chars.append(char)
        index += 1

    return "".join(chars)


def _repair_missing_commas_from_errors(text: str) -> str:
    """Use JSONDecodeError positions to repair omitted separators."""
    repaired = text
    for _ in range(20):
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError as exc:
            updated = repaired
            if "Expecting ',' delimiter" in exc.msg:
                insert_at = _missing_comma_insert_position(repaired, exc.pos)
                if insert_at is None:
                    return repaired
                updated = repaired[:insert_at] + "," + repaired[insert_at:]
            elif "Expecting property name enclosed in double quotes" in exc.msg:
                updated = _quote_unquoted_property_names(repaired)
            else:
                return repaired

            updated = _remove_trailing_commas(_quote_unquoted_property_names(updated))
            if updated == repaired:
                return repaired
            repaired = updated
    return repaired


def _missing_comma_insert_position(text: str, pos: int) -> int | None:
    """Return a safe comma insertion point for a JSON separator error."""
    if pos < 0 or pos >= len(text):
        return None

    insert_at = pos
    while insert_at > 0 and text[insert_at - 1].isspace():
        insert_at -= 1

    previous = insert_at - 1
    while previous >= 0 and text[previous].isspace():
        previous -= 1
    if previous < 0 or text[previous] not in '"0123456789}]el':
        return None

    next_index = insert_at
    while next_index < len(text) and text[next_index].isspace():
        next_index += 1
    if next_index >= len(text) or not (text[next_index] in '"{[' or text[next_index].isalpha() or text[next_index] == "_"):
        return None

    if _is_inside_json_string(text, insert_at):
        return None
    return insert_at


def _is_inside_json_string(text: str, pos: int) -> bool:
    """Return whether pos falls inside a JSON string literal."""
    in_string = False
    escaped = False
    for char in text[:pos]:
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
    return in_string


def _has_unclosed_json_container(text: str) -> bool:
    """Return whether a JSON-like string ends with unclosed object/array containers."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif char == "]":
            if stack and stack[-1] == "[":
                stack.pop()
    return bool(stack or in_string)


def _extract_json_object(text: str) -> str | None:
    """Return the substring from the first JSON object start to the last object end."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1].strip()


def _escape_unescaped_control_chars(text: str) -> str:
    """Escape literal newlines/tabs that appear inside JSON strings."""
    chars: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            chars.append(char)
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            chars.append(char)
            continue
        if in_string:
            if char == "\n":
                chars.append("\\n")
                continue
            if char == "\r":
                chars.append("\\r")
                continue
            if char == "\t":
                chars.append("\\t")
                continue
            if ord(char) < 0x20:
                chars.append(f"\\u{ord(char):04x}")
                continue
        chars.append(char)
    return "".join(chars)
