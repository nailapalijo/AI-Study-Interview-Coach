"""Robust JSON extraction from LLM output (handles code fences, prose, trailing commas, truncation)."""
from __future__ import annotations

import json
import re


class JSONExtractionError(ValueError):
    pass


def _balanced(text: str, start: int) -> str | None:
    stack: list[str] = []
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if not stack or stack[-1] != c:
                return None
            stack.pop()
            if not stack:
                return text[start:i + 1]
    # Output was cut off (token limit): try closing open structures.
    return text[start:] + ('"' if in_str else "") + "".join(reversed(stack))


def extract_json(text: str):
    if not text or not text.strip():
        raise JSONExtractionError("Empty model output.")
    cleaned = text.replace("```json", "```")
    for match in re.finditer(r"[{\[]", cleaned):
        candidate = _balanced(cleaned, match.start())
        if not candidate:
            continue
        for variant in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError:
                continue
    raise JSONExtractionError("No valid JSON found in model output.")
