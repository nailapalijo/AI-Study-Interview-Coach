"""Input validation helpers."""
from __future__ import annotations

import os
import re

import config


class ValidationError(ValueError):
    """Raised when user input is invalid. The message is safe to show to the user."""


def sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "").strip()
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base)
    return base[:120] or "file"


def validate_upload(filename: str, size_bytes: int) -> str:
    """Validate an uploaded file. Returns the lower-case extension (e.g. '.pdf')."""
    if not filename:
        raise ValidationError("The uploaded file has no name.")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(config.ALLOWED_EXTENSIONS)
        raise ValidationError(f"'{filename}': unsupported file type '{ext or 'none'}'. Allowed: {allowed}.")
    if size_bytes <= 0:
        raise ValidationError(f"'{filename}' is empty.")
    limit = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > limit:
        raise ValidationError(f"'{filename}' is {size_bytes / 1024 / 1024:.1f} MB; the limit is {config.MAX_FILE_SIZE_MB} MB.")
    return ext


def validate_text(value: str | None, label: str, min_len: int = 1, max_len: int = 4000) -> str:
    text = (value or "").strip()
    if len(text) < min_len:
        raise ValidationError(f"{label} is required" + (f" (at least {min_len} characters)." if min_len > 1 else "."))
    if len(text) > max_len:
        raise ValidationError(f"{label} is too long (max {max_len} characters).")
    return text


def validate_int(value, label: str, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{label} must be a whole number.") from None
    if not lo <= number <= hi:
        raise ValidationError(f"{label} must be between {lo} and {hi}.")
    return number
