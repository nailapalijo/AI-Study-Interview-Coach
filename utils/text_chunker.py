"""Paragraph/sentence-aware text chunker with character overlap."""
from __future__ import annotations

import re

import config


def _units(text: str, size: int) -> list[str]:
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            while len(sent) > size:
                units.append(sent[:size])
                sent = sent[size:]
            if sent:
                units.append(sent)
    return units


def _tail(text: str, n: int) -> str:
    if n <= 0 or len(text) <= n:
        return text if n > 0 else ""
    tail = text[-n:]
    cut = tail.find(" ")
    return tail[cut + 1:] if cut != -1 else tail


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None, min_chunk: int = 40) -> list[str]:
    """Split text into chunks of roughly ``chunk_size`` characters with ``overlap`` carried over."""
    chunk_size = config.CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    text = (text or "").replace("\r", "").strip()
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for unit in _units(text, chunk_size):
        if current and len(current) + len(unit) + 1 > chunk_size:
            chunks.append(current)
            tail = _tail(current, overlap)
            current = f"{tail} {unit}".strip() if tail else unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current.strip():
        chunks.append(current)
    if len(chunks) > 1:
        kept = [c for c in chunks if len(c.strip()) >= min_chunk]
        chunks = kept or chunks
    return [c.strip() for c in chunks]
