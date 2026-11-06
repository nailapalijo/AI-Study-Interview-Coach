import pytest

from utils.text_chunker import chunk_text


def test_empty():
    assert chunk_text("") == [] and chunk_text("   \n ") == []


def test_short_text_single_chunk():
    assert chunk_text("A short note about Python programming and its uses.") == ["A short note about Python programming and its uses."]


def test_respects_size_and_covers_all_content():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 3
    assert all(len(c) <= 500 + 100 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"Paragraph {i}." in joined for i in range(20))


def test_overlap_shared_between_neighbours():
    text = "\n\n".join(f"Sentence number {i} has some filler words here." for i in range(60))
    chunks = chunk_text(text, chunk_size=300, overlap=80)
    assert len(chunks) > 2
    assert chunks[1].split()[0] in chunks[0]  # carried-over tail appears in previous chunk


def test_huge_unbroken_text():
    chunks = chunk_text("x" * 5000, chunk_size=800, overlap=100)
    assert len(chunks) >= 6 and sum(len(c) for c in chunks) >= 5000


def test_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)
