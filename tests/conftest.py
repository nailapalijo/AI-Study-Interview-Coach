import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="coach_test_"))

import hashlib  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402


class HashEmbedder:
    """Deterministic bag-of-words embedder used ONLY to unit-test pipeline logic without model downloads."""
    dim = 128

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts, batch_size=32):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text):
        return self._vec(text)


class ScriptedLLM:
    """Test double with the LLMEngine interface that returns scripted raw text (tests parsing/validation logic)."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, messages, max_new_tokens=None, temperature=0.3, top_p=0.9):
        self.calls.append(messages)
        return self.outputs.pop(0)

    def generate_json(self, messages, max_new_tokens=None, validator=None, retries=None):
        from ai.llm_engine import LLMEngine
        return LLMEngine.generate_json(self, messages, max_new_tokens, validator, retries)


@pytest.fixture
def db(tmp_path):
    from database.db import Database
    return Database(tmp_path / "test.db")


@pytest.fixture
def embedder():
    return HashEmbedder()
