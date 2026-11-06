"""Sentence-transformer embeddings (all-MiniLM-L6-v2), lazily loaded and cached."""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

import config

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced. The message is safe to show to the user."""


class Embedder:
    def __init__(self, model_name: str | None = None, cache_size: int = 256):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                if isinstance(exc, ModuleNotFoundError) and exc.name == "sentence_transformers":
                    message = "The 'sentence-transformers' package is not installed. "
                else:
                    message = f"Could not import sentence-transformers because of a dependency error: {exc}. "
                raise EmbeddingError(message + "Run: pip install -r requirements.txt") from exc
            try:
                self._model = SentenceTransformer(self.model_name, device="cpu",
                                                  cache_folder=str(config.MODEL_CACHE_DIR / "st"))
            except Exception as exc:
                logger.exception("Embedding model load failed")
                raise EmbeddingError(f"Could not load embedding model '{self.model_name}': {str(exc)[:200]}. "
                                     "Check your internet connection on first run.") from exc
            return self._model

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalised float32 vectors, shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        try:
            vecs = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        except Exception as exc:
            logger.exception("Embedding failed")
            raise EmbeddingError(f"Embedding generation failed: {str(exc)[:200]}") from exc
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
        vec = self.embed([text])[0]
        with self._lock:
            self._cache[text] = vec
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return vec
