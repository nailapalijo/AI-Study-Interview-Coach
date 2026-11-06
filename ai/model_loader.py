"""Lazy, single-instance loader for the local Hugging Face causal LM (CPU-friendly)."""
from __future__ import annotations

import gc
import logging
import os
import threading

import config

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the LLM cannot be loaded. The message is safe to show to the user."""


class ModelManager:
    """Holds at most ONE model in memory. Loading a different model unloads the previous one."""

    def __init__(self, model_id: str | None = None):
        self._lock = threading.RLock()
        self._load_guard = threading.Lock()
        self.requested_id = model_id or config.DEFAULT_MODEL_ID
        self.loaded_id: str | None = None
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def is_ready(self) -> bool:
        return self.is_loaded() and self.loaded_id == self.requested_id

    def set_requested(self, model_id: str) -> None:
        with self._lock:
            self.requested_id = config.resolve_model_id(model_id)

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.tokenizer = None
            self.loaded_id = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _pick_dtype(torch, model_id: str, device: str):
        choice = config.LLM_DTYPE.lower()
        table = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        if choice in table:
            return table[choice]
        if device == "cuda":
            return torch.float16
        # CPU: bf16 halves memory for the 3B model; 1.5B/1.1B fit comfortably in fp32.
        return torch.bfloat16 if "3b" in model_id.lower() else torch.float32

    def _load_one(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = self._pick_dtype(torch, model_id, device)
        logger.info("Loading %s on %s (%s)", model_id, device, dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(config.MODEL_CACHE_DIR / "hub"))
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True, cache_dir=str(config.MODEL_CACHE_DIR / "hub"))
        model.to(device)
        model.eval()
        if device == "cpu":
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.model, self.tokenizer, self.device, self.loaded_id = model, tokenizer, device, model_id

    def load(self) -> None:
        """Load once without making other Streamlit sessions wait behind a long CPU load."""
        if not self._load_guard.acquire(blocking=False):
            raise ModelLoadError("The model is already loading in another session. Wait for it to finish, then retry.")
        try:
            self._load_with_guard()
        finally:
            self._load_guard.release()

    def _load_with_guard(self) -> None:
        """Load the requested model, falling back to the lightweight model if it fails."""
        with self._lock:
            if self.is_ready():
                return
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
            except ImportError as exc:
                raise ModelLoadError(
                    "The 'torch' and 'transformers' packages are not installed. Run: pip install -r requirements.txt"
                ) from exc
            self.unload()
            attempts = []
            for candidate in (self.requested_id, config.FALLBACK_MODEL_ID):
                if candidate and candidate not in attempts:
                    attempts.append(candidate)
            errors = []
            for model_id in attempts:
                try:
                    self._load_one(model_id)
                    self.last_error = None
                    if model_id != self.requested_id:
                        logger.warning("Fell back to %s (requested %s failed)", model_id, self.requested_id)
                        self.requested_id = model_id
                    return
                except MemoryError:
                    errors.append(f"{model_id}: out of memory")
                    self.unload()
                except Exception as exc:  # network, disk, corrupt cache, OOM RuntimeError...
                    logger.exception("Failed loading %s", model_id)
                    errors.append(f"{model_id}: {type(exc).__name__}: {str(exc)[:200]}")
                    self.unload()
            self.last_error = " | ".join(errors)
            raise ModelLoadError(
                "Could not load any language model. Check your internet connection (first run downloads the "
                f"weights), clear stale download cache, or choose the default Qwen 0.5B model. Details: {self.last_error}")
