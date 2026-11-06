"""Text generation on top of ModelManager. All inference is local (Transformers)."""
from __future__ import annotations

import logging
import threading

import config
from ai.model_loader import ModelLoadError, ModelManager
from utils.json_utils import extract_json

logger = logging.getLogger(__name__)

SYSTEM_DEFAULT = "You are a precise, helpful AI assistant."


class LLMError(RuntimeError):
    """Raised when generation fails. The message is safe to show to the user."""


class LLMEngine:
    def __init__(self, manager: ModelManager | None = None):
        self.manager = manager or ModelManager()
        self._gen_lock = threading.Lock()  # one generation at a time (CPU)

    # ---- lifecycle -----------------------------------------------------
    def is_loaded(self) -> bool:
        return self.manager.is_ready()

    def load(self) -> None:
        self.manager.load()

    @property
    def model_id(self) -> str | None:
        return self.manager.loaded_id

    # ---- prompt fitting --------------------------------------------------
    def _render(self, messages: list[dict]) -> str:
        return self.manager.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _count(self, messages: list[dict]) -> int:
        return len(self.manager.tokenizer(self._render(messages), add_special_tokens=False)["input_ids"])

    def fit_messages(self, messages: list[dict], limit: int | None = None) -> list[dict]:
        """Shrink the longest message until the prompt fits the token budget."""
        limit = limit or config.MAX_INPUT_TOKENS
        msgs = [dict(m) for m in messages]
        for _ in range(8):
            total = self._count(msgs)
            if total <= limit:
                return msgs
            idx = max(range(len(msgs)), key=lambda i: len(msgs[i]["content"]))
            content = msgs[idx]["content"]
            keep = int(len(content) * min(0.9, (limit / total) * 0.9))
            if keep < 200:
                break
            head = keep * 2 // 3
            msgs[idx]["content"] = content[:head] + "\n[...truncated...]\n" + content[-(keep - head):]
        return msgs  # generate() applies a hard token cut as a last resort

    # ---- generation ---------------------------------------------------------
    def generate(self, messages: list[dict], max_new_tokens: int | None = None, temperature: float = 0.3,
                 top_p: float = 0.9) -> str:
        import torch

        if not self._gen_lock.acquire(blocking=False):
            raise LLMError("The local model is busy generating another response. Wait for it to finish, then retry.")
        try:
            try:
                self.load()
            except ModelLoadError as exc:
                raise LLMError(str(exc)) from exc
            max_new = min(max_new_tokens or config.MAX_NEW_TOKENS, config.MAX_NEW_TOKENS * 2)
            tok, model = self.manager.tokenizer, self.manager.model
            msgs = self.fit_messages(messages)
            enc = tok(self._render(msgs), return_tensors="pt", add_special_tokens=False)
            ids, mask = enc["input_ids"], enc["attention_mask"]
            if ids.shape[1] > config.MAX_INPUT_TOKENS:  # hard cut, keep the end (the actual question)
                ids, mask = ids[:, -config.MAX_INPUT_TOKENS:], mask[:, -config.MAX_INPUT_TOKENS:]
            ids, mask = ids.to(self.manager.device), mask.to(self.manager.device)
            kwargs = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id, repetition_penalty=1.05)
            if temperature and temperature > 0:
                kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                kwargs.update(do_sample=False)
            with torch.inference_mode():
                out = model.generate(input_ids=ids, attention_mask=mask, **kwargs)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        except LLMError:
            raise
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover
            raise LLMError("The GPU ran out of memory. Choose a smaller model.") from exc
        except RuntimeError as exc:
            logger.exception("Generation failed")
            raise LLMError(f"Text generation failed ({str(exc)[:160]}). Try again or choose a smaller model.") from exc
        finally:
            self._gen_lock.release()
        text = text.strip()
        if not text:
            raise LLMError("The model returned an empty response. Please retry.")
        return text

    def generate_json(self, messages: list[dict], max_new_tokens: int | None = None, validator=None,
                      retries: int | None = None):
        """Generate and parse JSON. `validator(obj)` may normalise/raise ValueError to trigger a retry."""
        retries = config.JSON_RETRIES if retries is None else retries
        last: Exception | None = None
        current = list(messages)
        for attempt in range(retries + 1):
            raw = self.generate(current, max_new_tokens=max_new_tokens, temperature=0.3 if attempt == 0 else 0.1)
            try:
                obj = extract_json(raw)
                return validator(obj) if validator else obj
            except (ValueError, KeyError, TypeError) as exc:
                last = exc
                logger.warning("JSON attempt %d failed: %s", attempt + 1, exc)
                current = list(messages) + [
                    {"role": "assistant", "content": raw[:4000]},
                    {
                    "role": "user",
                    "content": f"Your previous reply was not usable ({exc}). Correct the specific issue. "
                               "Reply with ONLY the requested JSON, no commentary.",
                    },
                ]
        raise LLMError(f"The model did not return valid structured output after {retries + 1} attempts ({last}). "
                       "Please retry, or choose a larger model for better reliability.")
