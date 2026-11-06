"""Central configuration. Import this module before transformers/sentence-transformers
so that cache locations (HF_HOME) are applied."""
from __future__ import annotations

import logging
import logging.handlers
import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _resolve_data_dir() -> Path:
    """Pick a writable data directory.

    Priority: $DATA_DIR -> /data (Hugging Face persistent storage) -> ./data -> temp dir.
    """
    candidates = []
    if os.environ.get("DATA_DIR"):
        candidates.append(Path(os.environ["DATA_DIR"]))
    if Path("/data").is_dir() and os.access("/data", os.W_OK):
        candidates.append(Path("/data"))
    candidates.append(BASE_DIR / "data")
    candidates.append(Path(tempfile.gettempdir()) / "ai_coach_data")
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            return path
        except OSError:
            continue
    raise RuntimeError("No writable data directory available")


DATA_DIR = _resolve_data_dir()
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_CACHE_DIR = DATA_DIR / "hf_cache"
LOG_DIR = DATA_DIR / "logs"
DB_PATH = DATA_DIR / "coach.db"
for _d in (UPLOAD_DIR, MODEL_CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Windows can fail when the fast HF transfer path keeps a temporary file open while the cache
# is being moved/renamed. Force the standard download path to avoid intermittent file-lock errors.
if os.name == "nt":
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
STATIC_CSS = BASE_DIR / "assets" / "style.css"

# ---------------------------------------------------------------- models
MODEL_CHOICES = {
    "Qwen2.5-0.5B-Instruct (fast, recommended)": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "TinyLlama-1.1B-Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}
FALLBACK_MODEL_KEY = "Qwen2.5-0.5B-Instruct (fast, recommended)"


def resolve_model_id(value: str) -> str:
    """Accept either a friendly key from MODEL_CHOICES or a full Hugging Face repo id."""
    return MODEL_CHOICES.get(value, value)


DEFAULT_MODEL_KEY = os.environ.get("LLM_MODEL", "Qwen2.5-0.5B-Instruct (fast, recommended)")
DEFAULT_MODEL_ID = resolve_model_id(DEFAULT_MODEL_KEY)
FALLBACK_MODEL_ID = resolve_model_id(FALLBACK_MODEL_KEY)
LLM_DTYPE = os.environ.get("LLM_DTYPE", "auto")  # auto | float32 | bfloat16 | float16
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------- limits
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "10"))
ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt")
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "2048"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "600"))
JSON_RETRIES = 2
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.22"))
MAX_HISTORY_TURNS = 3
MAX_STUDY_DAYS = 14
MAX_QUESTIONS = 12


def setup_logging() -> None:
    root = logging.getLogger()
    if getattr(root, "_coach_configured", False):
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    try:
        fileh = logging.handlers.RotatingFileHandler(
            LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
        )
        fileh.setFormatter(fmt)
        root.addHandler(fileh)
    except OSError:
        pass
    root._coach_configured = True  # type: ignore[attr-defined]
