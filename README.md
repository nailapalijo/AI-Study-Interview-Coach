---
title: AI Study & Interview Coach
emoji: 🎓
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.40.2
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Study and interview coaching with local open-source AI
---

# 🎓 AI Study & Interview Coach

A Streamlit application that runs **open-source Hugging Face models locally** (no OpenAI/Gemini/OpenRouter/Ollama,
no external AI APIs) to help you study and prepare for interviews.

| Module | What it does |
|---|---|
| Dashboard | Real statistics from SQLite: documents, sessions, study progress, quiz accuracy, recent activity |
| Documents | Upload PDF / DOCX / TXT, validate, extract, preview, index (chunks → embeddings → FAISS), delete |
| Document Chat | RAG chat with source excerpts, multi-document or single-document scope, persistent history, clear chat. Says so when the answer is not in your documents |
| Interview Generator | Technical / HR / behavioral / scenario questions by role, level, domain, difficulty; optional CV personalisation |
| Mock Interview | One question at a time, LLM scoring on 6 criteria, strengths/weaknesses/improvements/improved answer, final summary |
| Answer Evaluator | Evaluate any answer on accuracy, completeness, clarity, technical depth, relevance |
| Study Planner | LLM agent: roadmap → daily tasks → projects → quiz topics; task tracking; regenerate with custom instructions |
| Quiz Generator | 4-option MCQs with explanations from documents or a topic; scoring and history |
| Progress | Charts for interview scores, quiz accuracy, completed tasks, learning history |

## Quick start (local)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU-only PyTorch is pulled via the extra index in requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.enableCORS true --server.enableXsrfProtection true
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.enableCORS true --server.enableXsrfProtection true
```

Open **http://127.0.0.1:8501** in your browser. Do not open `0.0.0.0`; that address is only for binding the server to interfaces (used by Spaces below).

### Open on a phone on the same Wi-Fi

Run `./run_mobile.ps1` in PowerShell on the PC. It prints the PC's Wi-Fi/LAN URL; open that URL on the phone while both devices use the same trusted Wi-Fi. Allow Python through Windows Firewall for **Private networks only** if prompted. This local app has no accounts, so do not expose it to public Wi-Fi or the internet.

The first AI action downloads the default open-weight Qwen2.5-0.5B model (about 1 GB) and the embedding model (about 90 MB) into `DATA_DIR/hf_cache`. The smaller model is faster on CPU but may produce less reliable structured answers than the larger options.

### Tests

```bash
pytest -q                                 # unit tests (no model downloads)
RUN_MODEL_TESTS=1 pytest -q tests/test_ai_logic.py::test_real_llm_generation_and_json   # real local inference (downloads weights)
```
Tests that need `sentence-transformers`, `faiss` or `streamlit` skip themselves if the package is missing.

## Deploying to Hugging Face Spaces

1. Create a new Space → SDK **Streamlit** → hardware **CPU basic** (16 GB RAM) or better.
2. Push this repository (the YAML block above is the Space configuration; `app.py` is the entry point).
3. Optional environment variables (Space → Settings → Variables): `LLM_MODEL`, `DATA_DIR`, `MAX_NEW_TOKENS`, `RELEVANCE_THRESHOLD`.
4. `.streamlit/config.toml` already binds `0.0.0.0:7860`, which is what Spaces expects.

> The `sdk_version: 1.40.2` pin matches `requirements.txt`. If Spaces rejects it, set both to a Streamlit version that Spaces currently lists.

### Persistent storage (important)

Spaces containers have **ephemeral disk**: uploaded files, the SQLite database and downloaded models are wiped on restart/rebuild.

* **Recommended:** attach Hugging Face **persistent storage** to the Space (paid add-on). It mounts at `/data`;
  `config.py` detects it automatically and stores `coach.db`, `uploads/`, `logs/` and `hf_cache/` there (override with `DATA_DIR`).
* **Free tier:** the app still works, but data and model cache reset on restart (models re-download on first use).
* All state is in SQLite + files under `DATA_DIR`; nothing important lives only in Streamlit session state.
* Note that a Space has **one shared database** — this app has no user accounts, so everyone using the Space shares data.
  Run it privately (private Space) or locally for personal data.

## Model selection

Choose via env var `LLM_MODEL` (key or any Hugging Face repo id) or the sidebar **Model settings**.
Only one LLM is held in memory at a time; switching unloads the previous one.

| Key | Repo | RAM (approx.) | Notes |
|---|---|---|---|
| `Qwen2.5-0.5B-Instruct (fast, recommended)` (default) | `Qwen/Qwen2.5-0.5B-Instruct` | ~2–3 GB fp32 | Smallest Qwen option; faster on CPU, weaker structured output |
| `Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | ~6 GB fp32 | Better response quality, slower CPU inference |
| `Qwen2.5-3B-Instruct` | `Qwen/Qwen2.5-3B-Instruct` | ~7 GB bf16 | Noticeably better JSON/structured output, roughly 2× slower |
| `TinyLlama-1.1B-Chat` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~4.5 GB fp32 | Alternative chat model; weaker at structured output |

Embeddings: `sentence-transformers/all-MiniLM-L6-v2`. Vector search: FAISS `IndexFlatIP` (exact cosine); if `faiss` is not importable the
app falls back to an exact NumPy search with identical results.

## Architecture

```
app.py                 navigation, theme, model settings
services.py            cached singletons (DB, LLM, embedder, RAG) + model-load handling
config.py              paths, limits, model choices (reads env vars)
modules/               one Streamlit page per feature
ai/model_loader.py     lazy single-model loader, dtype selection, fallback model
ai/llm_engine.py       chat-template prompting, input-token budget, generation, JSON retries
ai/embeddings.py       cached MiniLM embeddings
ai/rag_pipeline.py     chunks → embeddings → FAISS → relevance threshold → grounded answer
ai/prompts.py          all prompts
ai/generators.py       questions / quiz / study-plan generation + validation
ai/evaluation.py       evaluation parsing, clamping, interview aggregation
database/              SQLite schema + access layer
utils/                 file extraction, chunker, validators, JSON extraction, UI helpers
tests/                 pytest suite
```

How "not in the documents" is enforced: retrieved chunks below `RELEVANCE_THRESHOLD` (cosine, default 0.22) are discarded, and if none remain
the LLM is **not called** — the app answers that the information was not found. If chunks pass, the prompt instructs the model to answer
only from the excerpts or emit a sentinel, which the app converts into the same "not found" message. This reduces, but cannot eliminate,
hallucination by a small model; always check the shown source excerpts.

## Known limitations

* **Speed.** CPU inference of a 1.5B–3B model is slow (expect on the order of a few tokens per second on 2 vCPUs; faster on a modern laptop).
  A 10-question interview generation, a study plan or a quiz can take from tens of seconds to several minutes.
* **Quality.** Small models sometimes produce weak scores or invalid JSON. The app retries (twice) with a stricter prompt and then shows a clear
  error with a retry option rather than inventing results. The 3B model is more reliable.
* **Scores are LLM judgements**, not ground truth; "confidence" is inferred from written text only.
* **Context budget.** Inputs are capped at 2048 tokens (long answers/context are truncated) and outputs at 600 tokens to bound CPU time.
* **Scanned PDFs** (images without a text layer) are rejected — OCR is not included.
* **No authentication / multi-user separation** (see persistent-storage note).
* Study plans are limited to 7 or 14 days, quizzes to 10 questions and interviews to 12 questions to keep generation time reasonable.
