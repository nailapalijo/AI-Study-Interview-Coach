"""Document upload, validation, extraction, indexing, preview and deletion."""
from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

import streamlit as st

import config
from ai.embeddings import EmbeddingError
from database.db import DatabaseError
from services import get_db, get_rag
from utils import ui
from utils.file_processor import FileProcessingError, extract_text
from utils.validators import ValidationError, sanitize_filename, validate_upload

logger = logging.getLogger(__name__)


def _index(doc_id: int, name: str) -> tuple[bool, str]:
    """Index a document. Returns (ok, message)."""
    rag, db = get_rag(), get_db()
    try:
        with st.spinner(f"Indexing '{name}' (first run downloads the embedding model)…"):
            n = rag.ingest(doc_id)
        return True, f"Indexed '{name}': {n} chunks."
    except (EmbeddingError, ValueError, DatabaseError) as exc:
        db.set_document_status(doc_id, "extracted")
        return False, f"'{name}' was saved but could not be indexed: {exc} (use 'Index now' to retry)."


def _process_uploads(files) -> list[tuple[str, str]]:
    """Process files; returns [(level, message)] to display after the rerun."""
    db = get_db()
    msgs: list[tuple[str, str]] = []
    bar = st.progress(0.0, text="Processing uploads…")
    for i, f in enumerate(files):
        data = f.getvalue()
        try:
            ext = validate_upload(f.name, len(data))
            digest = hashlib.sha256(data).hexdigest()
            dup = db.find_document_by_hash(digest)
            if dup:
                msgs.append(("warning", f"'{f.name}' was already uploaded as '{dup['name']}' — skipped."))
                continue
            stored = Path(config.UPLOAD_DIR) / f"{uuid.uuid4().hex}{ext}"
            stored.write_bytes(data)
            try:
                result = extract_text(stored)
            except FileProcessingError:
                stored.unlink(missing_ok=True)
                raise
            doc_id = db.add_document(sanitize_filename(f.name), ext.lstrip("."), len(data), str(stored), result.text,
                                     digest, {"pages": result.pages, "extraction": result.method,
                                              "words": len(result.text.split())})
            bar.progress((i + 0.5) / len(files), text=f"Indexing {f.name}…")
            ok, message = _index(doc_id, f.name)
            msgs.append(("success" if ok else "warning", message))
        except (ValidationError, FileProcessingError) as exc:
            msgs.append(("error", str(exc)))
        except (DatabaseError, OSError) as exc:
            logger.exception("Upload failed")
            msgs.append(("error", f"Could not store '{f.name}': {exc}"))
        bar.progress((i + 1) / len(files), text=f"Processed {i + 1}/{len(files)}")
    bar.empty()
    return msgs


def render() -> None:
    db = get_db()
    ui.page_header("Document Management", "Upload PDFs, DOCX or TXT files. They are stored persistently and indexed for AI chat.")
    st.session_state.setdefault("upload_key", 0)
    for level, message in st.session_state.pop("upload_msgs", []):
        getattr(st, level)(message)
    files = st.file_uploader(f"Upload documents (PDF, DOCX, TXT · max {config.MAX_FILE_SIZE_MB} MB each)",
                             type=["pdf", "docx", "txt"], accept_multiple_files=True,
                             key=f"uploader_{st.session_state['upload_key']}")
    if files and st.button("⬆️ Process uploaded files", type="primary"):
        st.session_state["upload_msgs"] = _process_uploads(files)
        st.session_state["upload_key"] += 1  # resets the uploader widget
        st.rerun()

    st.markdown("### Your documents")
    docs = db.list_documents()
    if not docs:
        ui.empty_state("No documents yet. Upload a CV, notes or a PDF to begin.")
        return
    for d in docs:
        status = "✅ indexed" if d["status"] == "indexed" else "⚠️ not indexed"
        with st.expander(f"📄 {d['name']} — {status}"):
            full = db.get_document(d["id"]) or {}
            meta = full.get("metadata", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Type", d["file_type"].upper())
            c2.metric("Size", f"{d['size_bytes'] / 1024:.1f} KB")
            c3.metric("Words", meta.get("words", "—"))
            c4.metric("Chunks", d["num_chunks"])
            st.caption(f"Uploaded {d['created_at']} UTC · pages: {meta.get('pages', '—')} · extracted with {meta.get('extraction', '—')}")
            st.text_area("Extracted text preview", (full.get("text") or "")[:3000], height=200, disabled=True,
                         key=f"prev_{d['id']}")
            b1, b2, _ = st.columns([1, 1, 3])
            if d["status"] != "indexed" and b1.button("Index now", key=f"idx_{d['id']}"):
                ok, message = _index(d["id"], d["name"])
                if ok:
                    st.rerun()
                else:
                    st.error(message)
            if b2.button("🗑️ Delete", key=f"del_{d['id']}"):
                try:
                    db.delete_document(d["id"])
                    get_rag().invalidate()
                    st.toast("Document deleted", icon="🗑️")
                    st.rerun()
                except DatabaseError as exc:
                    st.error(str(exc))
