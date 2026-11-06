"""RAG chat over uploaded documents with persistent history and source references."""
from __future__ import annotations

import streamlit as st

import config
from ai.embeddings import EmbeddingError
from ai.llm_engine import LLMError
from database.db import DatabaseError
from services import ensure_llm, get_db, get_rag
from utils import ui


def _show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] {s['doc_name']}** · chunk {s['chunk_index'] + 1} · relevance {s['score']:.2f}")
            st.caption(s["text"][:600] + ("…" if len(s["text"]) > 600 else ""))


def render() -> None:
    db = get_db()
    ui.page_header("Document Chat", "Ask questions — answers are grounded in your documents, with sources.")
    docs = [d for d in db.list_documents() if d["status"] == "indexed"]
    if not docs:
        ui.empty_state("No indexed documents. Upload and index a document first.", "💬")
        st.button("Go to Documents", on_click=ui.goto, args=("📄 Documents",))
        return

    sessions = db.list_chat_sessions()
    top = st.columns([3, 1, 1])
    labels = {s["id"]: f"#{s['id']} · {s['title']}" for s in sessions}
    options = [0] + [s["id"] for s in sessions]
    current = st.session_state.get("chat_session_id", 0)
    if current not in options:
        current = 0
    selected = top[0].selectbox("Conversation", options, index=options.index(current),
                                format_func=lambda x: "➕ New conversation" if x == 0 else labels[x])
    st.session_state["chat_session_id"] = selected

    doc_map = {d["id"]: d["name"] for d in docs}
    session = db.get_chat_session(selected) if selected else None
    default_docs = [i for i in (session["doc_ids"] if session else list(doc_map)) if i in doc_map]
    scope = st.multiselect("Documents to search (select one for document-specific chat)", list(doc_map),
                           default=default_docs, format_func=lambda i: doc_map[i])
    if session and scope != session["doc_ids"] and scope:
        db.update_chat_docs(selected, scope)

    if session:
        if top[1].button("🧹 Clear chat"):
            db.clear_chat_messages(selected)
            st.rerun()
        if top[2].button("🗑️ Delete"):
            db.delete_chat_session(selected)
            st.session_state["chat_session_id"] = 0
            st.rerun()

    for m in (db.get_chat_messages(selected) if session else []):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                _show_sources(m["sources"])
    if not session:
        ui.empty_state("Ask your first question below to start a conversation.", "✨")

    question = st.chat_input("Ask something about your documents…")
    if not question:
        return
    if not scope:
        st.warning("Select at least one document to search.")
        return
    question = question.strip()[:2000]
    try:
        llm = ensure_llm()  # load first so a failed load doesn't leave an unanswered message
        if llm is None:
            return
        new_session = session is None
        if new_session:
            selected = db.create_chat_session(question[:60], scope)
            st.session_state["chat_session_id"] = selected
        history = [{"role": m["role"], "content": m["content"]}
                   for m in db.get_chat_messages(selected)][-2 * config.MAX_HISTORY_TURNS:]
        db.add_chat_message(selected, "user", question)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and thinking…"):
                result = get_rag().answer(question, llm, scope, history)
            st.markdown(result["answer"])
            _show_sources(result["sources"])
        db.add_chat_message(selected, "assistant", result["answer"], result["sources"])
        db.log_activity("chat", f"Asked: {question[:80]}")
        if new_session:
            st.rerun()
    except (LLMError, EmbeddingError, DatabaseError) as exc:
        st.error(str(exc))
