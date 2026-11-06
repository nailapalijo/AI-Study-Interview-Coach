"""Shared, cached service objects and LLM-availability helpers for the Streamlit UI."""
from __future__ import annotations

import streamlit as st

import config
from ai.embeddings import Embedder
from ai.llm_engine import LLMEngine
from ai.model_loader import ModelLoadError, ModelManager
from ai.rag_pipeline import RAGPipeline
from database.db import Database


@st.cache_resource(show_spinner=False)
def get_db() -> Database:
    return Database(config.DB_PATH)


@st.cache_resource(show_spinner=False)
def get_llm() -> LLMEngine:
    return LLMEngine(ModelManager())


@st.cache_resource(show_spinner=False)
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner=False)
def get_rag() -> RAGPipeline:
    return RAGPipeline(get_db(), get_embedder())


def ensure_llm() -> LLMEngine | None:
    """Return a loaded LLM engine, or show an error with a retry control and return None."""
    llm = get_llm()
    if llm.is_loaded():
        return llm
    with st.spinner(f"Loading {llm.manager.requested_id} into memory — cached weights load from disk; "
                    "only missing weights need downloading. CPU loading can take several minutes…"):
        try:
            llm.load()
            st.toast(f"Model ready: {llm.model_id}", icon="✅")
            return llm
        except ModelLoadError as exc:
            st.error(str(exc))
            st.caption("After the current load finishes—or after fixing a download error—submit your AI action again.")
            return None
