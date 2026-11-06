"""Small Streamlit UI helpers (HTML cards, headers, feedback rendering)."""
from __future__ import annotations

import html

import streamlit as st

from ai.evaluation import CRITERIA_LABELS


def load_css(path) -> None:
    try:
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except OSError:
        pass


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='page-header'><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>",
                unsafe_allow_html=True)


def metric_card(label: str, value, hint: str = "") -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{html.escape(label)}</div>"
        f"<div class='metric-value'>{html.escape(str(value))}</div>"
        f"<div class='metric-hint'>{html.escape(hint)}</div></div>", unsafe_allow_html=True)


def card(title: str, body: str = "") -> None:
    st.markdown(f"<div class='card'><h4>{html.escape(title)}</h4><p>{html.escape(body)}</p></div>",
                unsafe_allow_html=True)


def empty_state(message: str, icon: str = "📭") -> None:
    st.markdown(f"<div class='empty-state'><div class='empty-icon'>{icon}</div>{html.escape(message)}</div>",
                unsafe_allow_html=True)


def goto(page: str) -> None:
    """on_click callback: switch the sidebar navigation."""
    st.session_state["nav"] = page


def score_color(score: float | None) -> str:
    if score is None:
        return "#94A3B8"
    return "#22C55E" if score >= 7 else "#F59E0B" if score >= 5 else "#EF4444"


def score_badge(score: float | None, label: str = "Overall") -> None:
    text = "—" if score is None else f"{score:.1f}/10"
    st.markdown(f"<div class='score-badge' style='border-color:{score_color(score)};color:{score_color(score)}'>"
                f"{html.escape(label)}: <b>{text}</b></div>", unsafe_allow_html=True)


def render_feedback(fb: dict) -> None:
    """Render a normalised evaluation dict (see ai.evaluation.normalize_evaluation)."""
    score_badge(fb.get("overall_score"))
    if fb.get("feedback"):
        st.write(fb["feedback"])
    for key, val in (fb.get("scores") or {}).items():
        st.progress(min(max(val / 10, 0.0), 1.0), text=f"{CRITERIA_LABELS.get(key, key)}: {val}/10")
    cols = st.columns(2)
    with cols[0]:
        if fb.get("strengths"):
            st.markdown("**✅ Strengths**")
            for s in fb["strengths"]:
                st.markdown(f"- {s}")
    with cols[1]:
        if fb.get("weaknesses"):
            st.markdown("**⚠️ Weaknesses**")
            for s in fb["weaknesses"]:
                st.markdown(f"- {s}")
    if fb.get("suggestions"):
        st.markdown("**💡 Suggested improvements**")
        for s in fb["suggestions"]:
            st.markdown(f"- {s}")
    if fb.get("improved_answer"):
        with st.expander("Example improved answer"):
            st.write(fb["improved_answer"])
