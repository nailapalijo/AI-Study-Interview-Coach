"""Evaluate any interview answer the user types."""
from __future__ import annotations

import streamlit as st

from ai.evaluation import evaluate_user_answer
from ai.llm_engine import LLMError
from database.db import DatabaseError
from services import ensure_llm, get_db
from utils import ui
from utils.validators import ValidationError, validate_text


def render() -> None:
    db = get_db()
    ui.page_header("AI Answer Evaluator", "Paste an interview question and your answer for structured, actionable feedback.")
    with st.form("eval_form"):
        question = st.text_area("Interview question", height=90, placeholder="e.g. Explain the difference between a process and a thread.")
        answer = st.text_area("Your answer", height=220)
        context = st.text_input("Role / context (optional)", placeholder="e.g. Backend engineer, junior level")
        go = st.form_submit_button("🔍 Evaluate answer")
    if go:
        try:
            q = validate_text(question, "Question", 8, 1000)
            a = validate_text(answer, "Answer", 10, 4000)
            ctx = validate_text(context, "Context", 0, 200) if context.strip() else ""
        except ValidationError as exc:
            st.error(str(exc))
        else:
            llm = ensure_llm()
            if llm is not None:
                try:
                    with st.spinner("Evaluating…"):
                        fb = evaluate_user_answer(llm, q, a, ctx)
                    db.save_evaluation(q, a, fb["overall_score"], fb)
                    st.toast("Evaluation saved", icon="✅")
                    ui.render_feedback(fb)
                except (LLMError, DatabaseError) as exc:
                    st.error(str(exc))
    history = db.list_evaluations(10)
    st.markdown("### Previous evaluations")
    if not history:
        ui.empty_state("Your evaluated answers will appear here.", "📝")
    for h in history:
        with st.expander(f"{h['overall_score']}/10 · {h['question'][:80]} · {h['created_at'][:16]}"):
            st.markdown(f"**Answer:** {h['answer']}")
            ui.render_feedback(h["feedback"])
