"""Interactive mock interview: one question at a time, AI evaluation, final summary."""
from __future__ import annotations

import logging

import streamlit as st
import pandas as pd

import config
from ai.embeddings import EmbeddingError
from ai.evaluation import CRITERIA_LABELS, evaluate_interview_answer, narrative_summary, summarize_interview
from ai.generators import generate_questions
from ai.llm_engine import LLMError
from database.db import DatabaseError
from modules.interview_generator import DIFFICULTIES, LEVELS, profile_selector
from services import ensure_llm, get_db
from utils import ui
from utils.validators import ValidationError, validate_int, validate_text

logger = logging.getLogger(__name__)


def _setup(db) -> None:
    resumable = [s for s in db.list_interview_sessions(("mock",)) if s["status"] == "in_progress"]
    if resumable:
        st.markdown("#### ▶️ Resume an interview")
        for s in resumable[:3]:
            c1, c2 = st.columns([4, 1])
            c1.write(f"#{s['id']} · {s['job_title']} ({s['difficulty']}) · {s['created_at'][:16]}")
            if c2.button("Resume", key=f"resume_{s['id']}"):
                st.session_state["mock_session_id"] = s["id"]
                st.rerun()
    st.markdown("#### 🎤 Start a new interview")
    with st.form("mock_form"):
        c1, c2 = st.columns(2)
        job = c1.text_input("Job title", placeholder="e.g. Data Scientist")
        domain = c2.text_input("Technical domain", placeholder="e.g. machine learning, SQL")
        c3, c4, c5 = st.columns(3)
        level = c3.selectbox("Experience level", LEVELS, index=2)
        difficulty = c4.selectbox("Difficulty", DIFFICULTIES, index=1)
        n = c5.number_input("Number of questions", 1, config.MAX_QUESTIONS, 5)
        go = st.form_submit_button("🚀 Begin interview")
    profile = profile_selector(db, "mock")
    if not go:
        return
    try:
        job_v = validate_text(job, "Job title", 2, 100)
        domain_v = validate_text(domain, "Technical domain", 2, 100)
        n_v = validate_int(n, "Number of questions", 1, config.MAX_QUESTIONS)
    except ValidationError as exc:
        st.error(str(exc))
        return
    llm = ensure_llm()
    if llm is None:
        return
    try:
        with st.spinner("The interviewer is preparing your questions…"):
            qs = generate_questions(llm, job_v, level, domain_v, n_v, difficulty, profile)
        st.session_state["mock_session_id"] = db.create_interview_session(job_v, level, domain_v, difficulty, qs)
        st.session_state.pop("mock_feedback_qid", None)
        st.rerun()
    except (LLMError, DatabaseError, EmbeddingError) as exc:
        st.error(str(exc))


def _finish(db, session, questions) -> None:
    items = [{"question": q["question"], "category": q["category"], "score": q["score"], "feedback": q["feedback"]}
             for q in questions if q["answer_id"]]
    summary = summarize_interview(items)
    llm = ensure_llm()
    if llm is not None and items:
        try:
            with st.spinner("Writing your summary…"):
                summary["narrative"] = narrative_summary(llm, session["job_title"], items)
        except LLMError as exc:
            logger.warning("Narrative summary failed: %s", exc)
    db.complete_interview_session(session["id"], summary["overall"], summary)
    st.session_state.pop("mock_feedback_qid", None)
    st.rerun()


def _summary(db, session, questions) -> None:
    summary = session["summary"] or {}
    st.success("Interview complete 🎉")
    ui.score_badge(session["overall_score"], "Final score")
    if summary.get("narrative"):
        st.write(summary["narrative"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Average by criterion**")
        for k, v in (summary.get("criteria_avg") or {}).items():
            st.progress(min(v / 10, 1.0), text=f"{CRITERIA_LABELS.get(k, k)}: {v}/10")
    with c2:
        st.markdown("**Average by question type**")
        cats = summary.get("by_category") or {}
        if cats:
            st.bar_chart(pd.DataFrame({"score": cats}), y_label="score /10")
    st.markdown("### Question-by-question review")
    for i, q in enumerate(questions, 1):
        with st.expander(f"Q{i} [{q['category']}] — {q['score'] if q['score'] is not None else '—'}/10 · {q['question'][:80]}"):
            st.markdown(f"**Question:** {q['question']}")
            st.markdown(f"**Your answer:** {q['answer'] or '—'}")
            if q["feedback"]:
                ui.render_feedback(q["feedback"])
    b1, b2 = st.columns([1, 4])
    if b1.button("🔁 New interview"):
        st.session_state.pop("mock_session_id", None)
        st.rerun()


def _run(db, session, questions) -> None:
    answered = [q for q in questions if q["answer_id"]]
    idx = len(answered)
    total = len(questions)
    st.progress(idx / total, text=f"Question {min(idx + 1, total)} of {total} · {session['job_title']} ({session['difficulty']})")

    fb_qid = st.session_state.get("mock_feedback_qid")
    last = next((q for q in questions if q["id"] == fb_qid and q["answer_id"]), None)
    if last:
        st.markdown(f"**{last['category']} question:** {last['question']}")
        with st.chat_message("user"):
            st.write(last["answer"])
        with st.chat_message("assistant"):
            ui.render_feedback(last["feedback"] or {})
        if idx < total:
            if st.button("Next question →", type="primary"):
                st.session_state.pop("mock_feedback_qid", None)
                st.rerun()
        elif st.button("🏁 Finish interview", type="primary"):
            _finish(db, session, questions)
        return
    if idx >= total:
        _finish(db, session, questions)
        return

    q = questions[idx]
    with st.chat_message("assistant"):
        st.markdown(f"**{q['category']} question**")
        st.write(q["question"])
    with st.form(f"answer_form_{q['id']}"):
        answer = st.text_area("Your answer", height=200, placeholder="Type your answer as you would say it in an interview…")
        submit = st.form_submit_button("Submit answer")
    if st.button("✖ Abandon interview", key="abandon"):
        db.delete_interview_session(session["id"])
        st.session_state.pop("mock_session_id", None)
        st.rerun()
    if not submit:
        return
    try:
        answer_v = validate_text(answer, "Answer", 10, 4000)
    except ValidationError as exc:
        st.error(str(exc))
        return
    llm = ensure_llm()
    if llm is None:
        return
    try:
        with st.spinner("The interviewer is evaluating your answer…"):
            fb = evaluate_interview_answer(llm, q["question"], answer_v, session["job_title"],
                                           session["experience_level"], q["category"], session["domain"])
        db.save_interview_answer(q["id"], answer_v, fb["overall_score"], fb)
        st.session_state["mock_feedback_qid"] = q["id"]
        st.rerun()
    except (LLMError, DatabaseError) as exc:
        st.error(f"{exc}\n\nYour answer has not been lost — press Submit again to retry.")


def render() -> None:
    db = get_db()
    ui.page_header("Mock Interview", "An AI interviewer asks one question at a time and scores every answer.")
    sid = st.session_state.get("mock_session_id")
    session = db.get_interview_session(sid) if sid else None
    if not session:
        st.session_state.pop("mock_session_id", None)
        _setup(db)
        past = [s for s in db.list_interview_sessions(("mock",)) if s["status"] == "completed"]
        if past:
            st.markdown("#### 🗂️ Past interviews")
            for s in past[:8]:
                c1, c2 = st.columns([4, 1])
                c1.write(f"#{s['id']} · {s['job_title']} · **{s['overall_score']}/10** · {s['created_at'][:16]}")
                if c2.button("View", key=f"view_{s['id']}"):
                    st.session_state["mock_session_id"] = s["id"]
                    st.rerun()
        return
    questions = db.get_interview_questions(sid)
    if session["status"] == "completed":
        _summary(db, session, questions)
    else:
        _run(db, session, questions)
