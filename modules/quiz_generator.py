"""MCQ quizzes from uploaded documents or a technical topic, with scoring and history."""
from __future__ import annotations

import streamlit as st

from ai.embeddings import EmbeddingError
from ai.generators import generate_quiz
from ai.llm_engine import LLMError
from database.db import DatabaseError
from services import ensure_llm, get_db, get_rag
from utils import ui
from utils.validators import ValidationError, validate_int, validate_text

LETTERS = "ABCD"


def _setup(db) -> None:
    prefill = st.session_state.pop("quiz_prefill_topic", None)
    docs = [d for d in db.list_documents() if d["status"] == "indexed"]
    modes = ["Technical topic"] + (["My documents"] if docs else [])
    if prefill:
        st.session_state["quiz_mode"] = "Technical topic"
    mode = st.radio("Quiz source", modes, horizontal=True, key="quiz_mode")
    with st.form("quiz_form"):
        topic, chosen = "", []
        if mode == "Technical topic":
            topic = st.text_input("Topic", value=prefill or "", placeholder="e.g. Python decorators, SQL joins, REST APIs")
        else:
            names = {d["id"]: d["name"] for d in docs}
            chosen = st.multiselect("Documents", list(names), default=list(names)[:1], format_func=names.get)
            topic = st.text_input("Focus topic (optional)", placeholder="Leave blank to cover the whole document")
        c1, c2 = st.columns(2)
        n = c1.number_input("Number of questions", 1, 10, 5)
        difficulty = c2.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
        go = st.form_submit_button("✨ Generate quiz")
    if not go:
        return
    try:
        n_v = validate_int(n, "Number of questions", 1, 10)
        if mode == "Technical topic":
            topic_v, source_text = validate_text(topic, "Topic", 2, 150), None
        else:
            if not chosen:
                raise ValidationError("Select at least one document.")
            topic_v = topic.strip()[:150]
            source_text = get_rag().sample_context(chosen, max_chars=3000, topic=topic_v or None)
            if not source_text:
                raise ValidationError("No relevant passages were found in the selected documents for that topic.")
    except (ValidationError, EmbeddingError) as exc:
        st.error(str(exc))
        return
    llm = ensure_llm()
    if llm is None:
        return
    try:
        with st.spinner("Writing quiz questions…"):
            qs = generate_quiz(llm, n_v, difficulty, source_text=source_text, topic=topic_v or None)
        title = topic_v or f"Quiz on {', '.join(d['name'] for d in docs if d['id'] in chosen)[:60]}"
        st.session_state["quiz"] = {"title": title, "source": mode, "questions": qs, "result": None}
        st.session_state["quiz_run"] = st.session_state.get("quiz_run", 0) + 1
        st.rerun()
    except (LLMError, EmbeddingError) as exc:
        st.error(str(exc))


def _take(db, quiz: dict) -> None:
    st.markdown(f"### {quiz['title']}")
    qs = quiz["questions"]
    if quiz["result"] is None:
        with st.form(f"quiz_answers_{st.session_state['quiz_run']}"):
            picks = []
            for i, q in enumerate(qs):
                picks.append(st.radio(f"**{i + 1}. {q['question']}**", [0, 1, 2, 3], index=None,
                                      format_func=lambda k, q=q: f"{LETTERS[k]}. {q['options'][k]}", key=f"qa_{st.session_state['quiz_run']}_{i}"))
            submit = st.form_submit_button("Submit quiz")
        if submit:
            if any(p is None for p in picks):
                st.warning("Please answer every question.")
                return
            details = [{"question": q["question"], "options": q["options"], "answer_index": q["answer_index"],
                        "chosen": p, "explanation": q["explanation"]} for q, p in zip(qs, picks)]
            correct = sum(1 for d in details if d["chosen"] == d["answer_index"])
            try:
                db.save_quiz_result(quiz["title"], quiz["source"], len(qs), correct, details)
            except DatabaseError as exc:
                st.error(str(exc))
            quiz["result"] = {"correct": correct, "details": details}
            st.rerun()
    else:
        res = quiz["result"]
        pct = res["correct"] / len(qs) * 100
        ui.score_badge(res["correct"] / len(qs) * 10, f"Score {res['correct']}/{len(qs)} ({pct:.0f}%)")
        for i, d in enumerate(res["details"], 1):
            ok = d["chosen"] == d["answer_index"]
            st.markdown(f"**{i}. {d['question']}** {'✅' if ok else '❌'}")
            st.markdown(f"Your answer: {LETTERS[d['chosen']]}. {d['options'][d['chosen']]}")
            if not ok:
                st.markdown(f"Correct answer: **{LETTERS[d['answer_index']]}. {d['options'][d['answer_index']]}**")
            st.caption(d["explanation"])
        if st.button("🔁 New quiz"):
            st.session_state.pop("quiz", None)
            st.rerun()


def render() -> None:
    db = get_db()
    ui.page_header("Quiz Generator", "Multiple-choice quizzes from your documents or any technical topic.")
    quiz = st.session_state.get("quiz")
    if quiz:
        _take(db, quiz)
    else:
        _setup(db)
    st.markdown("### Quiz history")
    history = db.list_quiz_results(10)
    if not history:
        ui.empty_state("Completed quizzes will be listed here.", "❓")
    for h in history:
        st.markdown(f"<div class='activity-row'><b>{h['correct']}/{h['total']}</b> · {h['title']}<br>"
                    f"<small>{h['created_at'][:16]} UTC · {h['source']}</small></div>", unsafe_allow_html=True)
