"""Dashboard: real statistics from the database, recent activity, quick actions."""
from __future__ import annotations

import html

import streamlit as st

from services import get_db
from utils import ui


def render() -> None:
    db = get_db()
    stats = db.get_stats()
    ui.page_header("Welcome back 👋", "Upload your documents, practise interviews and follow a personalised study plan.")

    progress = (stats["tasks_done"] / stats["tasks_total"]) if stats["tasks_total"] else None
    cols = st.columns(4)
    with cols[0]:
        ui.metric_card("Documents", stats["documents"], f"{stats['indexed_chunks']} chunks indexed")
    with cols[1]:
        ui.metric_card("Interview sessions", stats["interviews_total"], f"{stats['interviews_completed']} completed")
    with cols[2]:
        ui.metric_card("Study progress", "—" if progress is None else f"{progress * 100:.0f}%",
                       f"{stats['tasks_done']}/{stats['tasks_total']} tasks" if stats["tasks_total"] else "No active plan")
    with cols[3]:
        acc = stats["quiz_accuracy"]
        ui.metric_card("Quiz accuracy", "—" if acc is None else f"{acc:.0f}%", f"{stats['quizzes']} quizzes taken")
    if progress is not None:
        st.progress(progress, text="Active study plan completion")

    st.markdown("### ⚡ Quick actions")
    actions = [("📄 Upload documents", "Add CVs, notes and PDFs", "📄 Documents"),
               ("💬 Chat with documents", "Ask grounded questions", "💬 Document Chat"),
               ("🎤 Start mock interview", "Practise with AI feedback", "🎤 Mock Interview"),
               ("🗺️ Plan my studies", "Personalised roadmap", "🗺️ Study Planner")]
    for col, (title, sub, page) in zip(st.columns(4), actions):
        with col:
            ui.card(title, sub)
            st.button("Open", key=f"qa_{page}", on_click=ui.goto, args=(page,), use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### 🕑 Recent activity")
        activity = db.recent_activity(8)
        if not activity:
            ui.empty_state("No activity yet — upload a document or start an interview to get going.")
        for a in activity:
            st.markdown(f"<div class='activity-row'>{html.escape(a['detail'])}<br><small>{a['created_at']} UTC · "
                        f"{html.escape(a['kind'])}</small></div>", unsafe_allow_html=True)
    with right:
        st.markdown("### 📊 Learning statistics")
        avg = stats["avg_interview_score"]
        rows = [("Average interview score", "—" if avg is None else f"{avg:.1f}/10"),
                ("Answers evaluated", stats["evaluations"]),
                ("Questions asked in chat", stats["chat_messages"]),
                ("Study tasks completed", stats["tasks_done"])]
        for label, value in rows:
            st.markdown(f"<div class='activity-row'>{label}: <b>{value}</b></div>", unsafe_allow_html=True)
