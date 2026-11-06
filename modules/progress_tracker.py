"""Progress charts: interview scores, quiz performance, completed study tasks, history."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services import get_db
from utils import ui


def render() -> None:
    db = get_db()
    ui.page_header("Progress Tracking", "Your interview, quiz and study performance over time.")
    stats = db.get_stats()
    cols = st.columns(4)
    with cols[0]:
        avg = stats["avg_interview_score"]
        ui.metric_card("Avg interview score", "—" if avg is None else f"{avg:.1f}/10", f"{stats['interviews_completed']} completed")
    with cols[1]:
        acc = stats["quiz_accuracy"]
        ui.metric_card("Quiz accuracy", "—" if acc is None else f"{acc:.0f}%", f"{stats['quizzes']} quizzes")
    with cols[2]:
        ui.metric_card("Tasks completed", f"{stats['tasks_done']}/{stats['tasks_total']}", "active plan")
    with cols[3]:
        ui.metric_card("Answers evaluated", stats["evaluations"], "")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Interview scores")
        hist = db.interview_score_history()
        if hist:
            df = pd.DataFrame(hist)
            df["session"] = [f"#{r['id']}" for r in hist]
            st.line_chart(df.set_index("session")["overall_score"], y_label="score /10")
        else:
            ui.empty_state("Complete a mock interview to see your trend.", "🎤")
    with c2:
        st.markdown("#### Quiz performance")
        qh = db.quiz_history()
        if qh:
            df = pd.DataFrame(qh)
            df["accuracy %"] = df["correct"] / df["total"] * 100
            df["quiz"] = [f"#{r['id']}" for r in qh]
            st.bar_chart(df.set_index("quiz")["accuracy %"], y_label="accuracy %")
        else:
            ui.empty_state("Take a quiz to see your results.", "❓")

    st.markdown("#### Study tasks completed per day")
    td = db.tasks_completed_by_day()
    if td:
        st.bar_chart(pd.DataFrame(td).set_index("day")["completed"], y_label="tasks")
    else:
        ui.empty_state("Tick off tasks in your study plan to track them here.", "✅")

    st.markdown("#### Learning history")
    activity = db.recent_activity(30)
    if activity:
        st.dataframe(pd.DataFrame(activity)[["created_at", "kind", "detail"]], use_container_width=True, hide_index=True)
    else:
        ui.empty_state("No history yet.", "🕑")
