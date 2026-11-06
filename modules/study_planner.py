"""AI study planner / agent: roadmap, daily tasks, projects, task tracking, regeneration."""
from __future__ import annotations

import streamlit as st

import config
from ai.generators import generate_study_plan
from ai.llm_engine import LLMError
from database.db import DatabaseError
from services import ensure_llm, get_db
from utils import ui
from utils.validators import ValidationError, validate_int, validate_text


def _create(db, goal, skills, weak, hours, days, notes) -> None:
    llm = ensure_llm()
    if llm is None:
        return
    try:
        with st.spinner("Your study agent is designing a roadmap and daily tasks… (this can take a few minutes on CPU)"):
            overview, tasks = generate_study_plan(llm, goal, skills, weak, hours, days, notes)
        db.create_study_plan(goal, skills, weak, hours, days, notes, overview, tasks)
        st.toast("Study plan created", icon="🗺️")
        st.rerun()
    except (LLMError, DatabaseError) as exc:
        st.error(str(exc))


def _on_toggle(db, task_id: int) -> None:
    db.set_task_completed(task_id, bool(st.session_state.get(f"task_{task_id}")))


def render() -> None:
    db = get_db()
    ui.page_header("Study Planner", "A local-LLM agent that turns your goal and weak areas into a day-by-day plan.")
    plans = db.list_study_plans()
    active = next((p for p in plans if p["is_active"]), None)

    with st.expander("➕ Create a new plan" if active else "➕ Create your plan", expanded=not active):
        with st.form("plan_form"):
            goal = st.text_input("Target job / career goal", value=active["goal"] if active else "",
                                 placeholder="e.g. Junior Data Scientist")
            c1, c2 = st.columns(2)
            skills = c1.text_area("Current skills", height=90, placeholder="Python, SQL basics, statistics")
            weak = c2.text_area("Weak areas", height=90, placeholder="Deep learning, system design")
            c3, c4 = st.columns(2)
            hours = c3.slider("Study hours per day", 0.5, 8.0, 2.0, 0.5)
            days = c4.select_slider("Plan length (days)", options=[7, 14], value=7)
            go = st.form_submit_button("🧠 Generate plan")
        if go:
            try:
                g = validate_text(goal, "Career goal", 3, 150)
                s = validate_text(skills, "Current skills", 0, 600) if skills.strip() else ""
                w = validate_text(weak, "Weak areas", 0, 600) if weak.strip() else ""
                d = validate_int(days, "Plan length", 1, config.MAX_STUDY_DAYS)
            except ValidationError as exc:
                st.error(str(exc))
            else:
                _create(db, g, s, w, hours, d, "")

    if not active:
        ui.empty_state("No study plan yet. Create one above.", "🗺️")
        return

    if len(plans) > 1:
        labels = {p["id"]: f"#{p['id']} · {p['goal']} · {p['created_at'][:10]}" for p in plans}
        pick = st.selectbox("Plan", [p["id"] for p in plans], index=[p["id"] for p in plans].index(active["id"]),
                            format_func=labels.get)
        if pick != active["id"]:
            db.activate_study_plan(pick)
            st.rerun()

    tasks = db.get_study_tasks(active["id"])
    done = sum(1 for t in tasks if t["completed"])
    st.markdown(f"### 🎯 {active['goal']}")
    st.progress(done / len(tasks) if tasks else 0.0, text=f"{done}/{len(tasks)} tasks completed")
    plan = active["plan"]
    if plan.get("summary"):
        st.write(plan["summary"])

    tab_road, tab_tasks, tab_proj, tab_custom = st.tabs(["🗺️ Roadmap", "✅ Daily tasks", "🛠️ Projects & quizzes", "🔧 Customize"])
    with tab_road:
        for p in plan.get("roadmap", []):
            with st.expander(p["phase"], expanded=True):
                st.write(p["focus"])
                for t in p.get("topics", []):
                    st.markdown(f"- {t}")
    with tab_tasks:
        current_day = None
        for t in tasks:
            if t["day"] != current_day:
                current_day = t["day"]
                st.markdown(f"**Day {current_day}**")
            label = f"{t['title']} · ~{t['est_hours']}h" if t["est_hours"] else t["title"]
            st.checkbox(label, value=bool(t["completed"]), key=f"task_{t['id']}", on_change=_on_toggle,
                        args=(db, t["id"]), help=t["description"] or None)
            if t["description"]:
                st.caption(t["description"])
    with tab_proj:
        if plan.get("projects"):
            for pr in plan["projects"]:
                ui.card(pr["title"], pr["description"])
        else:
            ui.empty_state("No projects were generated for this plan.")
        topics = plan.get("quiz_topics", [])
        if topics:
            st.markdown("**Quiz yourself on:**")
            for tpc in topics:
                def _quiz(tpc=tpc):
                    st.session_state["quiz_prefill_topic"] = tpc
                    ui.goto("❓ Quiz Generator")
                st.button(f"❓ {tpc}", key=f"qz_{tpc}", on_click=_quiz)
    with tab_custom:
        notes = st.text_area("Tell the agent what to change", placeholder="e.g. Focus more on SQL, add interview practice, lighter weekends")
        c1, c2 = st.columns(2)
        if c1.button("🔄 Regenerate with these changes"):
            _create(db, active["goal"], active["skills"] or "", active["weak_areas"] or "", active["hours_per_day"],
                    active["days"], notes.strip()[:500])
        if c2.button("🗑️ Delete this plan"):
            db.delete_study_plan(active["id"])
            st.rerun()
