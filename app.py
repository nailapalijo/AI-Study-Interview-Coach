"""AI Study & Interview Coach — Streamlit entry point.

Local: streamlit run app.py --server.address 127.0.0.1 --server.port 8501
Spaces: streamlit run app.py --server.address 0.0.0.0 --server.port 7860
"""
import config  # noqa: F401  (must be first: sets cache dirs before ML libraries load)

import logging

import streamlit as st

config.setup_logging()
st.set_page_config(page_title="AI Study & Interview Coach", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

import services  # noqa: E402
from modules import (answer_evaluator, dashboard, document_chat, document_manager, interview_generator,  # noqa: E402
                     mock_interview, progress_tracker, quiz_generator, study_planner)
from utils import ui  # noqa: E402

logger = logging.getLogger(__name__)

PAGES = {
    "🏠 Dashboard": dashboard.render,
    "📄 Documents": document_manager.render,
    "💬 Document Chat": document_chat.render,
    "🧩 Interview Generator": interview_generator.render,
    "🎤 Mock Interview": mock_interview.render,
    "📝 Answer Evaluator": answer_evaluator.render,
    "🗺️ Study Planner": study_planner.render,
    "❓ Quiz Generator": quiz_generator.render,
    "📈 Progress": progress_tracker.render,
}


def sidebar() -> str:
    with st.sidebar:
        st.markdown("<div class='brand'>🎓 AI Coach</div><div class='brand-sub'>Study &amp; Interview Coach</div>",
                    unsafe_allow_html=True)
        if st.session_state.get("nav") not in PAGES:
            st.session_state["nav"] = "🏠 Dashboard"
        page = st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed")
        st.divider()
        llm = services.get_llm()
        with st.expander("⚙️ Model settings"):
            choices = dict(config.MODEL_CHOICES)
            if llm.manager.requested_id not in choices.values():  # custom repo id from $LLM_MODEL
                choices[llm.manager.requested_id] = llm.manager.requested_id
            keys = list(choices)
            current = next(k for k, v in choices.items() if v == llm.manager.requested_id)
            choice = st.selectbox("Language model", keys, index=keys.index(current),
                                  help="Runs locally on this machine / Space. Smaller = faster, less accurate.")
            if choices[choice] != llm.manager.requested_id:
                llm.manager.set_requested(choices[choice])
                llm.manager.unload()  # only one LLM in memory at a time
                st.rerun()
            st.caption("Status: " + (f"✅ loaded ({llm.model_id})" if llm.is_loaded()
                                      else "⏳ loads on first AI action"))
            if llm.is_loaded() and st.button("Unload model", help="Free RAM"):
                llm.manager.unload()
                st.rerun()
            st.caption(f"Embeddings: {config.EMBEDDING_MODEL.split('/')[-1]}")
            st.caption(f"Data dir: `{config.DATA_DIR}`")
    return page


def main() -> None:
    ui.load_css(config.STATIC_CSS)
    try:
        page = sidebar()
        st.session_state["_page"] = page
        PAGES[page]()
    except Exception:  # last-resort guard so the app never white-screens
        logger.exception("Unhandled error in page")
        st.error("Something went wrong while rendering this page. Details were written to the log. "
                 "Try reloading, or switch to another page.")
        st.stop()


main()
