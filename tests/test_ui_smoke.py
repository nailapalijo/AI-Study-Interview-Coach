"""End-to-end UI smoke test using Streamlit's AppTest (skipped if streamlit is not installed).
Opens the real app and visits every navigation entry."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_every_page_renders_without_exceptions():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = list(at.sidebar.radio(key="nav").options)
    assert len(labels) == 9
    for label in labels:
        at.sidebar.radio(key="nav").set_value(label).run()
        assert not at.exception, (label, [e.value for e in at.exception])
