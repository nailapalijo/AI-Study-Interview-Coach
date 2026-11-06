"""Logic tests for LLM-output handling, scoring, generators. The LLM is a scripted double here:
these verify parsing/validation/retry/aggregation code, NOT model quality."""
import json

import pytest

from ai import generators as g
from ai.evaluation import (ANSWER_CRITERIA, INTERVIEW_CRITERIA, EvaluationError, evaluate_interview_answer,
                           normalize_evaluation, summarize_interview)
from ai.llm_engine import LLMEngine, LLMError
from ai.model_loader import ModelLoadError, ModelManager
from tests.conftest import ScriptedLLM
from utils.json_utils import JSONExtractionError, extract_json


# ---------------------------------------------------------------- JSON extraction
def test_extract_json_variants():
    assert extract_json('Sure!\n```json\n{"a": 1,}\n```') == {"a": 1}
    assert extract_json('text {"a": [1, 2]} more') == {"a": [1, 2]}
    assert extract_json('{"q": ["x", "y') == {"q": ["x", "y"]}  # truncated output is repaired
    with pytest.raises(JSONExtractionError):
        extract_json("no json here")
    with pytest.raises(JSONExtractionError):
        extract_json("")


# ---------------------------------------------------------------- interview scoring
GOOD = {"scores": {"technical_accuracy": 8, "communication": 7, "clarity": 9, "relevance": 8, "problem_solving": 6, "confidence": 7},
        "overall_score": 7.5, "strengths": ["clear"], "weaknesses": ["shallow"], "suggestions": ["add example"],
        "improved_answer": "Better answer"}


def test_normalize_evaluation_happy_path():
    r = normalize_evaluation(GOOD, INTERVIEW_CRITERIA)
    assert r["overall_score"] == 7.5 and r["scores"]["clarity"] == 9 and r["improved_answer"] == "Better answer"


def test_normalize_clamps_and_parses_strings():
    raw = {"scores": {"accuracy": "11/10", "completeness": -3, "clarity": "7", "technical_depth": 5.5, "relevance": 20},
           "feedback": "ok", "strengths": "- a\n- b"}
    r = normalize_evaluation(raw, ANSWER_CRITERIA)
    assert r["scores"] == {"accuracy": 10.0, "completeness": 0.0, "clarity": 7.0, "technical_depth": 5.5, "relevance": 10.0}
    assert r["strengths"] == ["a", "b"]
    assert r["overall_score"] == pytest.approx(sum(r["scores"].values()) / 5, abs=0.1)  # derived, not invented


def test_normalize_rejects_unusable():
    for bad in ["text", [], {}, {"scores": {"clarity": 5}, "feedback": "x"}, {"scores": {c: 5 for c in ANSWER_CRITERIA}}]:
        with pytest.raises(EvaluationError):
            normalize_evaluation(bad, ANSWER_CRITERIA)


def test_evaluate_retries_on_bad_json_then_succeeds():
    llm = ScriptedLLM(["I think the answer is fine.", json.dumps(GOOD)])
    r = evaluate_interview_answer(llm, "Q?", "A" * 30, "Dev", "Mid", "Technical", "Backend")
    assert r["overall_score"] == 7.5 and len(llm.calls) == 2
    assert "ONLY the requested JSON" in llm.calls[1][-1]["content"]
    assert llm.calls[1][-2]["role"] == "assistant"
    assert llm.calls[1][-2]["content"] == "I think the answer is fine."


def test_evaluate_gives_up_with_clear_error():
    with pytest.raises(LLMError):
        evaluate_interview_answer(ScriptedLLM(["nope"] * 3), "Q?", "A" * 30, "Dev", "Mid", "Technical", "Backend")


def test_llm_reports_busy_instead_of_queueing_cpu_generation():
    llm = LLMEngine()
    llm._gen_lock.acquire()
    try:
        with pytest.raises(LLMError, match="busy generating another response"):
            llm.generate([])
    finally:
        llm._gen_lock.release()


def test_model_load_reports_busy_instead_of_queueing_another_session():
    manager = ModelManager()
    manager._load_guard.acquire()
    try:
        with pytest.raises(ModelLoadError, match="already loading in another session"):
            manager.load()
    finally:
        manager._load_guard.release()


def test_summarize_interview():
    items = [{"question": "q1", "category": "Technical", "score": 8, "feedback": {"scores": {"clarity": 8}}},
             {"question": "q2", "category": "Technical", "score": 6, "feedback": {"scores": {"clarity": 6}}},
             {"question": "q3", "category": "HR", "score": 4, "feedback": {"scores": {"clarity": 4}}}]
    s = summarize_interview(items)
    assert s["overall"] == 6.0 and s["criteria_avg"]["clarity"] == 6.0
    assert s["by_category"] == {"Technical": 7.0, "HR": 4.0} and s["best"]["question"] == "q1" and s["worst"]["score"] == 4
    assert summarize_interview([])["overall"] is None


# ---------------------------------------------------------------- generators
def test_split_counts():
    assert sum(g.split_counts(10).values()) == 10 and g.split_counts(1) == {"Technical": 1}
    assert set(g.split_counts(5)) == {"Technical", "Behavioral", "Scenario", "HR"}
    with pytest.raises(ValueError):
        g.split_counts(0)


def test_generate_questions_by_category():
    n = 5
    outs = [json.dumps({"questions": [f"{cat} question number {i} about the role?" for i in range(c)]})
            for cat, c in g.split_counts(n).items()]
    qs = g.generate_questions(ScriptedLLM(outs), "Dev", "Mid", "Backend", n, "Medium")
    assert len(qs) == 5 and {q["category"] for q in qs} == {"Technical", "Behavioral", "Scenario", "HR"}


def test_generate_questions_too_few_triggers_retry():
    short = json.dumps({"questions": ["Only one decent question here?"]})
    ok = json.dumps({"questions": ["First technical question here?", "Second technical question here?"]})
    llm = ScriptedLLM([short, ok])
    qs = g.generate_questions(llm, "Dev", "Mid", "Backend", 2, "Easy")
    assert len(qs) == 2 and len(llm.calls) == 2


def test_normalize_quiz_question_variants():
    base = {"question": "What is 2+2?", "options": ["A) 3", "B) 4", "C) 5", "D) 6"], "explanation": "Math."}
    assert g.normalize_quiz_question({**base, "answer_index": 1})["options"][1] == "4"
    assert g.normalize_quiz_question({**base, "answer": "B"})["answer_index"] == 1
    assert g.normalize_quiz_question({**base, "correct_answer": "4"})["answer_index"] == 1
    for bad in [{**base, "answer_index": 7}, {**base, "options": ["a", "b", "c"], "answer_index": 0},
                {**base, "options": ["a", "a", "b", "c"], "answer_index": 0}, {**base, "answer_index": 0, "explanation": ""},
                {"question": "", "options": [], "answer_index": 0}, "str"]:
        with pytest.raises(ValueError):
            g.normalize_quiz_question(bad)


def test_generate_quiz_batches_and_dedupes():
    def q(i):
        return {"question": f"Question {i}?", "options": [f"a{i}", f"b{i}", f"c{i}", f"d{i}"], "answer_index": i % 4, "explanation": "e"}
    llm = ScriptedLLM([json.dumps({"questions": [q(0), q(1), q(2)]}),
                       json.dumps({"questions": [q(2), q(3)]})])  # q(2) duplicate is dropped
    quiz = g.generate_quiz(llm, 4, "Medium", topic="Python")
    assert [x["question"] for x in quiz] == [f"Question {i}?" for i in range(4)]
    with pytest.raises(ValueError):
        g.generate_quiz(llm, 2, "Easy")


def test_generate_study_plan():
    overview = {"summary": "s", "roadmap": [{"phase": "P1", "focus": "f", "topics": ["t"]}],
                "projects": [{"title": "Proj", "description": "d"}], "quiz_topics": ["SQL"]}
    t = lambda n: json.dumps({"tasks": [{"title": f"Task {i}", "description": "do", "est_hours": 2} for i in range(n)]})
    llm = ScriptedLLM([json.dumps(overview), t(7), t(3)])
    plan, tasks = g.generate_study_plan(llm, "Data Scientist", "py", "sql", 2.0, 10)
    assert plan["roadmap"][0]["phase"] == "P1" and [x["day"] for x in tasks] == list(range(1, 11))
    assert tasks[7]["title"] == "Task 0"


def test_study_plan_rejects_empty_roadmap():
    with pytest.raises(LLMError):
        g.generate_study_plan(ScriptedLLM([json.dumps({"roadmap": []})] * 3), "g", "", "", 2, 7)


# ---------------------------------------------------------------- real local generation (opt-in: downloads weights)
@pytest.mark.skipif(__import__("os").environ.get("RUN_MODEL_TESTS") != "1",
                    reason="set RUN_MODEL_TESTS=1 (needs torch, transformers, internet) to run real LLM inference")
def test_real_llm_generation_and_json():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from ai.llm_engine import LLMEngine
    llm = LLMEngine()
    out = llm.generate([{"role": "user", "content": "Reply with the single word: hello"}], max_new_tokens=10, temperature=0)
    assert isinstance(out, str) and out.strip()
    obj = llm.generate_json([{"role": "user", "content": 'Return JSON {"ok": true}'}], max_new_tokens=30)
    assert isinstance(obj, dict)


def test_hf_transfer_is_disabled_by_default_for_windows_stability():
    import os

    import config

    assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") in {"0", "false", "False"}
    assert config.MODEL_CACHE_DIR.exists()
