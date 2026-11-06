import numpy as np
import pytest

from database.db import DatabaseError


def test_document_lifecycle_and_cascade(db, tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    doc_id = db.add_document("x.txt", "txt", 5, str(f), "hello world", "h1", {"pages": 1})
    assert db.find_document_by_hash("h1")["id"] == doc_id
    assert db.get_document(doc_id)["metadata"]["pages"] == "1"
    db.replace_chunks(doc_id, ["a", "b"], np.eye(2, dtype=np.float32))
    assert db.get_document(doc_id)["status"] == "indexed" and len(db.get_chunks([doc_id])) == 2
    assert db.get_chunks([doc_id])[0]["vector"].dtype == np.float32
    db.delete_document(doc_id)
    assert db.list_documents() == [] and db.get_chunks() == [] and not f.exists()


def test_replace_chunks_mismatch(db):
    d = db.add_document("x", "txt", 1, None, "t")
    with pytest.raises(DatabaseError):
        db.replace_chunks(d, ["a"], np.zeros((2, 3), np.float32))


def test_chat(db):
    sid = db.create_chat_session("Hi", [1, 2])
    db.add_chat_message(sid, "user", "q")
    db.add_chat_message(sid, "assistant", "a", [{"doc_name": "x"}])
    msgs = db.get_chat_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["sources"][0]["doc_name"] == "x"
    assert db.get_chat_session(sid)["doc_ids"] == [1, 2]
    db.clear_chat_messages(sid)
    assert db.get_chat_messages(sid) == []
    db.delete_chat_session(sid)
    assert db.list_chat_sessions() == []


def test_interview_flow(db):
    qs = [{"category": "Technical", "question": "Q1?"}, {"category": "HR", "question": "Q2?"}]
    sid = db.create_interview_session("Dev", "Mid", "Backend", "Medium", qs)
    questions = db.get_interview_questions(sid)
    assert [q["position"] for q in questions] == [0, 1] and questions[0]["answer_id"] is None
    db.save_interview_answer(questions[0]["id"], "my answer", 7.5, {"scores": {"clarity": 7}})
    q0 = db.get_interview_questions(sid)[0]
    assert q0["score"] == 7.5 and q0["feedback"]["scores"]["clarity"] == 7
    db.complete_interview_session(sid, 7.5, {"overall": 7.5})
    s = db.get_interview_session(sid)
    assert s["status"] == "completed" and s["summary"]["overall"] == 7.5
    assert db.get_stats()["interviews_completed"] == 1 and db.get_stats()["avg_interview_score"] == 7.5
    with pytest.raises(DatabaseError):
        db.update_interview_session(sid, job_title="x")
    db.delete_interview_session(sid)
    assert db.get_interview_questions(sid) == []


def test_study_plan_and_tasks(db):
    tasks = [{"day": 1, "title": "T1", "description": "d", "est_hours": 2}, {"day": 2, "title": "T2"}]
    p1 = db.create_study_plan("goal", "s", "w", 2, 2, "", {"summary": "x"}, tasks)
    p2 = db.create_study_plan("goal2", "s", "w", 2, 2, "", {"summary": "y"}, tasks)
    plans = db.list_study_plans()
    assert [p["is_active"] for p in plans] == [1, 0]  # newest active only
    db.activate_study_plan(p1)
    assert next(p for p in db.list_study_plans() if p["id"] == p1)["is_active"] == 1
    t = db.get_study_tasks(p1)
    db.set_task_completed(t[0]["id"], True)
    assert db.get_stats()["tasks_done"] == 1 and db.tasks_completed_by_day()[0]["completed"] == 1
    db.set_task_completed(t[0]["id"], False)
    assert db.get_stats()["tasks_done"] == 0
    db.delete_study_plan(p2)
    assert len(db.list_study_plans()) == 1


def test_quiz_eval_activity_stats(db):
    db.save_quiz_result("Quiz", "topic", 5, 4, [{"q": 1}])
    db.save_evaluation("q", "a", 6.0, {"overall_score": 6.0})
    st = db.get_stats()
    assert st["quizzes"] == 1 and st["quiz_accuracy"] == 80 and st["evaluations"] == 1
    assert db.list_quiz_results()[0]["details"] == [{"q": 1}]
    assert len(db.recent_activity()) >= 2


def test_empty_stats(db):
    st = db.get_stats()
    assert st["documents"] == 0 and st["avg_interview_score"] is None and st["quiz_accuracy"] is None
