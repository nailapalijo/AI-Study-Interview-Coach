"""SQLite access layer."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from database.models import INTERVIEW_SESSION_UPDATABLE, SCHEMA_SQL

logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    pass


def _loads(value, default):
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    # ------------------------------------------------------------ plumbing
    @contextmanager
    def _connect(self):
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            except sqlite3.Error as exc:
                conn.rollback()
                logger.exception("Database error")
                raise DatabaseError(f"Database error: {exc}") from exc
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> int:
        with self._connect() as conn:
            return conn.execute(sql, params).lastrowid

    def _all(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple = ()) -> dict | None:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------ activity
    def log_activity(self, kind: str, detail: str) -> None:
        try:
            self._execute("INSERT INTO activity_log(kind, detail) VALUES (?, ?)", (kind, detail[:300]))
        except DatabaseError:
            pass

    def recent_activity(self, limit: int = 8) -> list[dict]:
        return self._all("SELECT * FROM activity_log ORDER BY id DESC LIMIT ?", (limit,))

    # ------------------------------------------------------------ documents
    def add_document(self, name, file_type, size_bytes, stored_path, text, content_hash=None, metadata=None) -> int:
        with self._connect() as conn:
            doc_id = conn.execute(
                "INSERT INTO documents(name, file_type, size_bytes, stored_path, content_hash, text, char_count) "
                "VALUES (?,?,?,?,?,?,?)",
                (name, file_type, size_bytes, stored_path, content_hash, text, len(text)),
            ).lastrowid
            for key, value in (metadata or {}).items():
                conn.execute("INSERT OR REPLACE INTO document_metadata(doc_id,key,value) VALUES (?,?,?)",
                             (doc_id, key, str(value)))
        self.log_activity("document", f"Uploaded {name}")
        return doc_id

    def find_document_by_hash(self, content_hash: str) -> dict | None:
        return self._one("SELECT id, name FROM documents WHERE content_hash = ?", (content_hash,))

    def list_documents(self) -> list[dict]:
        return self._all("SELECT id, name, file_type, size_bytes, char_count, num_chunks, status, created_at "
                         "FROM documents ORDER BY id DESC")

    def get_document(self, doc_id: int) -> dict | None:
        doc = self._one("SELECT * FROM documents WHERE id = ?", (doc_id,))
        if doc:
            rows = self._all("SELECT key, value FROM document_metadata WHERE doc_id = ?", (doc_id,))
            doc["metadata"] = {r["key"]: r["value"] for r in rows}
        return doc

    def set_document_status(self, doc_id: int, status: str, num_chunks: int | None = None) -> None:
        if num_chunks is None:
            self._execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))
        else:
            self._execute("UPDATE documents SET status = ?, num_chunks = ? WHERE id = ?", (status, num_chunks, doc_id))

    def delete_document(self, doc_id: int) -> None:
        doc = self._one("SELECT name, stored_path FROM documents WHERE id = ?", (doc_id,))
        if not doc:
            return
        self._execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        if doc["stored_path"]:
            try:
                Path(doc["stored_path"]).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove file %s", doc["stored_path"])
        self.log_activity("document", f"Deleted {doc['name']}")

    # ------------------------------------------------------------ chunks
    def replace_chunks(self, doc_id: int, chunks: list[str], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise DatabaseError("chunks and embeddings length mismatch")
        emb = np.asarray(embeddings, dtype=np.float32)
        dim = int(emb.shape[1]) if emb.ndim == 2 and len(emb) else 0
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                "INSERT INTO chunks(doc_id, chunk_index, text, embedding, dim) VALUES (?,?,?,?,?)",
                [(doc_id, i, c, emb[i].tobytes(), dim) for i, c in enumerate(chunks)],
            )
            conn.execute("UPDATE documents SET status='indexed', num_chunks=? WHERE id=?", (len(chunks), doc_id))

    def get_chunks(self, doc_ids: list[int] | None = None, with_embeddings: bool = True) -> list[dict]:
        cols = "c.id, c.doc_id, c.chunk_index, c.text, d.name AS doc_name" + (", c.embedding, c.dim" if with_embeddings else "")
        sql = f"SELECT {cols} FROM chunks c JOIN documents d ON d.id = c.doc_id"
        params: tuple = ()
        if doc_ids is not None:
            if not doc_ids:
                return []
            sql += " WHERE c.doc_id IN (%s)" % ",".join("?" * len(doc_ids))
            params = tuple(doc_ids)
        sql += " ORDER BY c.doc_id, c.chunk_index"
        rows = self._all(sql, params)
        if with_embeddings:
            for r in rows:
                blob = r.pop("embedding")
                r["vector"] = np.frombuffer(blob, dtype=np.float32) if blob else None
        return rows

    # ------------------------------------------------------------ chat
    def create_chat_session(self, title: str, doc_ids: list[int]) -> int:
        return self._execute("INSERT INTO chat_sessions(title, doc_ids) VALUES (?,?)", (title[:80], json.dumps(doc_ids)))

    def list_chat_sessions(self) -> list[dict]:
        rows = self._all("SELECT * FROM chat_sessions ORDER BY id DESC")
        for r in rows:
            r["doc_ids"] = _loads(r["doc_ids"], [])
        return rows

    def get_chat_session(self, session_id: int) -> dict | None:
        row = self._one("SELECT * FROM chat_sessions WHERE id = ?", (session_id,))
        if row:
            row["doc_ids"] = _loads(row["doc_ids"], [])
        return row

    def update_chat_docs(self, session_id: int, doc_ids: list[int]) -> None:
        self._execute("UPDATE chat_sessions SET doc_ids = ? WHERE id = ?", (json.dumps(doc_ids), session_id))

    def add_chat_message(self, session_id: int, role: str, content: str, sources: list | None = None) -> int:
        return self._execute("INSERT INTO chat_messages(session_id, role, content, sources) VALUES (?,?,?,?)",
                             (session_id, role, content, json.dumps(sources or [])))

    def get_chat_messages(self, session_id: int) -> list[dict]:
        rows = self._all("SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id", (session_id,))
        for r in rows:
            r["sources"] = _loads(r["sources"], [])
        return rows

    def clear_chat_messages(self, session_id: int) -> None:
        self._execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))

    def delete_chat_session(self, session_id: int) -> None:
        self._execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))

    # ------------------------------------------------------------ interviews
    def create_interview_session(self, job_title, level, domain, difficulty, questions: list[dict],
                                 mode: str = "mock", status: str = "in_progress") -> int:
        with self._connect() as conn:
            sid = conn.execute(
                "INSERT INTO interview_sessions(job_title, experience_level, domain, difficulty, mode, status) "
                "VALUES (?,?,?,?,?,?)", (job_title, level, domain, difficulty, mode, status)).lastrowid
            conn.executemany(
                "INSERT INTO interview_questions(session_id, position, category, question) VALUES (?,?,?,?)",
                [(sid, i, q["category"], q["question"]) for i, q in enumerate(questions)])
        self.log_activity("interview", f"{'Generated' if mode == 'generated' else 'Started'} interview: {job_title}")
        return sid

    def update_interview_session(self, session_id: int, **fields) -> None:
        bad = set(fields) - INTERVIEW_SESSION_UPDATABLE
        if bad or not fields:
            raise DatabaseError(f"Invalid interview session fields: {sorted(bad)}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._execute(f"UPDATE interview_sessions SET {sets} WHERE id = ?", (*fields.values(), session_id))

    def get_interview_session(self, session_id: int) -> dict | None:
        row = self._one("SELECT * FROM interview_sessions WHERE id = ?", (session_id,))
        if row:
            row["summary"] = _loads(row["summary"], None)
        return row

    def list_interview_sessions(self, modes: tuple[str, ...] | None = None) -> list[dict]:
        sql, params = "SELECT * FROM interview_sessions", ()
        if modes:
            sql += " WHERE mode IN (%s)" % ",".join("?" * len(modes))
            params = tuple(modes)
        rows = self._all(sql + " ORDER BY id DESC", params)
        for r in rows:
            r["summary"] = _loads(r["summary"], None)
        return rows

    def get_interview_questions(self, session_id: int) -> list[dict]:
        rows = self._all(
            "SELECT q.id, q.position, q.category, q.question, a.id AS answer_id, a.answer, a.score, a.feedback "
            "FROM interview_questions q LEFT JOIN interview_answers a ON a.question_id = q.id "
            "WHERE q.session_id = ? ORDER BY q.position", (session_id,))
        for r in rows:
            r["feedback"] = _loads(r["feedback"], None)
        return rows

    def save_interview_answer(self, question_id: int, answer: str, score: float | None, feedback: dict) -> None:
        self._execute(
            "INSERT OR REPLACE INTO interview_answers(question_id, answer, score, feedback) VALUES (?,?,?,?)",
            (question_id, answer, score, json.dumps(feedback)))

    def complete_interview_session(self, session_id: int, overall: float | None, summary: dict) -> None:
        self._execute(
            "UPDATE interview_sessions SET status='completed', overall_score=?, summary=?, completed_at=datetime('now') "
            "WHERE id=?", (overall, json.dumps(summary), session_id))
        s = self._one("SELECT job_title FROM interview_sessions WHERE id=?", (session_id,))
        self.log_activity("interview", f"Completed interview ({s['job_title'] if s else ''}) score {overall}")

    def delete_interview_session(self, session_id: int) -> None:
        self._execute("DELETE FROM interview_sessions WHERE id = ?", (session_id,))

    # ------------------------------------------------------------ evaluations
    def save_evaluation(self, question: str, answer: str, overall: float | None, feedback: dict) -> int:
        eid = self._execute("INSERT INTO evaluations(question, answer, overall_score, feedback) VALUES (?,?,?,?)",
                            (question, answer, overall, json.dumps(feedback)))
        self.log_activity("evaluation", f"Evaluated an answer ({overall}/10)")
        return eid

    def list_evaluations(self, limit: int = 20) -> list[dict]:
        rows = self._all("SELECT * FROM evaluations ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            r["feedback"] = _loads(r["feedback"], {})
        return rows

    # ------------------------------------------------------------ study plans
    def create_study_plan(self, goal, skills, weak_areas, hours_per_day, days, notes, plan: dict, tasks: list[dict]) -> int:
        with self._connect() as conn:
            conn.execute("UPDATE study_plans SET is_active = 0")
            pid = conn.execute(
                "INSERT INTO study_plans(goal, skills, weak_areas, hours_per_day, days, notes, plan_json) "
                "VALUES (?,?,?,?,?,?,?)",
                (goal, skills, weak_areas, hours_per_day, days, notes, json.dumps(plan))).lastrowid
            conn.executemany(
                "INSERT INTO study_tasks(plan_id, day, title, description, est_hours) VALUES (?,?,?,?,?)",
                [(pid, t["day"], t["title"], t.get("description", ""), t.get("est_hours")) for t in tasks])
        self.log_activity("study", f"Created study plan: {goal}")
        return pid

    def list_study_plans(self) -> list[dict]:
        rows = self._all("SELECT * FROM study_plans ORDER BY id DESC")
        for r in rows:
            r["plan"] = _loads(r.pop("plan_json"), {})
        return rows

    def activate_study_plan(self, plan_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE study_plans SET is_active = 0")
            conn.execute("UPDATE study_plans SET is_active = 1 WHERE id = ?", (plan_id,))

    def get_study_tasks(self, plan_id: int) -> list[dict]:
        return self._all("SELECT * FROM study_tasks WHERE plan_id = ? ORDER BY day, id", (plan_id,))

    def set_task_completed(self, task_id: int, completed: bool) -> None:
        self._execute("UPDATE study_tasks SET completed = ?, completed_at = CASE WHEN ? THEN datetime('now') ELSE NULL END "
                      "WHERE id = ?", (1 if completed else 0, 1 if completed else 0, task_id))
        if completed:
            t = self._one("SELECT title FROM study_tasks WHERE id = ?", (task_id,))
            self.log_activity("study", f"Completed task: {t['title'] if t else task_id}")

    def delete_study_plan(self, plan_id: int) -> None:
        self._execute("DELETE FROM study_plans WHERE id = ?", (plan_id,))

    # ------------------------------------------------------------ quizzes
    def save_quiz_result(self, title: str, source: str, total: int, correct: int, details: list[dict]) -> int:
        qid = self._execute("INSERT INTO quiz_results(title, source, total, correct, details) VALUES (?,?,?,?,?)",
                            (title[:120], source, total, correct, json.dumps(details)))
        self.log_activity("quiz", f"Quiz '{title[:40]}': {correct}/{total}")
        return qid

    def list_quiz_results(self, limit: int = 50) -> list[dict]:
        rows = self._all("SELECT * FROM quiz_results ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            r["details"] = _loads(r["details"], [])
        return rows

    # ------------------------------------------------------------ stats
    def get_stats(self) -> dict:
        def n(sql):
            row = self._one(sql)
            return list(row.values())[0] or 0 if row else 0
        tasks_total = n("SELECT COUNT(*) FROM study_tasks t JOIN study_plans p ON p.id=t.plan_id WHERE p.is_active=1")
        tasks_done = n("SELECT COUNT(*) FROM study_tasks t JOIN study_plans p ON p.id=t.plan_id "
                       "WHERE p.is_active=1 AND t.completed=1")
        quiz = self._one("SELECT SUM(total) t, SUM(correct) c, COUNT(*) n FROM quiz_results") or {}
        return {
            "documents": n("SELECT COUNT(*) FROM documents"),
            "indexed_chunks": n("SELECT COUNT(*) FROM chunks"),
            "interviews_total": n("SELECT COUNT(*) FROM interview_sessions WHERE mode='mock'"),
            "interviews_completed": n("SELECT COUNT(*) FROM interview_sessions WHERE mode='mock' AND status='completed'"),
            "avg_interview_score": self._one("SELECT AVG(overall_score) a FROM interview_sessions WHERE status='completed'")["a"],
            "evaluations": n("SELECT COUNT(*) FROM evaluations"),
            "tasks_total": tasks_total,
            "tasks_done": tasks_done,
            "quizzes": quiz.get("n") or 0,
            "quiz_accuracy": (quiz["c"] / quiz["t"] * 100) if quiz.get("t") else None,
            "chat_messages": n("SELECT COUNT(*) FROM chat_messages WHERE role='user'"),
        }

    def interview_score_history(self) -> list[dict]:
        return self._all("SELECT id, job_title, overall_score, completed_at FROM interview_sessions "
                         "WHERE status='completed' AND overall_score IS NOT NULL ORDER BY completed_at")

    def quiz_history(self) -> list[dict]:
        return self._all("SELECT id, title, total, correct, created_at FROM quiz_results ORDER BY created_at")

    def tasks_completed_by_day(self) -> list[dict]:
        return self._all("SELECT date(completed_at) AS day, COUNT(*) AS completed FROM study_tasks "
                         "WHERE completed=1 AND completed_at IS NOT NULL GROUP BY date(completed_at) ORDER BY day")
