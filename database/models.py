"""SQLite schema. All persistent state lives here (not in Streamlit session state)."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    stored_path TEXT,
    content_hash TEXT,
    text TEXT NOT NULL DEFAULT '',
    char_count INTEGER NOT NULL DEFAULT 0,
    num_chunks INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'extracted',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS document_metadata (
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY (doc_id, key)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    dim INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    doc_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS interview_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_title TEXT NOT NULL,
    experience_level TEXT NOT NULL,
    domain TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'mock',
    status TEXT NOT NULL DEFAULT 'in_progress',
    overall_score REAL,
    summary TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS interview_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES interview_sessions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    category TEXT NOT NULL,
    question TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS interview_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL UNIQUE REFERENCES interview_questions(id) ON DELETE CASCADE,
    answer TEXT NOT NULL,
    score REAL,
    feedback TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    overall_score REAL,
    feedback TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS study_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal TEXT NOT NULL,
    skills TEXT,
    weak_areas TEXT,
    hours_per_day REAL NOT NULL,
    days INTEGER NOT NULL,
    notes TEXT,
    plan_json TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS study_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL REFERENCES study_plans(id) ON DELETE CASCADE,
    day INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    est_hours REAL,
    completed INTEGER NOT NULL DEFAULT 0,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS quiz_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    source TEXT,
    total INTEGER NOT NULL,
    correct INTEGER NOT NULL,
    details TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

INTERVIEW_SESSION_UPDATABLE = {"mode", "status", "overall_score", "summary", "completed_at"}
