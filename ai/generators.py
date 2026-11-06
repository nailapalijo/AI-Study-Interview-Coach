"""LLM-driven generators: interview questions, quizzes, study plans (pure logic, UI-independent)."""
from __future__ import annotations

import re

from ai import prompts

CATEGORIES = ("Technical", "Behavioral", "Scenario", "HR")
_ORDER = ["Technical", "Technical", "Behavioral", "Scenario", "HR"]


def split_counts(n: int) -> dict[str, int]:
    """Distribute n questions over the four categories (Technical-heavy)."""
    if n < 1:
        raise ValueError("Number of questions must be at least 1.")
    counts = {c: 0 for c in CATEGORIES}
    for i in range(n):
        counts[_ORDER[i % len(_ORDER)]] += 1
    return {c: k for c, k in counts.items() if k > 0}


def _clean_q(q) -> str:
    if isinstance(q, dict):
        q = q.get("question") or q.get("text") or ""
    q = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", str(q)).strip()
    return q


def generate_questions(engine, job_title, level, domain, n, difficulty, profile="") -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for category, count in split_counts(n).items():
        def validator(obj, count=count):
            items = obj.get("questions") if isinstance(obj, dict) else obj
            if not isinstance(items, list):
                raise ValueError("questions must be a list")
            qs = []
            for it in items:
                q = _clean_q(it)
                if len(q) >= 15 and q.lower() not in seen and q.lower() not in [x.lower() for x in qs]:
                    qs.append(q)
            if len(qs) < count:
                raise ValueError(f"expected {count} questions, got {len(qs)}")
            return qs[:count]

        msgs = prompts.interview_questions_messages(job_title, level, domain, difficulty, category, count, profile)
        for q in engine.generate_json(msgs, max_new_tokens=100 * count + 60, validator=validator):
            seen.add(q.lower())
            out.append({"category": category, "question": q})
    return out


# ------------------------------------------------------------------ quiz
def normalize_quiz_question(q) -> dict:
    if not isinstance(q, dict):
        raise ValueError("question is not an object")
    text = str(q.get("question", "")).strip()
    opts = q.get("options") or q.get("choices")
    if isinstance(opts, dict):
        opts = [opts[k] for k in sorted(opts)]
    if not text or not isinstance(opts, list):
        raise ValueError("missing question or options")
    options = [re.sub(r"^\s*[A-Da-d][.)]\s+", "", str(o)).strip() for o in opts]
    if len(options) != 4 or any(not o for o in options) or len({o.lower() for o in options}) != 4:
        raise ValueError("need exactly 4 distinct options")
    key = next((k for k in ("answer_index", "correct_index", "answer", "correct_answer") if k in q), None)
    ans = q.get(key) if key else None
    idx = None
    if isinstance(ans, bool):
        idx = None
    elif isinstance(ans, int):
        idx = ans
    elif isinstance(ans, str):
        a = ans.strip()
        text_match = re.sub(r"^\s*[A-Da-d][.)]\s+", "", a).lower()
        by_text = next((i for i, o in enumerate(options) if o.lower() == text_match or o.lower() == a.lower()), None)
        if a.isdigit() and key in ("answer_index", "correct_index"):
            idx = int(a)
        elif re.fullmatch(r"[A-Da-d][.)]?", a) and by_text is None:
            idx = "abcd".index(a[0].lower())
        elif by_text is not None:
            idx = by_text
        elif a.isdigit():
            idx = int(a)
    if idx is None or not 0 <= idx <= 3:
        raise ValueError("invalid correct answer index")
    explanation = str(q.get("explanation", "")).strip()
    if not explanation:
        raise ValueError("missing explanation")
    return {"question": text, "options": options, "answer_index": idx, "explanation": explanation}


def generate_quiz(engine, n: int, difficulty: str, source_text: str | None = None, topic: str | None = None) -> list[dict]:
    if not source_text and not topic:
        raise ValueError("Provide document context or a topic.")
    questions: list[dict] = []
    guard = 0
    while len(questions) < n and guard < 6:
        guard += 1
        batch = min(3, n - len(questions))
        taken = [q["question"].lower() for q in questions]

        def validator(obj, batch=batch, taken=taken):
            items = obj.get("questions") if isinstance(obj, dict) else obj
            if not isinstance(items, list):
                raise ValueError("questions must be a list")
            good = []
            for it in items:
                try:
                    q = normalize_quiz_question(it)
                except ValueError:
                    continue
                if q["question"].lower() not in taken and q["question"].lower() not in [g["question"].lower() for g in good]:
                    good.append(q)
            if not good:
                raise ValueError("no valid questions")
            return good[:batch]

        msgs = prompts.quiz_messages(batch, difficulty, source_text, topic, [q["question"] for q in questions])
        questions.extend(engine.generate_json(msgs, max_new_tokens=200 * batch + 80, validator=validator))
    return questions[:n]


# ------------------------------------------------------------------ study plan
def _normalize_overview(obj) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("overview is not an object")
    roadmap = []
    for p in obj.get("roadmap") or []:
        if isinstance(p, dict) and (p.get("phase") or p.get("focus")):
            roadmap.append({"phase": str(p.get("phase", "")).strip() or "Phase",
                            "focus": str(p.get("focus", "")).strip(),
                            "topics": [str(t).strip() for t in (p.get("topics") or []) if str(t).strip()]})
    if not roadmap:
        raise ValueError("roadmap is empty")
    projects = [{"title": str(p.get("title", "")).strip(), "description": str(p.get("description", "")).strip()}
                for p in (obj.get("projects") or []) if isinstance(p, dict) and str(p.get("title", "")).strip()]
    topics = [str(t).strip() for t in (obj.get("quiz_topics") or []) if str(t).strip()]
    return {"summary": str(obj.get("summary", "")).strip(), "roadmap": roadmap, "projects": projects, "quiz_topics": topics}


def _normalize_tasks(obj, expected: int, start_day: int, hours: float) -> list[dict]:
    items = obj.get("tasks") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise ValueError("tasks must be a list")
    valid = [t for t in items if isinstance(t, dict) and str(t.get("title", "")).strip()]
    if len(valid) < expected:
        raise ValueError(f"expected {expected} tasks, got {len(valid)}")
    tasks = []
    for i, t in enumerate(valid[:expected]):
        try:
            est = float(t.get("est_hours"))
            est = est if 0 < est <= 12 else hours
        except (TypeError, ValueError):
            est = hours
        tasks.append({"day": start_day + i, "title": str(t["title"]).strip(),
                      "description": str(t.get("description", "")).strip(), "est_hours": est})
    return tasks


def generate_study_plan(engine, goal, skills, weak_areas, hours_per_day, days, notes="") -> tuple[dict, list[dict]]:
    overview = engine.generate_json(
        prompts.study_overview_messages(goal, skills, weak_areas, hours_per_day, days, notes),
        max_new_tokens=700, validator=_normalize_overview)
    tasks: list[dict] = []
    for start in range(1, days + 1, 7):
        end = min(start + 6, days)
        expected = end - start + 1
        msgs = prompts.study_tasks_messages(goal, weak_areas, hours_per_day, overview, start, end)
        tasks.extend(engine.generate_json(
            msgs, max_new_tokens=90 * expected + 60,
            validator=lambda o, e=expected, s=start: _normalize_tasks(o, e, s, hours_per_day)))
    return overview, tasks
