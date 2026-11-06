"""Prompt builders. Every prompt returns a chat `messages` list for the local LLM."""
from __future__ import annotations

NOT_FOUND_TOKEN = "NOT_FOUND_IN_DOCUMENTS"


def _sys(text: str) -> dict:
    return {"role": "system", "content": text}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def rag_messages(question: str, contexts: list[dict], history: list[dict]) -> list[dict]:
    ctx = "\n\n".join(f"[Source {i + 1}: {c['doc_name']}]\n{c['text']}" for i, c in enumerate(contexts))
    system = (
        "You answer questions using ONLY the provided document excerpts. "
        "If the excerpts do not contain the answer, reply with exactly " + NOT_FOUND_TOKEN + " and nothing else. "
        "Never invent facts that are not in the excerpts. Be concise and cite sources like [Source 1]."
    )
    msgs = [_sys(system)]
    for m in history[-6:]:
        msgs.append({"role": m["role"], "content": m["content"][:600]})
    msgs.append(_user(f"Document excerpts:\n{ctx}\n\nQuestion: {question}"))
    return msgs


def interview_questions_messages(job_title, level, domain, difficulty, category, count, profile="") -> list[dict]:
    guide = {
        "Technical": "technical questions that test knowledge and skills for the role",
        "HR": "HR / general questions about motivation, expectations, culture fit and career goals",
        "Behavioral": "behavioral questions answered with past experience (STAR method)",
        "Scenario": "realistic scenario-based questions describing a concrete work situation and asking what the candidate would do",
    }[category]
    prof = f"\nCandidate profile (from their documents):\n{profile[:1500]}\n" if profile else ""
    return [
        _sys("You are an expert interviewer. Reply with valid JSON only."),
        _user(
            f"Write exactly {count} {difficulty}-difficulty {guide}.\n"
            f"Role: {job_title}\nExperience level: {level}\nDomain: {domain}\n{prof}"
            "Each question must be specific, self-contained and different from the others. Keep each to one concise sentence; "
            "do not include answers or explanations.\n"
            'Return JSON: {"questions": ["question 1", "question 2"]}'),
    ]


def interview_evaluation_messages(question, answer, job_title, level, category, domain) -> list[dict]:
    return [
        _sys("You are a strict but fair interview coach. Judge ONLY what the candidate actually wrote. "
             "Confidence must be inferred from written communication only (assertiveness, hedging, structure). "
             "Reply with valid JSON only."),
        _user(
            f"Role: {job_title} ({level}), domain: {domain}\nQuestion type: {category}\n"
            f"Question: {question}\n\nCandidate answer:\n{answer}\n\n"
            "Score each criterion from 0 to 10 (integers). Return JSON with exactly these keys:\n"
            '{"scores": {"technical_accuracy": 0, "communication": 0, "clarity": 0, "relevance": 0, '
            '"problem_solving": 0, "confidence": 0}, "overall_score": 0, '
            '"strengths": ["..."], "weaknesses": ["..."], "suggestions": ["..."], '
            '"improved_answer": "a stronger example answer to this question"}'),
    ]


def answer_evaluation_messages(question, answer, context="") -> list[dict]:
    ctx = f"\nContext: {context}\n" if context else ""
    return [
        _sys("You are an expert technical interviewer evaluating a candidate's written answer. "
             "Judge ONLY what was written. Reply with valid JSON only."),
        _user(
            f"Question: {question}{ctx}\n\nAnswer to evaluate:\n{answer}\n\n"
            "Score each criterion from 0 to 10 (integers). Return JSON with exactly these keys:\n"
            '{"scores": {"accuracy": 0, "completeness": 0, "clarity": 0, "technical_depth": 0, "relevance": 0}, '
            '"overall_score": 0, "feedback": "2-3 sentence overall assessment", '
            '"strengths": ["..."], "weaknesses": ["..."], "suggestions": ["actionable step", "..."], '
            '"improved_answer": "a stronger version of the answer"}'),
    ]


def interview_summary_messages(job_title, items: list[dict]) -> list[dict]:
    lines = "\n".join(f"- [{i['category']}] score {i['score']}/10: {i['question'][:120]}" for i in items)
    return [
        _sys("You are an interview coach writing a short end-of-interview summary. Use only the data given."),
        _user(f"Role: {job_title}\nResults:\n{lines}\n\n"
              "Write a 3-4 sentence summary: overall performance, the strongest area, the weakest area, "
              "and the single most valuable next step. Plain text, no lists."),
    ]


def study_overview_messages(goal, skills, weak, hours, days, notes) -> list[dict]:
    extra = f"\nUser customisation request: {notes}" if notes else ""
    return [
        _sys("You are an expert career mentor and curriculum designer. Reply with valid JSON only."),
        _user(
            f"Career goal: {goal}\nCurrent skills: {skills or 'not specified'}\nWeak areas: {weak or 'not specified'}\n"
            f"Study time: {hours} hours/day for {days} days.{extra}\n\n"
            "Create a personalised study roadmap. Return JSON:\n"
            '{"summary": "2 sentences", "roadmap": [{"phase": "Phase name", "focus": "what and why", '
            '"topics": ["topic"]}], "projects": [{"title": "...", "description": "..."}], '
            '"quiz_topics": ["topic to be quizzed on"]}\n'
            "Use 3-4 roadmap phases, 2-3 practical projects and 3-5 quiz topics. Prioritise the weak areas."),
    ]


def study_tasks_messages(goal, weak, hours, overview: dict, start_day: int, end_day: int) -> list[dict]:
    phases = "; ".join(p.get("phase", "") + ": " + p.get("focus", "") for p in overview.get("roadmap", []))[:700]
    n = end_day - start_day + 1
    return [
        _sys("You are an expert career mentor. Reply with valid JSON only."),
        _user(
            f"Goal: {goal}\nWeak areas: {weak or 'not specified'}\nRoadmap: {phases}\n"
            f"Write one concrete learning task for each day from day {start_day} to day {end_day} "
            f"({n} tasks, about {hours} hours each). Tasks must progress logically and be actionable.\n"
            'Return JSON: {"tasks": [{"title": "short title", "description": "what exactly to do", "est_hours": 1.5}]}\n'
            f"The list must contain exactly {n} tasks in day order."),
    ]


def quiz_messages(count, difficulty, source_text=None, topic=None, avoid=None) -> list[dict]:
    if source_text:
        basis = f"Base every question strictly on this material:\n{source_text}\n"
    else:
        basis = f"Topic: {topic}\n"
    avoid_txt = ("\nDo not repeat these questions:\n- " + "\n- ".join(a[:100] for a in avoid)) if avoid else ""
    return [
        _sys("You write accurate multiple-choice quizzes. Reply with valid JSON only."),
        _user(
            f"{basis}Write exactly {count} {difficulty}-difficulty multiple-choice questions.{avoid_txt}\n"
            "Each has exactly 4 distinct options, exactly one correct, and a short explanation.\n"
            'Return JSON: {"questions": [{"question": "...", "options": ["A text", "B text", "C text", "D text"], '
            '"answer_index": 0, "explanation": "why it is correct"}]}\n'
            "answer_index is the 0-based index (0-3) of the correct option. Do not put letters like 'A)' in options."),
    ]
