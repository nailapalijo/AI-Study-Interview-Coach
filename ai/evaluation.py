"""Parsing/validation of LLM evaluations and interview score aggregation (pure Python, unit-testable)."""
from __future__ import annotations

import math
import re
from statistics import mean

from ai import prompts

INTERVIEW_CRITERIA = ["technical_accuracy", "communication", "clarity", "relevance", "problem_solving", "confidence"]
ANSWER_CRITERIA = ["accuracy", "completeness", "clarity", "technical_depth", "relevance"]
CRITERIA_LABELS = {
    "technical_accuracy": "Technical accuracy", "communication": "Communication", "clarity": "Clarity",
    "relevance": "Relevance", "problem_solving": "Problem solving", "confidence": "Confidence (written cues)",
    "accuracy": "Accuracy", "completeness": "Completeness", "technical_depth": "Technical depth",
}


class EvaluationError(ValueError):
    pass


def _num(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        return float(m.group()) if m else None
    return None


def _clamp(x: float) -> float:
    return max(0.0, min(10.0, x))


def _str_list(value) -> list[str]:
    if isinstance(value, str):
        value = [re.sub(r"^[\s\-*•\d.)]+", "", ln) for ln in value.splitlines()]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def normalize_evaluation(raw, criteria: list[str]) -> dict:
    """Validate and clamp an LLM evaluation. Raises EvaluationError if it is unusable (no fabricated defaults)."""
    if not isinstance(raw, dict):
        raise EvaluationError("Evaluation is not a JSON object.")
    src = raw["scores"] if isinstance(raw.get("scores"), dict) else raw
    scores = {}
    for c in criteria:
        n = _num(src.get(c))
        if n is not None:
            scores[c] = round(_clamp(n), 1)
    if len(scores) < math.ceil(len(criteria) * 0.6):
        raise EvaluationError("Evaluation is missing most criterion scores.")
    overall = _num(raw.get("overall_score", raw.get("score")))
    overall = round(_clamp(overall), 1) if overall is not None else round(mean(scores.values()), 1)
    result = {
        "scores": scores,
        "overall_score": overall,
        "feedback": str(raw.get("feedback") or raw.get("summary") or "").strip(),
        "strengths": _str_list(raw.get("strengths")),
        "weaknesses": _str_list(raw.get("weaknesses")),
        "suggestions": _str_list(raw.get("suggestions") or raw.get("improvements")),
        "improved_answer": str(raw.get("improved_answer") or "").strip(),
    }
    if not (result["feedback"] or result["strengths"] or result["weaknesses"] or result["suggestions"]):
        raise EvaluationError("Evaluation contains no written feedback.")
    return result


def evaluate_interview_answer(engine, question, answer, job_title, level, category, domain) -> dict:
    msgs = prompts.interview_evaluation_messages(question, answer, job_title, level, category, domain)
    return engine.generate_json(msgs, max_new_tokens=700, validator=lambda o: normalize_evaluation(o, INTERVIEW_CRITERIA))


def evaluate_user_answer(engine, question, answer, context="") -> dict:
    msgs = prompts.answer_evaluation_messages(question, answer, context)
    return engine.generate_json(msgs, max_new_tokens=700, validator=lambda o: normalize_evaluation(o, ANSWER_CRITERIA))


def summarize_interview(items: list[dict]) -> dict:
    """Aggregate scored answers. items: {question, category, score, feedback{scores}}"""
    scored = [i for i in items if i.get("score") is not None]
    if not scored:
        return {"count": len(items), "overall": None, "criteria_avg": {}, "by_category": {}, "best": None, "worst": None}
    per_criterion: dict[str, list[float]] = {}
    per_cat: dict[str, list[float]] = {}
    for i in scored:
        per_cat.setdefault(i["category"], []).append(i["score"])
        for k, v in ((i.get("feedback") or {}).get("scores") or {}).items():
            per_criterion.setdefault(k, []).append(v)
    best = max(scored, key=lambda i: i["score"])
    worst = min(scored, key=lambda i: i["score"])
    return {
        "count": len(items),
        "overall": round(mean(i["score"] for i in scored), 1),
        "criteria_avg": {k: round(mean(v), 1) for k, v in per_criterion.items()},
        "by_category": {k: round(mean(v), 1) for k, v in per_cat.items()},
        "best": {"question": best["question"], "score": best["score"]},
        "worst": {"question": worst["question"], "score": worst["score"]},
    }


def narrative_summary(engine, job_title: str, items: list[dict]) -> str:
    return engine.generate(prompts.interview_summary_messages(job_title, items), max_new_tokens=220, temperature=0.3)
