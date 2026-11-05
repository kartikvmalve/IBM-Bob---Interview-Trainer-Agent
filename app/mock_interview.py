"""
IBM Interview Trainer Agent — Mock Interview Session Engine
Manages stateful turn-by-turn interview sessions with adaptive questioning,
performance tracking, and comprehensive end-of-session reporting.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.granite_client import get_granite_client
from app.rag_pipeline import get_rag_pipeline

logger = logging.getLogger(__name__)


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionCategory(str, Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    HR = "hr"
    SITUATIONAL = "situational"
    SYSTEM_DESIGN = "system_design"


@dataclass
class Turn:
    """A single question-answer exchange in a mock interview."""
    turn_number: int
    question: str
    category: QuestionCategory
    difficulty: Difficulty
    user_answer: str = ""
    score: int = 0          # 1-10
    feedback: str = ""
    strengths: str = ""
    improvements: str = ""
    ideal_answer: str = ""
    time_taken_seconds: float = 0.0
    asked_at: float = field(default_factory=time.time)


@dataclass
class InterviewSession:
    """Full state for one mock interview session."""
    session_id: str
    candidate_name: str
    role: str
    experience_level: str
    total_questions: int
    resume_summary: str = ""
    turns: list[Turn] = field(default_factory=list)
    is_active: bool = True
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    final_report: dict[str, Any] = field(default_factory=dict)

    @property
    def current_turn_number(self) -> int:
        return len(self.turns) + 1

    @property
    def is_complete(self) -> bool:
        return len(self.turns) >= self.total_questions

    @property
    def answered_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.user_answer]

    @property
    def average_score(self) -> float:
        scored = [t for t in self.turns if t.score > 0]
        return round(sum(t.score for t in scored) / len(scored), 1) if scored else 0.0

    def current_difficulty(self) -> Difficulty:
        """Adaptive: adjust difficulty based on last 2 answers."""
        recent = self.answered_turns[-2:]
        if not recent:
            return Difficulty.EASY
        avg = sum(t.score for t in recent) / len(recent)
        if avg >= 7.5:
            return Difficulty.HARD
        if avg >= 5.0:
            return Difficulty.MEDIUM
        return Difficulty.EASY


# ── In-memory session store (replace with Redis for production) ───────────────
_sessions: dict[str, InterviewSession] = {}


def create_session(
    candidate_name: str,
    role: str,
    experience_level: str,
    total_questions: int = 8,
    resume_summary: str = "",
) -> InterviewSession:
    session = InterviewSession(
        session_id=uuid.uuid4().hex,
        candidate_name=candidate_name,
        role=role,
        experience_level=experience_level,
        total_questions=total_questions,
        resume_summary=resume_summary,
    )
    _sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> InterviewSession | None:
    return _sessions.get(session_id)


def end_session(session_id: str) -> InterviewSession | None:
    session = _sessions.get(session_id)
    if session:
        session.is_active = False
        session.ended_at = time.time()
    return session


# ── Prompt builders ───────────────────────────────────────────────────────────

_CATEGORY_CYCLE = [
    QuestionCategory.HR,
    QuestionCategory.TECHNICAL,
    QuestionCategory.BEHAVIORAL,
    QuestionCategory.TECHNICAL,
    QuestionCategory.SITUATIONAL,
    QuestionCategory.TECHNICAL,
    QuestionCategory.BEHAVIORAL,
    QuestionCategory.SYSTEM_DESIGN,
    QuestionCategory.TECHNICAL,
    QuestionCategory.HR,
]


def _pick_category(turn_number: int, total: int) -> QuestionCategory:
    idx = (turn_number - 1) % len(_CATEGORY_CYCLE)
    # Reserve last question for HR if near end
    if turn_number == total:
        return QuestionCategory.HR
    return _CATEGORY_CYCLE[idx]


def _build_question_generation_prompt(session: InterviewSession, category: QuestionCategory, difficulty: Difficulty) -> str:
    previous_questions = "\n".join(
        f"- Q{t.turn_number}: {t.question}" for t in session.turns
    )
    return f"""<|system|>
You are a professional interviewer conducting a real job interview. Ask exactly ONE interview question.
Do NOT provide answers, hints, or multiple questions. Just the question itself — clear, concise, professional.
<|user|>
Candidate: {session.candidate_name}
Role: {session.role}
Experience Level: {session.experience_level}
Background: {session.resume_summary or "Not provided"}

Interview progress: Question {session.current_turn_number} of {session.total_questions}
Category this question: {category.value}
Difficulty: {difficulty.value}

Questions already asked (do NOT repeat similar ones):
{previous_questions or "None yet."}

Generate ONE {difficulty.value} {category.value} interview question appropriate for a {session.experience_level} {session.role}.
Output only the question text, nothing else.
<|assistant|>
"""


def _build_evaluation_prompt(session: InterviewSession, turn: Turn) -> str:
    return f"""<|system|>
You are an expert interview coach evaluating a candidate's answer. Be specific, constructive, and encouraging.
<|user|>
Candidate: {session.candidate_name}
Role: {session.role}
Experience Level: {session.experience_level}
Question ({turn.category.value}, {turn.difficulty.value}): {turn.question}
Candidate's Answer: {turn.user_answer}

Evaluate this answer and respond in EXACTLY this format:

SCORE: [number 1-10]
STRENGTHS: [what was done well, 1-2 sentences]
IMPROVEMENTS: [what was missing or weak, 1-2 sentences]
IDEAL_ANSWER: [a model answer in 2-4 sentences]
FEEDBACK: [one encouraging coaching tip]
<|assistant|>
"""


def _build_final_report_prompt(session: InterviewSession) -> str:
    turns_summary = "\n".join(
        f"Q{t.turn_number} [{t.category.value}/{t.difficulty.value}]: Score {t.score}/10"
        for t in session.answered_turns
    )
    weak_areas = [t.question for t in session.answered_turns if t.score < 6]
    strong_areas = [t.question for t in session.answered_turns if t.score >= 8]

    return f"""<|system|>
You are a senior career coach delivering a post-interview performance report. Be honest, motivating, and specific.
<|user|>
Candidate: {session.candidate_name}
Role: {session.role}
Experience Level: {session.experience_level}
Overall Average Score: {session.average_score}/10

Question-by-Question Scores:
{turns_summary}

Strong areas (scored 8+): {len(strong_areas)} questions
Weak areas (scored below 6): {len(weak_areas)} questions

Write a concise performance report with these sections:
1. OVERALL_ASSESSMENT: 2-3 sentence summary of performance
2. TOP_STRENGTHS: 3 bullet points of what the candidate did well
3. KEY_IMPROVEMENTS: 3 bullet points of priority areas to work on
4. NEXT_STEPS: A 1-week action plan with 3 specific tasks
5. CONFIDENCE_MESSAGE: One motivating closing message personalized to {session.candidate_name}
<|assistant|>
"""


# ── Mock Interview Engine ─────────────────────────────────────────────────────

class MockInterviewEngine:
    """
    Manages the full lifecycle of a mock interview session:
    start → ask → evaluate → adapt → report
    """

    def __init__(self) -> None:
        self._llm = get_granite_client()
        self._rag = get_rag_pipeline()

    async def ask_next_question(self, session: InterviewSession) -> Turn:
        """Generate the next interview question and append it as a pending turn."""
        category = _pick_category(session.current_turn_number, session.total_questions)
        difficulty = session.current_difficulty()

        # Use RAG context to ground the question in real interview patterns
        query = f"{session.role} {category.value} {difficulty.value} interview question {session.experience_level}"
        context_docs = self._rag.retrieve(query, role=session.role.lower().replace(" ", "_"))
        rag_context = "\n".join(context_docs[:3])

        prompt = _build_question_generation_prompt(session, category, difficulty)
        # Inject top RAG context into the user section
        if rag_context:
            prompt = prompt.replace(
                "Generate ONE",
                f"Reference context (for inspiration only):\n{rag_context[:800]}\n\nGenerate ONE",
            )

        logger.info(
            "Session %s | Q%d | %s | %s",
            session.session_id[:8], session.current_turn_number, category.value, difficulty.value,
        )
        question_text = await self._llm.generate(prompt, max_new_tokens=150)
        # Strip any accidental multi-line or prefixed output
        question_text = question_text.strip().lstrip("Q1234567890.:) ").strip()

        turn = Turn(
            turn_number=session.current_turn_number,
            question=question_text,
            category=category,
            difficulty=difficulty,
        )
        session.turns.append(turn)
        return turn

    async def evaluate_answer(self, session: InterviewSession, user_answer: str, time_taken: float = 0.0) -> Turn:
        """Score the latest unanswered turn and return the evaluated turn."""
        # Find the last unanswered turn
        pending = next((t for t in reversed(session.turns) if not t.user_answer), None)
        if not pending:
            raise ValueError("No pending question to evaluate.")

        pending.user_answer = user_answer
        pending.time_taken_seconds = time_taken

        prompt = _build_evaluation_prompt(session, pending)
        raw = await self._llm.generate(prompt, max_new_tokens=400)

        # Parse structured response
        pending.score = _parse_field_int(raw, "SCORE", default=5)
        pending.strengths = _parse_field(raw, "STRENGTHS")
        pending.improvements = _parse_field(raw, "IMPROVEMENTS")
        pending.ideal_answer = _parse_field(raw, "IDEAL_ANSWER")
        pending.feedback = _parse_field(raw, "FEEDBACK")

        logger.info(
            "Session %s | Q%d evaluated | Score: %d/10",
            session.session_id[:8], pending.turn_number, pending.score,
        )
        return pending

    async def generate_final_report(self, session: InterviewSession) -> dict[str, Any]:
        """Generate comprehensive end-of-session performance report."""
        session.is_active = False
        session.ended_at = time.time()

        prompt = _build_final_report_prompt(session)
        raw = await self._llm.generate(prompt, max_new_tokens=600)

        duration_mins = round((session.ended_at - session.started_at) / 60, 1)

        # Score distribution
        scores = [t.score for t in session.answered_turns]
        score_dist = {
            "excellent": sum(1 for s in scores if s >= 8),
            "good": sum(1 for s in scores if 6 <= s < 8),
            "needs_work": sum(1 for s in scores if s < 6),
        }

        report = {
            "session_id": session.session_id,
            "candidate": session.candidate_name,
            "role": session.role,
            "experience_level": session.experience_level,
            "total_questions": len(session.answered_turns),
            "average_score": session.average_score,
            "score_distribution": score_dist,
            "duration_minutes": duration_mins,
            "performance_narrative": raw,
            "turns": [
                {
                    "turn": t.turn_number,
                    "category": t.category.value,
                    "difficulty": t.difficulty.value,
                    "question": t.question,
                    "answer": t.user_answer,
                    "score": t.score,
                    "feedback": t.feedback,
                    "strengths": t.strengths,
                    "improvements": t.improvements,
                    "ideal_answer": t.ideal_answer,
                    "time_taken_seconds": round(t.time_taken_seconds, 1),
                }
                for t in session.answered_turns
            ],
            "readiness_level": _readiness_label(session.average_score),
        }
        session.final_report = report
        return report


def _parse_field(text: str, field: str) -> str:
    """Extract a labelled field from the LLM structured response."""
    import re
    pattern = rf"{field}:\s*(.+?)(?=\n[A-Z_]+:|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_field_int(text: str, field: str, default: int = 5) -> int:
    import re
    m = re.search(rf"{field}:\s*(\d+)", text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        return max(1, min(10, val))
    return default


def _readiness_level(avg: float) -> str:
    if avg >= 8.0:
        return "Interview Ready"
    if avg >= 6.5:
        return "Almost Ready"
    if avg >= 5.0:
        return "Needs Practice"
    return "Significant Preparation Required"


# ── Singleton ─────────────────────────────────────────────────────────────────
_engine: MockInterviewEngine | None = None


def get_mock_engine() -> MockInterviewEngine:
    global _engine
    if _engine is None:
        _engine = MockInterviewEngine()
    return _engine
