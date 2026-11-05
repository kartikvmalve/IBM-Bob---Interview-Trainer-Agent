"""
IBM Interview Trainer Agent — Core Agent Logic
Orchestrates RAG retrieval + IBM Granite LLM generation to produce
tailored interview question sets, model answers, and coaching tips.
"""
from __future__ import annotations

import logging
from typing import Literal

from app.granite_client import get_granite_client
from app.rag_pipeline import get_rag_pipeline

logger = logging.getLogger(__name__)

ExperienceLevel = Literal["fresher", "junior", "mid", "senior", "lead"]

ROLE_NORMALIZER: dict[str, str] = {
    "software engineer": "software_engineer",
    "swe": "software_engineer",
    "developer": "software_engineer",
    "backend developer": "software_engineer",
    "frontend developer": "software_engineer",
    "fullstack developer": "software_engineer",
    "data scientist": "data_scientist",
    "ml engineer": "data_scientist",
    "machine learning engineer": "data_scientist",
    "product manager": "product_manager",
    "pm": "product_manager",
    "devops": "devops_engineer",
    "devops engineer": "devops_engineer",
    "sre": "devops_engineer",
    "cloud engineer": "devops_engineer",
}


def normalize_role(role: str) -> str:
    return ROLE_NORMALIZER.get(role.lower().strip(), "general")


# ── Prompt Templates ─────────────────────────────────────────────────────────

def _build_question_prompt(
    name: str,
    role: str,
    experience_level: str,
    resume_summary: str,
    context_docs: list[str],
    num_questions: int,
) -> str:
    context_block = "\n\n---\n\n".join(context_docs) if context_docs else "No additional context."
    return f"""<|system|>
You are an expert interview coach specializing in technical and behavioral interview preparation.
Your task is to generate a personalized, comprehensive interview question set with model answers and tips.
Always be encouraging, specific, and actionable.
<|user|>
## Candidate Profile
- Name: {name}
- Target Role: {role}
- Experience Level: {experience_level}
- Resume / Background Summary: {resume_summary or "Not provided."}

## Retrieved Interview Knowledge
{context_block}

## Your Task
Generate exactly {num_questions} interview questions tailored to this candidate.
For each question:
1. State the **Question** clearly.
2. Provide a **Model Answer** (2-4 sentences, using STAR where applicable).
3. Give a **Coaching Tip** specific to this candidate's experience level.

Cover a mix of: technical depth, behavioral/soft skills, role-specific scenarios, and at least one HR question.

Format each question block as:
### Q{{number}}: {{Question title}}
**Question:** {{full question text}}
**Model Answer:** {{answer}}
**Coaching Tip:** {{tip}}

<|assistant|>
"""


def _build_strategy_prompt(
    name: str,
    role: str,
    experience_level: str,
    resume_summary: str,
    context_docs: list[str],
) -> str:
    context_block = "\n\n---\n\n".join(context_docs) if context_docs else "No additional context."
    return f"""<|system|>
You are a senior career coach at a top recruitment consultancy.
Provide a concise, highly actionable interview preparation strategy.
<|user|>
## Candidate Profile
- Name: {name}
- Target Role: {role}
- Experience Level: {experience_level}
- Resume / Background: {resume_summary or "Not provided."}

## Industry Knowledge Context
{context_block}

## Your Task
Write a personalized **30-Day Interview Preparation Strategy** for {name} targeting the {role} role.
Include:
1. **Week 1 — Foundation**: Core technical topics to review, resources (books, courses, practice sites).
2. **Week 2 — Practice**: Mock interview schedule, key behavioral stories to prepare (STAR format).
3. **Week 3 — Refinement**: System design / case study practice, improving weak areas identified.
4. **Week 4 — Final Polish**: Company research, question bank review, confidence-building exercises.
5. **Day-of Checklist**: 5 things to do on interview day.
6. **Key Metrics**: How to measure readiness.

Be specific to this candidate's experience level ({experience_level}) and role ({role}).
<|assistant|>
"""


def _build_evaluation_prompt(
    question: str,
    user_answer: str,
    role: str,
    experience_level: str,
) -> str:
    return f"""<|system|>
You are an expert interview evaluator. Score and improve candidate answers with precision.
<|user|>
## Context
- Role: {role}
- Experience Level: {experience_level}

## Interview Question
{question}

## Candidate's Answer
{user_answer}

## Your Task
Evaluate the answer and provide:
1. **Score**: Rate out of 10 with justification.
2. **Strengths**: What was done well (be specific).
3. **Gaps**: What was missing or weak.
4. **Improved Answer**: Rewrite the answer to a 9/10 standard.
5. **One Actionable Tip**: The single most impactful thing to practice.
<|assistant|>
"""


# ── Agent Class ───────────────────────────────────────────────────────────────

class InterviewTrainerAgent:
    """
    RAG-powered interview trainer.
    Uses IBM Granite for generation and ChromaDB for retrieval.
    """

    def __init__(self) -> None:
        self._llm = get_granite_client()
        self._rag = get_rag_pipeline()

    async def generate_questions(
        self,
        name: str,
        role: str,
        experience_level: str,
        resume_text: str = "",
        num_questions: int = 10,
    ) -> dict:
        """Generate a tailored question set with model answers."""
        normalized_role = normalize_role(role)
        query = f"{role} interview questions {experience_level} technical behavioral"
        context_docs = self._rag.retrieve(query, role=normalized_role)

        # If resume provided, also retrieve based on resume keywords
        if resume_text:
            resume_query = resume_text[:500]  # first 500 chars as query
            resume_docs = self._rag.retrieve(resume_query)
            context_docs = list(dict.fromkeys(context_docs + resume_docs))[:6]

        prompt = _build_question_prompt(
            name=name,
            role=role,
            experience_level=experience_level,
            resume_summary=resume_text[:800] if resume_text else "",
            context_docs=context_docs,
            num_questions=num_questions,
        )

        logger.info("Generating %d questions for %s (%s, %s)…", num_questions, name, role, experience_level)
        generated = await self._llm.generate(prompt, max_new_tokens=2048)

        return {
            "candidate": {"name": name, "role": role, "experience_level": experience_level},
            "questions_and_answers": generated,
            "context_sources_count": len(context_docs),
            "model_used": self._llm.settings.ibm_granite_model_id,
        }

    async def generate_preparation_strategy(
        self,
        name: str,
        role: str,
        experience_level: str,
        resume_text: str = "",
    ) -> dict:
        """Generate a 30-day preparation strategy."""
        normalized_role = normalize_role(role)
        query = f"{role} {experience_level} interview preparation strategy tips"
        context_docs = self._rag.retrieve(query, role=normalized_role)

        prompt = _build_strategy_prompt(
            name=name,
            role=role,
            experience_level=experience_level,
            resume_summary=resume_text[:800] if resume_text else "",
            context_docs=context_docs,
        )

        logger.info("Generating prep strategy for %s…", name)
        generated = await self._llm.generate(prompt, max_new_tokens=2048)

        return {
            "candidate": {"name": name, "role": role, "experience_level": experience_level},
            "preparation_strategy": generated,
            "model_used": self._llm.settings.ibm_granite_model_id,
        }

    async def evaluate_answer(
        self,
        question: str,
        user_answer: str,
        role: str,
        experience_level: str,
    ) -> dict:
        """Score and improve a candidate's answer to an interview question."""
        prompt = _build_evaluation_prompt(
            question=question,
            user_answer=user_answer,
            role=role,
            experience_level=experience_level,
        )

        logger.info("Evaluating answer for role=%s, level=%s", role, experience_level)
        generated = await self._llm.generate(prompt, max_new_tokens=1024)

        return {
            "question": question,
            "user_answer": user_answer,
            "evaluation": generated,
            "model_used": self._llm.settings.ibm_granite_model_id,
        }

    async def ingest_resume(self, resume_text: str, metadata: dict | None = None) -> str:
        """Index a resume into the vector store for personalized retrieval."""
        return self._rag.ingest_text(resume_text, metadata=metadata or {"source": "resume"})


# ── Singleton ─────────────────────────────────────────────────────────────────
_agent: InterviewTrainerAgent | None = None


def get_agent() -> InterviewTrainerAgent:
    global _agent
    if _agent is None:
        _agent = InterviewTrainerAgent()
    return _agent
