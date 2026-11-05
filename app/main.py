"""
IBM Interview Trainer Agent — FastAPI Application
Full feature set: Question Generator, Strategy, Answer Evaluator,
Mock Interview (turn-by-turn), Resume Parser, Upload.
"""
from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import get_agent
from app.config import get_settings
from app.granite_client import get_granite_client
from app.mock_interview import (
    create_session, end_session, get_mock_engine, get_session,
)
from app.rag_pipeline import get_rag_pipeline
from app.resume_parser import get_resume_parser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting IBM Interview Trainer Agent...")
    get_rag_pipeline()
    logger.info("RAG pipeline ready.")
    yield
    logger.info("Shutting down...")
    await get_granite_client().close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IBM Interview Trainer Agent",
    description="Practice Smarter. Answer Better. Get Interview Ready with AI.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

import os
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response Models ─────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    role: str = Field(..., min_length=1, max_length=100)
    experience_level: str
    resume_text: str = Field(default="")
    num_questions: int = Field(default=10, ge=3, le=20)

class StrategyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    role: str = Field(..., min_length=1, max_length=100)
    experience_level: str
    resume_text: str = Field(default="")

class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=5)
    user_answer: str = Field(..., min_length=5)
    role: str = Field(..., min_length=1, max_length=100)
    experience_level: str = Field(default="mid")

class MockStartRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    role: str = Field(..., min_length=1, max_length=100)
    experience_level: str
    total_questions: int = Field(default=8, ge=3, le=15)
    resume_text: str = Field(default="")

class MockAnswerRequest(BaseModel):
    session_id: str
    answer: str = Field(..., min_length=1)
    time_taken_seconds: float = Field(default=0.0)


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_ui():
    path = os.path.join("static", "index.html")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>IBM Interview Trainer Agent v2</h1><p>Use /docs</p>")


# ── System ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    cfg = get_settings()
    return {
        "status": "healthy",
        "version": "2.0.0",
        "model": cfg.ibm_granite_model_id,
        "watsonx_configured": bool(cfg.ibm_watsonx_api_key and "your_" not in cfg.ibm_watsonx_api_key),
        "vector_store_docs": get_rag_pipeline()._collection.count(),
        "tagline": "Practice Smarter. Answer Better. Get Interview Ready with AI.",
    }


# ── Core Trainer Endpoints ────────────────────────────────────────────────────

@app.post("/api/questions", tags=["Trainer"])
async def generate_questions(req: QuestionRequest):
    """Generate a personalized question set with model answers and coaching tips."""
    try:
        return await get_agent().generate_questions(
            name=req.name, role=req.role, experience_level=req.experience_level,
            resume_text=req.resume_text, num_questions=req.num_questions,
        )
    except Exception as exc:
        logger.exception("generate_questions error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/strategy", tags=["Trainer"])
async def generate_strategy(req: StrategyRequest):
    """Generate a 30-day personalized preparation strategy."""
    try:
        return await get_agent().generate_preparation_strategy(
            name=req.name, role=req.role, experience_level=req.experience_level,
            resume_text=req.resume_text,
        )
    except Exception as exc:
        logger.exception("generate_strategy error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/evaluate", tags=["Trainer"])
async def evaluate_answer(req: EvaluateRequest):
    """Score and improve a candidate's answer to any interview question."""
    try:
        return await get_agent().evaluate_answer(
            question=req.question, user_answer=req.user_answer,
            role=req.role, experience_level=req.experience_level,
        )
    except Exception as exc:
        logger.exception("evaluate_answer error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Mock Interview Endpoints ──────────────────────────────────────────────────

@app.post("/api/mock/start", tags=["Mock Interview"])
async def mock_start(req: MockStartRequest):
    """
    Start a new mock interview session.
    Returns the session_id and the first question immediately.
    """
    try:
        session = create_session(
            candidate_name=req.name,
            role=req.role,
            experience_level=req.experience_level,
            total_questions=req.total_questions,
            resume_summary=req.resume_text[:500] if req.resume_text else "",
        )
        engine = get_mock_engine()
        first_turn = await engine.ask_next_question(session)
        return {
            "session_id": session.session_id,
            "candidate": req.name,
            "role": req.role,
            "experience_level": req.experience_level,
            "total_questions": req.total_questions,
            "turn_number": first_turn.turn_number,
            "question": first_turn.question,
            "category": first_turn.category.value,
            "difficulty": first_turn.difficulty.value,
            "questions_remaining": req.total_questions - 1,
        }
    except Exception as exc:
        logger.exception("mock_start error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/mock/answer", tags=["Mock Interview"])
async def mock_answer(req: MockAnswerRequest):
    """
    Submit an answer to the current question.
    Returns evaluation feedback and the next question (if any).
    """
    try:
        session = get_session(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found. Please start a new mock interview.")
        if not session.is_active:
            raise HTTPException(status_code=400, detail="Session is complete. Use /api/mock/report to get your results.")

        engine = get_mock_engine()

        # Evaluate current answer
        evaluated_turn = await engine.evaluate_answer(session, req.answer, req.time_taken_seconds)

        response: dict = {
            "session_id": session.session_id,
            "turn_number": evaluated_turn.turn_number,
            "score": evaluated_turn.score,
            "feedback": evaluated_turn.feedback,
            "strengths": evaluated_turn.strengths,
            "improvements": evaluated_turn.improvements,
            "ideal_answer": evaluated_turn.ideal_answer,
            "running_average": session.average_score,
            "is_complete": session.is_complete,
        }

        # If more questions remain, generate the next one
        if not session.is_complete:
            next_turn = await engine.ask_next_question(session)
            response.update({
                "next_question": next_turn.question,
                "next_category": next_turn.category.value,
                "next_difficulty": next_turn.difficulty.value,
                "questions_remaining": session.total_questions - len(session.turns),
            })
        else:
            response["next_question"] = None
            response["message"] = "Interview complete! Call /api/mock/report to get your full performance report."

        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("mock_answer error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/mock/report/{session_id}", tags=["Mock Interview"])
async def mock_report(session_id: str):
    """
    Generate the comprehensive end-of-session performance report.
    Includes score breakdown, strengths, improvements, and a 1-week action plan.
    """
    try:
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        if session.final_report:
            return session.final_report  # cached
        engine = get_mock_engine()
        return await engine.generate_final_report(session)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("mock_report error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/mock/status/{session_id}", tags=["Mock Interview"])
async def mock_status(session_id: str):
    """Get the current status of a mock interview session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "is_active": session.is_active,
        "is_complete": session.is_complete,
        "turns_completed": len(session.answered_turns),
        "total_questions": session.total_questions,
        "average_score": session.average_score,
        "candidate": session.candidate_name,
        "role": session.role,
    }


# ── Resume Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/parse-resume", tags=["Resume"])
async def parse_resume_text(resume_text: str = Form(...)):
    """Parse pasted resume text and return a structured candidate profile."""
    try:
        parser = get_resume_parser()
        profile = await parser.parse(resume_text)
        return {
            "name": profile.name,
            "email": profile.email,
            "current_role": profile.current_role,
            "total_experience_years": profile.total_experience_years,
            "tech_skills": profile.tech_skills,
            "soft_skills": profile.soft_skills,
            "education": profile.education,
            "companies": profile.companies,
            "certifications": profile.certifications,
            "projects": profile.projects,
            "suggested_role": profile.suggested_role,
            "suggested_level": profile.suggested_level,
            "ai_summary": profile.ai_summary,
        }
    except Exception as exc:
        logger.exception("parse_resume error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/upload-resume", tags=["Resume"])
async def upload_resume(
    file: UploadFile = File(...),
    role: str = Form(default="general"),
    experience_level: str = Form(default="mid"),
):
    """Upload a resume file (PDF, DOCX, TXT). Extracts text, parses profile, indexes for RAG."""
    try:
        content = await file.read()
        filename = (file.filename or "").lower()

        if filename.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(content))
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
            except ImportError:
                raise HTTPException(status_code=400, detail="Run: pip install pypdf")

        elif filename.endswith(".docx"):
            try:
                import zipfile, xml.etree.ElementTree as ET
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    with z.open("word/document.xml") as doc_xml:
                        tree = ET.parse(doc_xml)
                ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                paragraphs = []
                for para in tree.findall(f".//{{{ns}}}p"):
                    runs = "".join(r.text or "" for r in para.findall(f".//{{{ns}}}t"))
                    if runs.strip():
                        paragraphs.append(runs)
                text = "\n".join(paragraphs)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"DOCX parse error: {exc}")
        else:
            text = content.decode("utf-8", errors="ignore")

        text = text.strip()
        if not text:
            raise HTTPException(
                status_code=400,
                detail=f"No text extracted from '{file.filename}'. Try uploading a .txt version.",
            )

        # Parse profile (sync only — fast path)
        parser = get_resume_parser()
        profile = parser.parse_sync(text)

        # Index into vector store
        rag = get_rag_pipeline()
        index_result = rag.ingest_text(text, metadata={"source": "resume", "role": role, "level": experience_level})

        return {
            "status": "success",
            "filename": file.filename,
            "characters_indexed": len(text),
            "index_result": index_result,
            "profile": {
                "name": profile.name,
                "current_role": profile.current_role,
                "experience_years": profile.total_experience_years,
                "tech_skills": profile.tech_skills[:10],
                "suggested_role": profile.suggested_role,
                "suggested_level": profile.suggested_level,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("upload_resume error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
