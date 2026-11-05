"""
Tests for IBM Interview Trainer Agent
Run with: pytest tests/ -v
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the lru_cache so each test gets a fresh Settings instance."""
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── Config Tests ─────────────────────────────────────────────────────────────

def test_settings_defaults():
    """Settings load with defaults when no env file is present."""
    from app.config import Settings
    s = Settings(
        _env_file=None,
        IBM_WATSONX_API_KEY="test_key",
        IBM_WATSONX_PROJECT_ID="test_project",
    )
    assert s.ibm_watsonx_api_key == "test_key"
    assert s.ibm_granite_model_id == "ibm/granite-4-h-small"
    assert s.rag_top_k_results == 5
    assert s.rag_chunk_size == 512


# ── RAG Pipeline Tests ────────────────────────────────────────────────────────

def test_chunk_text_basic():
    from app.rag_pipeline import RAGPipeline
    chunks = RAGPipeline._chunk_text("word " * 600, chunk_size=512, overlap=64)
    assert len(chunks) >= 1
    for chunk in chunks:
        words = chunk.split()
        assert len(words) <= 512


def test_chunk_text_short():
    from app.rag_pipeline import RAGPipeline
    text = "Hello world"
    chunks = RAGPipeline._chunk_text(text, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_normalize_role():
    from app.agent import normalize_role
    assert normalize_role("Software Engineer") == "software_engineer"
    assert normalize_role("SWE") == "software_engineer"
    assert normalize_role("data scientist") == "data_scientist"
    assert normalize_role("unknown role xyz") == "general"


# ── API Endpoint Tests ────────────────────────────────────────────────────────

@pytest.fixture
def mock_agent():
    """Mock the InterviewTrainerAgent to avoid LLM calls in tests."""
    with patch("app.main.get_agent") as mock:
        agent = MagicMock()
        agent.generate_questions = AsyncMock(return_value={
            "candidate": {"name": "Test User", "role": "SWE", "experience_level": "mid"},
            "questions_and_answers": "### Q1: Tell me about yourself\n**Question:** ...",
            "context_sources_count": 3,
            "model_used": "ibm/granite-13b-chat-v2",
        })
        agent.generate_preparation_strategy = AsyncMock(return_value={
            "candidate": {"name": "Test User", "role": "SWE", "experience_level": "mid"},
            "preparation_strategy": "Week 1: Review fundamentals...",
            "model_used": "ibm/granite-13b-chat-v2",
        })
        agent.evaluate_answer = AsyncMock(return_value={
            "question": "What is a thread?",
            "user_answer": "A thread is...",
            "evaluation": "Score: 7/10. Strengths: ...",
            "model_used": "ibm/granite-13b-chat-v2",
        })
        mock.return_value = agent
        yield agent


@pytest.fixture
def mock_rag():
    with patch("app.main.get_rag_pipeline") as mock_main_rag, \
         patch("app.rag_pipeline.get_rag_pipeline") as mock_pipeline_rag:
        rag = MagicMock()
        rag._collection.count.return_value = 14
        mock_main_rag.return_value = rag
        mock_pipeline_rag.return_value = rag
        yield rag


@pytest.fixture
def mock_lifespan(mock_rag):
    """Patch lifespan so FastAPI startup doesn't hit the real vector store."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop(app):
        yield

    with patch("app.main.lifespan", _noop):
        yield


@pytest.mark.asyncio
async def test_health_endpoint(mock_rag, mock_lifespan):
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model" in data


@pytest.mark.asyncio
async def test_generate_questions_endpoint(mock_agent, mock_rag, mock_lifespan):
    from app.main import app
    payload = {
        "name": "Priya Sharma",
        "role": "Software Engineer",
        "experience_level": "mid",
        "num_questions": 5,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/questions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "questions_and_answers" in data
    assert "candidate" in data


@pytest.mark.asyncio
async def test_generate_strategy_endpoint(mock_agent, mock_rag, mock_lifespan):
    from app.main import app
    payload = {
        "name": "Arjun Mehta",
        "role": "Data Scientist",
        "experience_level": "senior",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/strategy", json=payload)
    assert resp.status_code == 200
    assert "preparation_strategy" in resp.json()


@pytest.mark.asyncio
async def test_evaluate_answer_endpoint(mock_agent, mock_rag, mock_lifespan):
    from app.main import app
    payload = {
        "question": "What is a thread?",
        "user_answer": "A thread is a lightweight process sharing memory.",
        "role": "Software Engineer",
        "experience_level": "mid",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/evaluate", json=payload)
    assert resp.status_code == 200
    assert "evaluation" in resp.json()


@pytest.mark.asyncio
async def test_missing_fields_returns_422(mock_rag, mock_lifespan):
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/questions", json={"name": "Only Name"})
    assert resp.status_code == 422
