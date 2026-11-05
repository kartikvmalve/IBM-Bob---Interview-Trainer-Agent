"""
IBM Interview Trainer Agent — Configuration
Loads and validates all environment variables.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── IBM watsonx.ai ─────────────────────────────────────────
    ibm_watsonx_api_key: str = Field(default="", alias="IBM_WATSONX_API_KEY")
    ibm_watsonx_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com", alias="IBM_WATSONX_URL"
    )
    ibm_watsonx_project_id: str = Field(default="", alias="IBM_WATSONX_PROJECT_ID")

    # ── IBM Granite Model ───────────────────────────────────────
    ibm_granite_model_id: str = Field(
        default="ibm/granite-4-h-small", alias="IBM_GRANITE_MODEL_ID"
    )
    ibm_granite_max_tokens: int = Field(default=1024, alias="IBM_GRANITE_MAX_TOKENS")
    ibm_granite_temperature: float = Field(default=0.7, alias="IBM_GRANITE_TEMPERATURE")
    ibm_granite_top_p: float = Field(default=0.9, alias="IBM_GRANITE_TOP_P")

    # ── IBM Watson Discovery (optional) ────────────────────────
    ibm_watson_discovery_api_key: str = Field(
        default="", alias="IBM_WATSON_DISCOVERY_API_KEY"
    )
    ibm_watson_discovery_url: str = Field(
        default="", alias="IBM_WATSON_DISCOVERY_URL"
    )
    ibm_watson_discovery_project_id: str = Field(
        default="", alias="IBM_WATSON_DISCOVERY_PROJECT_ID"
    )

    # ── Application ─────────────────────────────────────────────
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    app_secret_key: str = Field(
        default="default_secret_change_in_production", alias="APP_SECRET_KEY"
    )

    # ── ChromaDB Vector Store ───────────────────────────────────
    chroma_db_path: str = Field(default="./data/chroma_db", alias="CHROMA_DB_PATH")
    chroma_collection_name: str = Field(
        default="interview_knowledge_base", alias="CHROMA_COLLECTION_NAME"
    )

    # ── RAG ─────────────────────────────────────────────────────
    rag_top_k_results: int = Field(default=5, alias="RAG_TOP_K_RESULTS")
    rag_chunk_size: int = Field(default=512, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=64, alias="RAG_CHUNK_OVERLAP")

    model_config = {"env_file": ".env", "populate_by_name": True, "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
