from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session ID for conversation history")
    question: str = Field(..., min_length=1, max_length=2000)
    doc_id: Optional[str] = Field(None, description="Scope retrieval to a specific document")


class ClearHistoryRequest(BaseModel):
    session_id: str


class SourceDocument(BaseModel):
    source: str
    page: str | int
    chunk: str | int


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceDocument]


class IngestResponse(BaseModel):
    doc_id: str
    source: str
    total_pages: int
    total_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    collection_stats: dict


class SessionsResponse(BaseModel):
    active_sessions: List[str]
    total: int


class ClearHistoryResponse(BaseModel):
    session_id: str
    message: str


class CollectionStatsResponse(BaseModel):
    collection_name: str
    persist_dir: str
    total_chunks: int
