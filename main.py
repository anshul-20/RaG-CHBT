import os
import logging
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

import vector_store as vs
import chatbot
import history as hist
import pdf_loader
from config import get_settings
from schemas import (
    ChatRequest, ChatResponse,
    ClearHistoryRequest, ClearHistoryResponse,
    IngestResponse, HealthResponse,
    SessionsResponse, CollectionStatsResponse,
)
from utils import setup_logging, generate_session_id, validate_pdf_file

setup_logging("INFO")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== RAG Chatbot API starting ===")
    vs.init_vector_store()
    chatbot.init_llms()
    logger.info("All components ready.")
    yield
    logger.info("=== RAG Chatbot API stopped ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Chatbot API",
    description="PDF-grounded conversational AI with persistent chat history.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes — System
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(status="ok", collection_stats=vs.get_collection_stats())


@app.get("/stats", response_model=CollectionStatsResponse, tags=["System"])
async def collection_stats():
    return CollectionStatsResponse(**vs.get_collection_stats())


# ---------------------------------------------------------------------------
# Routes — Documents
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload a PDF, embed its chunks, and store them in ChromaDB."""
    try:
        validate_pdf_file(file.filename, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        chunks = pdf_loader.process_pdf(tmp_path, original_filename=file.filename)
        if not chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted from the PDF.")
        vs.add_documents(chunks)
        summary = pdf_loader.get_document_summary(chunks)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed for '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        doc_id=summary["doc_id"],
        source=summary["source"],
        total_pages=summary["total_pages"],
        total_chunks=summary["total_chunks"],
        message=f"Ingested '{file.filename}' — {summary['total_chunks']} chunks stored.",
    )


@app.delete("/document/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Remove all embeddings for a document by its doc_id."""
    try:
        vs.delete_by_doc_id(doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"doc_id": doc_id, "message": "Document deleted from vector store."}


# ---------------------------------------------------------------------------
# Routes — Chat
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Send a question and get a grounded answer from the ingested PDFs.
    Supply the same session_id across calls to retain conversation history.
    """
    try:
        result = chatbot.chat(
            session_id=request.session_id,
            question=request.question,
            doc_id=request.doc_id,
        )
    except Exception as e:
        logger.error(f"Chat error [session={request.session_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
    return ChatResponse(**result)


# ---------------------------------------------------------------------------
# Routes — Session
# ---------------------------------------------------------------------------

@app.post("/session/new", tags=["Session"])
async def new_session():
    """Generate a new session ID."""
    sid = generate_session_id()
    return {"session_id": sid, "message": "Use this session_id in /chat to track history."}


@app.get("/session/list", response_model=SessionsResponse, tags=["Session"])
async def list_sessions():
    sessions = hist.list_sessions()
    return SessionsResponse(active_sessions=sessions, total=len(sessions))


@app.post("/session/clear", response_model=ClearHistoryResponse, tags=["Session"])
async def clear_session(request: ClearHistoryRequest):
    hist.clear_session(request.session_id)
    return ClearHistoryResponse(
        session_id=request.session_id,
        message=f"History cleared for session '{request.session_id}'.",
    )
