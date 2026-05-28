import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import get_settings
from utils import generate_document_id

logger = logging.getLogger(__name__)


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_pdf_pages(file_path: str) -> List[Document]:
    """Load raw pages from a PDF file using PyPDFLoader."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    logger.info(f"Loading PDF: {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    return PyPDFLoader(str(path)).load()


def enrich_page_metadata(pages: List[Document], original_filename: str, doc_id: str) -> List[Document]:
    """Attach source, doc_id, page number, and total_pages to each page."""
    total = len(pages)
    for i, page in enumerate(pages):
        page.metadata.update({
            "source": original_filename,
            "doc_id": doc_id,
            "page": i + 1,
            "total_pages": total,
        })
    return pages


def split_documents(pages: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split pages into overlapping chunks and tag each with a chunk_index."""
    splitter = _make_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(pages)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
    logger.info(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def process_pdf(file_path: str, original_filename: str = "") -> List[Document]:
    """
    Full pipeline: load → enrich metadata → split into chunks.
    Returns a list of Document chunks ready for embedding.
    """
    settings = get_settings()
    fname = original_filename or Path(file_path).name
    doc_id = generate_document_id(fname)

    pages = load_pdf_pages(file_path)
    logger.info(f"Extracted {len(pages)} pages from '{fname}'")

    pages = enrich_page_metadata(pages, fname, doc_id)
    chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)
    return chunks


def get_document_summary(chunks: List[Document]) -> dict:
    """Return a metadata summary derived from the first chunk."""
    if not chunks:
        return {}
    meta = chunks[0].metadata
    return {
        "doc_id": meta.get("doc_id"),
        "source": meta.get("source"),
        "total_pages": meta.get("total_pages"),
        "total_chunks": len(chunks),
    }
