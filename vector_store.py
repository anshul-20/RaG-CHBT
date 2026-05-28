import logging
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.documents import Document

from config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — initialized once via init_vector_store()
# ---------------------------------------------------------------------------
_store: Optional[Chroma] = None
_embeddings: Optional[VoyageAIEmbeddings] = None


def init_vector_store() -> None:
    """Initialize the ChromaDB store and Voyage AI embeddings. Call once at startup."""
    global _store, _embeddings
    settings = get_settings()

    _embeddings = VoyageAIEmbeddings(
        model=settings.embedding_model,
        voyage_api_key=settings.voyage_api_key,
    )

    _store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=_embeddings,
        persist_directory=settings.chroma_persist_dir,
    )

    count = _store._collection.count()
    logger.info(
        f"ChromaDB ready | collection='{settings.chroma_collection_name}' "
        f"| persist='{settings.chroma_persist_dir}' | docs={count}"
    )


def _get_store() -> Chroma:
    if _store is None:
        raise RuntimeError("Vector store is not initialized. Call init_vector_store() first.")
    return _store


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def add_documents(documents: List[Document]) -> List[str]:
    """Embed and persist a list of Document chunks. Returns assigned IDs."""
    if not documents:
        logger.warning("add_documents called with empty list — skipping.")
        return []
    logger.info(f"Embedding {len(documents)} chunks...")
    ids = _get_store().add_documents(documents)
    logger.info(f"Stored {len(ids)} chunks in ChromaDB.")
    return ids


def similarity_search(
    query: str,
    k: Optional[int] = None,
    filter_dict: Optional[dict] = None,
) -> List[Document]:
    """Return top-k relevant chunks for a query, with optional metadata filter."""
    settings = get_settings()
    k = k or settings.top_k_retrieval
    results = _get_store().similarity_search(query, k=k, filter=filter_dict)
    logger.debug(f"Retrieved {len(results)} chunks for: '{query[:80]}'")
    return results


def similarity_search_with_score(
    query: str,
    k: Optional[int] = None,
) -> List[Tuple[Document, float]]:
    """Return chunks with cosine similarity scores."""
    settings = get_settings()
    k = k or settings.top_k_retrieval
    return _get_store().similarity_search_with_score(query, k=k)


def get_retriever(k: Optional[int] = None, filter_dict: Optional[dict] = None):
    """Return a LangChain-compatible retriever, optionally scoped by metadata filter."""
    settings = get_settings()
    k = k or settings.top_k_retrieval
    search_kwargs: dict = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict
    return _get_store().as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def delete_by_doc_id(doc_id: str) -> None:
    """Delete all chunks belonging to a specific document."""
    collection = _get_store()._collection
    results = collection.get(where={"doc_id": doc_id})
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id='{doc_id}'")
    else:
        logger.warning(f"No chunks found for doc_id='{doc_id}'")


def get_collection_stats() -> dict:
    """Return basic stats about the current ChromaDB collection."""
    settings = get_settings()
    return {
        "collection_name": settings.chroma_collection_name,
        "persist_dir": settings.chroma_persist_dir,
        "total_chunks": _get_store()._collection.count(),
    }
