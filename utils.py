import uuid
import re
import time
import logging
from pathlib import Path
from functools import wraps
from typing import Any, Callable


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_session_id() -> str:
    return str(uuid.uuid4())


def generate_document_id(filename: str) -> str:
    return f"{Path(filename).stem}_{uuid.uuid4().hex[:8]}"


def sanitize_collection_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    if len(sanitized) < 3:
        sanitized += "_col"
    return sanitized[:63]


def validate_pdf_file(filename: str, content_type: str) -> None:
    if not filename.lower().endswith(".pdf"):
        raise ValueError(f"'{filename}' is not a PDF.")
    if content_type not in ("application/pdf", "application/octet-stream"):
        raise ValueError(f"Invalid content type '{content_type}'.")


def format_sources(source_documents: list) -> list[dict]:
    seen, sources = set(), []
    for doc in source_documents:
        meta = doc.metadata
        key = (meta.get("source", ""), meta.get("page", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "N/A"),
                "chunk": meta.get("chunk_index", "N/A"),
            })
    return sources


def timeit(func: Callable) -> Callable:
    import asyncio
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        t = time.perf_counter()
        result = await func(*args, **kwargs)
        logger.info(f"{func.__name__} took {time.perf_counter() - t:.3f}s")
        return result

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        t = time.perf_counter()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} took {time.perf_counter() - t:.3f}s")
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
