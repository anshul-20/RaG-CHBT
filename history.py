import logging
from collections import defaultdict
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from config import get_settings

logger = logging.getLogger(__name__)

# Module-level session store: session_id → list of messages
_histories: dict[str, List[BaseMessage]] = defaultdict(list)


def _trim_history(session_id: str) -> None:
    """Keep only the last N message pairs (2 × window_size messages)."""
    max_msgs = get_settings().history_window_size * 2
    if len(_histories[session_id]) > max_msgs:
        _histories[session_id] = _histories[session_id][-max_msgs:]


def add_user_message(session_id: str, content: str) -> None:
    _histories[session_id].append(HumanMessage(content=content))
    _trim_history(session_id)


def add_ai_message(session_id: str, content: str) -> None:
    _histories[session_id].append(AIMessage(content=content))
    _trim_history(session_id)


def get_history(session_id: str) -> List[BaseMessage]:
    """Return the full (windowed) message list for a session."""
    return list(_histories[session_id])


def clear_session(session_id: str) -> None:
    if session_id in _histories:
        del _histories[session_id]
        logger.info(f"Cleared history for session '{session_id}'")


def session_exists(session_id: str) -> bool:
    return session_id in _histories


def list_sessions() -> List[str]:
    return list(_histories.keys())
