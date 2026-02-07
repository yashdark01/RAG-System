# app/core/chat_memory.py
"""Session-scoped short-term conversation memory using LangChain ConversationBufferWindowMemory."""

from langchain_classic.memory import ConversationBufferWindowMemory

_session_memories: dict[str, ConversationBufferWindowMemory] = {}


def get_memory(session_id: str, k: int = 5) -> ConversationBufferWindowMemory:
    """Get or create short-term memory for this session (last k turns)."""
    if session_id not in _session_memories:
        _session_memories[session_id] = ConversationBufferWindowMemory(
            k=k,
            return_messages=False,
            memory_key="chat_history",
        )
    return _session_memories[session_id]


def clear_memory(session_id: str) -> None:
    """Clear conversation history for the given session."""
    if session_id in _session_memories:
        del _session_memories[session_id]
