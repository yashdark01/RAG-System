"""
Chat service for conversational AI with RAG integration.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
import uuid
import time
import asyncio
from datetime import datetime
from collections import defaultdict

from app.config import settings
from app.logging.logger import get_logger
from app.llm.provider import LLMProvider, Message
from app.llm.prompt_templates import PromptManager
from app.services.rag_service import RAGService
from app.services.cache_service import CacheService

logger = get_logger(__name__)


class ChatService:
    """
    Service for managing conversations with RAG integration.
    Supports multi-turn conversations, context management, and streaming.
    """
    
    def __init__(
        self,
        rag_service: RAGService,
        llm_provider: LLMProvider,
        cache_service: Optional[CacheService] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize chat service.
        
        Args:
            rag_service: RAG service for document retrieval
            llm_provider: LLM provider
            cache_service: Optional cache service
            prompt_manager: Optional prompt manager
        """
        self.rag = rag_service
        self.llm = llm_provider
        self.cache = cache_service
        self.prompt_manager = prompt_manager or PromptManager()
        
        # Conversation storage (in-memory, should be replaced with DB in production)
        self._conversations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._conversation_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_history_length = 20
        self.context_window_size = 10
        
        # Metrics
        self._chat_count = 0
        self._total_time = 0.0
        
        logger.info("ChatService initialized")
    
    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        use_rag: bool = True,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a chat message and return response.
        
        Args:
            message: User message
            conversation_id: Conversation ID for context
            history: Previous messages
            system_prompt: Custom system prompt
            use_rag: Whether to use RAG for context
            temperature: LLM temperature
            **kwargs: Additional parameters
            
        Returns:
            Response with answer and metadata
        """
        start_time = time.time()
        self._chat_count += 1
        
        # Get or create conversation
        conv_id = conversation_id or str(uuid.uuid4())
        
        if conv_id not in self._conversations:
            self._conversations[conv_id] = []
            self._conversation_metadata[conv_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "message_count": 0
            }
        
        # Add user message to history
        if history:
            self._conversations[conv_id] = history.copy()
        
        self._conversations[conv_id].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Retrieve context if RAG enabled
        context = ""
        sources = []
        
        if use_rag:
            rag_result = await self.rag.query(
                query=message,
                return_sources=True
            )
            context = self._extract_context_from_rag(rag_result)
            sources = rag_result.get("sources", [])
        
        # Build messages for LLM
        messages = await self._build_messages(
            conversation_id=conv_id,
            message=message,
            context=context,
            system_prompt=system_prompt
        )
        
        # Generate response
        response = await self.llm.generate(
            messages=messages,
            temperature=temperature
        )
        
        # Add assistant response to history
        self._conversations[conv_id].append({
            "role": "assistant",
            "content": response.content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update metadata
        self._conversation_metadata[conv_id]["message_count"] += 2
        self._conversation_metadata[conv_id]["last_activity"] = datetime.utcnow().isoformat()
        
        # Trim history if too long
        if len(self._conversations[conv_id]) > self.max_history_length:
            self._conversations[conv_id] = self._conversations[conv_id][-self.max_history_length:]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self._total_time += processing_time
        
        return {
            "response": response.content,
            "conversation_id": conv_id,
            "sources": sources,
            "processing_time": processing_time,
            "usage": response.usage if hasattr(response, 'usage') else None
        }
    
    async def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        use_rag: bool = True,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a chat response.
        
        Args:
            message: User message
            conversation_id: Conversation ID
            history: Previous messages
            system_prompt: Custom system prompt
            use_rag: Whether to use RAG
            temperature: LLM temperature
            
        Yields:
            Streaming chunks
        """
        # Get or create conversation
        conv_id = conversation_id or str(uuid.uuid4())
        
        if conv_id not in self._conversations:
            self._conversations[conv_id] = []
        
        if history:
            self._conversations[conv_id] = history.copy()
        
        # Add user message
        self._conversations[conv_id].append({
            "role": "user",
            "content": message
        })
        
        # Retrieve context
        context = ""
        if use_rag:
            rag_result = await self.rag.query(query=message)
            context = self._extract_context_from_rag(rag_result)
        
        # Build messages
        messages = await self._build_messages(
            conversation_id=conv_id,
            message=message,
            context=context,
            system_prompt=system_prompt
        )
        
        # Stream response
        full_response = ""
        
        async for chunk in self.llm.stream(
            messages=messages,
            temperature=temperature
        ):
            full_response += chunk.content
            
            yield {
                "content": chunk.content,
                "conversation_id": conv_id,
                "is_final": chunk.is_final
            }
        
        # Add complete response to history
        self._conversations[conv_id].append({
            "role": "assistant",
            "content": full_response
        })
    
    async def _build_messages(
        self,
        conversation_id: str,
        message: str,
        context: str = "",
        system_prompt: Optional[str] = None
    ) -> List[Message]:
        """
        Build message list for LLM.
        
        Args:
            conversation_id: Conversation ID
            message: Current message
            context: RAG context
            system_prompt: System prompt override
            
        Returns:
            List of Message objects
        """
        messages = []
        
        # Build system prompt
        if system_prompt:
            system_content = system_prompt
        else:
            system_content = (
                "You are a helpful AI assistant with access to a knowledge base. "
                "Answer questions accurately based on the provided context and conversation history. "
                "If you're unsure, acknowledge the uncertainty."
            )
        
        if context:
            system_content += f"\n\nRelevant context from knowledge base:\n{context}"
        
        messages.append(Message(role="system", content=system_content))
        
        # Add conversation history
        history = self._conversations.get(conversation_id, [])
        
        # Only include recent history (last N messages)
        recent_history = history[-self.context_window_size:] if len(history) > self.context_window_size else history
        
        for msg in recent_history:
            if msg.get("content") != message:  # Skip current message (will be added last)
                messages.append(Message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                ))
        
        # Add current message if not in history
        if not history or history[-1].get("content") != message:
            messages.append(Message(role="user", content=message))
        
        return messages
    
    def _extract_context_from_rag(self, rag_result: Dict[str, Any]) -> str:
        """Extract context string from RAG result."""
        sources = rag_result.get("sources", [])
        
        if not sources:
            return ""
        
        context_parts = []
        for i, source in enumerate(sources[:5], 1):
            context_parts.append(
                f"[{i}] {source.get('text', '')[:500]}"
            )
        
        return "\n\n".join(context_parts)
    
    async def get_conversation(
        self,
        conversation_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get conversation history.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of messages or None
        """
        return self._conversations.get(conversation_id)
    
    async def delete_conversation(
        self,
        conversation_id: str
    ) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            True if deleted
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            if conversation_id in self._conversation_metadata:
                del self._conversation_metadata[conversation_id]
            return True
        return False
    
    async def list_conversations(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List conversations.
        
        Args:
            limit: Number per page
            offset: Pagination offset
            
        Returns:
            List of conversation summaries
        """
        conversations = []
        
        for conv_id, metadata in list(self._conversation_metadata.items())[offset:offset + limit]:
            messages = self._conversations.get(conv_id, [])
            last_message = messages[-1] if messages else None
            
            conversations.append({
                "conversation_id": conv_id,
                "message_count": len(messages),
                "created_at": metadata.get("created_at"),
                "last_activity": metadata.get("last_activity"),
                "last_message": last_message.get("content", "")[:100] if last_message else ""
            })
        
        return conversations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chat service statistics."""
        avg_time = self._total_time / self._chat_count if self._chat_count > 0 else 0
        
        return {
            "total_chats": self._chat_count,
            "avg_response_time": avg_time,
            "active_conversations": len(self._conversations),
            "max_history_length": self.max_history_length
        }