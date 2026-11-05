"""
Chat repository for conversation and message operations.
"""

from typing import List, Optional
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload

from app.database.repositories.base import BaseRepository
from app.database.models import Conversation, Message


class ChatRepository:
    """Repository for chat-related operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.conversation_repo = BaseRepository(Conversation, session)
        self.message_repo = BaseRepository(Message, session)
    
    # Conversation methods
    
    async def create_conversation(
        self,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            title: Conversation title
            user_id: User identifier
            metadata: Additional metadata
            
        Returns:
            Created conversation
        """
        return await self.conversation_repo.create(
            title=title,
            user_id=user_id,
            metadata=metadata or {}
        )
    
    async def get_conversation(
        self,
        conversation_id: uuid.UUID
    ) -> Optional[Conversation]:
        """
        Get conversation with messages.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation with messages or None
        """
        query = (
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.messages))
        )
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Conversation]:
        """
        Get conversations for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum results
            
        Returns:
            List of conversations
        """
        query = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update_conversation(
        self,
        conversation_id: uuid.UUID,
        **kwargs
    ) -> Optional[Conversation]:
        """Update conversation."""
        return await self.conversation_repo.update(conversation_id, **kwargs)
    
    async def delete_conversation(
        self,
        conversation_id: uuid.UUID
    ) -> bool:
        """Delete conversation."""
        return await self.conversation_repo.delete(conversation_id)
    
    # Message methods
    
    async def add_message(
        self,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        token_count: Optional[int] = None,
        sources: Optional[list] = None,
        metadata: Optional[dict] = None
    ) -> Message:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user, assistant, system)
            content: Message content
            token_count: Token count
            sources: RAG sources
            metadata: Additional metadata
            
        Returns:
            Created message
        """
        # Create message
        message = await self.message_repo.create(
            role=role,
            content=content,
            token_count=token_count,
            sources=sources,
            metadata=metadata or {}
        )
        
        # Add to conversation
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            conversation.messages.append(message)
            conversation.message_count = len(conversation.messages)
            if token_count:
                conversation.total_tokens = (conversation.total_tokens or 0) + token_count
            await self.session.flush()
        
        return message
    
    async def get_conversation_messages(
        self,
        conversation_id: uuid.UUID,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """
        Get messages for a conversation.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum messages
            offset: Pagination offset
            
        Returns:
            List of messages
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.messages
        messages.sort(key=lambda m: m.created_at)
        
        if offset:
            messages = messages[offset:]
        if limit:
            messages = messages[:limit]
        
        return messages
    
    async def get_conversation_stats(self) -> dict:
        """
        Get conversation statistics.
        
        Returns:
            Statistics dictionary
        """
        total_query = select(func.count(Conversation.id))
        total_result = await self.session.execute(total_query)
        total_conversations = total_result.scalar()
        
        messages_query = select(func.count(Message.id))
        messages_result = await self.session.execute(messages_query)
        total_messages = messages_result.scalar()
        
        active_query = select(func.count(Conversation.id)).where(
            Conversation.status == "active"
        )
        active_result = await self.session.execute(active_query)
        active_conversations = active_result.scalar()
        
        return {
            "total_conversations": total_conversations,
            "active_conversations": active_conversations,
            "total_messages": total_messages
        }