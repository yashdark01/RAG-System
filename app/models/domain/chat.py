"""
Domain models for chat-related entities.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class MessageRole(str, Enum):
    """Roles for messages in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Represents a single message in a conversation."""
    id: str = Field(..., description="Unique message identifier")
    role: MessageRole = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    token_count: Optional[int] = Field(None, description="Estimated token count")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "msg_123",
                "role": "user",
                "content": "Hello, how can you help me?",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
    )


class ConversationMetadata(BaseModel):
    """Metadata for a conversation."""
    message_count: int = Field(default=0, description="Number of messages in conversation")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    user_id: Optional[str] = Field(None, description="User ID if applicable")
    title: Optional[str] = Field(None, description="Conversation title")
    tags: List[str] = Field(default_factory=list, description="Conversation tags")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message_count": 5,
                "last_activity": "2024-01-01T00:00:00Z",
                "created_at": "2024-01-01T00:00:00Z",
                "title": "Help with Python"
            }
        }
    )


class Conversation(BaseModel):
    """Represents a conversation between user and assistant."""
    id: str = Field(..., description="Unique conversation identifier")
    messages: List[Message] = Field(default_factory=list, description="List of messages")
    metadata: ConversationMetadata = Field(default_factory=ConversationMetadata, description="Conversation metadata")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    is_active: bool = Field(default=True, description="Whether conversation is active")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "conv_123",
                "messages": [
                    {
                        "id": "msg_1",
                        "role": "user",
                        "content": "Hello"
                    },
                    {
                        "id": "msg_2",
                        "role": "assistant",
                        "content": "Hi! How can I help?"
                    }
                ],
                "is_active": True
            }
        }
    )

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.metadata.message_count = len(self.messages)
        self.metadata.last_activity = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def get_last_n_messages(self, n: int) -> List[Message]:
        """Get the last n messages from the conversation."""
        return self.messages[-n:] if len(self.messages) >= n else self.messages

    def get_messages_by_role(self, role: MessageRole) -> List[Message]:
        """Get all messages of a specific role."""
        return [msg for msg in self.messages if msg.role == role]
