"""
Chat endpoints for conversational AI.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import json

from app.dependencies import dependencies
from app.models.schemas.request import ChatRequest, ChatHistoryRequest
from app.models.schemas.response import ChatResponse
from app.logging.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Chat with AI",
    description="Send a message and get AI response with RAG context"
)
async def chat(
    request: ChatRequest,
    chat_service = Depends(dependencies.get_chat_service)
) -> ChatResponse:
    """
    Chat with the AI assistant.
    
    - **message**: User message
    - **conversation_id**: Optional conversation ID for context
    - **history**: Previous messages in conversation
    - **system_prompt**: Custom system prompt
    - **use_rag**: Whether to use RAG for context
    - **temperature**: LLM temperature
    """
    try:
        result = await chat_service.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            history=request.history,
            system_prompt=request.system_prompt,
            use_rag=request.use_rag,
            temperature=request.temperature
        )
        
        return ChatResponse(
            message=result["response"],
            conversation_id=result["conversation_id"],
            sources=result.get("sources", []),
            processing_time=result.get("processing_time"),
            usage=result.get("usage"),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/stream",
    summary="Stream chat response",
    description="Chat with streaming response"
)
async def chat_stream(
    request: ChatRequest,
    chat_service = Depends(dependencies.get_chat_service)
):
    """
    Chat with streaming response.
    
    - **message**: User message
    - **conversation_id**: Optional conversation ID
    - **history**: Previous messages
    - **system_prompt**: Custom system prompt
    - **use_rag**: Whether to use RAG
    - **temperature**: LLM temperature
    """
    async def generate():
        try:
            async for chunk in chat_service.chat_stream(
                message=request.message,
                conversation_id=request.conversation_id,
                history=request.history,
                system_prompt=request.system_prompt,
                use_rag=request.use_rag,
                temperature=request.temperature
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
                
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get(
    "/conversations/{conversation_id}",
    summary="Get conversation history",
    description="Retrieve conversation history by ID"
)
async def get_conversation(
    conversation_id: str,
    chat_service = Depends(dependencies.get_chat_service)
) -> Dict[str, Any]:
    """
    Get conversation history.
    
    - **conversation_id**: Conversation identifier
    """
    try:
        history = await chat_service.get_conversation(conversation_id)
        
        if not history:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "conversation_id": conversation_id,
            "messages": history,
            "message_count": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/conversations/{conversation_id}",
    summary="Delete conversation",
    description="Delete a conversation and its history"
)
async def delete_conversation(
    conversation_id: str,
    chat_service = Depends(dependencies.get_chat_service)
) -> Dict[str, str]:
    """
    Delete conversation.
    
    - **conversation_id**: Conversation identifier
    """
    try:
        success = await chat_service.delete_conversation(conversation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"status": "deleted", "conversation_id": conversation_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/conversations",
    summary="List conversations",
    description="List all conversations"
)
async def list_conversations(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    chat_service = Depends(dependencies.get_chat_service)
) -> Dict[str, Any]:
    """
    List conversations.
    
    - **limit**: Number of conversations per page
    - **offset**: Pagination offset
    """
    try:
        conversations = await chat_service.list_conversations(limit, offset)
        
        return {
            "conversations": conversations,
            "total": len(conversations),
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Conversation listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))