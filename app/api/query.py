# app/api/query.py
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from app.core.embeddings import get_embedding_model
from app.core.vector_store import load_vector_store
from app.core.llm import get_llm
from app.core.chat_memory import get_memory
from app.services.retriever import get_retriever
from app.services.rag_chain import build_rag_chain

# Create an instance of APIRouter to handle API routes
router = APIRouter()


class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., min_length=1, description="The question to ask")
    k: int = Field(default=4, ge=1, le=10, description="Number of documents to retrieve")
    session_id: Optional[str] = Field(default=None, description="Conversation session id for chat memory; omit to start a new conversation")


class AnswerResponse(BaseModel):
    """Response model for answers"""
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_chunks: int
    session_id: Optional[str] = Field(default=None, description="Session id to send with follow-up messages for the same conversation")


@router.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question based on uploaded documents.
    
    Args:
        request: QuestionRequest containing the question and retrieval parameters
        
    Returns:
        AnswerResponse: The answer and source documents
    """
    try:
        session_id = request.session_id or uuid.uuid4().hex

        # Get session-scoped short-term memory and load chat history
        memory = get_memory(session_id)
        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        # Generate embeddings for the input question
        embeddings = get_embedding_model()
        
        # Load the vector store using the generated embeddings
        vector_db = load_vector_store(embeddings)

        # Retrieve relevant documents from the vector store
        retriever = get_retriever(vector_db, k=request.k)
        
        # Get the language model for processing the question
        llm = get_llm()

        # Build the RAG chain using the language model and retriever
        rag_chain = build_rag_chain(llm, retriever)

        # Run the RAG chain with question and chat history
        answer = rag_chain.invoke({
            "question": request.question,
            "chat_history": chat_history,
        })

        # Save this turn to session memory
        memory.save_context({"input": request.question}, {"output": answer})

        # Get relevant documents for sources
        docs = retriever.invoke(request.question)

        return AnswerResponse(
            answer=answer,
            sources=[doc.metadata for doc in docs],
            retrieved_chunks=len(docs),
            session_id=session_id,
        )
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="Vector database not found. Please upload documents first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )