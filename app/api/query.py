# app/api/query.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app.core.embeddings import get_embedding_model
from app.core.vector_store import load_vector_store
from app.core.llm import get_llm
from app.services.retriever import get_retriever
from app.services.rag_chain import build_rag_chain

# Create an instance of APIRouter to handle API routes
router = APIRouter()


class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., min_length=1, description="The question to ask")
    k: int = Field(default=4, ge=1, le=10, description="Number of documents to retrieve")


class AnswerResponse(BaseModel):
    """Response model for answers"""
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_chunks: int


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

        # Run the RAG chain with the question
        answer = rag_chain.invoke(request.question)
        
        # Get relevant documents for sources using invoke() instead of get_relevant_documents()
        docs = retriever.invoke(request.question)

        # Return the answer and the sources of the documents used
        return AnswerResponse(
            answer=answer,
            sources=[doc.metadata for doc in docs],
            retrieved_chunks=len(docs)
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