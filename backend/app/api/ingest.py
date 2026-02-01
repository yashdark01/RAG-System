# app/api/ingest.py
from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from app.services.loader import load_pdf
from app.services.chunker import chunk_documents
from app.core.embeddings import get_embedding_model
from app.core.vector_store import create_vector_store

# Create an APIRouter instance for organizing routes
router = APIRouter()

# Define the directory where uploaded documents will be stored
UPLOAD_FOLDER = "data/documents"


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document and index it in the vector store.
    
    Args:
        file: The PDF file to upload
        
    Returns:
        dict: Success message with filename
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Construct the file path by joining folder and filename
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # Create the upload directory if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save the uploaded file to disk
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract text content from the PDF file
        documents = load_pdf(file_path)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No content found in PDF")
        
        # Split documents into smaller chunks for processing
        chunks = chunk_documents(documents)

        # Get the embedding model
        embedding_model = get_embedding_model()

        # Store chunks and embeddings in the vector database
        create_vector_store(chunks, embedding_model)

        # Return success message to the client
        return {
            "message": f"{file.filename} uploaded and indexed successfully.",
            "chunks_created": len(chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")