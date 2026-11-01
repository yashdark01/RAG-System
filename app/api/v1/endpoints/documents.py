"""
Document management endpoints.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from fastapi import (
    APIRouter, 
    Depends, 
    HTTPException, 
    UploadFile, 
    File, 
    Form,
    Query
)
from fastapi.responses import JSONResponse

from app.dependencies import dependencies
from app.models.schemas.request import DocumentMetadataUpdate
from app.models.schemas.response import DocumentResponse, DocumentListResponse
from app.logging.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentResponse,
    summary="Upload document",
    description="Upload a document for processing and indexing"
)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    document_service = Depends(dependencies.get_document_service)
) -> DocumentResponse:
    """
    Upload and process a document.
    
    - **file**: Document file to upload (PDF, TXT, MD, etc.)
    - **metadata**: Optional JSON metadata string
    - **chunk_size**: Override default chunk size
    - **chunk_overlap**: Override default chunk overlap
    """
    try:
        # Parse metadata
        import json
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON")
        
        # Read file content
        content = await file.read()
        
        # Process document
        result = await document_service.process_and_index(
            content=content,
            filename=file.filename,
            metadata=meta_dict,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        return DocumentResponse(
            document_id=result["document_id"],
            filename=file.filename,
            status="processed",
            chunks_count=result["chunks_count"],
            metadata=meta_dict,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/upload-batch",
    response_model=List[DocumentResponse],
    summary="Upload multiple documents",
    description="Upload multiple documents for batch processing"
)
async def upload_documents_batch(
    files: List[UploadFile] = File(...),
    document_service = Depends(dependencies.get_document_service)
) -> List[DocumentResponse]:
    """
    Upload and process multiple documents.
    
    - **files**: List of document files
    """
    try:
        results = []
        
        for file in files:
            content = await file.read()
            
            result = await document_service.process_and_index(
                content=content,
                filename=file.filename
            )
            
            results.append(DocumentResponse(
                document_id=result["document_id"],
                filename=file.filename,
                status="processed",
                chunks_count=result["chunks_count"],
                created_at=datetime.now(timezone.utc).isoformat()
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Batch document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document",
    description="Get document by ID"
)
async def get_document(
    document_id: str,
    vector_store = Depends(dependencies.get_vector_store)
) -> DocumentResponse:
    """
    Retrieve document by ID.
    
    - **document_id**: Document identifier
    """
    try:
        doc = await vector_store.get_document(document_id)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentResponse(
            document_id=doc["id"],
            filename=doc.get("metadata", {}).get("source", "unknown"),
            status="active",
            chunks_count=1,
            metadata=doc.get("metadata", {}),
            created_at=doc.get("created_at", datetime.now(timezone.utc).isoformat())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/{document_id}",
    summary="Delete document",
    description="Delete a document and its chunks"
)
async def delete_document(
    document_id: str,
    vector_store = Depends(dependencies.get_vector_store)
) -> Dict[str, str]:
    """
    Delete document by ID.
    
    - **document_id**: Document identifier
    """
    try:
        success = await vector_store.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"status": "deleted", "document_id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Update document",
    description="Update document metadata or content"
)
async def update_document(
    document_id: str,
    update: DocumentMetadataUpdate,
    vector_store = Depends(dependencies.get_vector_store)
) -> DocumentResponse:
    """
    Update document metadata.
    
    - **document_id**: Document identifier
    - **update**: Update data
    """
    try:
        success = await vector_store.update_document(
            document_id=document_id,
            metadata=update.metadata
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc = await vector_store.get_document(document_id)
        
        return DocumentResponse(
            document_id=document_id,
            filename=doc.get("metadata", {}).get("source", "unknown"),
            status="updated",
            chunks_count=1,
            metadata=doc.get("metadata", {}),
            created_at=doc.get("created_at", datetime.now(timezone.utc).isoformat())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List documents",
    description="List all documents with pagination"
)
async def list_documents(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    vector_store = Depends(dependencies.get_vector_store)
) -> DocumentListResponse:
    """
    List documents with pagination.
    
    - **limit**: Number of documents per page
    - **offset**: Pagination offset
    - **status**: Filter by status
    """
    try:
        # This is a simplified implementation
        # In production, you'd query the database
        total = await vector_store.count_documents()
        
        # Get all document IDs (limited)
        all_docs = []
        for doc_id in list(vector_store._metadata_index.keys())[offset:offset + limit]:
            doc = await vector_store.get_document(doc_id)
            if doc:
                all_docs.append(DocumentResponse(
                    document_id=doc["id"],
                    filename=doc.get("metadata", {}).get("source", "unknown"),
                    status="active",
                    chunks_count=1,
                    metadata=doc.get("metadata", {}),
                    created_at=doc.get("created_at", datetime.now(timezone.utc).isoformat())
                ))
        
        return DocumentListResponse(
            documents=all_docs,
            total=total,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))