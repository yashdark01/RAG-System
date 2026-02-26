import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.production_ocr_wrapper import ProductionOCRService
from app.core.llm import get_llm


logger = logging.getLogger(__name__)

router = APIRouter()


UPLOAD_FOLDER = "data/ocr_uploads"

# Reuse a single OCR service instance so that cache and metrics are shared
_ocr_service = ProductionOCRService()

# In-memory store for looking up results by document_id
_ocr_results: Dict[str, Dict[str, Any]] = {}


async def _save_upload_file(file: UploadFile) -> str:
    """
    Save an uploaded file to disk and return the saved path.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    suffix = Path(file.filename).suffix.lower()
    safe_name = Path(file.filename).name
    target_path = Path(UPLOAD_FOLDER) / safe_name

    # If file already exists, append a numeric suffix
    counter = 1
    while target_path.exists():
        target_path = Path(UPLOAD_FOLDER) / f"{Path(safe_name).stem}_{counter}{suffix}"
        counter += 1

    content = await file.read()
    with target_path.open("wb") as f:
        f.write(content)

    return str(target_path)


@router.post("/process")
async def process_document(
    file: UploadFile = File(...),
    document_type: str = "general",
    force_reprocess: bool = False,
) -> Dict[str, Any]:
    """
    Process a single document with the production OCR service.
    """
    try:
        file_path = await _save_upload_file(file)
        result = _ocr_service.process_document(
            file_path=file_path,
            document_type=document_type,
            force_reprocess=force_reprocess,
        )

        document_id = result.get("document_id")
        if result.get("status") == "success" and document_id:
            _ocr_results[document_id] = result

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing OCR document")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")


@router.post("/batch")
async def batch_process_documents(
    files: List[UploadFile] = File(...),
    document_type: str = "general",
) -> Dict[str, Any]:
    """
    Process multiple documents with the production OCR service.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    saved_paths: List[str] = []
    try:
        for file in files:
            saved_paths.append(await _save_upload_file(file))

        results: List[Dict[str, Any]] = []
        for path in saved_paths:
            result = _ocr_service.process_document(path, document_type=document_type)
            results.append(result)

            document_id = result.get("document_id")
            if result.get("status") == "success" and document_id:
                _ocr_results[document_id] = result

        successful = sum(1 for r in results if r.get("status") == "success")
        failed = sum(1 for r in results if r.get("status") == "error")

        return {
            "status": "success",
            "total_documents": len(results),
            "successful": successful,
            "failed": failed,
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error batch processing OCR documents")
        raise HTTPException(status_code=500, detail=f"OCR batch processing failed: {e}")


@router.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    """
    Get OCR service metrics (documents processed, cache stats, success rate, etc.).
    """
    return _ocr_service.get_metrics()


@router.get("/health")
def health_check() -> Dict[str, Any]:
    """
    Simple health check for the OCR service.
    """
    metrics = _ocr_service.get_metrics()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "documents_processed": metrics.get("total_documents", 0),
        "success_rate": metrics.get("success_rate", 0),
        "avg_processing_time": round(metrics.get("avg_processing_time", 0), 2),
    }


@router.post("/cache/clear")
def clear_cache() -> Dict[str, Any]:
    """
    Clear the OCR result cache.
    """
    _ocr_service.clear_cache()
    return {"status": "success", "message": "OCR cache cleared"}


@router.get("/status/{document_id}")
def get_document_status(document_id: str) -> Dict[str, Any]:
    """
    Retrieve the last OCR result for a document by its document_id.
    """
    if document_id not in _ocr_results:
        raise HTTPException(status_code=404, detail="Document ID not found")

    return _ocr_results[document_id]


@router.post("/llm-extract")
async def llm_extract(
    file: UploadFile = File(...),
    document_type: str = "general",
    task: Optional[str] = None,
    schema_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run OCR on a document and use an LLM to produce enriched structured JSON.
    """
    try:
        file_path = await _save_upload_file(file)
        ocr_result = _ocr_service.process_document(
            file_path=file_path,
            document_type=document_type,
            force_reprocess=False,
        )

        if ocr_result.get("status") != "success":
            raise HTTPException(
                status_code=500,
                detail=f"OCR processing failed: {ocr_result.get('error', 'unknown error')}",
            )

        raw_text = ocr_result.get("raw_text", "")
        existing_fields = ocr_result.get("extracted_fields", {})
        entities = ocr_result.get("entities", {})

        instruction = task or "Extract and normalize key fields from this document."

        schema_description = schema_hint or (
            "Return a JSON object with keys that match the document type "
            f"'{document_type}' where possible (for example: invoice_number, "
            "date, amount, merchant, items, passport_number, etc.)."
        )

        llm = get_llm(temperature=0.1)

        prompt = (
            "You are a strict JSON information extraction engine.\n"
            "You receive OCR text from a document plus some preliminary extracted fields and entities.\n\n"
            f"Document type: {document_type}\n"
            f"Task: {instruction}\n"
            f"Schema hint: {schema_description}\n\n"
            "Preliminary extracted_fields (from rule-based OCR):\n"
            f"{json.dumps(existing_fields, indent=2)}\n\n"
            "Preliminary entities:\n"
            f"{json.dumps(entities, indent=2)}\n\n"
            "Full OCR text:\n"
            f"{raw_text}\n\n"
            "Respond with a SINGLE JSON object only, no prose, no markdown.\n"
            "Use this shape at minimum:\n"
            "{\n"
            '  \"document_type\": string,\n'
            '  \"fields\": { /* normalized key fields */ },\n'
            '  \"entities\": { /* optional refined entities */ },\n'
            '  \"confidence\": number\n'
            "}\n"
        )

        llm_response = llm.invoke(prompt)
        llm_content = getattr(llm_response, "content", str(llm_response))

        try:
            llm_json = json.loads(llm_content)
        except json.JSONDecodeError:
            llm_json = {
                "error": "Failed to parse LLM response as JSON",
                "raw_response": llm_content,
            }

        return {
            "ocr_result": ocr_result,
            "llm_result": llm_json,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error running OCR LLM extraction")
        raise HTTPException(status_code=500, detail=f"OCR LLM extraction failed: {e}")


