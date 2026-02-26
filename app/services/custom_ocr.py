"""
Custom OCR Service - Document Processing & Text Extraction
A complete OCR service with document ingestion, processing, and structured JSON output
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import hashlib
import re
import uuid
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[OCR Service] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES & ENUMS
# ============================================================================

class DocumentType(Enum):
    """Supported document types"""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BUSINESS_CARD = "business_card"
    INSURANCE = "insurance"
    RESUME = "resume"
    MEDICAL = "medical"
    GENERAL = "general"


class ConfidenceLevel(Enum):
    """Confidence levels for extracted data"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class TextBlock:
    """Represents a text block extracted from document"""
    text: str
    confidence: float
    bounding_box: Optional[Dict] = None
    font_size: Optional[int] = None
    is_bold: bool = False
    is_italic: bool = False
    language: str = "en"
    block_type: str = "text"  # text, heading, list, table, etc.


@dataclass
class Table:
    """Represents a table extracted from document"""
    id: str
    rows: List[List[str]]
    headers: Optional[List[str]] = None
    col_count: int = 0
    row_count: int = 0
    confidence: float = 0.0


@dataclass
class Image:
    """Represents an image extracted from document"""
    id: str
    description: str
    size: Optional[Tuple[int, int]] = None
    format: str = "unknown"
    confidence: float = 0.0


@dataclass
class Page:
    """Represents a single page from document"""
    page_number: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    full_text: str = ""
    language: str = "en"
    skew_angle: float = 0.0
    is_blank: bool = False
    quality_score: float = 0.0


@dataclass
class DocumentMetadata:
    """Metadata about the document"""
    document_id: str
    document_type: DocumentType
    file_name: str
    file_size: int
    file_format: str
    upload_time: str
    processing_time: float
    ocr_engine: str
    ocr_version: str
    language: str
    page_count: int
    total_words: int
    confidence_score: float


@dataclass
class ExtractedField:
    """Represents an extracted field with value and confidence"""
    name: str
    value: Any
    confidence: float = 0.0
    field_type: str = "text"  # text, number, date, currency, etc.
    raw_text: str = ""
    normalized_value: Optional[Any] = None


@dataclass
class OCRResult:
    """Complete OCR result structure"""
    metadata: DocumentMetadata
    pages: List[Page]
    extracted_fields: Dict[str, ExtractedField] = field(default_factory=dict)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    relationships: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# OCR PROCESSORS
# ============================================================================

class BaseOCRProcessor(ABC):
    """Base class for OCR processors"""
    
    def __init__(self, name: str):
        self.name = name
        self.config = {}
    
    @abstractmethod
    def process(self, file_path: str) -> OCRResult:
        """Process document and return OCR result"""
        pass


class TextExtractor(BaseOCRProcessor):
    """Extracts text from documents"""
    
    def __init__(self):
        super().__init__("TextExtractor")
    
    def process(self, file_path: str) -> List[TextBlock]:
        """Extract text blocks from document"""
        logger.info(f"Extracting text from {file_path}")
        
        # Mock OCR extraction - in real scenario, use pytesseract, paddleOCR, etc.
        text_blocks = [
            TextBlock(
                text="INVOICE #2024-001",
                confidence=0.95,
                is_bold=True,
                block_type="heading"
            ),
            TextBlock(
                text="Date: January 15, 2024",
                confidence=0.92,
                block_type="text"
            ),
            TextBlock(
                text="Customer Name: John Doe",
                confidence=0.88,
                block_type="text"
            ),
            TextBlock(
                text="Amount: $1,250.00",
                confidence=0.94,
                block_type="text"
            )
        ]
        
        return text_blocks


class EntityExtractor(BaseOCRProcessor):
    """Extracts named entities from documents"""
    
    def __init__(self):
        super().__init__("EntityExtractor")
        self.entity_patterns = {
            'email': r'[\w\.-]+@[\w\.-]+\.\w+',
            'phone': r'[\+]?[0-9]{1,3}?[-.\s]?[\(]?[0-9]{3}[\)]?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'currency': r'[\$£€¥]\s?[\d,]+\.?\d*',
            'url': r'https?://[^\s]+',
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_type] = list(set(matches))
        
        return entities


class FieldExtractor(BaseOCRProcessor):
    """Extracts specific fields based on document type"""
    
    def __init__(self):
        super().__init__("FieldExtractor")
    
    def extract_invoice_fields(self, text: str) -> Dict[str, ExtractedField]:
        """Extract invoice-specific fields"""
        fields = {}
        
        # Invoice Number
        invoice_match = re.search(r'(?:Invoice|INV)[#:\s]*([A-Z0-9\-]+)', text, re.IGNORECASE)
        if invoice_match:
            fields['invoice_number'] = ExtractedField(
                name='invoice_number',
                value=invoice_match.group(1),
                confidence=0.92,
                field_type='text',
                raw_text=invoice_match.group(0)
            )
        
        # Date
        date_match = re.search(r'(?:Date|DATE)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
        if date_match:
            fields['date'] = ExtractedField(
                name='date',
                value=date_match.group(1),
                confidence=0.95,
                field_type='date',
                raw_text=date_match.group(0)
            )
        
        # Amount
        amount_match = re.search(r'(?:Total|Amount)[:\s]*(\$?\s?[\d,]+\.?\d*)', text, re.IGNORECASE)
        if amount_match:
            amount_str = amount_match.group(1).replace('$', '').replace(',', '').strip()
            fields['amount'] = ExtractedField(
                name='amount',
                value=amount_str,
                confidence=0.94,
                field_type='currency',
                raw_text=amount_match.group(0),
                normalized_value=float(amount_str)
            )
        
        return fields
    
    def extract_receipt_fields(self, text: str) -> Dict[str, ExtractedField]:
        """Extract receipt-specific fields"""
        fields = {}
        
        # Merchant name
        merchant_match = re.search(r'(?:Merchant|Store)[:\s]*([^\n]+)', text, re.IGNORECASE)
        if merchant_match:
            fields['merchant'] = ExtractedField(
                name='merchant',
                value=merchant_match.group(1).strip(),
                confidence=0.85,
                field_type='text'
            )
        
        return fields


class DocumentQualityAnalyzer(BaseOCRProcessor):
    """Analyzes document quality"""
    
    def __init__(self):
        super().__init__("DocumentQualityAnalyzer")
    
    def analyze(self, text_blocks: List[TextBlock]) -> Tuple[float, List[str]]:
        """
        Analyze document quality
        Returns: (quality_score, issues)
        """
        issues = []
        scores = []
        
        # Check average confidence
        avg_confidence = sum(b.confidence for b in text_blocks) / len(text_blocks) if text_blocks else 0
        scores.append(avg_confidence)
        
        if avg_confidence < 0.7:
            issues.append("Low average confidence score")
        
        # Check for blank pages
        empty_blocks = sum(1 for b in text_blocks if len(b.text.strip()) == 0)
        if empty_blocks > len(text_blocks) * 0.3:
            issues.append("Too many empty text blocks")
        
        # Overall quality score
        quality_score = sum(scores) / len(scores) if scores else 0.5
        
        return quality_score, issues


# ============================================================================
# MAIN OCR SERVICE
# ============================================================================

class OCRService:
    """Main OCR Service for document processing"""
    
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.entity_extractor = EntityExtractor()
        self.field_extractor = FieldExtractor()
        self.quality_analyzer = DocumentQualityAnalyzer()
        self.processing_log = []
    
    def process_document(
        self,
        file_path: str,
        document_type: DocumentType = DocumentType.GENERAL,
        language: str = "en"
    ) -> Dict:
        """
        Main method to process a document
        
        Args:
            file_path: Path to the document
            document_type: Type of document
            language: Document language
        
        Returns:
            Complete OCR result as dictionary
        """
        logger.info(f"Starting OCR processing: {file_path}")
        
        process_start = datetime.now()
        
        try:
            # Get file metadata
            file_path_obj = Path(file_path)
            file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
            file_format = file_path_obj.suffix.lower()
            
            # Create document ID
            document_id = self._generate_document_id(file_path)
            
            # Extract text
            text_blocks = self.text_extractor.process(file_path)
            
            # Create page from text blocks
            page = Page(
                page_number=1,
                text_blocks=text_blocks,
                language=language
            )
            
            # Combine text
            full_text = " ".join([b.text for b in text_blocks])
            page.full_text = full_text
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(full_text)
            
            # Extract fields based on document type
            extracted_fields = self._extract_fields_by_type(full_text, document_type)
            
            # Analyze quality
            quality_score, issues = self.quality_analyzer.analyze(text_blocks)
            page.quality_score = quality_score
            
            # Calculate processing time
            processing_time = (datetime.now() - process_start).total_seconds()
            
            # Create metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                document_type=document_type,
                file_name=file_path_obj.name,
                file_size=file_size,
                file_format=file_format,
                upload_time=datetime.now().isoformat(),
                processing_time=processing_time,
                ocr_engine="CustomOCR",
                ocr_version="1.0.0",
                language=language,
                page_count=1,
                total_words=len(full_text.split()),
                confidence_score=quality_score
            )
            
            # Create OCR result
            result = OCRResult(
                metadata=metadata,
                pages=[page],
                extracted_fields=extracted_fields,
                entities=entities,
                warnings=issues
            )
            
            # Log processing
            self.processing_log.append({
                'document_id': document_id,
                'timestamp': datetime.now().isoformat(),
                'file_name': file_path_obj.name,
                'status': 'success'
            })
            
            logger.info(f"✓ Successfully processed {file_path}")
            
            return self._convert_to_json_structure(result)
            
        except Exception as e:
            logger.error(f"✗ Error processing {file_path}: {str(e)}")
            return self._error_response(str(e), file_path)
    
    def _extract_fields_by_type(
        self,
        text: str,
        document_type: DocumentType
    ) -> Dict[str, ExtractedField]:
        """Extract fields based on document type"""
        
        if document_type == DocumentType.INVOICE:
            return self.field_extractor.extract_invoice_fields(text)
        elif document_type == DocumentType.RECEIPT:
            return self.field_extractor.extract_receipt_fields(text)
        else:
            return {}
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID"""
        hash_input = f"{file_path}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _convert_to_json_structure(self, result: OCRResult) -> Dict:
        """Convert OCRResult to JSON-serializable dictionary"""
        
        return {
            "metadata": {
                "document_id": result.metadata.document_id,
                "document_type": result.metadata.document_type.value,
                "file_name": result.metadata.file_name,
                "file_size": result.metadata.file_size,
                "file_format": result.metadata.file_format,
                "upload_time": result.metadata.upload_time,
                "processing_time_seconds": round(result.metadata.processing_time, 2),
                "ocr_engine": result.metadata.ocr_engine,
                "ocr_version": result.metadata.ocr_version,
                "language": result.metadata.language,
                "page_count": result.metadata.page_count,
                "total_words": result.metadata.total_words,
                "confidence_score": round(result.metadata.confidence_score, 2)
            },
            "pages": [
                {
                    "page_number": page.page_number,
                    "language": page.language,
                    "quality_score": round(page.quality_score, 2),
                    "skew_angle": page.skew_angle,
                    "is_blank": page.is_blank,
                    "text_blocks": [
                        {
                            "text": block.text,
                            "confidence": round(block.confidence, 2),
                            "block_type": block.block_type,
                            "formatting": {
                                "is_bold": block.is_bold,
                                "is_italic": block.is_italic,
                                "font_size": block.font_size
                            },
                            "bounding_box": block.bounding_box
                        }
                        for block in page.text_blocks
                    ],
                    "tables": [
                        {
                            "id": table.id,
                            "headers": table.headers,
                            "rows": table.rows,
                            "col_count": table.col_count,
                            "row_count": table.row_count,
                            "confidence": round(table.confidence, 2)
                        }
                        for table in page.tables
                    ],
                    "images": [
                        {
                            "id": img.id,
                            "description": img.description,
                            "size": img.size,
                            "format": img.format,
                            "confidence": round(img.confidence, 2)
                        }
                        for img in page.images
                    ],
                    "full_text": page.full_text
                }
                for page in result.pages
            ],
            "extracted_fields": {
                field_name: {
                    "name": field.name,
                    "value": field.value,
                    "confidence": round(field.confidence, 2),
                    "field_type": field.field_type,
                    "raw_text": field.raw_text,
                    "normalized_value": field.normalized_value
                }
                for field_name, field in result.extracted_fields.items()
            },
            "entities": result.entities,
            "relationships": result.relationships,
            "warnings": result.warnings,
            "errors": result.errors
        }
    
    def _error_response(self, error: str, file_path: str) -> Dict:
        """Generate error response"""
        return {
            "status": "error",
            "error": error,
            "file": file_path,
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# JSON OUTPUT EXAMPLES
# ============================================================================

# Example 1: Invoice Document Output
INVOICE_OUTPUT_EXAMPLE = {
    "metadata": {
        "document_id": "a1b2c3d4e5f6g7h8",
        "document_type": "invoice",
        "file_name": "invoice_2024_001.pdf",
        "file_size": 245632,
        "file_format": ".pdf",
        "upload_time": "2024-01-15T10:30:45.123456",
        "processing_time_seconds": 2.45,
        "ocr_engine": "CustomOCR",
        "ocr_version": "1.0.0",
        "language": "en",
        "page_count": 2,
        "total_words": 1250,
        "confidence_score": 0.92
    },
    "pages": [
        {
            "page_number": 1,
            "language": "en",
            "quality_score": 0.94,
            "skew_angle": 0.5,
            "is_blank": False,
            "text_blocks": [
                {
                    "text": "INVOICE #2024-001",
                    "confidence": 0.98,
                    "block_type": "heading",
                    "formatting": {
                        "is_bold": True,
                        "is_italic": False,
                        "font_size": 24
                    },
                    "bounding_box": {
                        "x": 50,
                        "y": 20,
                        "width": 300,
                        "height": 40
                    }
                },
                {
                    "text": "Date: January 15, 2024",
                    "confidence": 0.95,
                    "block_type": "text",
                    "formatting": {
                        "is_bold": False,
                        "is_italic": False,
                        "font_size": 12
                    }
                },
                {
                    "text": "Bill To: John Doe, 123 Main St",
                    "confidence": 0.91,
                    "block_type": "text",
                    "formatting": {
                        "is_bold": False,
                        "is_italic": False,
                        "font_size": 12
                    }
                }
            ],
            "tables": [
                {
                    "id": "table_1",
                    "headers": ["Item", "Quantity", "Unit Price", "Total"],
                    "rows": [
                        ["Widget A", "5", "$100", "$500"],
                        ["Widget B", "3", "$250", "$750"],
                        ["Service", "1", "$0", "$0"]
                    ],
                    "col_count": 4,
                    "row_count": 3,
                    "confidence": 0.89
                }
            ],
            "images": [],
            "full_text": "INVOICE #2024-001 Date: January 15, 2024 Bill To: John Doe, 123 Main St..."
        },
        {
            "page_number": 2,
            "language": "en",
            "quality_score": 0.90,
            "skew_angle": 0.2,
            "is_blank": False,
            "text_blocks": [],
            "tables": [],
            "images": [],
            "full_text": "..."
        }
    ],
    "extracted_fields": {
        "invoice_number": {
            "name": "invoice_number",
            "value": "2024-001",
            "confidence": 0.97,
            "field_type": "text",
            "raw_text": "INVOICE #2024-001",
            "normalized_value": "2024-001"
        },
        "date": {
            "name": "date",
            "value": "01/15/2024",
            "confidence": 0.96,
            "field_type": "date",
            "raw_text": "Date: January 15, 2024",
            "normalized_value": "2024-01-15"
        },
        "amount": {
            "name": "amount",
            "value": "$1250.00",
            "confidence": 0.94,
            "field_type": "currency",
            "raw_text": "Total Amount: $1250.00",
            "normalized_value": 1250.00
        },
        "bill_to": {
            "name": "bill_to",
            "value": "John Doe, 123 Main St",
            "confidence": 0.88,
            "field_type": "text",
            "raw_text": "Bill To: John Doe, 123 Main St",
            "normalized_value": None
        }
    },
    "entities": {
        "email": ["john.doe@example.com"],
        "phone": ["+1-234-567-8900"],
        "date": ["01/15/2024"],
        "currency": ["$1250.00", "$500", "$750"],
        "url": []
    },
    "relationships": [
        {
            "type": "invoice_for",
            "subject": "invoice_2024_001",
            "object": "john_doe",
            "confidence": 0.85
        }
    ],
    "warnings": [],
    "errors": []
}


# Example 2: Receipt Document Output
RECEIPT_OUTPUT_EXAMPLE = {
    "metadata": {
        "document_id": "r1e2c3e4i5p6t7o8",
        "document_type": "receipt",
        "file_name": "receipt_starbucks_20240115.jpg",
        "file_size": 156234,
        "file_format": ".jpg",
        "upload_time": "2024-01-15T15:45:30.654321",
        "processing_time_seconds": 1.23,
        "ocr_engine": "CustomOCR",
        "ocr_version": "1.0.0",
        "language": "en",
        "page_count": 1,
        "total_words": 145,
        "confidence_score": 0.89
    },
    "pages": [
        {
            "page_number": 1,
            "language": "en",
            "quality_score": 0.88,
            "skew_angle": 1.2,
            "is_blank": False,
            "text_blocks": [
                {
                    "text": "STARBUCKS COFFEE",
                    "confidence": 0.94,
                    "block_type": "heading",
                    "formatting": {"is_bold": True, "is_italic": False, "font_size": 18}
                },
                {
                    "text": "Store #1234",
                    "confidence": 0.91,
                    "block_type": "text",
                    "formatting": {"is_bold": False, "is_italic": False, "font_size": 10}
                },
                {
                    "text": "2024-01-15 14:30",
                    "confidence": 0.96,
                    "block_type": "text",
                    "formatting": {"is_bold": False, "is_italic": False, "font_size": 10}
                },
                {
                    "text": "VENTI LATTE           $6.45",
                    "confidence": 0.87,
                    "block_type": "text",
                    "formatting": {"is_bold": False, "is_italic": False, "font_size": 10}
                },
                {
                    "text": "Total                 $6.45",
                    "confidence": 0.92,
                    "block_type": "text",
                    "formatting": {"is_bold": True, "is_italic": False, "font_size": 11}
                }
            ],
            "tables": [],
            "images": [],
            "full_text": "STARBUCKS COFFEE Store #1234 2024-01-15 14:30 VENTI LATTE $6.45..."
        }
    ],
    "extracted_fields": {
        "merchant": {
            "name": "merchant",
            "value": "Starbucks Coffee",
            "confidence": 0.96,
            "field_type": "text",
            "raw_text": "STARBUCKS COFFEE",
            "normalized_value": "Starbucks Coffee"
        },
        "store_id": {
            "name": "store_id",
            "value": "1234",
            "confidence": 0.93,
            "field_type": "text",
            "raw_text": "Store #1234",
            "normalized_value": "1234"
        },
        "datetime": {
            "name": "datetime",
            "value": "2024-01-15 14:30",
            "confidence": 0.97,
            "field_type": "datetime",
            "raw_text": "2024-01-15 14:30",
            "normalized_value": "2024-01-15T14:30:00"
        },
        "items": {
            "name": "items",
            "value": [{"name": "VENTI LATTE", "price": "$6.45"}],
            "confidence": 0.85,
            "field_type": "list",
            "raw_text": "VENTI LATTE $6.45",
            "normalized_value": None
        },
        "total": {
            "name": "total",
            "value": "$6.45",
            "confidence": 0.98,
            "field_type": "currency",
            "raw_text": "Total $6.45",
            "normalized_value": 6.45
        }
    },
    "entities": {
        "email": [],
        "phone": [],
        "date": ["2024-01-15"],
        "currency": ["$6.45"],
        "url": []
    },
    "relationships": [],
    "warnings": ["Some items may be partially obscured"],
    "errors": []
}


# Example 3: Passport Document Output
PASSPORT_OUTPUT_EXAMPLE = {
    "metadata": {
        "document_id": "p1a2s3s4p5o6r7t8",
        "document_type": "passport",
        "file_name": "passport_john_doe.pdf",
        "file_size": 567890,
        "file_format": ".pdf",
        "upload_time": "2024-01-15T09:15:20.345678",
        "processing_time_seconds": 3.45,
        "ocr_engine": "CustomOCR",
        "ocr_version": "1.0.0",
        "language": "en",
        "page_count": 1,
        "total_words": 89,
        "confidence_score": 0.88
    },
    "pages": [
        {
            "page_number": 1,
            "language": "en",
            "quality_score": 0.87,
            "skew_angle": 0.0,
            "is_blank": False,
            "text_blocks": [
                {
                    "text": "PASSPORT",
                    "confidence": 0.99,
                    "block_type": "heading",
                    "formatting": {"is_bold": True, "is_italic": False, "font_size": 20}
                },
                {
                    "text": "Given Names: JOHN",
                    "confidence": 0.94,
                    "block_type": "text",
                    "formatting": {"is_bold": False, "is_italic": False, "font_size": 12}
                },
                {
                    "text": "Surname: DOE",
                    "confidence": 0.97,
                    "block_type": "text",
                    "formatting": {"is_bold": False, "is_italic": False, "font_size": 12}
                }
            ],
            "tables": [],
            "images": [
                {
                    "id": "img_1",
                    "description": "Portrait photo",
                    "size": [150, 200],
                    "format": "jpg",
                    "confidence": 0.92
                }
            ],
            "full_text": "PASSPORT Given Names: JOHN Surname: DOE..."
        }
    ],
    "extracted_fields": {
        "first_name": {
            "name": "first_name",
            "value": "JOHN",
            "confidence": 0.96,
            "field_type": "text",
            "raw_text": "Given Names: JOHN",
            "normalized_value": "John"
        },
        "last_name": {
            "name": "last_name",
            "value": "DOE",
            "confidence": 0.98,
            "field_type": "text",
            "raw_text": "Surname: DOE",
            "normalized_value": "Doe"
        },
        "passport_number": {
            "name": "passport_number",
            "value": "AB123456",
            "confidence": 0.91,
            "field_type": "text",
            "raw_text": "Passport: AB123456",
            "normalized_value": "AB123456"
        },
        "issue_date": {
            "name": "issue_date",
            "value": "01/15/2019",
            "confidence": 0.94,
            "field_type": "date",
            "raw_text": "Date of Issue: 01/15/2019",
            "normalized_value": "2019-01-15"
        },
        "expiry_date": {
            "name": "expiry_date",
            "value": "01/15/2029",
            "confidence": 0.94,
            "field_type": "date",
            "raw_text": "Date of Expiry: 01/15/2029",
            "normalized_value": "2029-01-15"
        },
        "nationality": {
            "name": "nationality",
            "value": "United States",
            "confidence": 0.92,
            "field_type": "text",
            "raw_text": "Nationality: United States",
            "normalized_value": "US"
        }
    },
    "entities": {
        "email": [],
        "phone": [],
        "date": ["01/15/2019", "01/15/2029"],
        "currency": [],
        "url": []
    },
    "relationships": [],
    "warnings": [],
    "errors": []
}


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_ocr():
    """Example 1: Basic OCR processing"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic OCR Processing")
    print("="*70)
    
    ocr_service = OCRService()
    
    # Process invoice
    result = ocr_service.process_document(
        file_path="/path/to/invoice.pdf",
        document_type=DocumentType.INVOICE,
        language="en"
    )
    
    # Print result
    print("\n✓ Invoice Processing Result:")
    print(json.dumps(result, indent=2)[:500] + "...")


def example_multiple_document_types():
    """Example 2: Processing multiple document types"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Document Types")
    print("="*70)
    
    ocr_service = OCRService()
    
    documents = [
        ("/path/to/invoice.pdf", DocumentType.INVOICE),
        ("/path/to/receipt.jpg", DocumentType.RECEIPT),
        ("/path/to/passport.pdf", DocumentType.PASSPORT),
    ]
    
    results = []
    for file_path, doc_type in documents:
        result = ocr_service.process_document(file_path, doc_type)
        results.append(result)
        print(f"✓ Processed: {file_path}")
    
    print(f"\nTotal documents processed: {len(results)}")


def example_json_output_structures():
    """Example 3: Show JSON output structures"""
    print("\n" + "="*70)
    print("EXAMPLE 3: JSON Output Structure Examples")
    print("="*70)
    
    # Invoice
    print("\n--- INVOICE OUTPUT STRUCTURE ---")
    print(json.dumps(INVOICE_OUTPUT_EXAMPLE, indent=2)[:1000] + "...\n")
    
    # Receipt
    print("--- RECEIPT OUTPUT STRUCTURE ---")
    print(json.dumps(RECEIPT_OUTPUT_EXAMPLE, indent=2)[:800] + "...\n")
    
    # Passport
    print("--- PASSPORT OUTPUT STRUCTURE ---")
    print(json.dumps(PASSPORT_OUTPUT_EXAMPLE, indent=2)[:800] + "...\n")


def example_field_extraction():
    """Example 4: Field extraction showcase"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Extracted Fields from Invoice")
    print("="*70)
    
    invoice_fields = INVOICE_OUTPUT_EXAMPLE["extracted_fields"]
    
    print("\nExtracted Fields:")
    for field_name, field_data in invoice_fields.items():
        print(f"\n{field_name}:")
        print(f"  Value: {field_data['value']}")
        print(f"  Confidence: {field_data['confidence']}")
        print(f"  Type: {field_data['field_type']}")
        print(f"  Raw Text: {field_data['raw_text']}")


def example_entity_extraction():
    """Example 5: Entity extraction showcase"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Extracted Entities")
    print("="*70)
    
    entities = INVOICE_OUTPUT_EXAMPLE["entities"]
    
    print("\nDetected Entities:")
    for entity_type, values in entities.items():
        if values:
            print(f"\n{entity_type.upper()}:")
            for value in values:
                print(f"  - {value}")


def example_error_handling():
    """Example 6: Error handling"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Error Handling")
    print("="*70)
    
    ocr_service = OCRService()
    
    # Process non-existent file
    result = ocr_service.process_document(
        file_path="/path/to/nonexistent.pdf",
        document_type=DocumentType.INVOICE
    )
    
    print("\nError Response:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CUSTOM OCR SERVICE - DOCUMENT PROCESSING")
    print("="*70)
    
    # Run examples
    example_json_output_structures()
    example_field_extraction()
    example_entity_extraction()
    example_basic_ocr()
    example_multiple_document_types()
    example_error_handling()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)