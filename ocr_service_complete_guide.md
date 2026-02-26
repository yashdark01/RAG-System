# Custom OCR Service - Complete Guide & Implementation

## Table of Contents
1. [Architecture](#architecture)
2. [JSON Output Formats](#json-output-formats)
3. [Real OCR Engine Integrations](#real-ocr-engine-integrations)
4. [Implementation Details](#implementation-details)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)

---

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    OCR SERVICE                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Document    │───→│  Processors  │───→│   Output     │  │
│  │  Ingestion   │    │              │    │  Formatting  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                     │          │
│         ├─ File Validation  ├─ Text Extractor   ├─ JSON    │
│         ├─ Format Detection │ ├─ Entity Extractor  ├─ XML   │
│         └─ Metadata Extract │ ├─ Field Extractor  └─ CSV    │
│                             ├─ Quality Analyzer              │
│                             └─ Relationship Mapper           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
PDF/Image
   ↓
[Preprocessing] - Remove noise, rotate, enhance
   ↓
[OCR Engine] - Extract text with coordinates
   ↓
[Text Processing] - Clean, normalize, structure
   ↓
[Entity Recognition] - Find emails, dates, amounts
   ↓
[Field Extraction] - Extract domain-specific fields
   ↓
[Quality Analysis] - Confidence scoring, issue detection
   ↓
[Output Formatting] - JSON, XML, CSV
   ↓
[Structured Data]
```

---

## JSON Output Formats

### Format 1: Standard Invoice Output

```json
{
  "metadata": {
    "document_id": "string",
    "document_type": "invoice|receipt|passport|...",
    "file_name": "string",
    "file_size": number,
    "file_format": ".pdf|.jpg|.png|...",
    "upload_time": "ISO 8601 timestamp",
    "processing_time_seconds": number,
    "ocr_engine": "string",
    "ocr_version": "string",
    "language": "en|fr|es|...",
    "page_count": number,
    "total_words": number,
    "confidence_score": number (0-1)
  },
  "pages": [
    {
      "page_number": number,
      "language": "string",
      "quality_score": number (0-1),
      "skew_angle": number (degrees),
      "is_blank": boolean,
      "text_blocks": [
        {
          "text": "string",
          "confidence": number (0-1),
          "block_type": "heading|text|list|table_cell|...",
          "formatting": {
            "is_bold": boolean,
            "is_italic": boolean,
            "font_size": number
          },
          "bounding_box": {
            "x": number,
            "y": number,
            "width": number,
            "height": number
          }
        }
      ],
      "tables": [
        {
          "id": "string",
          "headers": ["col1", "col2", ...],
          "rows": [["cell1", "cell2"], ...],
          "col_count": number,
          "row_count": number,
          "confidence": number (0-1)
        }
      ],
      "images": [
        {
          "id": "string",
          "description": "string",
          "size": [width, height],
          "format": "jpg|png|...",
          "confidence": number (0-1)
        }
      ],
      "full_text": "string - entire page text"
    }
  ],
  "extracted_fields": {
    "field_name": {
      "name": "string",
      "value": "any",
      "confidence": number (0-1),
      "field_type": "text|number|date|currency|...",
      "raw_text": "original extracted text",
      "normalized_value": "any - standardized value"
    }
  },
  "entities": {
    "email": ["string"],
    "phone": ["string"],
    "date": ["string"],
    "currency": ["string"],
    "url": ["string"],
    "person_name": ["string"],
    "organization": ["string"],
    "location": ["string"]
  },
  "relationships": [
    {
      "type": "string - relationship type",
      "subject": "string",
      "object": "string",
      "confidence": number (0-1)
    }
  ],
  "warnings": ["string - issues that don't prevent processing"],
  "errors": ["string - critical errors if any"]
}
```

### Format 2: Compact Output (for simple use cases)

```json
{
  "document_id": "string",
  "success": boolean,
  "extracted_data": {
    "field_name": "value"
  },
  "entities": {},
  "raw_text": "string",
  "confidence": number,
  "processing_time_ms": number,
  "errors": []
}
```

### Format 3: Structured Data Output (for database storage)

```json
{
  "id": "document_id",
  "type": "invoice|receipt|...",
  "source": {
    "file_name": "string",
    "upload_time": "ISO timestamp",
    "file_format": "string"
  },
  "content": {
    "text": "full document text",
    "pages": number,
    "language": "string"
  },
  "structured_data": {
    "invoice_number": "value",
    "date": "value",
    "amount": "value",
    "vendor": "value",
    "line_items": [
      {"description": "string", "quantity": number, "price": number}
    ]
  },
  "quality": {
    "overall_score": number,
    "issues": ["string"],
    "warnings": ["string"]
  },
  "processing": {
    "ocr_engine": "string",
    "processing_time": number,
    "version": "string"
  }
}
```

---

## Real OCR Engine Integrations

### Option 1: Tesseract (Free, Open Source)

```python
import pytesseract
from PIL import Image

class TesseractOCR:
    """Tesseract-based OCR processor"""
    
    def __init__(self):
        self.engine = "tesseract"
        self.version = pytesseract.get_tesseract_version()
    
    def extract_text(self, image_path):
        """Extract text from image"""
        img = Image.open(image_path)
        
        # Preprocess
        img = self._preprocess(img)
        
        # Extract text with coordinates
        data = pytesseract.image_to_data(img, output_type='dict')
        
        text_blocks = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                text_blocks.append({
                    'text': data['text'][i],
                    'confidence': int(data['conf'][i]) / 100.0,
                    'bbox': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                })
        
        return text_blocks
    
    def _preprocess(self, img):
        """Preprocess image for better OCR"""
        # Convert to grayscale
        img = img.convert('L')
        
        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        
        # Resize if too small
        if img.size[0] < 300:
            img = img.resize((img.size[0] * 2, img.size[1] * 2))
        
        return img
```

**Installation:**
```bash
pip install pytesseract pillow
# Also need to install tesseract-ocr binary
# On Ubuntu: sudo apt-get install tesseract-ocr
# On macOS: brew install tesseract
# On Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

### Option 2: PaddleOCR (Better accuracy, supports multiple languages)

```python
from paddleocr import PaddleOCR

class PaddleOCREngine:
    """PaddleOCR-based processor"""
    
    def __init__(self, language='en'):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=language)
        self.language = language
    
    def extract_text(self, image_path):
        """Extract text from image"""
        result = self.ocr.ocr(image_path, cls=True)
        
        text_blocks = []
        for line in result:
            for word_info in line:
                coords, (text, confidence) = word_info
                
                # Convert coordinates
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]
                
                text_blocks.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': {
                        'x': min(x_coords),
                        'y': min(y_coords),
                        'width': max(x_coords) - min(x_coords),
                        'height': max(y_coords) - min(y_coords)
                    }
                })
        
        return text_blocks
    
    def extract_tables(self, image_path):
        """Extract tables from image"""
        # Use table detection model
        from paddleocr import PaddleOCR
        table_ocr = PaddleOCR(use_gpu=True)
        result = table_ocr.ocr(image_path)
        # Process table structure...
        return []
```

**Installation:**
```bash
pip install paddleocr paddlepaddle
```

### Option 3: EasyOCR (Fast, good accuracy)

```python
import easyocr

class EasyOCREngine:
    """EasyOCR-based processor"""
    
    def __init__(self, language=['en']):
        self.reader = easyocr.Reader(language, gpu=True)
    
    def extract_text(self, image_path):
        """Extract text from image"""
        result = self.reader.readtext(image_path)
        
        text_blocks = []
        for detection in result:
            bbox, text, confidence = detection
            
            # Convert bbox format
            x_coords = [coord[0] for coord in bbox]
            y_coords = [coord[1] for coord in bbox]
            
            text_blocks.append({
                'text': text,
                'confidence': confidence,
                'bbox': {
                    'x': min(x_coords),
                    'y': min(y_coords),
                    'width': max(x_coords) - min(x_coords),
                    'height': max(y_coords) - min(y_coords)
                }
            })
        
        return text_blocks
```

**Installation:**
```bash
pip install easyocr torch
```

### Option 4: Cloud Services

#### Google Cloud Vision API

```python
from google.cloud import vision

class GoogleVisionOCR:
    """Google Cloud Vision OCR processor"""
    
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
    
    def extract_text(self, image_path):
        """Extract text using Google Cloud Vision"""
        with open(image_path, 'rb') as f:
            content = f.read()
        
        image = vision.Image(content=content)
        response = self.client.document_text_detection(image=image)
        
        text_blocks = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        text = ''.join([s.text for s in word.symbols])
                        confidence = word.confidence
                        
                        text_blocks.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': self._convert_bbox(word.bounding_box)
                        })
        
        return text_blocks
    
    def _convert_bbox(self, bounding_box):
        vertices = bounding_box.vertices
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        
        return {
            'x': min(x_coords),
            'y': min(y_coords),
            'width': max(x_coords) - min(x_coords),
            'height': max(y_coords) - min(y_coords)
        }
```

**Installation:**
```bash
pip install google-cloud-vision
# Need to set up Google Cloud credentials
```

#### AWS Textract

```python
import boto3

class AWSTextractOCR:
    """AWS Textract OCR processor"""
    
    def __init__(self, region='us-east-1'):
        self.client = boto3.client('textract', region_name=region)
    
    def extract_text(self, image_path):
        """Extract text using AWS Textract"""
        with open(image_path, 'rb') as f:
            document_bytes = f.read()
        
        response = self.client.detect_document_text(
            Document={'Bytes': document_bytes}
        )
        
        text_blocks = []
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                text_blocks.append({
                    'text': block['Text'],
                    'confidence': block['Confidence'] / 100.0,
                    'bbox': self._convert_bbox(block['Geometry'])
                })
        
        return text_blocks
    
    def _convert_bbox(self, geometry):
        bbox = geometry['BoundingBox']
        return {
            'x': int(bbox['Left'] * 100),
            'y': int(bbox['Top'] * 100),
            'width': int(bbox['Width'] * 100),
            'height': int(bbox['Height'] * 100)
        }
```

---

## Implementation Details

### Complete OCR Pipeline

```python
class CompleteOCRPipeline:
    """Complete OCR processing pipeline"""
    
    def __init__(self, ocr_engine='paddle'):
        self.ocr_engine = self._init_engine(ocr_engine)
        self.preprocessor = ImagePreprocessor()
        self.text_processor = TextProcessor()
        self.entity_extractor = EntityExtractor()
        self.field_extractor = FieldExtractor()
        self.validator = DataValidator()
    
    def process(self, file_path, document_type='general'):
        """Complete processing pipeline"""
        
        # Step 1: Validate and prepare file
        validated_path = self.preprocessor.validate_and_prepare(file_path)
        
        # Step 2: Extract text using OCR
        text_blocks = self.ocr_engine.extract_text(validated_path)
        
        # Step 3: Process and structure text
        processed_text = self.text_processor.process(text_blocks)
        
        # Step 4: Extract entities
        entities = self.entity_extractor.extract(processed_text)
        
        # Step 5: Extract domain-specific fields
        fields = self.field_extractor.extract(processed_text, document_type)
        
        # Step 6: Validate extracted data
        validation_result = self.validator.validate(fields, document_type)
        
        # Step 7: Generate output
        return self._format_output(
            text_blocks, processed_text, entities, fields, validation_result
        )
    
    def _init_engine(self, engine_name):
        if engine_name == 'paddle':
            return PaddleOCREngine()
        elif engine_name == 'tesseract':
            return TesseractOCR()
        elif engine_name == 'easyocr':
            return EasyOCREngine()
        else:
            raise ValueError(f"Unknown engine: {engine_name}")
    
    def _format_output(self, text_blocks, processed_text, entities, fields, validation):
        return {
            'metadata': {...},
            'pages': [...],
            'extracted_fields': fields,
            'entities': entities,
            'validation': validation,
            'warnings': [],
            'errors': []
        }
```

---

## Advanced Features

### 1. Table Detection & Extraction

```python
class TableDetector:
    """Detect and extract tables from documents"""
    
    def detect_tables(self, image_path):
        """Detect table regions in image"""
        # Use models like TableNet or CascadeTabNet
        pass
    
    def extract_table_structure(self, table_region):
        """Extract table structure (rows, columns)"""
        # Parse table cells and content
        pass
```

### 2. Document Layout Analysis

```python
class LayoutAnalyzer:
    """Analyze document layout and structure"""
    
    def analyze_layout(self, text_blocks):
        """Understand document structure"""
        
        regions = {
            'header': [],
            'body': [],
            'footer': [],
            'sidebar': []
        }
        
        # Classify regions based on position
        for block in text_blocks:
            y_pos = block['bbox']['y']
            
            if y_pos < 100:
                regions['header'].append(block)
            elif y_pos > 900:
                regions['footer'].append(block)
            else:
                regions['body'].append(block)
        
        return regions
```

### 3. Handwriting Detection

```python
class HandwritingDetector:
    """Detect and handle handwritten text"""
    
    def detect_handwriting(self, text_blocks):
        """Identify handwritten text"""
        
        handwritten = []
        for block in text_blocks:
            if self._is_handwritten(block):
                handwritten.append(block)
                block['confidence'] *= 0.7  # Reduce confidence
        
        return handwritten
    
    def _is_handwritten(self, block):
        """Use ML to detect if text is handwritten"""
        # Use handwriting detection model
        return False
```

### 4. Language Detection

```python
from langdetect import detect, detect_langs

class LanguageDetector:
    """Detect languages in document"""
    
    def detect_language(self, text):
        """Detect language of text"""
        try:
            lang = detect(text)
            probabilities = detect_langs(text)
            return {
                'primary': lang,
                'probabilities': {str(p).split(':')[0]: float(str(p).split(':')[1]) 
                                 for p in probabilities}
            }
        except:
            return {'primary': 'unknown', 'probabilities': {}}
```

### 5. Document Quality Scoring

```python
class QualityScorer:
    """Score document quality for OCR"""
    
    def score_quality(self, image):
        """Calculate quality score (0-1)"""
        
        scores = {}
        
        # Brightness
        scores['brightness'] = self._check_brightness(image)
        
        # Contrast
        scores['contrast'] = self._check_contrast(image)
        
        # Sharpness
        scores['sharpness'] = self._check_sharpness(image)
        
        # Skew
        scores['skew'] = self._check_skew(image)
        
        # Overall
        overall = sum(scores.values()) / len(scores)
        
        return overall, scores
    
    def _check_brightness(self, image):
        import numpy as np
        brightness = np.mean(image) / 255.0
        # Optimal brightness around 0.5
        return 1.0 - abs(brightness - 0.5)
```

---

## Best Practices

### 1. Error Handling

```python
class RobustOCRService:
    """OCR service with robust error handling"""
    
    def process_with_fallback(self, file_path, primary_engine='paddle'):
        """Process with fallback to other engines"""
        
        engines = ['paddle', 'easyocr', 'tesseract']
        
        for engine in engines:
            try:
                ocr = self._get_engine(engine)
                result = ocr.process(file_path)
                result['engine_used'] = engine
                return result
            except Exception as e:
                logger.warning(f"{engine} failed: {e}")
                continue
        
        # All engines failed
        return {'error': 'All OCR engines failed', 'file': file_path}
```

### 2. Caching

```python
import hashlib

class CachedOCRService:
    """OCR service with result caching"""
    
    def __init__(self):
        self.cache = {}
    
    def process(self, file_path):
        """Process with caching"""
        
        # Generate cache key
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Check cache
        if file_hash in self.cache:
            logger.info("Using cached result")
            return self.cache[file_hash]
        
        # Process
        result = self.ocr_engine.process(file_path)
        
        # Cache result
        self.cache[file_hash] = result
        
        return result
```

### 3. Batch Processing

```python
import concurrent.futures

class BatchOCRService:
    """Process multiple documents efficiently"""
    
    def process_batch(self, file_paths, max_workers=4):
        """Process multiple documents in parallel"""
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_single, path): path 
                for path in file_paths
            }
            
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
        
        return results
```

### 4. Monitoring & Logging

```python
import logging

class MonitoredOCRService:
    """OCR service with monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger('OCRService')
        self.metrics = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'avg_time': 0,
            'avg_confidence': 0
        }
    
    def process(self, file_path):
        """Process with monitoring"""
        
        start_time = time.time()
        
        try:
            result = self.ocr_engine.process(file_path)
            
            # Update metrics
            self.metrics['total_processed'] += 1
            self.metrics['successful'] += 1
            
            processing_time = time.time() - start_time
            self.logger.info(f"Processed {file_path} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.metrics['failed'] += 1
            self.logger.error(f"Failed to process {file_path}: {e}")
            raise
```

---

## Configuration Management

```python
from dataclasses import dataclass

@dataclass
class OCRConfig:
    """OCR service configuration"""
    
    # Engine settings
    engine: str = "paddle"  # paddle, tesseract, easyocr
    language: str = "en"
    use_gpu: bool = True
    
    # Processing settings
    preprocessing: bool = True
    deskew: bool = True
    denoise: bool = True
    
    # Quality settings
    min_confidence: float = 0.5
    quality_threshold: float = 0.6
    
    # Performance settings
    max_workers: int = 4
    cache_enabled: bool = True
    batch_size: int = 10
    
    # Output settings
    output_format: str = "json"  # json, xml, csv
    include_images: bool = False
    include_coordinates: bool = True
```

---

## Comparison: OCR Engines

| Feature | Tesseract | PaddleOCR | EasyOCR | Google Vision | AWS Textract |
|---------|-----------|-----------|---------|--------------|--------------|
| Cost | Free | Free | Free | $$$ | $$$ |
| Setup | Local | Local | Local | Cloud | Cloud |
| Accuracy | Good | Excellent | Excellent | Excellent | Excellent |
| Speed | Fast | Fast | Medium | Slow | Slow |
| Languages | 100+ | 80+ | 80+ | 100+ | 100+ |
| Tables | No | Yes | No | Yes | Yes |
| Handwriting | No | No | No | Yes | Yes |
| Offline | Yes | Yes | Yes | No | No |

---

## Example: FastAPI Integration (RAG Backend)

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

def process_invoice_with_ocr():
    with open("invoice.pdf", "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/ocr/process",
            files={"file": f},
            params={"document_type": "invoice"},
        )
    resp.raise_for_status()
    result = resp.json()
    print("✓ OCR complete")
    print("Document ID:", result["document_id"])
    print("Confidence:", result["confidence"])
    return result


def process_invoice_with_ocr_and_llm():
    with open("invoice.pdf", "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/ocr/llm-extract",
            files={"file": f},
            params={
                "document_type": "invoice",
                "task": "Extract and normalize all key invoice fields.",
            },
        )
    resp.raise_for_status()
    result = resp.json()
    print("✓ OCR + LLM extraction complete")
    print("LLM fields:", result["llm_result"].get("fields"))
    return result


if __name__ == "__main__":
    ocr_only = process_invoice_with_ocr()
    ocr_and_llm = process_invoice_with_ocr_and_llm()
```

Good luck with your OCR service! 🚀