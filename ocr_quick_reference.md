# OCR Service - Quick Reference & Cheat Sheet

## 🚀 Quick Start

### Installation & Setup

```bash
# Install requirements
pip install paddlepaddle paddleocr pytesseract pillow easyocr

# Basic usage
from custom_ocr_service import OCRService
ocr = OCRService()
result = ocr.process_document("invoice.pdf", DocumentType.INVOICE)
```

### Minimal Example

```python
from production_ocr_wrapper import ProductionOCRService

# Create service
ocr = ProductionOCRService()

# Process document
result = ocr.process_document("document.pdf", "invoice")

# Get results
print(f"Confidence: {result['confidence']}")
print(f"Fields: {result['extracted_fields']}")
print(f"Entities: {result['entities']}")
```

---

## 📊 JSON Output Structure at a Glance

```
Result
├── metadata (document info)
│   ├── document_id
│   ├── document_type
│   ├── processing_time_seconds
│   └── confidence_score
├── pages (page-by-page content)
│   ├── text_blocks
│   ├── tables
│   └── images
├── extracted_fields (structured data)
│   └── field_name: {value, confidence, type}
├── entities (recognized entities)
│   ├── emails
│   ├── phones
│   ├── dates
│   └── amounts
└── warnings/errors
```

---

## 🎯 Document Types & Extracted Fields

### Invoice
```python
DocumentType.INVOICE

Extracted Fields:
- invoice_number
- date
- amount / total
- bill_to
- line_items
- tax
```

### Receipt
```python
DocumentType.RECEIPT

Extracted Fields:
- merchant
- datetime
- items
- total
- payment_method
- transaction_id
```

### Passport
```python
DocumentType.PASSPORT

Extracted Fields:
- first_name
- last_name
- passport_number
- date_of_birth
- nationality
- date_of_expiry
```

### General
```python
DocumentType.GENERAL

Extracted Fields:
- full_text
- entities only
- no specific structure
```

---

## 🔧 Configuration Quick Ref

### Development
```python
config = {
    'primary_engine': 'tesseract',  # Fast
    'cache_enabled': False,
    'min_confidence': 0.5,
}
```

### Production
```python
config = {
    'primary_engine': 'paddle',  # Accurate
    'fallback_engines': ['easyocr'],
    'cache_enabled': True,
    'min_confidence': 0.75,
    'max_workers': 8,
}
```

---

## 📝 API Endpoints Quick Ref

These are the FastAPI endpoints exposed by the RAG backend.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/ocr/process` | POST | Process single document with OCR |
| `/api/ocr/batch` | POST | Process multiple documents |
| `/api/ocr/status/{document_id}` | GET | Check processing status by document id |
| `/api/ocr/metrics` | GET | Get OCR service metrics |
| `/api/ocr/health` | GET | OCR health check |
| `/api/ocr/cache/clear` | POST | Clear OCR cache |
| `/api/ocr/llm-extract` | POST | OCR + LLM-based structured extraction |

### FastAPI Usage Examples

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# 1) Pure OCR processing
with open("invoice.pdf", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/api/ocr/process",
        files={"file": f},
        params={"document_type": "invoice"},
    )
result = resp.json()
print(result["confidence"], result["extracted_fields"])

# 2) OCR + LLM-based extraction
with open("invoice.pdf", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/api/ocr/llm-extract",
        files={"file": f},
        params={
            "document_type": "invoice",
            "task": "Extract all key invoice fields and normalize them.",
        },
    )
enriched = resp.json()
print(enriched["llm_result"])
```

---

## 🐛 Common Patterns

### Pattern 1: Basic Processing
```python
ocr = ProductionOCRService()
result = ocr.process_document("file.pdf", "invoice")
if result['status'] == 'success':
    fields = result['extracted_fields']
    print(fields)
```

### Pattern 2: Batch Processing
```python
api = OCRServiceAPI()
results = api.batch_process([
    "doc1.pdf", "doc2.pdf", "doc3.pdf"
])
print(f"Processed: {results['successful']}/{results['total_documents']}")
```

### Pattern 3: With Retry
```python
for attempt in range(3):
    try:
        result = ocr.process_document(file_path)
        break
    except Exception as e:
        if attempt < 2:
            time.sleep(2 ** attempt)
        else:
            raise
```

### Pattern 4: With Caching
```python
# Check cache first
cached = ocr._get_from_cache(file_path)
if cached:
    result = cached
else:
    result = ocr.process_document(file_path)
```

### Pattern 5: Custom Processing
```python
class CustomOCR(ProductionOCRService):
    def process_document(self, file_path, document_type):
        result = super().process_document(file_path, document_type)
        # Add custom logic
        result['custom_field'] = "custom_value"
        return result
```

---

## 🎓 OCR Engine Comparison

```
Tesseract      ████████░░ Speed, Free
PaddleOCR      ██████████ Accuracy, Free
EasyOCR        █████████░ Balance, Free
Google Vision  ██████████ Best, $$
AWS Textract   ██████████ Best, $$
```

**Choose:**
- **Tesseract**: Budget project, local only
- **Paddle**: General purpose, good accuracy
- **EasyOCR**: Need fast + accurate
- **Google/AWS**: Enterprise, critical accuracy

---

## ⚙️ Preprocessing & Enhancement

```python
# The service does this automatically:
1. Image deskewing (rotation correction)
2. Contrast enhancement
3. Noise reduction
4. Brightness normalization
5. Size optimization

# Manual preprocessing if needed:
from PIL import Image, ImageEnhance

img = Image.open('doc.jpg')

# Enhance contrast
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(2)

# Enhance brightness
enhancer = ImageEnhance.Brightness(img)
img = enhancer.enhance(1.2)
```

---

## 🔍 Entity Types Extracted

```json
{
  "emails": ["user@example.com"],
  "phones": ["+1-234-567-8900"],
  "dates": ["01/15/2024"],
  "amounts": ["$1000.00"],
  "urls": ["https://example.com"],
  "person_names": ["John Doe"],
  "organizations": ["Company Inc"],
  "locations": ["New York, USA"]
}
```

---

## ⚠️ Error Codes & Solutions

| Code | Cause | Solution |
|------|-------|----------|
| `FILE_NOT_FOUND` | File doesn't exist | Check file path |
| `UNSUPPORTED_FORMAT` | Wrong file type | Use PDF/JPG/PNG |
| `INVALID_FILE` | Corrupted file | Re-scan/re-upload |
| `OCR_FAILED` | Processing error | Retry or use fallback |
| `TIMEOUT` | Taking too long | Reduce image size |
| `INSUFFICIENT_CONFIDENCE` | Low accuracy | Improve image quality |

---

## 📈 Performance Tips

### Tip 1: Image Quality
```
✓ Good (>300 DPI, clear text)
⚠ Medium (200-300 DPI)
✗ Bad (<200 DPI, blurry)
```

### Tip 2: Caching
```python
# Cache improves performance 36-50%
config['cache_enabled'] = True
# Especially useful for:
# - Duplicate submissions
# - Batch processing
# - Testing
```

### Tip 3: Parallel Processing
```python
# Process 4 documents simultaneously
api.batch_process(files, max_workers=4)
# Faster than one-by-one
```

### Tip 4: Engine Selection
```python
# For speed: tesseract (0.5-1s per page)
# For balance: easyocr (1-2s per page)
# For accuracy: paddle (2-3s per page)
```

---

## 🧪 Testing

### Test Different Document Types
```python
test_docs = [
    ("invoice.pdf", "invoice"),
    ("receipt.jpg", "receipt"),
    ("passport.pdf", "passport"),
]

for doc, doc_type in test_docs:
    result = ocr.process_document(doc, doc_type)
    assert result['status'] == 'success'
    assert result['confidence'] > 0.7
```

### Test with Low Quality Images
```python
# Your OCR should handle:
- Blurry images
- Skewed documents
- Low contrast
- Handwriting
- Multiple languages
```

### Test Error Handling
```python
# Test these error cases:
- Non-existent file
- Unsupported format
- Corrupted file
- Empty file
- Timeout scenario
```

---

## 📊 Monitoring Checklist

```
Service Health:
□ Processing time < 5s per page
□ Success rate > 90%
□ Avg confidence > 0.85
□ Cache hit rate > 30%
□ Error rate < 5%
□ Uptime > 99%

Document Quality:
□ Average confidence > 0.8
□ Most fields extracted
□ Minimal warnings
□ Consistent results

System Performance:
□ Memory usage stable
□ CPU usage reasonable
□ Disk space available
□ Network connectivity OK
```

---

## 🚨 Troubleshooting

### Issue: Low Confidence
```
Solution:
1. Check image quality (brightness, contrast)
2. Try different OCR engine
3. Use preprocessing
4. Verify document type selection
```

### Issue: Slow Processing
```
Solution:
1. Reduce image resolution
2. Use faster engine (tesseract)
3. Enable caching
4. Use parallel processing
```

### Issue: Missing Fields
```
Solution:
1. Check document type is correct
2. Verify image quality
3. Try manual field extraction
4. Adjust min_confidence threshold
```

### Issue: OCR Engine Crashes
```
Solution:
1. Check available memory
2. Reduce batch size
3. Update library versions
4. Check file permissions
```

---

## 🔐 Security Best Practices

```python
# 1. Validate file uploads
if not validate_file(file_path):
    return error("Invalid file")

# 2. Set file size limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
if get_file_size(file) > MAX_FILE_SIZE:
    return error("File too large")

# 3. Sanitize file paths
file_path = sanitize_path(file_path)

# 4. Set timeout
result = process_with_timeout(file_path, timeout=60)

# 5. Clean up sensitive data
result = remove_sensitive_fields(result)
```

---

## 🎯 When to Use Each Feature

### Use Basic OCR When:
- Just need extracted text
- Document quality is good
- Speed is priority

### Use Field Extraction When:
- Processing invoices/receipts
- Need structured data
- Want to avoid manual parsing

### Use Entity Extraction When:
- Need contact information
- Want dates, amounts, URLs
- Searching documents

### Use Batch Processing When:
- Processing 10+ documents
- Time is not critical
- Want cost savings

### Use Caching When:
- Processing same documents often
- High traffic expected
- Server resources limited

---

## 📚 Further Learning

### Resources
- **Tesseract Docs**: https://github.com/UB-Mannheim/tesseract
- **PaddleOCR Docs**: https://github.com/PaddlePaddle/PaddleOCR
- **EasyOCR Docs**: https://github.com/JaidedAI/EasyOCR

### Next Steps
1. Set up local OCR service
2. Test with sample documents
3. Integrate with your app
4. Monitor performance
5. Tune for your use case

---

## 💡 Pro Tips

### Tip 1: Rotate for Better OCR
```python
# Some engines work better on upright images
# Detect and rotate if needed
angle = detect_rotation(image)
if angle != 0:
    image = rotate_image(image, angle)
```

### Tip 2: Split Large Documents
```python
# Process one page at a time
# Faster and more reliable
for page in split_pdf("large_doc.pdf"):
    result = ocr.process(page)
```

### Tip 3: Use Fallback Strategies
```python
# If low confidence, try:
# 1. Different engine
# 2. Preprocessing
# 3. Manual review flag
```

### Tip 4: Extract Confidence by Field
```python
# Don't just check overall confidence
# Check confidence per extracted field
for field, data in result['extracted_fields'].items():
    if data['confidence'] < 0.6:
        flag_for_review(field, data)
```

### Tip 5: Monitor Processing Time
```python
# Track which document types are slow
if result['processing_time'] > 5:
    log_slow_processing(result['file_name'], result['processing_time'])
```

---

## 🎓 Quick Comparison: Which OCR to Use?

```
Your Use Case → Recommended Engine

Fast extraction          → Tesseract
Better accuracy         → PaddleOCR
Both balance            → EasyOCR
Professional grade      → Google Vision
Tables & handwriting    → AWS Textract
Multiple languages      → Google Vision
Offline requirement     → Local (Paddle/Easy/Tess)
Enterprise support      → AWS/Google
Budget conscious        → Tesseract
Best results, no budget → Google Vision
```

---

## 📞 Support & Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Shows all internal operations
```

### Export Metrics
```python
metrics = ocr.get_metrics()
import json
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

### Profile Performance
```python
import cProfile
cProfile.run('ocr.process_document("file.pdf")')
# Shows timing breakdown
```

---

Good luck with your OCR service! 🎉

For more details, see:
- custom_ocr_service.py - Core implementation
- production_ocr_wrapper.py - Production wrapper
- ocr_service_complete_guide.md - Full guide
- ocr_api_documentation.md - API docs