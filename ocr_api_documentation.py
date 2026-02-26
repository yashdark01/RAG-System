"""
Production-Ready OCR Service Wrapper
Complete implementation with examples, integrations, and production patterns
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ProductionOCRService')


# ============================================================================
# PRODUCTION OCR SERVICE WRAPPER
# ============================================================================

class ProductionOCRService:
    """
    Production-ready OCR service wrapper
    Supports multiple engines with fallback, caching, and monitoring
    """
    
    def __init__(self, config: Dict = None):
        """Initialize service with configuration"""
        self.config = config or self._default_config()
        self.cache = {} if self.config.get('cache_enabled') else None
        self.metrics = self._init_metrics()
        self.engines = self._init_engines()
    
    @staticmethod
    def _default_config() -> Dict:
        """Default configuration"""
        return {
            'primary_engine': 'paddle',
            'fallback_engines': ['easyocr', 'tesseract'],
            'cache_enabled': True,
            'cache_dir': '.ocr_cache',
            'min_confidence': 0.5,
            'output_format': 'json',
            'max_retries': 3,
            'timeout': 60,
            'enable_monitoring': True,
        }
    
    def _init_metrics(self) -> Dict:
        """Initialize metrics tracking"""
        return {
            'total_documents': 0,
            'successful': 0,
            'failed': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'avg_confidence': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
    
    def _init_engines(self) -> Dict:
        """Initialize OCR engines"""
        # In production, import actual OCR libraries
        return {
            'paddle': self._mock_ocr_engine('paddle'),
            'easyocr': self._mock_ocr_engine('easyocr'),
            'tesseract': self._mock_ocr_engine('tesseract'),
        }
    
    @staticmethod
    def _mock_ocr_engine(engine_name: str) -> callable:
        """Mock OCR engine for demonstration"""
        def mock_process(file_path: str) -> Dict:
            return {
                'text': f'Extracted text from {file_path} using {engine_name}',
                'confidence': 0.87,
                'engine': engine_name,
                'blocks': []
            }
        return mock_process
    
    def process_document(
        self,
        file_path: str,
        document_type: str = 'general',
        force_reprocess: bool = False
    ) -> Dict:
        """
        Process a document with the OCR service
        
        Args:
            file_path: Path to document
            document_type: Type of document (invoice, receipt, etc.)
            force_reprocess: Ignore cache and reprocess
        
        Returns:
            OCR result dictionary
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate file
            if not self._validate_file(file_path):
                return self._error_response(f"Invalid file: {file_path}")
            
            # Step 2: Check cache
            if not force_reprocess and self.cache is not None:
                cached = self._get_from_cache(file_path)
                if cached:
                    logger.info(f"Cache hit for {file_path}")
                    self.metrics['cache_hits'] += 1
                    return cached
            
            self.metrics['cache_misses'] += 1
            
            # Step 3: Process with primary engine
            result = self._process_with_engine(
                file_path,
                self.config['primary_engine']
            )
            
            # Step 4: If confidence is low, try fallback engines
            if result.get('confidence', 0) < self.config['min_confidence']:
                logger.warning(f"Low confidence ({result['confidence']:.2f}), trying fallback engines")
                for fallback_engine in self.config['fallback_engines']:
                    fallback_result = self._process_with_engine(file_path, fallback_engine)
                    if fallback_result.get('confidence', 0) > result.get('confidence', 0):
                        result = fallback_result
                        logger.info(f"Using fallback engine: {fallback_engine}")
            
            # Step 5: Extract fields based on document type
            result['extracted_fields'] = self._extract_fields(
                result, document_type
            )
            
            # Step 6: Extract entities
            result['entities'] = self._extract_entities(result)
            
            # Step 7: Cache result
            if self.cache is not None:
                self._save_to_cache(file_path, result)
            
            # Step 8: Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(result, processing_time)
            
            logger.info(f"✓ Successfully processed {file_path} in {processing_time:.2f}s")
            
            return {
                'status': 'success',
                'document_id': self._generate_id(file_path),
                'file_name': Path(file_path).name,
                'file_size': Path(file_path).stat().st_size,
                'document_type': document_type,
                'processing_time_seconds': round(processing_time, 2),
                'raw_text': result.get('text', ''),
                'confidence': round(result.get('confidence', 0), 2),
                'engine_used': result.get('engine', 'unknown'),
                'extracted_fields': result.get('extracted_fields', {}),
                'entities': result.get('entities', {}),
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'total_words': len(result.get('text', '').split()),
                    'quality_score': result.get('confidence', 0),
                    'warnings': result.get('warnings', []),
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            self.metrics['failed'] += 1
            return self._error_response(str(e))
    
    def _validate_file(self, file_path: str) -> bool:
        """Validate file exists and is supported"""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        supported_formats = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        if path.suffix.lower() not in supported_formats:
            logger.error(f"Unsupported format: {path.suffix}")
            return False
        
        return True
    
    def _get_from_cache(self, file_path: str) -> Optional[Dict]:
        """Get result from cache"""
        if self.cache is None:
            return None
        
        file_hash = self._generate_id(file_path)
        return self.cache.get(file_hash)
    
    def _save_to_cache(self, file_path: str, result: Dict) -> None:
        """Save result to cache"""
        if self.cache is None:
            return
        
        file_hash = self._generate_id(file_path)
        self.cache[file_hash] = result
    
    def _process_with_engine(self, file_path: str, engine: str) -> Dict:
        """Process document with specific engine"""
        try:
            if engine in self.engines:
                result = self.engines[engine](file_path)
                result['engine'] = engine
                return result
            else:
                logger.error(f"Unknown engine: {engine}")
                return {'confidence': 0, 'text': '', 'engine': engine}
        except Exception as e:
            logger.error(f"Error with engine {engine}: {str(e)}")
            return {'confidence': 0, 'text': '', 'engine': engine}
    
    def _extract_fields(self, result: Dict, document_type: str) -> Dict:
        """Extract structured fields based on document type"""
        text = result.get('text', '')
        fields = {}
        
        if document_type == 'invoice':
            # Extract invoice fields
            import re
            
            # Invoice number
            match = re.search(r'(?:Invoice|INV)[#:\s]*([A-Z0-9\-]+)', text, re.IGNORECASE)
            if match:
                fields['invoice_number'] = match.group(1)
            
            # Amount
            match = re.search(r'(?:Total|Amount)[:\s]*(\$?\s?[\d,]+\.?\d*)', text, re.IGNORECASE)
            if match:
                fields['amount'] = match.group(1)
            
            # Date
            match = re.search(r'(?:Date|DATE)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
            if match:
                fields['date'] = match.group(1)
        
        elif document_type == 'receipt':
            import re
            
            # Merchant
            match = re.search(r'(?:Merchant|Store)[:\s]*([^\n]+)', text, re.IGNORECASE)
            if match:
                fields['merchant'] = match.group(1).strip()
            
            # Total
            match = re.search(r'(?:Total)[:\s]*(\$?\s?[\d,]+\.?\d*)', text, re.IGNORECASE)
            if match:
                fields['total'] = match.group(1)
        
        return fields
    
    def _extract_entities(self, result: Dict) -> Dict:
        """Extract named entities from text"""
        import re
        text = result.get('text', '')
        
        entities = {
            'emails': re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text),
            'phones': re.findall(r'[\+]?[0-9]{1,3}?[-.\s]?[\(]?[0-9]{3}[\)]?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', text),
            'urls': re.findall(r'https?://[^\s]+', text),
            'dates': re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text),
            'amounts': re.findall(r'[\$£€¥]\s?[\d,]+\.?\d*', text),
        }
        
        return entities
    
    @staticmethod
    def _generate_id(file_path: str) -> str:
        """Generate unique ID for file"""
        return hashlib.md5(file_path.encode()).hexdigest()[:16]
    
    def _update_metrics(self, result: Dict, processing_time: float) -> None:
        """Update service metrics"""
        self.metrics['total_documents'] += 1
        self.metrics['successful'] += 1
        self.metrics['total_processing_time'] += processing_time
        
        # Update averages
        if self.metrics['successful'] > 0:
            self.metrics['avg_processing_time'] = (
                self.metrics['total_processing_time'] / self.metrics['successful']
            )
    
    @staticmethod
    def _error_response(error: str) -> Dict:
        """Generate error response"""
        return {
            'status': 'error',
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics(self) -> Dict:
        """Get service metrics"""
        return {
            **self.metrics,
            'cache_enabled': self.cache is not None,
            'cache_size': len(self.cache) if self.cache else 0,
            'success_rate': (
                self.metrics['successful'] / self.metrics['total_documents']
                if self.metrics['total_documents'] > 0
                else 0
            )
        }
    
    def clear_cache(self) -> None:
        """Clear the cache"""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")


# ============================================================================
# OCR SERVICE API
# ============================================================================

class OCRServiceAPI:
    """REST API wrapper for OCR service"""
    
    def __init__(self):
        self.service = ProductionOCRService()
    
    def process(self, file_path: str, document_type: str = 'general') -> Dict:
        """Process document and return result"""
        return self.service.process_document(file_path, document_type)
    
    def batch_process(self, file_paths: List[str], document_type: str = 'general') -> List[Dict]:
        """Process multiple documents"""
        results = []
        for file_path in file_paths:
            result = self.service.process_document(file_path, document_type)
            results.append(result)
        
        return {
            'status': 'success',
            'total_documents': len(file_paths),
            'successful': sum(1 for r in results if r.get('status') == 'success'),
            'failed': sum(1 for r in results if r.get('status') == 'error'),
            'results': results
        }
    
    def get_metrics(self) -> Dict:
        """Get service metrics"""
        return self.service.get_metrics()
    
    def health_check(self) -> Dict:
        """Health check endpoint"""
        metrics = self.service.get_metrics()
        
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'documents_processed': metrics['total_documents'],
            'success_rate': metrics['success_rate'],
            'avg_processing_time': round(metrics['avg_processing_time'], 2)
        }


# ============================================================================
# EXAMPLES
# ============================================================================

def example_1_basic_processing():
    """Example 1: Basic document processing"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Document Processing")
    print("="*70)
    
    ocr = ProductionOCRService()
    
    # Process invoice
    result = ocr.process_document(
        file_path="invoice.pdf",
        document_type="invoice"
    )
    
    print("\nResult:")
    print(json.dumps(result, indent=2)[:500] + "...")
    
    # Show metrics
    print("\n\nMetrics:")
    metrics = ocr.get_metrics()
    print(f"  Documents processed: {metrics['total_documents']}")
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"  Avg processing time: {metrics['avg_processing_time']:.2f}s")


def example_2_batch_processing():
    """Example 2: Batch document processing"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Document Processing")
    print("="*70)
    
    api = OCRServiceAPI()
    
    documents = [
        "invoice1.pdf",
        "invoice2.pdf",
        "receipt1.jpg",
    ]
    
    results = api.batch_process(documents, document_type="invoice")
    
    print(f"\nProcessed {results['total_documents']} documents")
    print(f"✓ Successful: {results['successful']}")
    print(f"✗ Failed: {results['failed']}")
    
    # Show first result
    if results['results']:
        print(f"\nFirst result:")
        print(json.dumps(results['results'][0], indent=2)[:400] + "...")


def example_3_with_fallback():
    """Example 3: Processing with fallback engines"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Processing with Fallback Engines")
    print("="*70)
    
    config = {
        'primary_engine': 'paddle',
        'fallback_engines': ['easyocr', 'tesseract'],
        'min_confidence': 0.6,
        'cache_enabled': True,
    }
    
    ocr = ProductionOCRService(config)
    
    # First time - processes with primary engine
    result1 = ocr.process_document("document.pdf")
    print(f"\nFirst processing:")
    print(f"  Engine: {result1.get('engine_used', 'unknown')}")
    print(f"  Confidence: {result1.get('confidence', 0):.2f}")
    
    # Second time - uses cache
    result2 = ocr.process_document("document.pdf")
    print(f"\nSecond processing (cached):")
    print(f"  From cache: {result2.get('status') == 'success'}")
    
    metrics = ocr.get_metrics()
    print(f"\nCache stats:")
    print(f"  Cache hits: {metrics['cache_hits']}")
    print(f"  Cache misses: {metrics['cache_misses']}")


def example_4_document_types():
    """Example 4: Processing different document types"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Different Document Types")
    print("="*70)
    
    ocr = ProductionOCRService()
    
    documents = [
        ("invoice.pdf", "invoice"),
        ("receipt.jpg", "receipt"),
        ("passport.pdf", "passport"),
        ("document.pdf", "general"),
    ]
    
    results = []
    for file_path, doc_type in documents:
        result = ocr.process_document(file_path, doc_type)
        results.append((doc_type, result))
        
        print(f"\n{doc_type.upper()}:")
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Fields extracted: {len(result.get('extracted_fields', {}))}")
        print(f"  Entities found: {sum(len(v) for v in result.get('entities', {}).values())}")


def example_5_api_usage():
    """Example 5: Using the REST API wrapper"""
    print("\n" + "="*70)
    print("EXAMPLE 5: REST API Usage")
    print("="*70)
    
    api = OCRServiceAPI()
    
    # Process document
    result = api.process("invoice.pdf", "invoice")
    print(f"\nProcess result:")
    print(f"  Status: {result.get('status', 'unknown')}")
    print(f"  Document ID: {result.get('document_id', 'N/A')}")
    
    # Health check
    health = api.health_check()
    print(f"\nHealth check:")
    print(f"  Status: {health['status']}")
    print(f"  Success rate: {health['success_rate']:.1%}")
    print(f"  Avg time: {health['avg_processing_time']:.2f}s")


def example_6_field_extraction():
    """Example 6: Field extraction"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Field Extraction")
    print("="*70)
    
    ocr = ProductionOCRService()
    
    # Process invoice
    result = ocr.process_document("invoice.pdf", "invoice")
    
    if result.get('status') == 'success':
        fields = result.get('extracted_fields', {})
        
        print("\nExtracted Fields:")
        for field_name, field_value in fields.items():
            print(f"  {field_name}: {field_value}")
        
        # Extracted entities
        entities = result.get('entities', {})
        print("\nExtracted Entities:")
        for entity_type, values in entities.items():
            if values:
                print(f"  {entity_type}: {', '.join(values[:2])}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PRODUCTION-READY OCR SERVICE")
    print("="*70)
    
    # Run examples
    example_1_basic_processing()
    example_2_batch_processing()
    example_3_with_fallback()
    example_4_document_types()
    example_5_api_usage()
    example_6_field_extraction()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)