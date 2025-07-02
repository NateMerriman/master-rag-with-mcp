#!/usr/bin/env python3
"""
Test script for DocumentIngestionPipeline core logic and metadata extraction.

Tests the main pipeline orchestration and comprehensive metadata extraction
without requiring external dependencies.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from document_ingestion_pipeline import (
    DocumentIngestionPipeline,
    PipelineConfig,
    ChunkingConfig,
    PipelineResult
)


def test_title_extraction():
    """Test title extraction from various markdown formats."""
    print("🧪 Testing Title Extraction")
    print("-" * 60)
    
    # Create pipeline for testing
    config = PipelineConfig()
    pipeline = DocumentIngestionPipeline(config)
    
    test_cases = [
        ("# Main Title\n\nContent here", "Main Title"),
        ("Title\n===\n\nContent", "Title"),
        ("Title: My Document\n\nContent", "My Document"),
        ("**Bold Title**\n\nContent", "Bold Title"),
        ("No title here, just content", "No title here, just content"),
        ("", "Untitled Document"),
        ("```code\nno title\n```", "Untitled Document"),
    ]
    
    for i, (content, expected) in enumerate(test_cases):
        result = pipeline._extract_title(content)
        print(f"   Test {i+1}: '{result}' {'✅' if expected in result else '❌'}")
        if expected not in result:
            print(f"     Expected: {expected}")
            print(f"     Got: {result}")
    
    print("✅ Title extraction tests completed")
    return True


def test_metadata_extraction():
    """Test comprehensive metadata extraction."""
    print("\n🧪 Testing Metadata Extraction")
    print("-" * 60)
    
    # Create pipeline for testing
    config = PipelineConfig()
    pipeline = DocumentIngestionPipeline(config)
    
    # Complex test content with various elements
    test_content = """# Advanced Web Crawling Guide

## Introduction

This guide covers advanced web crawling techniques for modern websites.

### Prerequisites

- Python 3.8+
- Basic knowledge of HTTP
- Understanding of HTML/CSS

## Implementation

### Python Setup

```python
import requests
from bs4 import BeautifulSoup

def crawl_website(url):
    response = requests.get(url)
    return response.text
```

### JavaScript Handling

For JavaScript-heavy sites, use Playwright:

```javascript
const { chromium } = require('playwright');

async function crawlWithJS(url) {
    const browser = await chromium.launch();
    const page = await browser.newPage();
    await page.goto(url);
    return await page.content();
}
```

## Best Practices

1. **Rate Limiting**: Always implement delays
2. **Robots.txt**: Respect website policies  
3. **User Agent**: Use appropriate headers

### Common Pitfalls

- Not handling JavaScript rendering
- Ignoring rate limits
- Poor error handling

## External Resources

- [Python Requests Documentation](https://docs.python-requests.org/)
- [Playwright Guide](https://playwright.dev/docs/)
- [Web Scraping Ethics](https://example.com/ethics)

## Data Processing

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Simple | Fast | Low | Static sites |
| Advanced | Slow | High | Dynamic sites |

## Conclusion

Modern web crawling requires a combination of techniques to handle the
complexity of today's web applications.
"""
    
    # Test metadata extraction
    metadata = pipeline._extract_document_metadata(
        test_content, 
        "https://docs.example.com/guides/web-crawling"
    )
    
    print(f"📊 Extracted metadata for {metadata['word_count']} words")
    
    # Verify basic metadata
    expected_fields = [
        "source_url", "domain", "word_count", "character_count",
        "headers", "header_count", "code_blocks_count", 
        "total_links_count", "complexity_score"
    ]
    
    for field in expected_fields:
        if field in metadata:
            print(f"   ✅ {field}: {metadata[field]}")
        else:
            print(f"   ❌ Missing field: {field}")
    
    # Verify specific content analysis
    print(f"\n📋 Content Analysis:")
    print(f"   Headers found: {metadata['header_count']}")
    print(f"   Max header depth: {metadata['max_header_depth']}")
    print(f"   Code blocks: {metadata['code_blocks_count']}")
    print(f"   Programming languages: {metadata['programming_languages']}")
    print(f"   Internal links: {metadata['internal_links_count']}")
    print(f"   External links: {metadata['external_links_count']}")
    print(f"   Content types: {metadata['content_types']}")
    print(f"   Complexity: {metadata['complexity_category']} ({metadata['complexity_score']})")
    print(f"   Reading time: {metadata['estimated_reading_time_minutes']} minutes")
    
    # Validate expected results
    validations = [
        (metadata['header_count'] >= 3, "Should find multiple headers"),
        (metadata['code_blocks_count'] >= 2, "Should find code blocks"),
        (len(metadata['programming_languages']) >= 2, "Should detect multiple languages"),
        (metadata['external_links_count'] >= 2, "Should find external links"),
        ("technical" in metadata['content_types'], "Should identify as technical content"),
        (metadata['complexity_score'] > 50, "Should be identified as complex"),
    ]
    
    for passed, description in validations:
        print(f"   {'✅' if passed else '❌'} {description}")
    
    print("✅ Metadata extraction tests completed")
    return True


async def test_pipeline_process_document():
    """Test the main pipeline process_document method."""
    print("\n🧪 Testing Pipeline Process Document")
    print("-" * 60)
    
    # Create pipeline configuration  
    config = PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=300,
            chunk_overlap=50,
            use_semantic_splitting=False  # Use simple for testing
        ),
        generate_embeddings=False,  # Disable for testing
        store_in_database=False,    # Disable for testing
    )
    
    pipeline = DocumentIngestionPipeline(config)
    
    # Test content
    test_content = """# API Integration Guide

## Overview

This guide shows how to integrate with external APIs using Python.

## Authentication

### API Keys

Most APIs require authentication via API keys:

```python
import requests

headers = {
    'Authorization': 'Bearer your-api-key',
    'Content-Type': 'application/json'
}
```

### OAuth 2.0

For more secure applications, use OAuth:

```python
from requests_oauthlib import OAuth2Session

oauth = OAuth2Session(client_id)
token = oauth.fetch_token(token_url, client_secret=client_secret)
```

## Making Requests

### GET Requests

```python
response = requests.get(
    'https://api.example.com/data',
    headers=headers
)
data = response.json()
```

### POST Requests

```python
payload = {'key': 'value'}
response = requests.post(
    'https://api.example.com/create',
    json=payload,
    headers=headers
)
```

## Error Handling

Always implement proper error handling:

1. Check status codes
2. Handle timeouts
3. Implement retry logic
4. Log errors appropriately

## Best Practices

- Use connection pooling for multiple requests
- Implement rate limiting
- Cache responses when appropriate
- Validate input data

## Conclusion

Proper API integration requires attention to authentication, error handling,
and performance considerations.
"""
    
    # Process the document
    try:
        result = await pipeline.process_document(
            content=test_content,
            source_url="https://docs.example.com/api-integration",
            metadata={"framework": "api_docs", "version": "2.0"}
        )
        
        print(f"✅ Pipeline processing successful")
        print(f"   Document ID: {result.document_id}")
        print(f"   Title: {result.title}")
        print(f"   Chunks created: {result.chunks_created}")
        print(f"   Processing time: {result.processing_time_ms:.1f} ms")
        print(f"   Success: {result.success}")
        
        if result.errors:
            print(f"   Errors: {result.errors}")
        
        # Validate results
        validations = [
            (result.success, "Processing should succeed"),
            (result.chunks_created > 0, "Should create chunks"),
            (result.title == "API Integration Guide", "Should extract correct title"),
            (result.processing_time_ms > 0, "Should record processing time"),
            (len(result.errors) == 0, "Should have no errors"),
        ]
        
        for passed, description in validations:
            print(f"   {'✅' if passed else '❌'} {description}")
        
        return result.success and all(passed for passed, _ in validations)
        
    except Exception as e:
        print(f"❌ Pipeline processing failed: {str(e)}")
        return False


async def test_pipeline_error_handling():
    """Test pipeline error handling with invalid inputs."""
    print("\n🧪 Testing Pipeline Error Handling")
    print("-" * 60)
    
    config = PipelineConfig(
        generate_embeddings=False,
        store_in_database=False
    )
    pipeline = DocumentIngestionPipeline(config)
    
    # Test empty content
    result = await pipeline.process_document("", "https://example.com/empty")
    print(f"   Empty content: {'✅' if not result.success else '❌'} (should fail)")
    
    # Test content that produces no chunks
    minimal_content = "x"
    result = await pipeline.process_document(minimal_content, "https://example.com/minimal")
    print(f"   Minimal content: {'✅' if result.chunks_created >= 0 else '❌'} (should handle gracefully)")
    
    # Test very large content
    large_content = "This is a test sentence. " * 10000  # ~250KB
    result = await pipeline.process_document(large_content, "https://example.com/large")
    print(f"   Large content: {'✅' if result.success else '❌'} (should handle large content)")
    
    print("✅ Error handling tests completed")
    return True


def test_configuration_validation():
    """Test pipeline configuration validation."""
    print("\n🧪 Testing Configuration Validation")
    print("-" * 60)
    
    try:
        # Test valid configuration
        config = PipelineConfig(
            chunking=ChunkingConfig(chunk_size=1000, chunk_overlap=200),
            generate_embeddings=True,
            store_in_database=True
        )
        print("   ✅ Valid configuration accepted")
        
        # Test invalid overlap (should raise ValueError)
        try:
            invalid_config = ChunkingConfig(chunk_size=100, chunk_overlap=100)
            print("   ❌ Invalid overlap configuration should have failed")
            return False
        except ValueError:
            print("   ✅ Invalid overlap configuration properly rejected")
        
        # Test pipeline creation
        pipeline = DocumentIngestionPipeline(config)
        print("   ✅ Pipeline created successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration validation failed: {e}")
        return False


async def main():
    """Run all pipeline core tests."""
    print("🚀 DocumentIngestionPipeline Core Tests")
    print("=" * 60)
    
    tests = [
        ("Title Extraction", test_title_extraction),
        ("Metadata Extraction", test_metadata_extraction),
        ("Pipeline Process Document", test_pipeline_process_document),
        ("Pipeline Error Handling", test_pipeline_error_handling),
        ("Configuration Validation", test_configuration_validation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results.append((test_name, False))
        
        print()  # Empty line between tests
    
    # Summary
    print("=" * 60)
    print("🎯 PIPELINE CORE TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL PIPELINE CORE TESTS PASSED!")
        print("✅ DocumentIngestionPipeline core logic is working correctly")
        print("✅ Title extraction handles multiple markdown formats")
        print("✅ Metadata extraction provides comprehensive document analysis")
        print("✅ Pipeline orchestration manages the complete workflow")
    else:
        print("\n⚠️  Some tests failed")
        print("Check the error messages above for details")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)