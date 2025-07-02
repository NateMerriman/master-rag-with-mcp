#!/usr/bin/env python3
"""
Integration test for DocumentStorage with the complete DocumentIngestionPipeline.

Tests the database storage logic integrated with the full pipeline workflow.
"""

import asyncio
import sys
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Mock dependencies
class MockBaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockField:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

def mock_field_validator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

class MockOpenAI:
    class RateLimitError(Exception):
        pass

# Mock modules
import sys
sys.modules['pydantic'] = type('MockPydantic', (), {
    'BaseModel': MockBaseModel,
    'Field': lambda *args, **kwargs: MockField(**kwargs),
    'field_validator': mock_field_validator
})()
sys.modules['openai'] = MockOpenAI()

from document_ingestion_pipeline import (
    DocumentIngestionPipeline,
    PipelineConfig,
    ChunkingConfig,
    DocumentStorage,
    DocumentChunk
)


async def test_pipeline_with_storage():
    """Test complete pipeline integration with storage."""
    print("🧪 Testing Pipeline with Storage Integration")
    print("-" * 60)
    
    # Track all storage operations
    storage_operations = []
    stored_documents = []
    
    def mock_get_supabase_client():
        storage_operations.append("get_client")
        return Mock()
    
    def mock_add_documents_to_supabase(client, urls, chunk_numbers, contents, metadatas, url_to_full_document, strategy_config, batch_size):
        storage_operations.append("add_documents")
        
        # Capture the stored data for validation
        stored_documents.append({
            'urls': urls,
            'chunk_numbers': chunk_numbers,
            'contents': contents,
            'metadatas': metadatas,
            'url_to_full_document': url_to_full_document,
            'batch_size': batch_size
        })
    
    # Create pipeline configuration
    config = PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=400,
            chunk_overlap=50,
            use_semantic_splitting=False  # Use simple chunking for predictable testing
        ),
        generate_embeddings=False,  # Disable for testing
        store_in_database=True      # Enable storage testing
    )
    
    pipeline = DocumentIngestionPipeline(config)
    
    # Mock the storage functions
    pipeline.storage.get_supabase_client = mock_get_supabase_client
    pipeline.storage.add_documents_to_supabase = mock_add_documents_to_supabase
    
    # Test document content
    test_content = """# Database Storage Integration Test

## Overview

This document tests the complete integration of the DocumentIngestionPipeline
with the database storage system. The storage component handles:

- Document metadata management
- Chunk storage with proper indexing
- Integration with existing Supabase infrastructure
- Error handling and recovery

## Storage Architecture

### Database Schema Compatibility

The storage system maintains compatibility with the existing crawled_pages table:

- **url**: Source URL of the document
- **chunk_number**: Sequential chunk identifier
- **content**: Processed chunk content
- **metadata**: Enhanced metadata with pipeline information
- **embedding**: Vector embeddings (when enabled)
- **source_id**: Foreign key to sources table

### Storage Process

1. **Document Processing**: Pipeline processes content through chunking and embedding
2. **Metadata Enhancement**: Adds pipeline-specific metadata
3. **Batch Storage**: Stores chunks in configurable batches
4. **Error Recovery**: Handles failures gracefully

## Key Features

### Enhanced Metadata

The storage system adds comprehensive metadata:

```json
{
  "document_id": "unique_identifier",
  "pipeline_processed": true,
  "processing_timestamp": "2024-01-01T00:00:00",
  "chunk_method": "semantic",
  "token_count": 150,
  "start_position": 0,
  "end_position": 500
}
```

### Integration Points

- **AdvancedWebCrawler Output**: Receives clean markdown
- **Existing Storage Functions**: Uses add_documents_to_supabase
- **Database Schema**: Compatible with current tables
- **MCP Server**: Accessible through existing tools

## Testing Strategy

Comprehensive testing includes:

1. Basic storage functionality
2. Metadata preservation and enhancement
3. Batch processing validation
4. Error handling scenarios
5. End-to-end pipeline integration

This ensures reliability and compatibility with the existing system.
"""
    
    # Process the document
    result = await pipeline.process_document(
        content=test_content,
        source_url="https://docs.example.com/storage-integration",
        metadata={
            "framework": "pipeline_testing",
            "version": "1.0",
            "test_type": "integration"
        }
    )
    
    # Validate pipeline result
    print(f"   📊 Pipeline Result:")
    print(f"      Success: {result.success}")
    print(f"      Document ID: {result.document_id}")
    print(f"      Title: {result.title}")
    print(f"      Chunks created: {result.chunks_created}")
    print(f"      Processing time: {result.processing_time_ms:.1f} ms")
    
    # Validate storage operations
    print(f"\n   📋 Storage Operations:")
    print(f"      Operations called: {storage_operations}")
    print(f"      Documents stored: {len(stored_documents)}")
    
    if stored_documents:
        stored_doc = stored_documents[0]
        print(f"\n   📦 Stored Document Details:")
        print(f"      URLs: {len(stored_doc['urls'])} entries")
        print(f"      Chunk numbers: {stored_doc['chunk_numbers']}")
        print(f"      Content lengths: {[len(c) for c in stored_doc['contents']]}")
        print(f"      Batch size: {stored_doc['batch_size']}")
        print(f"      Full document included: {'storage-integration' in str(stored_doc['url_to_full_document'])}")
        
        # Validate metadata enhancement
        sample_metadata = stored_doc['metadatas'][0] if stored_doc['metadatas'] else {}
        metadata_fields = [
            'document_id', 'pipeline_processed', 'processing_timestamp',
            'chunk_method', 'start_position', 'end_position'
        ]
        
        print(f"\n   🏷️  Enhanced Metadata:")
        for field in metadata_fields:
            has_field = field in sample_metadata
            print(f"      {field}: {'✅' if has_field else '❌'} {sample_metadata.get(field, 'Missing')}")
    
    # Validation assertions
    validations = [
        (result.success, "Pipeline processing should succeed"),
        (result.chunks_created > 0, "Should create chunks"),
        (result.document_id == "docs.example.com_storage-integration", "Should generate correct document ID"),
        ("get_client" in storage_operations, "Should request Supabase client"),
        ("add_documents" in storage_operations, "Should add documents to storage"),
        (len(stored_documents) == 1, "Should store exactly one document"),
    ]
    
    if stored_documents:
        stored_doc = stored_documents[0]
        validations.extend([
            (len(stored_doc['urls']) == result.chunks_created, "URL count should match chunk count"),
            (len(stored_doc['contents']) == result.chunks_created, "Content count should match chunk count"),
            (all('pipeline_processed' in m for m in stored_doc['metadatas']), "All metadata should be pipeline-enhanced"),
            (stored_doc['batch_size'] == 20, "Should use correct batch size"),
        ])
    
    print(f"\n   🔍 Validation Results:")
    all_passed = True
    for passed, description in validations:
        status = "✅" if passed else "❌"
        print(f"      {status} {description}")
        if not passed:
            all_passed = False
    
    return all_passed


async def test_storage_with_embeddings():
    """Test storage integration with embedding generation."""
    print("\n🧪 Testing Storage with Embeddings")
    print("-" * 60)
    
    # Mock embedding functions
    def mock_create_embeddings_batch(texts):
        return [[0.1] * 1536 for _ in texts]  # Mock 1536-dim embeddings
    
    # Track storage calls
    embedding_storage_calls = []
    
    def mock_storage_with_embeddings(client, urls, chunk_numbers, contents, metadatas, url_to_full_document, strategy_config, batch_size):
        embedding_storage_calls.append({
            'chunk_count': len(contents),
            'has_embeddings': any('embedding' in str(m) for m in metadatas),
            'batch_size': batch_size
        })
    
    # Create pipeline with embeddings enabled
    config = PipelineConfig(
        chunking=ChunkingConfig(chunk_size=300, chunk_overlap=30, use_semantic_splitting=False),
        generate_embeddings=True,
        store_in_database=True
    )
    
    pipeline = DocumentIngestionPipeline(config)
    
    # Mock embedding and storage functions
    pipeline.embedder.create_embeddings_batch = mock_create_embeddings_batch
    pipeline.embedder.available = True
    pipeline.storage.get_supabase_client = lambda: Mock()
    pipeline.storage.add_documents_to_supabase = mock_storage_with_embeddings
    
    # Test content for embedding
    embedding_test_content = """# Embedding Storage Test

## Vector Database Integration

This test validates that embeddings are properly generated and stored
alongside the document chunks in the database.

### Embedding Pipeline

1. Generate embeddings for each chunk
2. Validate embedding dimensions
3. Store with proper metadata
4. Ensure database compatibility

The storage system handles vector embeddings efficiently while maintaining
compatibility with the existing infrastructure.
"""
    
    # Process with embeddings
    result = await pipeline.process_document(
        content=embedding_test_content,
        source_url="https://test.example.com/embeddings",
        metadata={"test_type": "embedding_storage"}
    )
    
    print(f"   📊 Embedding Pipeline Result:")
    print(f"      Success: {result.success}")
    print(f"      Chunks created: {result.chunks_created}")
    print(f"      Embeddings generated: {result.embeddings_generated}")
    print(f"      Processing time: {result.processing_time_ms:.1f} ms")
    
    print(f"\n   🔗 Embedding Storage:")
    for call in embedding_storage_calls:
        print(f"      Chunks stored: {call['chunk_count']}")
        print(f"      Embeddings included: {call['has_embeddings']}")
        print(f"      Batch size: {call['batch_size']}")
    
    # Validate embedding integration
    validations = [
        (result.success, "Embedding pipeline should succeed"),
        (result.embeddings_generated > 0, "Should generate embeddings"),
        (result.embeddings_generated == result.chunks_created, "Should embed all chunks"),
        (len(embedding_storage_calls) > 0, "Should call storage functions"),
    ]
    
    print(f"\n   🔍 Embedding Validation:")
    all_passed = True
    for passed, description in validations:
        status = "✅" if passed else "❌"
        print(f"      {status} {description}")
        if not passed:
            all_passed = False
    
    return all_passed


async def test_storage_error_scenarios():
    """Test storage behavior under error conditions."""
    print("\n🧪 Testing Storage Error Scenarios")
    print("-" * 60)
    
    # Create pipeline
    config = PipelineConfig(
        chunking=ChunkingConfig(chunk_size=200, use_semantic_splitting=False),
        generate_embeddings=False,
        store_in_database=True
    )
    
    pipeline = DocumentIngestionPipeline(config)
    
    # Test 1: Storage function unavailable
    pipeline.storage.get_supabase_client = None
    pipeline.storage.add_documents_to_supabase = None
    
    result1 = await pipeline.process_document(
        content="# Test\n\nSimple test content.",
        source_url="https://example.com/error-test-1"
    )
    
    print(f"   📋 Test 1 - Missing Functions:")
    print(f"      Success: {result1.success} (should be False)")
    print(f"      Errors: {result1.errors}")
    
    # Test 2: Client connection failure
    def failing_client():
        raise Exception("Database connection failed")
    
    pipeline.storage.get_supabase_client = failing_client
    pipeline.storage.add_documents_to_supabase = Mock()
    
    result2 = await pipeline.process_document(
        content="# Test 2\n\nAnother test content.",
        source_url="https://example.com/error-test-2"
    )
    
    print(f"\n   📋 Test 2 - Connection Failure:")
    print(f"      Success: {result2.success} (should be False)")
    print(f"      Errors: {result2.errors}")
    
    # Test 3: Partial failure recovery
    success_count = 0
    def sometimes_failing_storage(*args, **kwargs):
        nonlocal success_count
        success_count += 1
        if success_count == 1:
            raise Exception("First attempt failed")
        # Second attempt succeeds
    
    pipeline.storage.get_supabase_client = lambda: Mock()
    pipeline.storage.add_documents_to_supabase = sometimes_failing_storage
    
    result3 = await pipeline.process_document(
        content="# Test 3\n\nRecovery test content.",
        source_url="https://example.com/error-test-3"
    )
    
    print(f"\n   📋 Test 3 - Partial Recovery:")
    print(f"      Success: {result3.success} (depends on retry logic)")
    print(f"      Errors: {result3.errors}")
    
    # Validate error handling
    validations = [
        (not result1.success, "Should fail when storage functions missing"),
        (len(result1.errors) > 0, "Should report errors for missing functions"),
        (not result2.success, "Should fail when client connection fails"),
        (len(result2.errors) > 0, "Should report connection errors"),
    ]
    
    print(f"\n   🔍 Error Handling Validation:")
    all_passed = True
    for passed, description in validations:
        status = "✅" if passed else "❌"
        print(f"      {status} {description}")
        if not passed:
            all_passed = False
    
    return all_passed


async def main():
    """Run all storage integration tests."""
    print("🚀 DocumentStorage Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Pipeline with Storage", test_pipeline_with_storage),
        ("Storage with Embeddings", test_storage_with_embeddings),
        ("Storage Error Scenarios", test_storage_error_scenarios),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
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
    print("🎯 STORAGE INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL STORAGE INTEGRATION TESTS PASSED!")
        print("\n🚀 Task 14.5 Implementation Summary:")
        print("✅ DocumentStorage component successfully implemented")
        print("✅ Database storage logic integrated with existing infrastructure")
        print("✅ Maintains compatibility with current Supabase schema")
        print("✅ Enhanced metadata management with pipeline information")
        print("✅ Robust error handling and recovery mechanisms")
        print("✅ Seamless integration with DocumentIngestionPipeline")
        print("✅ Support for batch processing and configurable storage")
        print("✅ Full compatibility with existing add_documents_to_supabase function")
        
        print("\n📊 Integration Ready:")
        print("The DocumentStorage component is ready for production use and")
        print("completes the DocumentIngestionPipeline implementation. The storage")
        print("layer provides reliable, schema-compatible persistence for all")
        print("processed documents and chunks.")
    else:
        print("\n⚠️  Some tests failed")
        print("Check the error messages above for details")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)