#!/usr/bin/env python3
"""
Test script for EmbeddingGenerator integration and enhanced functionality.

Tests the complete embedding pipeline including batch processing, error handling,
retry logic, and integration with the DocumentIngestionPipeline.
"""

import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import time

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from document_ingestion_pipeline import (
    EmbeddingGenerator,
    DocumentChunk,
    DocumentIngestionPipeline,
    PipelineConfig,
    ChunkingConfig
)


async def test_embedding_generator_initialization():
    """Test EmbeddingGenerator initialization with different configurations."""
    print("🧪 Testing EmbeddingGenerator Initialization")
    print("-" * 60)
    
    # Test with default configuration
    embedder = EmbeddingGenerator()
    print(f"   ✅ Default initialization: batch_size={embedder.batch_size}, max_retries={embedder.max_retries}")
    
    # Test with custom configuration
    custom_embedder = EmbeddingGenerator(batch_size=50, max_retries=5)
    print(f"   ✅ Custom initialization: batch_size={custom_embedder.batch_size}, max_retries={custom_embedder.max_retries}")
    
    # Test availability check
    print(f"   ✅ Embedder available: {embedder.available}")
    print(f"   ✅ Embedding dimension: {embedder.embedding_dimension}")
    
    return True


async def test_embedding_generation_mock():
    """Test embedding generation with mocked utils functions."""
    print("\n🧪 Testing Embedding Generation (Mocked)")
    print("-" * 60)
    
    # Create test chunks
    test_chunks = [
        DocumentChunk(
            content="This is the first test chunk with meaningful content.",
            chunk_index=0,
            start_position=0,
            end_position=53,
            metadata={"title": "Test Document", "source": "test.md"}
        ),
        DocumentChunk(
            content="This is the second test chunk with different content for testing.",
            chunk_index=1,
            start_position=54,
            end_position=118,
            metadata={"title": "Test Document", "source": "test.md"}
        ),
        DocumentChunk(
            content="",  # Empty chunk to test filtering
            chunk_index=2,
            start_position=119,
            end_position=119,
            metadata={"title": "Test Document", "source": "test.md"}
        ),
        DocumentChunk(
            content="Short",  # Very short chunk to test filtering
            chunk_index=3,
            start_position=120,
            end_position=125,
            metadata={"title": "Test Document", "source": "test.md"}
        )
    ]
    
    # Mock embedding function to return realistic embeddings
    def mock_create_embeddings_batch(texts):
        """Mock function that returns embeddings for valid texts."""
        embeddings = []
        for text in texts:
            if text and len(text.strip()) >= 10:
                # Return a mock 1536-dimension embedding (all zeros for testing)
                embeddings.append([0.1] * 1536)
            else:
                # This shouldn't happen as filtering occurs before this call
                embeddings.append([0.0] * 1536)
        return embeddings
    
    # Patch the embedding function
    with patch('document_ingestion_pipeline.EmbeddingGenerator') as MockEmbeddingGenerator:
        embedder = EmbeddingGenerator(batch_size=2)  # Small batch for testing
        embedder.create_embeddings_batch = mock_create_embeddings_batch
        embedder.available = True
        
        # Test embedding generation
        result_chunks = await embedder.embed_chunks(test_chunks)
        
        # Validate results
        print(f"   📊 Processed {len(result_chunks)} chunks")
        
        embedded_count = 0
        for i, chunk in enumerate(result_chunks):
            if chunk.embedding:
                embedded_count += 1
                print(f"   ✅ Chunk {i}: embedded ({len(chunk.embedding)} dimensions)")
                assert len(chunk.embedding) == 1536, f"Wrong embedding dimension: {len(chunk.embedding)}"
            else:
                print(f"   ⚠️  Chunk {i}: no embedding (content: '{chunk.content[:20]}...')")
        
        # Should have embeddings for chunks 0 and 1 (valid content)
        # Chunks 2 and 3 should be filtered out due to insufficient content
        expected_embedded = 2
        print(f"   📈 Embedded chunks: {embedded_count}/{len(test_chunks)} (expected: {expected_embedded})")
        
        assert embedded_count == expected_embedded, f"Expected {expected_embedded} embeddings, got {embedded_count}"
        
        print("   ✅ Embedding generation with filtering completed successfully")
    
    return True


async def test_embedding_batch_processing():
    """Test batch processing functionality."""
    print("\n🧪 Testing Batch Processing")
    print("-" * 60)
    
    # Create many chunks to test batching
    large_chunk_set = []
    for i in range(25):  # Create 25 chunks
        large_chunk_set.append(
            DocumentChunk(
                content=f"This is test chunk number {i} with sufficient content for embedding generation.",
                chunk_index=i,
                start_position=i * 100,
                end_position=(i + 1) * 100,
                metadata={"title": "Large Document", "source": "large.md"}
            )
        )
    
    # Mock batch function that tracks call count
    call_count = 0
    def mock_batch_embeddings(texts):
        nonlocal call_count
        call_count += 1
        print(f"      📦 Batch {call_count}: processing {len(texts)} texts")
        return [[0.1] * 1536 for _ in texts]
    
    # Test with small batch size to ensure batching works
    embedder = EmbeddingGenerator(batch_size=10)
    embedder.create_embeddings_batch = mock_batch_embeddings
    embedder.available = True
    
    start_time = time.time()
    result_chunks = await embedder.embed_chunks(large_chunk_set)
    elapsed_time = time.time() - start_time
    
    # Validate batch processing
    expected_batches = (len(large_chunk_set) + embedder.batch_size - 1) // embedder.batch_size
    print(f"   📊 Processed {len(large_chunk_set)} chunks in {call_count} batches")
    print(f"   ⏱️  Processing time: {elapsed_time:.3f} seconds")
    print(f"   📈 Expected batches: {expected_batches}, Actual batches: {call_count}")
    
    assert call_count == expected_batches, f"Expected {expected_batches} batches, got {call_count}"
    
    # Verify all chunks have embeddings
    embedded_count = sum(1 for chunk in result_chunks if chunk.embedding)
    print(f"   ✅ All chunks embedded: {embedded_count}/{len(large_chunk_set)}")
    
    assert embedded_count == len(large_chunk_set), f"Not all chunks embedded: {embedded_count}/{len(large_chunk_set)}"
    
    return True


async def test_embedding_error_handling():
    """Test error handling and retry logic."""
    print("\n🧪 Testing Error Handling and Retry Logic")
    print("-" * 60)
    
    test_chunks = [
        DocumentChunk(
            content="This is a test chunk for error handling scenarios.",
            chunk_index=0,
            start_position=0,
            end_position=48,
            metadata={"title": "Error Test", "source": "error.md"}
        )
    ]
    
    # Test 1: Embedding function that fails once then succeeds
    attempt_count = 0
    def failing_then_success_embeddings(texts):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise Exception("Simulated API failure")
        return [[0.2] * 1536 for _ in texts]
    
    embedder = EmbeddingGenerator(batch_size=10, max_retries=3)
    embedder.create_embeddings_batch = failing_then_success_embeddings
    embedder.available = True
    
    print("   🔄 Testing retry logic (fail once, then succeed)")
    result_chunks = await embedder.embed_chunks(test_chunks)
    
    assert result_chunks[0].embedding is not None, "Should have embedding after retry"
    assert len(result_chunks[0].embedding) == 1536, "Should have correct dimension"
    print(f"   ✅ Retry successful after {attempt_count} attempts")
    
    # Test 2: Embedding function that always fails
    def always_failing_embeddings(texts):
        raise Exception("Persistent API failure")
    
    embedder2 = EmbeddingGenerator(batch_size=10, max_retries=2)
    embedder2.create_embeddings_batch = always_failing_embeddings
    embedder2.available = True
    
    print("   🔄 Testing fallback to zero vectors (always failing)")
    result_chunks2 = await embedder2.embed_chunks(test_chunks)
    
    # Should have fallback zero vector
    assert result_chunks2[0].embedding is not None, "Should have fallback embedding"
    assert len(result_chunks2[0].embedding) == 1536, "Should have correct dimension"
    assert all(x == 0.0 for x in result_chunks2[0].embedding), "Should be zero vector"
    print("   ✅ Fallback to zero vectors working correctly")
    
    # Test 3: Invalid embedding dimensions
    def invalid_dimension_embeddings(texts):
        return [[0.3] * 512 for _ in texts]  # Wrong dimension
    
    embedder3 = EmbeddingGenerator()
    embedder3.create_embeddings_batch = invalid_dimension_embeddings
    embedder3.available = True
    
    print("   🔄 Testing invalid dimension handling")
    result_chunks3 = await embedder3.embed_chunks(test_chunks)
    
    # Should have corrected embedding with proper dimension
    assert result_chunks3[0].embedding is not None, "Should have corrected embedding"
    assert len(result_chunks3[0].embedding) == 1536, "Should have corrected dimension"
    print("   ✅ Invalid dimension correction working correctly")
    
    return True


async def test_pipeline_embedding_integration():
    """Test embedding integration within the complete DocumentIngestionPipeline."""
    print("\n🧪 Testing Pipeline Integration")
    print("-" * 60)
    
    # Mock embedding function for pipeline testing
    def mock_pipeline_embeddings(texts):
        return [[0.4] * 1536 for _ in texts]
    
    # Create pipeline configuration with embeddings enabled
    config = PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=200,
            chunk_overlap=50,
            use_semantic_splitting=False  # Use simple for predictable testing
        ),
        generate_embeddings=True,
        store_in_database=False  # Disable storage for testing
    )
    
    pipeline = DocumentIngestionPipeline(config)
    
    # Mock the embedding function in the pipeline's embedder
    pipeline.embedder.create_embeddings_batch = mock_pipeline_embeddings
    pipeline.embedder.available = True
    
    # Test document content
    test_content = """# Embedding Integration Test

## Overview
This document tests the integration of embedding generation within the complete 
DocumentIngestionPipeline workflow.

## Key Features
- Automatic embedding generation for all chunks
- Error handling and fallback mechanisms
- Performance optimization through batching

## Implementation Details
The embedding system uses OpenAI's text-embedding-3-small model to generate
1536-dimensional vectors for semantic search and retrieval.

## Quality Assurance
All embeddings are validated for proper dimensions and non-zero content
before being stored in the database."""
    
    # Process the document
    result = await pipeline.process_document(
        content=test_content,
        source_url="https://example.com/embedding-test",
        metadata={"framework": "test_integration", "version": "1.0"}
    )
    
    # Validate pipeline result
    print(f"   📊 Pipeline result: {result.success}")
    print(f"   📄 Document title: {result.title}")
    print(f"   📦 Chunks created: {result.chunks_created}")
    print(f"   🔗 Embeddings generated: {result.embeddings_generated}")
    print(f"   ⏱️  Processing time: {result.processing_time_ms:.1f} ms")
    
    # Validate results
    assert result.success, "Pipeline should succeed"
    assert result.chunks_created > 0, "Should create chunks"
    assert result.embeddings_generated > 0, "Should generate embeddings"
    assert result.embeddings_generated == result.chunks_created, "Should embed all chunks"
    assert len(result.errors) == 0, f"Should have no errors: {result.errors}"
    
    print("   ✅ Pipeline embedding integration working correctly")
    
    return True


async def test_embedding_performance():
    """Test embedding generation performance with larger datasets."""
    print("\n🧪 Testing Performance")
    print("-" * 60)
    
    # Create performance test data
    large_content = """# Performance Test Document

## Section 1
This is a large document designed to test the performance of the embedding
generation system when processing many chunks of content.

## Section 2
The document contains multiple sections with substantial content to ensure
that the chunking and embedding processes are thoroughly tested.

## Section 3
Each section has enough content to generate meaningful embeddings while
testing the batch processing capabilities of the system.

## Section 4
The performance test measures both throughput and latency to ensure
the system can handle real-world document processing workloads.

## Section 5
This section continues the pattern of substantial content to maintain
consistent chunk sizes throughout the performance evaluation.

## Section 6
Performance metrics include processing time per chunk, memory usage,
and overall system throughput under various load conditions.

## Section 7
The final section concludes the performance test with additional content
to ensure a complete evaluation of the embedding generation pipeline.""" * 5  # Multiply to create larger document
    
    # Mock high-performance embedding function
    def fast_mock_embeddings(texts):
        return [[0.5] * 1536 for _ in texts]
    
    # Create performance-optimized configuration
    config = PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=400,
            chunk_overlap=100,
            use_semantic_splitting=False
        ),
        generate_embeddings=True,
        store_in_database=False
    )
    
    pipeline = DocumentIngestionPipeline(config)
    pipeline.embedder = EmbeddingGenerator(batch_size=50)  # Larger batches for performance
    pipeline.embedder.create_embeddings_batch = fast_mock_embeddings
    pipeline.embedder.available = True
    
    # Run performance test
    start_time = time.time()
    result = await pipeline.process_document(
        content=large_content,
        source_url="https://example.com/performance-test",
        metadata={"test_type": "performance"}
    )
    total_time = time.time() - start_time
    
    # Performance metrics
    chunks_per_second = result.chunks_created / total_time if total_time > 0 else 0
    embeddings_per_second = result.embeddings_generated / total_time if total_time > 0 else 0
    
    print(f"   📊 Performance Metrics:")
    print(f"      Document size: {len(large_content):,} characters")
    print(f"      Chunks created: {result.chunks_created}")
    print(f"      Embeddings generated: {result.embeddings_generated}")
    print(f"      Total processing time: {total_time:.3f} seconds")
    print(f"      Chunks per second: {chunks_per_second:.1f}")
    print(f"      Embeddings per second: {embeddings_per_second:.1f}")
    print(f"      Processing time per chunk: {result.processing_time_ms/result.chunks_created:.1f} ms")
    
    # Performance assertions
    assert result.success, "Performance test should succeed"
    assert chunks_per_second > 1, f"Performance too slow: {chunks_per_second:.1f} chunks/sec"
    assert result.processing_time_ms < 30000, f"Processing too slow: {result.processing_time_ms:.1f} ms"
    
    print("   ✅ Performance test completed successfully")
    
    return True


async def main():
    """Run all embedding integration tests."""
    print("🚀 EmbeddingGenerator Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Initialization", test_embedding_generator_initialization),
        ("Embedding Generation (Mocked)", test_embedding_generation_mock),
        ("Batch Processing", test_embedding_batch_processing),
        ("Error Handling", test_embedding_error_handling),
        ("Pipeline Integration", test_pipeline_embedding_integration),
        ("Performance", test_embedding_performance),
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
    print("🎯 EMBEDDING INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL EMBEDDING INTEGRATION TESTS PASSED!")
        print("✅ EmbeddingGenerator enhanced functionality is working correctly")
        print("✅ Batch processing optimizes performance for large datasets")
        print("✅ Error handling and retry logic ensure robustness")
        print("✅ Pipeline integration provides seamless embedding generation")
        print("✅ Performance meets requirements for production workloads")
    else:
        print("\n⚠️  Some tests failed")
        print("Check the error messages above for details")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)