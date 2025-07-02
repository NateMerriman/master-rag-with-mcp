#!/usr/bin/env python3
"""
Basic test script for SemanticChunker to verify functionality
without requiring external dependencies.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from document_ingestion_pipeline import (
    SemanticChunker,
    ChunkingConfig,
    DocumentChunk
)


async def test_basic_chunking():
    """Test basic chunking functionality."""
    print("🧪 Testing SemanticChunker Basic Functionality")
    print("-" * 60)
    
    # Create configuration
    config = ChunkingConfig(
        chunk_size=300,
        chunk_overlap=50,
        use_semantic_splitting=False  # Use simple chunking for reliable testing
    )
    
    # Create chunker
    chunker = SemanticChunker(config)
    print(f"✅ SemanticChunker initialized with config: {config}")
    
    # Test content
    test_content = """# Document Processing Pipeline

## Introduction

This document describes the implementation of a document processing pipeline
that takes clean markdown content and processes it for storage and retrieval.
The pipeline is designed to be modular and extensible.

## Components

### Semantic Chunker

The semantic chunker intelligently splits documents into coherent pieces.
It uses both rule-based and LLM-powered approaches to ensure optimal
chunking for downstream processing.

### Embedding Generator

The embedding generator creates vector representations of text chunks.
These embeddings enable semantic search capabilities across the document
collection.

### Storage System

The storage system persists processed chunks in the database for efficient
retrieval. It maintains compatibility with existing schemas while adding
enhanced metadata.

## Processing Flow

1. Input validation and preprocessing
2. Title and metadata extraction
3. Semantic chunking with overlap handling
4. Embedding generation for vector search
5. Database storage with transaction management

## Quality Assurance

The pipeline includes comprehensive quality checks at each stage to ensure
data integrity and processing reliability. Error handling and fallback
mechanisms are implemented throughout."""
    
    print(f"\n📄 Test content: {len(test_content)} characters")
    
    # Test chunking
    try:
        chunks = await chunker.chunk_document(
            content=test_content,
            title="Document Processing Pipeline",
            source="https://example.com/pipeline-docs",
            metadata={"framework": "documentation", "language": "en"}
        )
        
        print(f"✅ Chunking successful: {len(chunks)} chunks created")
        
        # Analyze chunks
        total_chars = sum(len(chunk.content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        print(f"\n📊 Chunking Statistics:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Total characters: {total_chars}")
        print(f"   Average chunk size: {avg_chunk_size:.0f} characters")
        print(f"   Target chunk size: {config.chunk_size}")
        
        # Show chunk details
        print(f"\n📋 Chunk Details:")
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {len(chunk.content)} chars, {chunk.token_count} tokens")
            print(f"            Positions: {chunk.start_position}-{chunk.end_position}")
            print(f"            Preview: {chunk.content[:80]}...")
        
        # Validate chunk metadata
        print(f"\n🔍 Metadata Validation:")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i, f"Chunk {i} has wrong index: {chunk.chunk_index}"
            assert chunk.metadata["title"] == "Document Processing Pipeline"
            assert chunk.metadata["source"] == "https://example.com/pipeline-docs"
            assert chunk.metadata["framework"] == "documentation"
            assert chunk.token_count > 0
        
        print("✅ All metadata validation passed")
        
        # Test content preservation
        reconstructed = "\n\n".join(chunk.content for chunk in chunks)
        assert "Document Processing Pipeline" in reconstructed
        assert "Semantic Chunker" in reconstructed
        assert "Quality Assurance" in reconstructed
        
        print("✅ Content preservation verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Chunking failed: {str(e)}")
        return False


async def test_structural_splitting():
    """Test structural splitting functionality."""
    print("\n🧪 Testing Structural Splitting")
    print("-" * 60)
    
    config = ChunkingConfig()
    chunker = SemanticChunker(config)
    
    # Test content with clear structure
    structured_content = """# Main Title

This is content under the main title.

## Section 1

Content for section 1 with multiple paragraphs.

This is the second paragraph in section 1.

### Subsection 1.1

More detailed content in a subsection.

- List item 1
- List item 2
- List item 3

## Section 2

Content for section 2.

1. Numbered item 1
2. Numbered item 2

```python
def example_code():
    return "This is a code block"
```

Final content."""
    
    # Test structural splitting
    sections = chunker._split_on_structure(structured_content)
    
    print(f"✅ Structural splitting created {len(sections)} sections")
    
    # Validate sections
    headers_found = sum(1 for section in sections if section.strip().startswith('#'))
    lists_found = sum(1 for section in sections if ('- ' in section or '1. ' in section))
    code_found = sum(1 for section in sections if '```' in section)
    
    print(f"   Headers found: {headers_found}")
    print(f"   Lists found: {lists_found}")
    print(f"   Code blocks found: {code_found}")
    
    assert headers_found >= 2, "Should find multiple headers"
    assert lists_found >= 1, "Should find lists"
    assert code_found >= 1, "Should find code blocks"
    
    print("✅ Structural splitting validation passed")
    
    return True


async def test_simple_splitting():
    """Test simple rule-based splitting."""
    print("\n🧪 Testing Simple Splitting")
    print("-" * 60)
    
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
    chunker = SemanticChunker(config)
    
    # Long text that needs splitting
    long_text = "This is a test sentence. " * 20  # ~500 characters
    
    chunks = chunker._simple_split(long_text)
    
    print(f"✅ Simple splitting created {len(chunks)} chunks from {len(long_text)} characters")
    
    # Validate chunk sizes
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: {len(chunk)} characters")
        if i < len(chunks) - 1:  # Not the last chunk
            assert len(chunk) <= config.chunk_size + 50, f"Chunk {i} too large: {len(chunk)}"
    
    # Verify content preservation
    reconstructed = " ".join(chunks)
    assert "This is a test sentence." in reconstructed
    
    print("✅ Simple splitting validation passed")
    
    return True


async def main():
    """Run all basic tests."""
    print("🚀 SemanticChunker Basic Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Chunking", test_basic_chunking),
        ("Structural Splitting", test_structural_splitting),
        ("Simple Splitting", test_simple_splitting),
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
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ SemanticChunker basic functionality is working correctly")
    else:
        print("\n⚠️  Some tests failed")
        print("Check the error messages above for details")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)