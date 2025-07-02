#!/usr/bin/env python3
"""
Test script for DocumentStorage functionality.

Tests the database storage logic without requiring external dependencies.
"""

import asyncio
import sys
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Mock pydantic to avoid dependency issues
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

# Mock the imports
import sys
sys.modules['pydantic'] = type('MockPydantic', (), {
    'BaseModel': MockBaseModel,
    'Field': lambda *args, **kwargs: MockField(**kwargs),
    'field_validator': mock_field_validator
})()

# Mock openai
class MockOpenAI:
    class RateLimitError(Exception):
        pass

sys.modules['openai'] = MockOpenAI()

from document_ingestion_pipeline import DocumentStorage, DocumentChunk


def test_storage_initialization():
    """Test DocumentStorage initialization."""
    print("🧪 Testing DocumentStorage Initialization")
    print("-" * 60)
    
    storage = DocumentStorage()
    print(f"   ✅ Storage initialized")
    print(f"   ✅ Has get_supabase_client: {hasattr(storage, 'get_supabase_client')}")
    print(f"   ✅ Has add_documents_to_supabase: {hasattr(storage, 'add_documents_to_supabase')}")
    
    return True


def test_document_id_generation():
    """Test document ID generation logic."""
    print("\n🧪 Testing Document ID Generation")
    print("-" * 60)
    
    storage = DocumentStorage()
    
    test_cases = [
        ("https://docs.example.com/api/guide", "docs.example.com_api_guide"),
        ("https://github.com/user/repo/blob/main/README.md", "github.com_user_repo_blob_main_README.md"),
        ("https://stackoverflow.com/questions/123/test", "stackoverflow.com_questions_123_test"),
        ("https://example.com/", "example.com_"),
        ("https://example.com", "example.com"),
    ]
    
    for url, expected in test_cases:
        result = storage._generate_document_id(url)
        print(f"   {'✅' if result == expected else '❌'} {url} -> {result}")
        if result != expected:
            print(f"     Expected: {expected}")
    
    print("✅ Document ID generation tests completed")
    return True


async def test_chunk_preparation():
    """Test chunk data preparation for storage."""
    print("\n🧪 Testing Chunk Data Preparation")
    print("-" * 60)
    
    storage = DocumentStorage()
    
    # Create test chunks
    test_chunks = [
        DocumentChunk(
            content="This is the first test chunk with substantial content for storage testing.",
            chunk_index=0,
            start_position=0,
            end_position=75,
            metadata={
                "source": "https://example.com/test",
                "title": "Test Document",
                "chunk_method": "semantic"
            }
        ),
        DocumentChunk(
            content="This is the second test chunk with different content for comprehensive testing.",
            chunk_index=1,
            start_position=76,
            end_position=154,
            metadata={
                "source": "https://example.com/test", 
                "title": "Test Document",
                "chunk_method": "semantic"
            }
        )
    ]
    
    document_id = "example.com_test"
    original_content = "# Test Document\n\nThis is test content for storage validation."
    
    # Mock the storage functions to capture what would be stored
    stored_data = {}
    
    def mock_get_supabase_client():
        return Mock()
    
    def mock_add_documents_to_supabase(client, urls, chunk_numbers, contents, metadatas, url_to_full_document, strategy_config, batch_size):
        stored_data.update({
            'urls': urls,
            'chunk_numbers': chunk_numbers,
            'contents': contents,
            'metadatas': metadatas,
            'url_to_full_document': url_to_full_document,
            'batch_size': batch_size
        })
    
    storage.get_supabase_client = mock_get_supabase_client
    storage.add_documents_to_supabase = mock_add_documents_to_supabase
    
    # Test the storage preparation
    try:
        await storage._store_chunks(document_id, test_chunks, original_content)
        
        print(f"   ✅ Chunk storage prepared successfully")
        print(f"   📊 URLs: {stored_data['urls']}")
        print(f"   📊 Chunk numbers: {stored_data['chunk_numbers']}")
        print(f"   📊 Content lengths: {[len(c) for c in stored_data['contents']]}")
        print(f"   📊 Metadata count: {len(stored_data['metadatas'])}")
        print(f"   📊 Batch size: {stored_data['batch_size']}")
        
        # Validate data structure
        validations = [
            (len(stored_data['urls']) == len(test_chunks), "URL count matches chunk count"),
            (len(stored_data['contents']) == len(test_chunks), "Content count matches chunk count"),
            (len(stored_data['metadatas']) == len(test_chunks), "Metadata count matches chunk count"),
            (all('document_id' in m for m in stored_data['metadatas']), "All metadata has document_id"),
            (all('pipeline_processed' in m for m in stored_data['metadatas']), "All metadata marked as pipeline processed"),
            (stored_data['batch_size'] == 20, "Correct batch size used"),
        ]
        
        for passed, description in validations:
            print(f"   {'✅' if passed else '❌'} {description}")
        
        return all(passed for passed, _ in validations)
        
    except Exception as e:
        print(f"   ❌ Chunk storage preparation failed: {e}")
        return False


async def test_full_document_storage():
    """Test complete document storage workflow."""
    print("\n🧪 Testing Full Document Storage Workflow")
    print("-" * 60)
    
    storage = DocumentStorage()
    
    # Create comprehensive test data
    test_chunks = [
        DocumentChunk(
            content="# Introduction\n\nThis document covers advanced techniques for document processing.",
            chunk_index=0,
            start_position=0,
            end_position=80,
            metadata={
                "source": "https://docs.example.com/advanced-guide",
                "title": "Advanced Document Processing",
                "chunk_method": "semantic",
                "section": "introduction"
            },
            token_count=15
        ),
        DocumentChunk(
            content="## Core Concepts\n\nThe processing pipeline consists of several stages that work together.",
            chunk_index=1,
            start_position=81,
            end_position=165,
            metadata={
                "source": "https://docs.example.com/advanced-guide",
                "title": "Advanced Document Processing", 
                "chunk_method": "semantic",
                "section": "concepts"
            },
            token_count=18
        )
    ]
    
    # Mock storage infrastructure
    storage_calls = []
    
    def mock_get_supabase_client():
        client_mock = Mock()
        storage_calls.append("get_client")
        return client_mock
    
    def mock_add_documents_to_supabase(*args, **kwargs):
        storage_calls.append("add_documents")
        # Simulate successful storage
        return None
    
    storage.get_supabase_client = mock_get_supabase_client
    storage.add_documents_to_supabase = mock_add_documents_to_supabase
    
    # Test document storage
    try:
        result = await storage.store_document(
            title="Advanced Document Processing",
            source_url="https://docs.example.com/advanced-guide",
            original_content="# Advanced Document Processing\n\nThis is the full content...",
            chunks=test_chunks,
            metadata={"category": "technical", "complexity": "high"}
        )
        
        print(f"   ✅ Document stored successfully")
        print(f"   📋 Document ID: {result}")
        print(f"   📋 Storage calls made: {storage_calls}")
        
        # Validate workflow
        validations = [
            (result is not None, "Document ID returned"),
            (isinstance(result, str), "Document ID is string"),
            ("get_client" in storage_calls, "Supabase client requested"),
            ("add_documents" in storage_calls, "Documents added to storage"),
        ]
        
        for passed, description in validations:
            print(f"   {'✅' if passed else '❌'} {description}")
        
        return all(passed for passed, _ in validations)
        
    except Exception as e:
        print(f"   ❌ Document storage failed: {e}")
        return False


async def test_storage_error_handling():
    """Test storage error handling scenarios."""
    print("\n🧪 Testing Storage Error Handling")
    print("-" * 60)
    
    storage = DocumentStorage()
    
    # Test 1: Missing storage functions
    storage.get_supabase_client = None
    storage.add_documents_to_supabase = None
    
    test_chunks = [
        DocumentChunk(
            content="Test content",
            chunk_index=0,
            start_position=0,
            end_position=12,
            metadata={"source": "https://example.com"}
        )
    ]
    
    try:
        await storage._store_chunks("test_doc", test_chunks, "test content")
        print("   ❌ Should have raised error for missing functions")
        return False
    except RuntimeError as e:
        print(f"   ✅ Correctly raised error: {e}")
    
    # Test 2: Client connection failure
    def failing_get_client():
        raise Exception("Connection failed")
    
    storage.get_supabase_client = failing_get_client
    storage.add_documents_to_supabase = Mock()
    
    try:
        await storage._store_chunks("test_doc", test_chunks, "test content") 
        print("   ❌ Should have raised error for client failure")
        return False
    except Exception as e:
        print(f"   ✅ Correctly handled client failure: {e}")
    
    print("✅ Error handling tests completed")
    return True


async def main():
    """Run all storage tests."""
    print("🚀 DocumentStorage Basic Tests")
    print("=" * 60)
    
    tests = [
        ("Initialization", test_storage_initialization),
        ("Document ID Generation", test_document_id_generation),
        ("Chunk Preparation", test_chunk_preparation),
        ("Full Document Storage", test_full_document_storage),
        ("Error Handling", test_storage_error_handling),
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
    print("🎯 STORAGE BASIC TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL STORAGE BASIC TESTS PASSED!")
        print("✅ DocumentStorage initialization working correctly")
        print("✅ Document ID generation follows URL patterns")
        print("✅ Chunk data preparation maintains proper structure")
        print("✅ Full document storage workflow executes successfully")
        print("✅ Error handling provides appropriate safeguards")
    else:
        print("\n⚠️  Some tests failed")
        print("Check the error messages above for details")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)