#!/usr/bin/env python3
"""
Core Functionality Tests - No External Dependencies Required

This test module validates basic functionality that can be tested
without external libraries like pytest, pydantic, crawl4ai, etc.
"""

import sys
import json
import sqlite3
import tempfile
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

# Basic test result tracking
test_results = []


def log_test_result(test_name: str, passed: bool, error: str = None):
    """Log test result."""
    test_results.append({
        'test_name': test_name,
        'passed': passed,
        'error': error,
        'timestamp': time.time()
    })
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} {test_name}")
    if error:
        print(f"    Error: {error}")


def test_python_environment():
    """Test basic Python environment."""
    try:
        # Test basic imports
        import json
        import sqlite3
        import asyncio
        import pathlib
        
        # Test basic functionality
        test_data = {"test": "data", "number": 42}
        json_str = json.dumps(test_data)
        parsed = json.loads(json_str)
        
        assert parsed == test_data, "JSON serialization failed"
        
        log_test_result("Python Environment", True)
        return True
    except Exception as e:
        log_test_result("Python Environment", False, str(e))
        return False


def test_sqlite_database():
    """Test SQLite database functionality."""
    try:
        # Create in-memory database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Create test table
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert test data
        test_data = [
            ("Test content 1", '{"type": "test"}'),
            ("Test content 2", '{"type": "validation"}')
        ]
        
        cursor.executemany(
            "INSERT INTO test_table (content, metadata) VALUES (?, ?)",
            test_data
        )
        
        # Query data
        cursor.execute("SELECT COUNT(*) FROM test_table")
        count = cursor.fetchone()[0]
        
        assert count == 2, f"Expected 2 records, got {count}"
        
        # Test JSON in metadata
        cursor.execute("SELECT metadata FROM test_table WHERE id = 1")
        metadata_str = cursor.fetchone()[0]
        metadata = json.loads(metadata_str)
        
        assert metadata["type"] == "test", "Metadata JSON parsing failed"
        
        conn.close()
        log_test_result("SQLite Database", True)
        return True
    except Exception as e:
        log_test_result("SQLite Database", False, str(e))
        return False


def test_file_operations():
    """Test basic file operations."""
    try:
        # Test file creation and writing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            test_content = "This is test content\nWith multiple lines\n"
            f.write(test_content)
            temp_file_path = f.name
        
        # Test file reading
        with open(temp_file_path, 'r') as f:
            read_content = f.read()
        
        assert read_content == test_content, "File read/write mismatch"
        
        # Test Path operations
        path_obj = Path(temp_file_path)
        assert path_obj.exists(), "Path.exists() failed"
        assert path_obj.is_file(), "Path.is_file() failed"
        
        # Cleanup
        path_obj.unlink()
        
        log_test_result("File Operations", True)
        return True
    except Exception as e:
        log_test_result("File Operations", False, str(e))
        return False


async def test_async_functionality():
    """Test async/await functionality."""
    try:
        async def async_operation(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"Completed after {delay}s"
        
        # Test single async operation
        start_time = time.time()
        result = await async_operation(0.1)
        elapsed = time.time() - start_time
        
        assert "Completed" in result, "Async operation failed"
        assert 0.08 <= elapsed <= 0.15, f"Timing issue: {elapsed}s"
        
        # Test concurrent operations
        start_time = time.time()
        tasks = [async_operation(0.05) for _ in range(3)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        assert len(results) == 3, "Concurrent operations failed"
        assert elapsed < 0.15, f"Concurrent timing issue: {elapsed}s"
        
        log_test_result("Async Functionality", True)
        return True
    except Exception as e:
        log_test_result("Async Functionality", False, str(e))
        return False


def test_text_processing():
    """Test basic text processing functionality."""
    try:
        # Test markdown-like content processing
        markdown_content = """# Test Document

## Introduction

This is a test document with **bold** text and *italic* text.

### Code Example

```python
def hello_world():
    return "Hello, World!"
```

## Lists

- Item 1
- Item 2
- Item 3

### Numbered List

1. First item
2. Second item
3. Third item

## Links

Here's a [link](https://example.com) and another [internal link](#section).
"""
        
        # Basic text analysis
        lines = markdown_content.split('\n')
        headers = [line for line in lines if line.startswith('#')]
        code_blocks = []
        in_code_block = False
        
        for line in lines:
            if line.startswith('```'):
                in_code_block = not in_code_block
                if not in_code_block:
                    code_blocks.append(line)
            elif in_code_block:
                code_blocks.append(line)
        
        # Validation
        assert len(headers) >= 3, f"Expected at least 3 headers, got {len(headers)}"
        assert len(code_blocks) > 0, "No code blocks found"
        assert "Hello, World!" in '\n'.join(code_blocks), "Code content not found"
        
        # Word count
        words = markdown_content.split()
        word_count = len([w for w in words if w.strip()])
        assert word_count > 30, f"Expected >30 words, got {word_count}"
        
        log_test_result("Text Processing", True)
        return True
    except Exception as e:
        log_test_result("Text Processing", False, str(e))
        return False


def test_data_structures():
    """Test data structure operations."""
    try:
        # Test document-like data structure
        document = {
            'id': 'test-doc-1',
            'title': 'Test Document',
            'content': 'This is test content',
            'metadata': {
                'source': 'test',
                'created_at': time.time(),
                'tags': ['test', 'validation']
            },
            'chunks': []
        }
        
        # Test chunk creation
        content = document['content']
        chunk_size = 50
        chunks = []
        
        for i in range(0, len(content), chunk_size):
            chunk = {
                'index': len(chunks),
                'content': content[i:i+chunk_size],
                'start_pos': i,
                'end_pos': min(i + chunk_size, len(content))
            }
            chunks.append(chunk)
        
        document['chunks'] = chunks
        
        # Validation
        assert len(chunks) > 0, "No chunks created"
        total_content = ''.join(chunk['content'] for chunk in chunks)
        assert total_content == content, "Chunk reconstruction failed"
        
        # Test JSON serialization
        json_str = json.dumps(document, default=str)
        reconstructed = json.loads(json_str)
        
        assert reconstructed['title'] == document['title'], "JSON serialization failed"
        assert len(reconstructed['chunks']) == len(document['chunks']), "Chunks not preserved"
        
        log_test_result("Data Structures", True)
        return True
    except Exception as e:
        log_test_result("Data Structures", False, str(e))
        return False


class MockStorage:
    """Mock storage class for testing."""
    
    def __init__(self):
        self.data = []
        self.db_path = ":memory:"
        self.connection = None
    
    async def __aenter__(self):
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
    
    async def store_document(self, title: str, content: str, metadata: Dict[str, Any]) -> int:
        """Store a document and return ID."""
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO documents (title, content, metadata) VALUES (?, ?, ?)",
            (title, content, json.dumps(metadata))
        )
        self.connection.commit()
        return cursor.lastrowid
    
    async def get_documents(self) -> List[Dict[str, Any]]:
        """Get all documents."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id, title, content, metadata, created_at FROM documents")
        rows = cursor.fetchall()
        
        documents = []
        for row in rows:
            doc = {
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'metadata': json.loads(row[3]) if row[3] else {},
                'created_at': row[4]
            }
            documents.append(doc)
        
        return documents


async def test_mock_pipeline():
    """Test mock pipeline functionality."""
    try:
        async with MockStorage() as storage:
            # Test document storage
            doc_id = await storage.store_document(
                title="Test Document",
                content="This is test content for the mock pipeline.",
                metadata={"source": "test", "type": "validation"}
            )
            
            assert doc_id is not None, "Document storage failed"
            assert isinstance(doc_id, int), "Document ID not integer"
            
            # Test document retrieval
            documents = await storage.get_documents()
            
            assert len(documents) == 1, f"Expected 1 document, got {len(documents)}"
            
            doc = documents[0]
            assert doc['title'] == "Test Document", "Title mismatch"
            assert doc['metadata']['source'] == "test", "Metadata mismatch"
            
            # Test multiple documents
            for i in range(3):
                await storage.store_document(
                    title=f"Document {i+2}",
                    content=f"Content for document {i+2}",
                    metadata={"source": "bulk_test", "index": i}
                )
            
            all_docs = await storage.get_documents()
            assert len(all_docs) == 4, f"Expected 4 documents, got {len(all_docs)}"
        
        log_test_result("Mock Pipeline", True)
        return True
    except Exception as e:
        log_test_result("Mock Pipeline", False, str(e))
        return False


def test_error_handling():
    """Test error handling capabilities."""
    try:
        error_scenarios = []
        
        # Test division by zero handling
        try:
            result = 10 / 0
        except ZeroDivisionError as e:
            error_scenarios.append(("division_by_zero", True, str(e)))
        
        # Test file not found handling
        try:
            with open("/nonexistent/file.txt", 'r') as f:
                content = f.read()
        except FileNotFoundError as e:
            error_scenarios.append(("file_not_found", True, str(e)))
        
        # Test JSON parsing error handling
        try:
            invalid_json = '{"invalid": json content}'
            parsed = json.loads(invalid_json)
        except json.JSONDecodeError as e:
            error_scenarios.append(("json_decode_error", True, str(e)))
        
        # Test database error handling
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM nonexistent_table")
        except sqlite3.OperationalError as e:
            error_scenarios.append(("database_error", True, str(e)))
            conn.close()
        
        # Validation
        assert len(error_scenarios) >= 3, f"Expected at least 3 error scenarios, got {len(error_scenarios)}"
        
        all_handled = all(scenario[1] for scenario in error_scenarios)
        assert all_handled, "Not all errors were properly handled"
        
        log_test_result("Error Handling", True)
        return True
    except Exception as e:
        log_test_result("Error Handling", False, str(e))
        return False


async def run_all_tests():
    """Run all tests and generate report."""
    print("🧪 Running Core Functionality Tests")
    print("=" * 60)
    print("Testing components that don't require external dependencies...")
    print()
    
    # Run synchronous tests
    sync_tests = [
        test_python_environment,
        test_sqlite_database,
        test_file_operations,
        test_text_processing,
        test_data_structures,
        test_error_handling
    ]
    
    for test_func in sync_tests:
        test_func()
    
    # Run async tests
    async_tests = [
        test_async_functionality,
        test_mock_pipeline
    ]
    
    for test_func in async_tests:
        await test_func()
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed_tests = [r for r in test_results if r['passed']]
    failed_tests = [r for r in test_results if not r['passed']]
    
    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {len(passed_tests)/len(test_results):.1%}")
    
    if failed_tests:
        print("\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"  - {test['test_name']}: {test['error']}")
    
    # Save results
    report = {
        'timestamp': time.time(),
        'summary': {
            'total_tests': len(test_results),
            'passed_tests': len(passed_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(passed_tests)/len(test_results)
        },
        'test_results': test_results
    }
    
    report_path = Path(__file__).parent / "core_functionality_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Report saved to: {report_path}")
    
    # Determine overall success
    overall_success = len(failed_tests) == 0
    
    if overall_success:
        print("\n🎉 All Core Functionality Tests: PASSED")
        return 0
    else:
        print("\n💥 Core Functionality Tests: FAILED")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run_all_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest runner failed with error: {e}")
        sys.exit(1)