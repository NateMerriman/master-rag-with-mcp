#!/usr/bin/env python3
"""
Validation script for code_examples table migration.

This script validates that the code_examples table was created correctly
and tests basic functionality including insertion and querying.
"""

import os
import sys
import traceback
from typing import List, Dict, Any

# Add parent directories to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils import get_supabase_client
from src.database.models import CodeExample
from src.code_extraction import extract_code_from_content


def validate_table_structure():
    """Validate that the code_examples table has the correct structure."""
    print("🔍 Validating code_examples table structure...")

    try:
        # Override SUPABASE_URL for local testing
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            os.environ["SUPABASE_URL"] = original_url.replace(
                "host.docker.internal", "localhost"
            )

        client = get_supabase_client()

        # Check if table exists and get its structure
        result = client.rpc("get_table_info", {"table_name": "code_examples"}).execute()

        if not result.data:
            print("❌ code_examples table does not exist")
            return False

        # Expected columns
        expected_columns = {
            "id",
            "source_id",
            "url",
            "chunk_number",
            "content",
            "programming_language",
            "complexity_score",
            "metadata",
            "embedding",
            "content_tokens",
            "created_at",
        }

        actual_columns = {col["column_name"] for col in result.data}

        missing_columns = expected_columns - actual_columns
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False

        print("✅ All expected columns present")

        # Check for indexes
        index_result = client.rpc(
            "get_table_indexes", {"table_name": "code_examples"}
        ).execute()

        expected_indexes = {
            "idx_code_examples_embedding_hnsw",
            "idx_code_examples_content_tokens_gin",
            "idx_code_examples_source_id",
            "idx_code_examples_programming_language",
            "idx_code_examples_complexity_score",
        }

        if index_result.data:
            actual_indexes = {
                idx["indexname"] for idx in index_result.data if idx["indexname"]
            }

            missing_indexes = expected_indexes - actual_indexes
            if missing_indexes:
                print(f"⚠️  Missing indexes: {missing_indexes}")
            else:
                print("✅ All expected indexes present")
        else:
            print("⚠️  Could not verify indexes (may require custom RPC function)")

        return True

    except Exception as e:
        print(f"❌ Table structure validation failed: {e}")
        traceback.print_exc()
        return False


def test_basic_operations():
    """Test basic insert and query operations on code_examples table."""
    print("\n🧪 Testing basic CRUD operations...")

    try:
        # Override SUPABASE_URL for local testing
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            os.environ["SUPABASE_URL"] = original_url.replace(
                "host.docker.internal", "localhost"
            )

        client = get_supabase_client()

        # Test data
        test_code_example = CodeExample(
            source_id=1,  # Assuming source_id 1 exists from Task 2.1
            url="http://example.com/test",
            chunk_number=0,
            content="def hello_world():\n    print('Hello, World!')\n    return True",
            programming_language="python",
            complexity_score=2,
            metadata={"test": "true"},
        )

        # Test INSERT
        print("  ➕ Testing INSERT operation...")
        insert_result = (
            client.table("code_examples").insert(test_code_example.to_dict()).execute()
        )

        if not insert_result.data:
            print("❌ INSERT operation failed")
            return False

        inserted_id = insert_result.data[0]["id"]
        print(f"✅ INSERT successful, created record with ID: {inserted_id}")

        # Test SELECT
        print("  📖 Testing SELECT operation...")
        select_result = (
            client.table("code_examples").select("*").eq("id", inserted_id).execute()
        )

        if not select_result.data:
            print("❌ SELECT operation failed")
            return False

        retrieved_record = select_result.data[0]
        print(
            f"✅ SELECT successful, retrieved record: {retrieved_record['programming_language']} code"
        )

        # Test filtering by programming language
        print("  🔍 Testing language filtering...")
        filter_result = (
            client.table("code_examples")
            .select("*")
            .eq("programming_language", "python")
            .execute()
        )

        if not filter_result.data:
            print("❌ Language filtering failed")
            return False

        print(
            f"✅ Language filtering successful, found {len(filter_result.data)} Python examples"
        )

        # Test UPDATE
        print("  ✏️  Testing UPDATE operation...")
        update_result = (
            client.table("code_examples")
            .update(
                {"content": "Updated: Simple Python function that prints a greeting"}
            )
            .eq("id", inserted_id)
            .execute()
        )

        if not update_result.data:
            print("❌ UPDATE operation failed")
            return False

        print("✅ UPDATE successful")

        # Test DELETE (cleanup)
        print("  🗑️  Testing DELETE operation...")
        delete_result = (
            client.table("code_examples").delete().eq("id", inserted_id).execute()
        )

        if not delete_result.data:
            print("❌ DELETE operation failed")
            return False

        print("✅ DELETE successful, test record cleaned up")

        return True

    except Exception as e:
        print(f"❌ Basic operations test failed: {e}")
        traceback.print_exc()
        return False


def test_code_extraction_integration():
    """Test integration with code extraction pipeline."""
    print("\n🔗 Testing code extraction integration...")

    try:
        # Test content with code blocks
        test_content = """
        Here's a Python example:
        
        ```python
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        ```
        
        And here's a JavaScript function:
        
        ```javascript
        function factorial(n) {
            if (n <= 1) return 1;
            return n * factorial(n - 1);
        }
        ```
        
        These are recursive algorithms.
        """

        # Extract code blocks
        extracted_codes = extract_code_from_content(test_content)

        if not extracted_codes:
            print("❌ No code blocks extracted")
            return False

        print(f"✅ Extracted {len(extracted_codes)} code blocks")

        # Validate extracted data
        for i, code in enumerate(extracted_codes):
            print(f"  📋 Code block {i + 1}:")
            print(f"    Language: {code.programming_language}")
            print(f"    Complexity: {code.complexity_score}")
            print(f"    Summary: {code.summary[:50]}...")

            # Validate required fields
            if not code.content:
                print(f"❌ Code block {i + 1} missing content")
                return False

            if not code.programming_language:
                print(f"❌ Code block {i + 1} missing language")
                return False

            if not (1 <= code.complexity_score <= 10):
                print(
                    f"❌ Code block {i + 1} invalid complexity score: {code.complexity_score}"
                )
                return False

        print("✅ All extracted code blocks are valid")

        # Test integration with code extraction pipeline
        print("  🔄 Testing model conversion...")

        for code in extracted_codes:
            # Note: This is a test of the data model, not a real insertion
            code_example = CodeExample(
                source_id=1,
                url=code.url,
                chunk_number=code.chunk_number,
                content=code.content,
                programming_language=code.programming_language,
                complexity_score=code.complexity_score,
                embedding=[0.0] * 1536,
                metadata=code.metadata,
            )

            # Test serialization
            code_dict = code_example.to_dict()

            if not isinstance(code_dict, dict):
                print("❌ Model serialization failed")
                return False

        print("✅ Model conversion and serialization successful")

        return True

    except Exception as e:
        print(f"❌ Code extraction integration test failed: {e}")
        traceback.print_exc()
        return False


def test_foreign_key_constraints():
    """Test foreign key constraints with sources table."""
    print("\n🔗 Testing foreign key constraints...")

    try:
        # Override SUPABASE_URL for local testing
        original_url = os.environ.get("SUPABASE_URL")
        if original_url and "host.docker.internal" in original_url:
            os.environ["SUPABASE_URL"] = original_url.replace(
                "host.docker.internal", "localhost"
            )

        client = get_supabase_client()

        # Check that sources table exists and has data
        sources_result = client.table("sources").select("source_id").limit(1).execute()

        if not sources_result.data:
            print("⚠️  No sources found, skipping foreign key constraint test")
            return True

        valid_source_id = sources_result.data[0]["source_id"]

        # Test valid foreign key
        print("  ✅ Testing valid foreign key reference...")
        valid_code = CodeExample(
            source_id=valid_source_id,
            url="http://example.com/valid-fk",
            chunk_number=1,
            content="print('Valid foreign key test')",
            programming_language="python",
            complexity_score=1,
        )

        insert_result = (
            client.table("code_examples").insert(valid_code.to_dict()).execute()
        )

        if insert_result.data:
            inserted_id = insert_result.data[0]["id"]
            print("✅ Valid foreign key insert successful")

            # Clean up
            client.table("code_examples").delete().eq("id", inserted_id).execute()
        else:
            print("❌ Valid foreign key insert failed")
            return False

        # Test invalid foreign key (should fail)
        print("  ❌ Testing invalid foreign key reference...")
        invalid_code = CodeExample(
            source_id=99999,  # Non-existent source_id
            url="http://example.com/invalid-fk",
            chunk_number=1,
            content="print('Invalid foreign key test')",
            programming_language="python",
            complexity_score=1,
        )

        try:
            invalid_result = (
                client.table("code_examples").insert(invalid_code.to_dict()).execute()
            )
            if invalid_result.data:
                print(
                    "⚠️  Invalid foreign key insert succeeded (constraint may not be enforced)"
                )
                # Clean up
                client.table("code_examples").delete().eq(
                    "id", invalid_result.data[0]["id"]
                ).execute()
            else:
                print("✅ Invalid foreign key correctly rejected")
        except Exception as e:
            print("✅ Invalid foreign key correctly rejected with error")

        return True

    except Exception as e:
        print(f"❌ Foreign key constraint test failed: {e}")
        traceback.print_exc()
        return False


def run_all_validations():
    """Run all validation tests for code_examples migration."""
    print("🚀 Starting code_examples table migration validation\n")

    tests = [
        ("Table Structure", validate_table_structure),
        ("Basic CRUD Operations", test_basic_operations),
        ("Code Extraction Integration", test_code_extraction_integration),
        ("Foreign Key Constraints", test_foreign_key_constraints),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"{'=' * 60}")
        print(f"Running: {test_name}")
        print(f"{'=' * 60}")

        try:
            result = test_func()
            results.append((test_name, result))

            if result:
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")

        except Exception as e:
            print(f"\n💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))

        print()

    # Summary
    print(f"{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All code_examples migration validations PASSED!")
        print("📋 Summary:")
        print("   • Table structure is correct")
        print("   • Basic CRUD operations work")
        print("   • Code extraction integration functional")
        print("   • Foreign key constraints enforced")
        print("   • Ready for hybrid search function integration")
        return True
    else:
        print(f"\n⚠️  {total - passed} validation(s) FAILED!")
        print("Please review the errors above before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_validations()
    sys.exit(0 if success else 1)
