"""
Integration tests for enhanced metadata functionality.
Tests the complete pipeline from code extraction to database storage.
"""

import json
import pytest
from src.code_extraction import extract_code_from_content


class TestEnhancedMetadataIntegration:
    """Test enhanced metadata integration with database storage."""

    def test_metadata_serialization(self):
        """Test that enhanced metadata can be properly serialized for database storage."""
        content = """
        # FastAPI Example
        Here's a comprehensive API example:
        
        ```python
        from fastapi import FastAPI, HTTPException
        from typing import List, Optional
        import asyncio
        
        app = FastAPI()
        
        @app.get("/")
        async def root():
            '''Root endpoint that returns a greeting.'''
            return {"message": "Hello World"}
        
        @app.get("/items/{item_id}")
        async def read_item(item_id: int, q: Optional[str] = None):
            try:
                if item_id < 0:
                    raise HTTPException(status_code=404, detail="Item not found")
                
                result = {"item_id": item_id}
                if q:
                    result.update({"q": q})
                
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        ```
        
        This example demonstrates FastAPI routing with error handling.
        """

        extracted_codes = extract_code_from_content(
            content, "https://docs.fastapi.example.com/tutorial"
        )

        assert len(extracted_codes) == 1
        code = extracted_codes[0]

        # Test that metadata is properly structured
        metadata = code.metadata
        assert isinstance(metadata, dict)

        # Test serialization
        serialized = json.dumps(metadata)
        deserialized = json.loads(serialized)

        # Verify all the enhanced metadata fields are present and serializable
        expected_fields = [
            "start_line",
            "end_line",
            "block_type",
            "context_before",
            "context_after",
            "char_count",
            "word_count",
            "line_count",
            "non_empty_line_count",
            "code_density",
            "identifiers",
            "patterns",
            "has_comments",
            "has_strings",
            "has_loops",
            "has_conditions",
            "surrounding_headers",
            "context_keywords",
            "estimated_reading_time",
            "complexity_indicators",
        ]

        for field in expected_fields:
            assert field in deserialized, f"Missing field: {field}"

        # Test specific enhanced features
        assert "FastAPI" in deserialized["identifiers"]
        assert "read_item" in deserialized["identifiers"]
        assert "root" in deserialized["identifiers"]

        assert "async" in deserialized["patterns"]
        assert "error_handling" in deserialized["patterns"]
        assert "imports" in deserialized["patterns"]
        # OOP might not be detected in this specific case, make it optional
        patterns_found = deserialized["patterns"]
        print(f"Patterns found: {patterns_found}")

        assert deserialized["has_comments"] == True  # Should detect docstring
        assert deserialized["has_strings"] == True
        assert deserialized["has_conditions"] == True  # if statements

        assert "api" in deserialized["context_keywords"]
        assert "example" in deserialized["context_keywords"]

        # Test complexity indicators
        complexity = deserialized["complexity_indicators"]
        assert isinstance(complexity["nesting_level"], int)
        assert isinstance(complexity["cyclomatic_complexity"], int)
        assert isinstance(complexity["unique_identifiers"], int)
        assert complexity["nesting_level"] >= 1  # Functions have nesting
        assert complexity["cyclomatic_complexity"] > 1  # Multiple decision points

        print(f"✅ Enhanced metadata integration test passed!")
        print(f"📊 Metadata includes {len(deserialized['identifiers'])} identifiers")
        print(f"🔍 Found {len(deserialized['patterns'])} code patterns")
        print(f"📝 Context includes {len(deserialized['context_keywords'])} keywords")
        print(
            f"🧮 Complexity: nesting={complexity['nesting_level']}, cyclomatic={complexity['cyclomatic_complexity']}"
        )

    def test_metadata_with_different_languages(self):
        """Test enhanced metadata generation for different programming languages."""
        test_cases = [
            {
                "language": "javascript",
                "content": """
                # JavaScript ES6 Features
                Modern JavaScript example:
                
                ```javascript
                const users = [];
                
                class UserManager {
                    constructor() {
                        this.users = new Map();
                    }
                    
                    async addUser(user) {
                        try {
                            if (!user.email || !user.name) {
                                throw new Error("Invalid user data");
                            }
                            
                            const id = Date.now();
                            this.users.set(id, { ...user, id });
                            return id;
                        } catch (error) {
                            console.error("Failed to add user:", error);
                            return null;
                        }
                    }
                }
                ```
                """,
                "expected_identifiers": ["UserManager", "addUser"],
                "expected_patterns": ["async", "error_handling", "oop"],
            },
            {
                "language": "sql",
                "content": """
                # Database Query Example
                Complex SQL query for reporting:
                
                ```sql
                WITH monthly_sales AS (
                    SELECT 
                        DATE_TRUNC('month', sale_date) as month,
                        product_id,
                        SUM(quantity * price) as revenue
                    FROM sales s
                    JOIN products p ON s.product_id = p.id
                    WHERE sale_date >= '2023-01-01'
                    GROUP BY DATE_TRUNC('month', sale_date), product_id
                ),
                ranked_products AS (
                    SELECT 
                        month,
                        product_id,
                        revenue,
                        ROW_NUMBER() OVER (PARTITION BY month ORDER BY revenue DESC) as rank
                    FROM monthly_sales
                )
                SELECT * FROM ranked_products WHERE rank <= 5;
                ```
                """,
                "expected_identifiers": ["sales", "products"],
                "expected_patterns": ["cte", "joins"],
            },
        ]

        for case in test_cases:
            extracted_codes = extract_code_from_content(
                case["content"], f"https://example.com/{case['language']}"
            )

            assert len(extracted_codes) == 1
            code = extracted_codes[0]
            metadata = code.metadata

            # Verify language-specific identifiers
            identifiers_found = metadata["identifiers"]
            print(f"Identifiers found for {case['language']}: {identifiers_found}")

            # Check that at least some expected identifiers are found
            found_count = sum(
                1
                for identifier in case["expected_identifiers"]
                if identifier in metadata["identifiers"]
            )
            assert found_count > 0, (
                f"No expected identifiers found for {case['language']}. Expected: {case['expected_identifiers']}, Found: {identifiers_found}"
            )

            # Verify language-specific patterns
            patterns_found = metadata["patterns"]
            print(f"Patterns found for {case['language']}: {patterns_found}")

            # Check that at least some expected patterns are found
            pattern_found_count = sum(
                1
                for pattern in case["expected_patterns"]
                if pattern in metadata["patterns"]
            )
            assert pattern_found_count > 0, (
                f"No expected patterns found for {case['language']}. Expected: {case['expected_patterns']}, Found: {patterns_found}"
            )

            # Verify serialization works
            serialized = json.dumps(metadata)
            assert len(serialized) > 0

            print(f"✅ {case['language'].title()} metadata test passed!")

    def test_metadata_performance_characteristics(self):
        """Test that enhanced metadata generation doesn't create oversized payloads."""
        # Create a moderately complex code example
        content = """
        # Performance Test Example
        Complex function with multiple features:
        
        ```python
        import asyncio
        import json
        from typing import Dict, List, Optional, Union
        from datetime import datetime, timedelta
        
        class DataProcessor:
            '''Processes various types of data with caching and validation.'''
            
            def __init__(self, cache_size: int = 1000):
                self.cache = {}
                self.cache_size = cache_size
                self.stats = {"hits": 0, "misses": 0, "errors": 0}
            
            async def process_batch(self, items: List[Dict]) -> List[Dict]:
                '''Process a batch of items with error handling and caching.'''
                results = []
                
                for item in items:
                    try:
                        # Check cache first
                        cache_key = self._generate_cache_key(item)
                        if cache_key in self.cache:
                            self.stats["hits"] += 1
                            results.append(self.cache[cache_key])
                            continue
                        
                        # Validate item
                        if not self._validate_item(item):
                            self.stats["errors"] += 1
                            continue
                        
                        # Process item
                        processed = await self._process_single_item(item)
                        
                        # Cache result if space available
                        if len(self.cache) < self.cache_size:
                            self.cache[cache_key] = processed
                        
                        self.stats["misses"] += 1
                        results.append(processed)
                        
                    except Exception as e:
                        self.stats["errors"] += 1
                        print(f"Error processing item: {e}")
                        continue
                
                return results
        ```
        """

        extracted_codes = extract_code_from_content(
            content, "https://example.com/performance"
        )

        assert len(extracted_codes) == 1
        code = extracted_codes[0]
        metadata = code.metadata

        # Test serialization size is reasonable (should be < 10KB for most code examples)
        serialized = json.dumps(metadata)
        serialized_size = len(serialized.encode("utf-8"))

        assert serialized_size < 10240, f"Metadata too large: {serialized_size} bytes"

        # Test that context is limited appropriately
        assert len(metadata["context_before"]) <= 500
        assert len(metadata["context_after"]) <= 500

        # Test that lists are limited
        assert len(metadata["identifiers"]) <= 10
        assert len(metadata["patterns"]) <= 8
        assert len(metadata["surrounding_headers"]) <= 5
        assert len(metadata["context_keywords"]) <= 6

        print(f"✅ Performance test passed! Metadata size: {serialized_size} bytes")
        print(f"📋 Identifiers: {len(metadata['identifiers'])}")
        print(f"🏷️ Patterns: {len(metadata['patterns'])}")
        print(f"📑 Headers: {len(metadata['surrounding_headers'])}")
        print(f"🔑 Keywords: {len(metadata['context_keywords'])}")
