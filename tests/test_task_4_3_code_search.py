"""
Tests for Task 4.3: Agentic RAG Tools Implementation - Code Search Functionality
Tests the search_code_examples tool implementation with hybrid search capabilities.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json
import pytest
import sys
import os
import asyncio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import RAGStrategy


class TestCodeSearchImplementation(unittest.TestCase):
    """Test the search_code_examples tool implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock Supabase client
        self.mock_supabase = Mock()

        # Mock context and strategy manager
        self.mock_ctx = Mock()
        self.mock_strategy_manager = Mock()
        self.mock_ctx.request_context.lifespan_context.supabase_client = (
            self.mock_supabase
        )
        self.mock_ctx.request_context.lifespan_context.strategy_manager = (
            self.mock_strategy_manager
        )

        # Configure strategy manager to allow the tool
        self.mock_strategy_manager.should_tool_be_available.return_value = True
        self.mock_strategy_manager.get_enabled_strategies.return_value = [
            RAGStrategy.AGENTIC_RAG
        ]

    def _setup_supabase_mock(self, data):
        """Helper method to set up the Supabase mock with proper call chain."""
        mock_rpc_result = Mock()
        mock_final_response = Mock()
        mock_final_response.data = data
        mock_rpc_result.execute.return_value = mock_final_response
        self.mock_supabase.rpc.return_value = mock_rpc_result

    @patch("utils.create_embedding")
    def test_successful_code_search(self, mock_create_embedding):
        """Test successful code search with results."""
        # Mock embedding creation
        mock_create_embedding.return_value = [0.1] * 1536

        # Set up Supabase mock with sample data
        self._setup_supabase_mock(
            [
                {
                    "id": 1,
                    "url": "https://example.com/doc",
                    "content": "def hello_world():\n    print('Hello, World!')",
                    "summary": "A simple Python function that prints Hello World",
                    "programming_language": "python",
                    "complexity_score": 2,
                    "similarity": 0.8543,
                    "rrf_score": 0.7231,
                    "semantic_rank": 1,
                    "full_text_rank": 2,
                    "metadata": {"patterns": ["function"], "keywords": ["hello"]},
                },
                {
                    "id": 2,
                    "url": "https://example.com/doc2",
                    "content": "function greet(name) {\n  return `Hello, ${name}!`;\n}",
                    "summary": "JavaScript function that greets a person by name",
                    "programming_language": "javascript",
                    "complexity_score": 3,
                    "similarity": 0.7892,
                    "rrf_score": 0.6845,
                    "semantic_rank": 2,
                    "full_text_rank": 1,
                    "metadata": {
                        "patterns": ["function", "template_literal"],
                        "keywords": ["greet"],
                    },
                },
            ]
        )

        # Import and test the function
        from crawl4ai_mcp import search_code_examples

        # Call the function (async)
        result = asyncio.run(
            search_code_examples(
                self.mock_ctx,
                "hello world function",
                programming_language="python",
                complexity_min=1,
                complexity_max=5,
                match_count=5,
            )
        )

        # Parse result
        result_data = json.loads(result)

        # Assertions
        self.assertTrue(result_data["success"])
        self.assertEqual(result_data["query"], "hello world function")
        self.assertEqual(
            result_data["enhanced_query"],
            "Code example: hello world function in python",
        )
        self.assertEqual(len(result_data["results"]), 2)

        # Check first result formatting
        first_result = result_data["results"][0]
        self.assertEqual(first_result["id"], 1)
        self.assertEqual(first_result["programming_language"], "python")
        self.assertEqual(first_result["complexity_score"], 2)
        self.assertAlmostEqual(first_result["similarity"], 0.8543, places=4)
        self.assertEqual(first_result["ranking"]["position"], 1)

        # Check search stats
        stats = result_data["search_stats"]
        self.assertEqual(stats["total_found"], 2)
        self.assertEqual(stats["returned"], 2)
        self.assertEqual(stats["search_type"], "hybrid_rrf")

        # Verify RPC was called with correct parameters
        self.mock_supabase.rpc.assert_called_once()
        call_args = self.mock_supabase.rpc.call_args
        self.assertEqual(call_args[0][0], "hybrid_search_code_examples")

        params = call_args[0][1]
        self.assertEqual(params["query_text"], "hello world function")
        self.assertEqual(params["language_filter"], "python")
        self.assertEqual(params["max_complexity"], 5)
        self.assertEqual(params["match_count"], 5)

    @patch("utils.create_embedding")
    def test_no_results_found(self, mock_create_embedding):
        """Test when no code examples match the search criteria."""
        mock_create_embedding.return_value = [0.1] * 1536

        # Set up empty response
        self._setup_supabase_mock([])

        from crawl4ai_mcp import search_code_examples

        result = asyncio.run(
            search_code_examples(
                self.mock_ctx, "nonexistent code pattern", match_count=10
            )
        )

        result_data = json.loads(result)

        self.assertTrue(result_data["success"])
        self.assertEqual(len(result_data["results"]), 0)
        self.assertEqual(result_data["count"], 0)
        self.assertIn("No code examples found", result_data["message"])

    @patch("utils.create_embedding")
    def test_complexity_filtering(self, mock_create_embedding):
        """Test that complexity filtering works correctly."""
        mock_create_embedding.return_value = [0.1] * 1536

        # Set up mock response simulating what the SQL function would return
        # The SQL function filters by max_complexity, so with complexity_max=5,
        # it should only return items with complexity_score <= 5
        # We're testing that the client-side min filtering also works
        self._setup_supabase_mock(
            [
                {
                    "id": 1,
                    "content": "simple",
                    "summary": "Simple code",
                    "complexity_score": 2,
                    "programming_language": "python",
                    "similarity": 0.9,
                    "rrf_score": 0.8,
                    "semantic_rank": 1,
                    "full_text_rank": 1,
                    "metadata": {},
                }
                # Item with complexity_score=8 would be filtered out by SQL (max_complexity=5)
                # so we don't include it in the mock response
            ]
        )

        from crawl4ai_mcp import search_code_examples

        # Search with complexity filter
        result = asyncio.run(
            search_code_examples(
                self.mock_ctx, "test code", complexity_min=1, complexity_max=5
            )
        )

        result_data = json.loads(result)

        self.assertTrue(result_data["success"])
        self.assertEqual(len(result_data["results"]), 1)  # Only simple code should pass
        self.assertEqual(result_data["results"][0]["id"], 1)
        self.assertEqual(result_data["search_stats"]["after_complexity_filter"], 1)

    @patch("utils.create_embedding")
    def test_complexity_filtering_min_filter(self, mock_create_embedding):
        """Test that client-side min complexity filtering works correctly."""
        mock_create_embedding.return_value = [0.1] * 1536

        # Set up mock response with items that would pass SQL max filter
        # but should be filtered by client-side min filter
        self._setup_supabase_mock(
            [
                {
                    "id": 1,
                    "content": "very simple",
                    "summary": "Very simple code",
                    "complexity_score": 1,
                    "programming_language": "python",
                    "similarity": 0.9,
                    "rrf_score": 0.8,
                    "semantic_rank": 1,
                    "full_text_rank": 1,
                    "metadata": {},
                },
                {
                    "id": 2,
                    "content": "medium",
                    "summary": "Medium complexity code",
                    "complexity_score": 4,
                    "programming_language": "python",
                    "similarity": 0.85,
                    "rrf_score": 0.75,
                    "semantic_rank": 2,
                    "full_text_rank": 2,
                    "metadata": {},
                },
            ]
        )

        from crawl4ai_mcp import search_code_examples

        # Search with min complexity filter that should exclude the first item
        result = asyncio.run(
            search_code_examples(
                self.mock_ctx,
                "test code",
                complexity_min=3,  # Should exclude item with complexity=1
                complexity_max=10,
            )
        )

        result_data = json.loads(result)

        self.assertTrue(result_data["success"])
        self.assertEqual(
            len(result_data["results"]), 1
        )  # Only medium complexity should pass
        self.assertEqual(result_data["results"][0]["id"], 2)
        self.assertEqual(
            result_data["search_stats"]["total_found"], 2
        )  # SQL returned 2
        self.assertEqual(
            result_data["search_stats"]["after_complexity_filter"], 1
        )  # Client filtered to 1

    @patch("utils.create_embedding")
    def test_parameter_validation(self, mock_create_embedding):
        """Test that parameters are validated and clamped correctly."""
        mock_create_embedding.return_value = [0.1] * 1536

        self._setup_supabase_mock([])

        from crawl4ai_mcp import search_code_examples

        # Test with invalid parameters that should be clamped
        result = asyncio.run(
            search_code_examples(
                self.mock_ctx,
                "test",
                complexity_min=-5,  # Should clamp to 1
                complexity_max=15,  # Should clamp to 10
                match_count=50,  # Should clamp to 30
            )
        )

        # Verify RPC call parameters were clamped
        call_args = self.mock_supabase.rpc.call_args[0][1]
        self.assertEqual(call_args["max_complexity"], 10)
        self.assertEqual(call_args["match_count"], 30)

        result_data = json.loads(result)
        filters = result_data["filters"]
        self.assertEqual(filters["complexity_range"], [1, 10])
        self.assertEqual(filters["match_count"], 30)

    @patch("utils.create_embedding")
    def test_metadata_parsing(self, mock_create_embedding):
        """Test that metadata is parsed correctly from various formats."""
        mock_create_embedding.return_value = [0.1] * 1536

        # Mock response with different metadata formats
        self._setup_supabase_mock(
            [
                {
                    "id": 1,
                    "content": "test",
                    "summary": "Test",
                    "complexity_score": 1,
                    "programming_language": "python",
                    "similarity": 0.9,
                    "rrf_score": 0.8,
                    "semantic_rank": 1,
                    "full_text_rank": 1,
                    "metadata": {"key": "value"},  # Dict format
                },
                {
                    "id": 2,
                    "content": "test2",
                    "summary": "Test2",
                    "complexity_score": 2,
                    "programming_language": "python",
                    "similarity": 0.85,
                    "rrf_score": 0.75,
                    "semantic_rank": 2,
                    "full_text_rank": 2,
                    "metadata": '{"json": "string"}',  # JSON string format
                },
                {
                    "id": 3,
                    "content": "test3",
                    "summary": "Test3",
                    "complexity_score": 3,
                    "programming_language": "python",
                    "similarity": 0.8,
                    "rrf_score": 0.7,
                    "semantic_rank": 3,
                    "full_text_rank": 3,
                    "metadata": "invalid json",  # Invalid JSON
                },
            ]
        )

        from crawl4ai_mcp import search_code_examples

        result = asyncio.run(search_code_examples(self.mock_ctx, "test"))
        result_data = json.loads(result)

        # Check metadata parsing
        results = result_data["results"]
        self.assertEqual(results[0]["metadata"], {"key": "value"})
        self.assertEqual(results[1]["metadata"], {"json": "string"})
        self.assertEqual(results[2]["metadata"], {})  # Should fallback to empty dict

    @patch("utils.create_embedding")
    def test_error_handling(self, mock_create_embedding):
        """Test error handling when search fails."""
        mock_create_embedding.return_value = [0.1] * 1536

        # Mock Supabase RPC to raise an exception
        self.mock_supabase.rpc.side_effect = Exception("Database connection error")

        from crawl4ai_mcp import search_code_examples

        result = asyncio.run(search_code_examples(self.mock_ctx, "test query"))
        result_data = json.loads(result)

        self.assertFalse(result_data["success"])
        self.assertIn("Code search failed", result_data["error"])
        self.assertEqual(result_data["query"], "test query")

    def test_enhanced_query_generation(self):
        """Test that enhanced queries are generated correctly."""
        from crawl4ai_mcp import search_code_examples

        with patch("utils.create_embedding") as mock_embedding:
            mock_embedding.return_value = [0.1] * 1536

            # Set up empty response for both test cases
            self._setup_supabase_mock([])

            # Test without language filter
            result1 = asyncio.run(search_code_examples(self.mock_ctx, "sort algorithm"))
            data1 = json.loads(result1)
            self.assertEqual(data1["enhanced_query"], "Code example: sort algorithm")

            # Reset mock for second call
            self._setup_supabase_mock([])

            # Test with language filter
            result2 = asyncio.run(
                search_code_examples(
                    self.mock_ctx, "sort algorithm", programming_language="python"
                )
            )
            data2 = json.loads(result2)
            self.assertEqual(
                data2["enhanced_query"], "Code example: sort algorithm in python"
            )

    @patch("utils.create_embedding")
    def test_result_ranking_preservation(self, mock_create_embedding):
        """Test that ranking information is preserved and formatted correctly."""
        mock_create_embedding.return_value = [0.1] * 1536

        self._setup_supabase_mock(
            [
                {
                    "id": 1,
                    "content": "first",
                    "summary": "First result",
                    "complexity_score": 1,
                    "programming_language": "python",
                    "similarity": 0.95,
                    "rrf_score": 0.9,
                    "semantic_rank": 1,
                    "full_text_rank": 3,
                    "metadata": {},
                },
                {
                    "id": 2,
                    "content": "second",
                    "summary": "Second result",
                    "complexity_score": 2,
                    "programming_language": "python",
                    "similarity": 0.85,
                    "rrf_score": 0.8,
                    "semantic_rank": 2,
                    "full_text_rank": 1,
                    "metadata": {},
                },
            ]
        )

        from crawl4ai_mcp import search_code_examples

        result = asyncio.run(search_code_examples(self.mock_ctx, "test"))
        result_data = json.loads(result)

        results = result_data["results"]

        # Check first result ranking
        first_ranking = results[0]["ranking"]
        self.assertEqual(first_ranking["position"], 1)
        self.assertEqual(first_ranking["semantic_rank"], 1)
        self.assertEqual(first_ranking["full_text_rank"], 3)

        # Check second result ranking
        second_ranking = results[1]["ranking"]
        self.assertEqual(second_ranking["position"], 2)
        self.assertEqual(second_ranking["semantic_rank"], 2)
        self.assertEqual(second_ranking["full_text_rank"], 1)


class TestCodeSearchIntegration(unittest.TestCase):
    """Integration tests for code search functionality."""

    @pytest.mark.integration
    def test_conditional_tool_availability(self):
        """Test that the tool is only available when AGENTIC_RAG is enabled."""

        mock_ctx = Mock()
        mock_strategy_manager = Mock()
        mock_ctx.request_context.lifespan_context.strategy_manager = (
            mock_strategy_manager
        )

        # Test when tool is not available
        mock_strategy_manager.should_tool_be_available.return_value = False
        mock_strategy_manager.get_enabled_strategies.return_value = []

        from crawl4ai_mcp import search_code_examples

        # The @conditional_tool decorator should prevent execution
        result = asyncio.run(search_code_examples(mock_ctx, "test"))
        result_data = json.loads(result)

        self.assertFalse(result_data["success"])
        self.assertIn("requires strategies", result_data["error"])
        self.assertEqual(result_data["tool"], "search_code_examples")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
