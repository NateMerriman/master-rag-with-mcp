#!/usr/bin/env python3
"""
Integration tests for automatic source creation functionality.
Tests the new get_or_create_source() function and its integration with add_documents_to_supabase().
"""

import pytest
import os
from unittest.mock import Mock, patch
from pathlib import Path

# Add src directory to path for testing
import sys
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils import get_or_create_source, add_documents_to_supabase


class TestSourceCreation:
    """Test source creation and integration functionality."""

    def test_get_or_create_source_creates_new_source(self):
        """Test that get_or_create_source creates a new source when none exists."""
        # Mock Supabase client
        mock_client = Mock()
        
        # Mock that no existing source is found
        mock_client.table().select().eq().limit().execute.return_value.data = []
        
        # Mock successful source creation
        mock_client.table().upsert().execute.return_value.data = [{"source_id": 123}]
        
        # Test data
        url = "https://example.com/test"
        contents = ["Test content chunk 1", "Test content chunk 2"]
        metadatas = [{"test": True}, {"test": True}]
        
        # Call function
        result = get_or_create_source(mock_client, url, contents, metadatas)
        
        # Verify result
        assert result == 123
        
        # Verify that upsert was called with correct data
        mock_client.table.assert_called_with("sources")
        upsert_call = mock_client.table().upsert.call_args
        assert upsert_call[0][0]["url"] == url
        assert upsert_call[0][0]["total_word_count"] == 8  # "Test content chunk 1" + "Test content chunk 2" = 4 + 4 = 8 words
        assert upsert_call[1]["on_conflict"] == "url"

    def test_get_or_create_source_returns_existing_source(self):
        """Test that get_or_create_source returns existing source_id when source exists."""
        # Mock Supabase client
        mock_client = Mock()
        
        # Mock that existing source is found
        mock_client.table().select().eq().limit().execute.return_value.data = [{"source_id": 456}]
        
        # Test data
        url = "https://example.com/existing"
        contents = ["Existing content"]
        metadatas = [{"test": True}]
        
        # Call function
        result = get_or_create_source(mock_client, url, contents, metadatas)
        
        # Verify result
        assert result == 456
        
        # Verify that upsert was NOT called (existing source found)
        mock_client.table().upsert.assert_not_called()

    def test_get_or_create_source_calculates_word_count_correctly(self):
        """Test that word count is calculated correctly from content chunks."""
        # Mock Supabase client
        mock_client = Mock()
        mock_client.table().select().eq().limit().execute.return_value.data = []
        mock_client.table().upsert().execute.return_value.data = [{"source_id": 789}]
        
        # Test data with different word counts
        url = "https://example.com/wordcount"
        contents = [
            "This has four words",           # 4 words
            "Short",                         # 1 word  
            "This is a longer sentence with more words in it"  # 10 words
        ]
        metadatas = [{}, {}, {}]
        
        # Call function
        result = get_or_create_source(mock_client, url, contents, metadatas)
        
        # Verify word count calculation (4 + 1 + 10 = 15)
        upsert_call = mock_client.table().upsert.call_args
        assert upsert_call[0][0]["total_word_count"] == 15

    def test_get_or_create_source_handles_errors_gracefully(self):
        """Test that get_or_create_source handles database errors gracefully."""
        # Mock Supabase client
        mock_client = Mock()
        
        # Mock initial query returns no results
        mock_client.table().select().eq().limit().execute.return_value.data = []
        
        # Mock upsert throws an exception
        mock_client.table().upsert().execute.side_effect = Exception("Database error")
        
        # Mock retry query also fails
        mock_client.table().select().eq().limit().execute.side_effect = Exception("Retry failed")
        
        # Test data
        url = "https://example.com/error"
        contents = ["Error test content"]
        metadatas = [{"test": True}]
        
        # Call function
        result = get_or_create_source(mock_client, url, contents, metadatas)
        
        # Should return None on error
        assert result is None

    @patch('utils._should_use_agentic_rag')
    @patch('utils.get_or_create_source')
    def test_add_documents_to_supabase_creates_sources(self, mock_get_or_create_source, mock_should_use_agentic_rag):
        """Test that add_documents_to_supabase automatically creates sources."""
        # Disable agentic RAG for simpler testing
        mock_should_use_agentic_rag.return_value = False
        
        # Mock source creation
        mock_get_or_create_source.side_effect = lambda client, url, contents, metadatas: {
            "https://example.com/page1": 100,
            "https://example.com/page2": 200,
        }.get(url)
        
        # Mock Supabase client
        mock_client = Mock()
        mock_client.table().delete().in_().execute.return_value = None
        mock_client.table().insert().execute.return_value = None
        
        # Test data
        urls = ["https://example.com/page1", "https://example.com/page1", "https://example.com/page2"]
        chunk_numbers = [0, 1, 0]
        contents = ["Content 1", "Content 2", "Content 3"]
        metadatas = [{"chunk": 1}, {"chunk": 2}, {"chunk": 3}]
        url_to_full_document = {
            "https://example.com/page1": "Full doc 1",
            "https://example.com/page2": "Full doc 2"
        }
        
        with patch('utils._should_use_contextual_embeddings', return_value=False), \
             patch('utils.create_embeddings_batch', return_value=[[0.1]*1536, [0.2]*1536, [0.3]*1536]):
            
            # Call function
            add_documents_to_supabase(
                mock_client, urls, chunk_numbers, contents, metadatas, url_to_full_document
            )
        
        # Verify source creation was called for unique URLs
        assert mock_get_or_create_source.call_count == 2
        
        # Verify insert was called with source_id values
        insert_call = mock_client.table().insert.call_args[0][0]
        assert insert_call[0]["source_id"] == 100  # page1
        assert insert_call[1]["source_id"] == 100  # page1 chunk 2
        assert insert_call[2]["source_id"] == 200  # page2


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])