"""
Unit tests for the reranking module.

Tests cover model loading, caching, batch processing, error handling,
and fallback behavior for the cross-encoder reranking functionality.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reranking import (
    ResultReranker, 
    SearchResult, 
    RerankingResult,
    get_reranker,
    is_reranking_available,
    get_reranking_status,
    rerank_search_results,
    SENTENCE_TRANSFORMERS_AVAILABLE
)
from config import reset_config


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test basic SearchResult creation."""
        result = SearchResult(
            content="Test content",
            url="https://example.com",
            title="Test Title", 
            chunk_index=0,
            original_score=0.8
        )
        
        assert result.content == "Test content"
        assert result.url == "https://example.com"
        assert result.title == "Test Title"
        assert result.chunk_index == 0
        assert result.original_score == 0.8
        assert result.metadata == {}
    
    def test_search_result_with_metadata(self):
        """Test SearchResult with custom metadata."""
        metadata = {"source": "test", "category": "docs"}
        result = SearchResult(
            content="Test",
            url="https://example.com",
            title="Title",
            chunk_index=1,
            original_score=0.5,
            metadata=metadata
        )
        
        assert result.metadata == metadata


class TestResultReranker:
    """Test ResultReranker class."""
    
    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()
        # Clear global reranker
        import reranking
        reranking._reranker = None
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    })
    def test_reranker_initialization_without_sentence_transformers(self):
        """Test reranker initialization when sentence-transformers not available."""
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            reranker = ResultReranker()
            
            assert not reranker.is_available()
            assert reranker.load_error == "sentence-transformers not available"
            assert not reranker.model_loaded
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false", 
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    })
    def test_reranker_initialization_with_sentence_transformers(self):
        """Test reranker initialization when sentence-transformers available."""
        mock_model = Mock()
        
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder', return_value=mock_model):
            
            reranker = ResultReranker()
            
            assert reranker.is_available()
            assert reranker.model_loaded
            assert reranker.model == mock_model
            assert reranker.load_error is None
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    })
    def test_reranker_model_loading_failure(self):
        """Test graceful handling of model loading failure."""
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder', side_effect=Exception("Model load failed")):
            
            reranker = ResultReranker()
            
            assert not reranker.is_available()
            assert not reranker.model_loaded
            assert "Model load failed" in reranker.load_error
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "custom-model"
    })
    def test_custom_model_name(self):
        """Test using custom model name."""
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder') as mock_ce:
            
            reranker = ResultReranker("my-custom-model")
            
            assert reranker.model_name == "my-custom-model"
            mock_ce.assert_called_with("my-custom-model")
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    })
    def test_get_status(self):
        """Test get_status method."""
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder'):
            
            reranker = ResultReranker()
            status = reranker.get_status()
            
            assert isinstance(status, dict)
            assert "sentence_transformers_available" in status
            assert "model_loaded" in status
            assert "model_name" in status
            assert "available" in status
            assert "load_error" in status
    
    def test_create_query_document_pairs(self):
        """Test creation of query-document pairs."""
        reranker = ResultReranker()
        
        results = [
            SearchResult("Content 1", "url1", "Title 1", 0, 0.8),
            SearchResult("Content 2", "url2", "", 1, 0.7)  # No title
        ]
        
        pairs = reranker._create_query_document_pairs("test query", results)
        
        assert len(pairs) == 2
        assert pairs[0] == ("test query", "Title 1\nContent 1")
        assert pairs[1] == ("test query", "Content 2")
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    })
    def test_score_pairs_batch_model_available(self):
        """Test batch scoring when model is available."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7, 0.5]
        
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder', return_value=mock_model):
            
            reranker = ResultReranker()
            pairs = [("q", "d1"), ("q", "d2"), ("q", "d3")]
            scores = reranker._score_pairs_batch(pairs)
            
            assert scores == [0.9, 0.7, 0.5]
            mock_model.predict.assert_called_once_with(pairs)
    
    def test_score_pairs_batch_model_unavailable(self):
        """Test batch scoring when model is unavailable."""
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            reranker = ResultReranker()
            pairs = [("q", "d1"), ("q", "d2")]
            scores = reranker._score_pairs_batch(pairs)
            
            assert scores == [0.0, 0.0]
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "MAX_RERANKING_RESULTS": "5"
    })
    def test_rerank_results_fallback(self):
        """Test rerank_results with fallback behavior."""
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            reranker = ResultReranker()
            
            results = [
                {"content": "doc1", "url": "url1", "title": "title1", "score": 0.8},
                {"content": "doc2", "url": "url2", "title": "title2", "score": 0.6}
            ]
            
            reranking_result = reranker.rerank_results("test query", results)
            
            assert isinstance(reranking_result, RerankingResult)
            assert reranking_result.fallback_used
            assert len(reranking_result.results) == 2
            assert reranking_result.total_scored == 2
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "MAX_RERANKING_RESULTS": "5"
    })
    def test_rerank_results_with_model(self):
        """Test rerank_results with working model."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.5]  # First doc scores higher
        
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder', return_value=mock_model):
            
            reranker = ResultReranker()
            
            results = [
                {"content": "doc1", "url": "url1", "title": "title1", "score": 0.6},
                {"content": "doc2", "url": "url2", "title": "title2", "score": 0.8}
            ]
            
            reranking_result = reranker.rerank_results("test query", results)
            
            assert not reranking_result.fallback_used
            assert len(reranking_result.results) == 2
            # First result should be reranked higher
            assert reranking_result.results[0].content == "doc1"
            assert reranking_result.results[0].metadata['reranking_score'] == 0.9
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    })
    def test_rerank_simple(self):
        """Test simple reranking interface."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.7, 0.9, 0.5]
        
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder', return_value=mock_model):
            
            reranker = ResultReranker()
            documents = ["doc1", "doc2", "doc3"]
            
            result = reranker.rerank_simple("query", documents)
            
            # Should be sorted by score (descending)
            assert result[0] == ("doc2", 0.9)
            assert result[1] == ("doc1", 0.7)
            assert result[2] == ("doc3", 0.5)
    
    def test_rerank_simple_fallback(self):
        """Test simple reranking with fallback."""
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            reranker = ResultReranker()
            documents = ["doc1", "doc2"]
            
            result = reranker.rerank_simple("query", documents)
            
            assert result == [("doc1", 0.0), ("doc2", 0.0)]


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def setup_method(self):
        """Reset global state before each test."""
        reset_config()
        import reranking
        reranking._reranker = None
    
    @patch.dict(os.environ, {"USE_RERANKING": "false"})
    def test_get_reranker_singleton(self):
        """Test that get_reranker returns singleton instance."""
        reranker1 = get_reranker()
        reranker2 = get_reranker()
        
        assert reranker1 is reranker2
    
    @patch.dict(os.environ, {"USE_RERANKING": "false"})
    def test_is_reranking_available(self):
        """Test is_reranking_available function."""
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            assert not is_reranking_available()
        
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder'):
            # Need to reset global reranker to test availability
            import reranking
            reranking._reranker = None
            assert is_reranking_available()
    
    @patch.dict(os.environ, {"USE_RERANKING": "false"})
    def test_get_reranking_status(self):
        """Test get_reranking_status function."""
        status = get_reranking_status()
        
        assert isinstance(status, dict)
        assert "available" in status
        assert "model_name" in status
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    })
    def test_rerank_search_results_convenience(self):
        """Test convenience function for reranking."""
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            results = [{"content": "test", "url": "url", "title": "title", "score": 0.5}]
            
            reranking_result = rerank_search_results("query", results)
            
            assert isinstance(reranking_result, RerankingResult)
            assert reranking_result.fallback_used


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()
        import reranking
        reranking._reranker = None
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "RERANKING_TIMEOUT_MS": "100"
    })
    def test_timeout_warning(self):
        """Test reranking completes even with short timeout setting."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.5]
        
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder', return_value=mock_model):
            
            reranker = ResultReranker()
            results = [{"content": "test", "url": "url", "title": "title", "score": 0.5}]
            
            # Should complete successfully
            reranking_result = reranker.rerank_results("query", results)
            assert not reranking_result.fallback_used
            assert len(reranking_result.results) == 1
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    })
    def test_empty_results(self):
        """Test reranking with empty results."""
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder'):
            
            reranker = ResultReranker()
            reranking_result = reranker.rerank_results("query", [])
            
            assert len(reranking_result.results) == 0
            assert reranking_result.total_scored == 0
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    })
    def test_scoring_error_handling(self):
        """Test error handling during scoring."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Scoring failed")
        
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder', return_value=mock_model):
            
            reranker = ResultReranker()
            pairs = [("query", "doc")]
            scores = reranker._score_pairs_batch(pairs)
            
            # Should return zero scores on error
            assert scores == [0.0]
    
    @patch.dict(os.environ, {
        "USE_RERANKING": "false",
        "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "MAX_RERANKING_RESULTS": "2"
    })
    def test_max_results_limiting(self):
        """Test that results are limited by max_results."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.8]  # Only 2 scores for 2 results
        
        with patch('reranking.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('reranking.CrossEncoder', return_value=mock_model):
            
            reranker = ResultReranker()
            
            # Provide 3 results but max is 2
            results = [
                {"content": "doc1", "url": "url1", "title": "title1", "score": 0.5},
                {"content": "doc2", "url": "url2", "title": "title2", "score": 0.6},
                {"content": "doc3", "url": "url3", "title": "title3", "score": 0.7}
            ]
            
            reranking_result = reranker.rerank_results("query", results)
            
            # Should only rerank first 2 results
            assert len(reranking_result.results) == 2
            assert reranking_result.total_scored == 2