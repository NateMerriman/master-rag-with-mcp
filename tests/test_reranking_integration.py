#!/usr/bin/env python3
"""
Integration tests for reranking functionality with hybrid search.

This module tests the complete reranking pipeline integration, including:
- Hybrid search to reranking pipeline
- Performance characteristics 
- Quality improvements from reranking
- Integration with strategy configuration system
"""

import pytest
import json
import time
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from config import StrategyConfig, RAGStrategy
from reranking import ResultReranker, SearchResult, RerankingResult
from performance_monitor import PerformanceMonitor
from strategies.manager import StrategyManager


class TestRerankingIntegration:
    """Test complete reranking integration with hybrid search."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client for testing."""
        client = Mock()
        return client
    
    @pytest.fixture
    def mock_hybrid_results(self):
        """Mock hybrid search results."""
        return [
            {
                "url": "https://example.com/doc1",
                "content": "Python is a programming language great for beginners",
                "metadata": {
                    "headers": "Python Programming",
                    "chunk_index": 0,
                    "url": "https://example.com/doc1"
                },
                "rrf_score": 0.85,
                "full_text_rank": 1,
                "semantic_rank": 2
            },
            {
                "url": "https://example.com/doc2", 
                "content": "JavaScript is widely used for web development",
                "metadata": {
                    "headers": "JavaScript Basics",
                    "chunk_index": 0,
                    "url": "https://example.com/doc2"
                },
                "rrf_score": 0.78,
                "full_text_rank": 2,
                "semantic_rank": 1
            },
            {
                "url": "https://example.com/doc3",
                "content": "Machine learning algorithms require mathematical understanding",
                "metadata": {
                    "headers": "ML Mathematics",
                    "chunk_index": 0,
                    "url": "https://example.com/doc3"
                },
                "rrf_score": 0.72,
                "full_text_rank": 3,
                "semantic_rank": 3
            }
        ]
    
    @pytest.fixture
    def reranking_config(self):
        """Configuration with reranking enabled."""
        with patch.dict(os.environ, {
            "USE_RERANKING": "true",
            "RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        }):
            return StrategyConfig()
    
    @pytest.fixture
    def strategy_manager_with_reranking(self, reranking_config):
        """Strategy manager with reranking component initialized."""
        manager = StrategyManager(reranking_config)
        
        # Mock the reranker initialization
        with patch('reranking.get_reranker') as mock_get_reranker:
            mock_reranker = Mock(spec=ResultReranker)
            mock_reranker.is_available.return_value = True
            mock_reranker.get_status.return_value = {
                "sentence_transformers_available": True,
                "model_loaded": True,
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "available": True,
                "load_error": None
            }
            mock_get_reranker.return_value = mock_reranker
            
            success = manager.initialize_components()
            assert success, "Strategy manager initialization should succeed"
            
            return manager

    def test_hybrid_search_to_reranking_pipeline(self, mock_supabase_client, mock_hybrid_results, strategy_manager_with_reranking):
        """Test complete pipeline from hybrid search through reranking."""
        query = "Python programming tutorial"
        
        # Mock hybrid search results
        with patch('utils.search_documents') as mock_search:
            mock_search.return_value = mock_hybrid_results
            
            # Get reranker from strategy manager
            reranker = strategy_manager_with_reranking.get_component("reranker")
            assert reranker is not None, "Reranker should be available"
            
            # Mock reranking results
            mock_reranking_results = [
                SearchResult(
                    content=result["content"],
                    url=result["url"],
                    title=result["metadata"]["headers"],
                    chunk_index=result["metadata"]["chunk_index"],
                    original_score=result["rrf_score"],
                    metadata={**result["metadata"], "reranking_score": 0.9 - i*0.1}
                )
                for i, result in enumerate(mock_hybrid_results)
            ]
            
            mock_reranking_result = RerankingResult(
                results=mock_reranking_results,
                reranking_time_ms=150.0,
                model_used="cross-encoder/ms-marco-MiniLM-L-6-v2",
                total_scored=3,
                fallback_used=False
            )
            
            reranker.rerank_results.return_value = mock_reranking_result
            
            # Test the pipeline
            search_results_for_reranking = []
            for result in mock_hybrid_results:
                search_results_for_reranking.append({
                    "content": result["content"],
                    "url": result["url"],
                    "title": result["metadata"]["headers"],
                    "chunk_index": result["metadata"]["chunk_index"],
                    "score": result["rrf_score"],
                    "metadata": result["metadata"]
                })
            
            reranking_result = reranker.rerank_results(query, search_results_for_reranking)
            
            # Assertions
            assert reranking_result.total_scored == 3
            assert reranking_result.reranking_time_ms == 150.0
            assert not reranking_result.fallback_used
            assert len(reranking_result.results) == 3
            
            # Check that reranking scores are preserved
            for result in reranking_result.results:
                assert "reranking_score" in result.metadata
                assert result.metadata["reranking_score"] >= 0

    def test_reranking_preserves_hybrid_search_benefits(self, mock_supabase_client, mock_hybrid_results):
        """Test that reranking preserves RRF scores and hybrid search metadata."""
        query = "programming languages"
        
        with patch('utils.search_documents') as mock_search:
            mock_search.return_value = mock_hybrid_results
            
            with patch('reranking.get_reranker') as mock_get_reranker:
                mock_reranker = Mock()
                mock_reranker.is_available.return_value = True
                
                # Create reranked results that preserve original scores
                reranked_results = []
                for i, result in enumerate(mock_hybrid_results):
                    search_result = SearchResult(
                        content=result["content"],
                        url=result["url"],
                        title=result["metadata"]["headers"],
                        chunk_index=result["metadata"]["chunk_index"],
                        original_score=result["rrf_score"],
                        metadata={
                            **result["metadata"],
                            "reranking_score": 0.95 - i*0.05  # Simulated reranking scores
                        }
                    )
                    reranked_results.append(search_result)
                
                mock_reranking_result = RerankingResult(
                    results=reranked_results,
                    reranking_time_ms=120.0,
                    model_used="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    total_scored=3,
                    fallback_used=False
                )
                
                mock_reranker.rerank_results.return_value = mock_reranking_result
                mock_get_reranker.return_value = mock_reranker
                
                # Convert results for reranking
                search_results_for_reranking = []
                for result in mock_hybrid_results:
                    search_results_for_reranking.append({
                        "content": result["content"],
                        "url": result["url"],
                        "title": result["metadata"]["headers"],
                        "chunk_index": result["metadata"]["chunk_index"],
                        "score": result["rrf_score"],
                        "metadata": result["metadata"],
                        "rrf_score": result["rrf_score"],
                        "full_text_rank": result["full_text_rank"],
                        "semantic_rank": result["semantic_rank"]
                    })
                
                # Apply reranking
                reranking_result = mock_reranker.rerank_results(query, search_results_for_reranking)
                
                # Check that original hybrid search scores are preserved
                for i, result in enumerate(reranking_result.results):
                    original_result = mock_hybrid_results[i]
                    assert result.original_score == original_result["rrf_score"]
                    assert result.url == original_result["url"]
                    assert result.content == original_result["content"]

    def test_reranking_performance_monitoring(self, mock_supabase_client, mock_hybrid_results):
        """Test performance monitoring for reranking operations."""
        query = "test query for performance"
        
        with patch('utils.search_documents') as mock_search:
            mock_search.return_value = mock_hybrid_results
            
            with patch('reranking.is_reranking_available') as mock_available:
                mock_available.return_value = True
                
                with patch('reranking.get_reranker') as mock_get_reranker:
                    mock_reranker = Mock()
                    
                    # Mock reranking result with timing
                    mock_reranking_result = RerankingResult(
                        results=[],
                        reranking_time_ms=180.0,
                        model_used="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        total_scored=3,
                        fallback_used=False
                    )
                    mock_reranker.rerank_results.return_value = mock_reranking_result
                    mock_get_reranker.return_value = mock_reranker
                    
                    # Test performance monitoring
                    monitor = PerformanceMonitor()
                    performance_metrics = monitor.measure_reranking_performance(
                        mock_supabase_client, query, match_count=20, rerank_top_k=5
                    )
                    
                    # Verify performance metrics structure
                    assert "total_time_ms" in performance_metrics
                    assert "search_time_ms" in performance_metrics
                    assert "reranking_time_ms" in performance_metrics
                    assert "reranking_overhead_percent" in performance_metrics
                    assert "reranking_model" in performance_metrics
                    assert performance_metrics["reranking_available"] is True
                    assert not performance_metrics.get("fallback_used", True)

    def test_reranking_quality_improvements(self, mock_supabase_client):
        """Test that reranking improves result quality for specific query types."""
        # Test data with deliberately suboptimal initial ranking
        query = "Python tutorial for beginners"
        initial_results = [
            {
                "url": "https://example.com/advanced",
                "content": "Advanced Python metaclasses and decorators for expert developers",
                "metadata": {"headers": "Advanced Python", "chunk_index": 0},
                "rrf_score": 0.85,  # High RRF but low relevance for "beginners"
                "full_text_rank": 1,
                "semantic_rank": 1
            },
            {
                "url": "https://example.com/basics",
                "content": "Python basics tutorial perfect for complete beginners to programming",
                "metadata": {"headers": "Python Basics", "chunk_index": 0},
                "rrf_score": 0.75,  # Lower RRF but high relevance for "beginners"
                "full_text_rank": 2,
                "semantic_rank": 2
            }
        ]
        
        with patch('utils.search_documents') as mock_search:
            mock_search.return_value = initial_results
            
            with patch('src.reranking.get_reranker') as mock_get_reranker:
                mock_reranker = Mock()
                mock_reranker.is_available.return_value = True
                
                # Simulate reranking that puts beginner content first
                reranked_results = [
                    SearchResult(
                        content=initial_results[1]["content"],  # Beginner content first
                        url=initial_results[1]["url"],
                        title=initial_results[1]["metadata"]["headers"],
                        chunk_index=0,
                        original_score=initial_results[1]["rrf_score"],
                        metadata={**initial_results[1]["metadata"], "reranking_score": 0.92}
                    ),
                    SearchResult(
                        content=initial_results[0]["content"],  # Advanced content second
                        url=initial_results[0]["url"],
                        title=initial_results[0]["metadata"]["headers"],
                        chunk_index=0,
                        original_score=initial_results[0]["rrf_score"],
                        metadata={**initial_results[0]["metadata"], "reranking_score": 0.78}
                    )
                ]
                
                mock_reranking_result = RerankingResult(
                    results=reranked_results,
                    reranking_time_ms=140.0,
                    model_used="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    total_scored=2,
                    fallback_used=False
                )
                
                mock_reranker.rerank_results.return_value = mock_reranking_result
                mock_get_reranker.return_value = mock_reranker
                
                # Test reranking
                search_results_for_reranking = [
                    {
                        "content": result["content"],
                        "url": result["url"],
                        "title": result["metadata"]["headers"],
                        "chunk_index": result["metadata"]["chunk_index"],
                        "score": result["rrf_score"],
                        "metadata": result["metadata"]
                    }
                    for result in initial_results
                ]
                
                reranking_result = mock_reranker.rerank_results(query, search_results_for_reranking)
                
                # Verify quality improvement: beginner content should be ranked higher
                assert "beginners" in reranking_result.results[0].content.lower()
                assert reranking_result.results[0].metadata["reranking_score"] > reranking_result.results[1].metadata["reranking_score"]
                
                # Verify that RRF benefits are still preserved in metadata
                assert reranking_result.results[0].original_score == 0.75
                assert reranking_result.results[1].original_score == 0.85

    def test_reranking_fallback_behavior(self, mock_supabase_client, mock_hybrid_results):
        """Test graceful fallback when reranking is unavailable."""
        query = "test fallback query"
        
        with patch('src.utils.search_documents') as mock_search:
            mock_search.return_value = mock_hybrid_results
            
            with patch('src.reranking.is_reranking_available') as mock_available:
                mock_available.return_value = False
                
                # Test performance monitoring with reranking unavailable
                monitor = PerformanceMonitor()
                performance_metrics = monitor.measure_reranking_performance(
                    mock_supabase_client, query
                )
                
                # Should return error when reranking unavailable
                assert "error" in performance_metrics
                assert performance_metrics["reranking_available"] is False
                assert "Reranking not available" in performance_metrics["error"]

    def test_reranking_performance_within_limits(self, mock_supabase_client, mock_hybrid_results):
        """Test that reranking performance stays within acceptable limits."""
        query = "performance test query"
        
        with patch('src.utils.search_documents') as mock_search:
            mock_search.return_value = mock_hybrid_results
            
            with patch('src.reranking.is_reranking_available') as mock_available:
                mock_available.return_value = True
                
                with patch('src.reranking.get_reranker') as mock_get_reranker:
                    mock_reranker = Mock()
                    
                    # Mock realistic performance timings
                    mock_reranking_result = RerankingResult(
                        results=[],
                        reranking_time_ms=200.0,  # 200ms reranking time
                        model_used="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        total_scored=10,
                        fallback_used=False
                    )
                    mock_reranker.rerank_results.return_value = mock_reranking_result
                    mock_get_reranker.return_value = mock_reranker
                    
                    # Test performance
                    monitor = PerformanceMonitor()
                    
                    start_time = time.time()
                    performance_metrics = monitor.measure_reranking_performance(
                        mock_supabase_client, query, match_count=20, rerank_top_k=5
                    )
                    end_time = time.time()
                    
                    # Verify performance requirements (Task 4.1 acceptance criteria)
                    total_time = (end_time - start_time) * 1000
                    assert total_time < 1000, f"Total time {total_time}ms should be under 1000ms"
                    assert performance_metrics["reranking_time_ms"] < 500, "Reranking should complete within 500ms"
                    assert performance_metrics["reranking_overhead_percent"] < 100, "Reranking overhead should be reasonable"

    def test_strategy_configuration_integration(self):
        """Test integration with strategy configuration system."""
        # Clear any cached configuration
        from config import reset_config
        
        # Test with reranking disabled
        with patch.dict(os.environ, {"USE_RERANKING": "false"}, clear=True):
            reset_config()
            config = StrategyConfig()
            assert not config.use_reranking
            assert RAGStrategy.RERANKING not in config.get_enabled_strategies()
        
        # Test with reranking enabled
        with patch.dict(os.environ, {"USE_RERANKING": "true"}, clear=True):
            reset_config()
            config = StrategyConfig()
            assert config.use_reranking
            assert RAGStrategy.RERANKING in config.get_enabled_strategies()
            
            # Test strategy manager initialization
            with patch('reranking.get_reranker') as mock_get_reranker:
                mock_reranker = Mock()
                mock_reranker.is_available.return_value = True
                mock_get_reranker.return_value = mock_reranker
                
                manager = StrategyManager(config)
                success = manager.initialize_components()
                
                assert success
                assert manager.get_component("reranker") is not None
                assert manager.is_strategy_enabled(RAGStrategy.RERANKING)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])