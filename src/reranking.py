"""
Cross-encoder reranking functionality for improving search result quality.

This module provides a cross-encoder based reranking system that can improve
the relevance ordering of search results by scoring query-document pairs.
"""

import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None

from config import get_config


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with content and metadata."""
    content: str
    url: str
    title: str
    chunk_index: int
    original_score: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class RerankingResult:
    """Result of reranking operation with timing and metadata."""
    results: List[SearchResult]
    reranking_time_ms: float
    model_used: str
    total_scored: int
    fallback_used: bool = False


class ResultReranker:
    """
    Cross-encoder based result reranker for improving search quality.
    
    Uses sentence-transformers cross-encoder models to score query-document pairs
    and reorder results based on relevance. Includes caching, batch processing,
    and graceful fallback behavior.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use.
                       If None, uses model from configuration.
        """
        self.config = get_config()
        self.model_name = model_name or self.config.reranking_model
        self.model: Optional[CrossEncoder] = None
        self.model_loaded = False
        self.load_error: Optional[str] = None
        
        # Initialize model if sentence-transformers is available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            self.load_error = "sentence-transformers not available"
            logger.warning("sentence-transformers not installed, reranking will use fallback")
    
    def _load_model(self) -> None:
        """Load the cross-encoder model with error handling."""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            start_time = time.time()
            
            self.model = CrossEncoder(self.model_name)
            self.model_loaded = True
            
            load_time = (time.time() - start_time) * 1000
            logger.info(f"Model loaded successfully in {load_time:.2f}ms")
            
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
            self.model = None
            self.model_loaded = False
    
    def is_available(self) -> bool:
        """Check if reranking is available."""
        return SENTENCE_TRANSFORMERS_AVAILABLE and self.model_loaded and self.model is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information about the reranker."""
        return {
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "available": self.is_available(),
            "load_error": self.load_error
        }
    
    def _create_query_document_pairs(self, query: str, results: List[SearchResult]) -> List[Tuple[str, str]]:
        """Create query-document pairs for scoring."""
        pairs = []
        for result in results:
            # Combine title and content for better context
            document_text = f"{result.title}\n{result.content}" if result.title else result.content
            pairs.append((query, document_text))
        return pairs
    
    def _score_pairs_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score query-document pairs in batch."""
        if not self.is_available():
            # Return original scores (no reranking)
            return [0.0] * len(pairs)
        
        try:
            # Score pairs with timeout consideration
            scores = self.model.predict(pairs)
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            
        except Exception as e:
            logger.error(f"Error during batch scoring: {e}")
            return [0.0] * len(pairs)
    
    def rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        max_results: Optional[int] = None
    ) -> RerankingResult:
        """
        Rerank search results using cross-encoder scoring.
        
        Args:
            query: The search query
            results: List of search result dictionaries
            max_results: Maximum number of results to rerank (uses config if None)
            
        Returns:
            RerankingResult with reranked results and metadata
        """
        start_time = time.time()
        max_results = max_results or self.config.max_reranking_results
        
        # Convert to SearchResult objects
        search_results = []
        for i, result in enumerate(results[:max_results]):
            search_result = SearchResult(
                content=result.get('content', ''),
                url=result.get('url', ''),
                title=result.get('title', ''),
                chunk_index=result.get('chunk_index', i),
                original_score=result.get('score', 0.0),
                metadata=result.get('metadata', {})
            )
            search_results.append(search_result)
        
        # Check if reranking is available
        if not self.is_available():
            logger.warning("Reranking not available, returning original order")
            reranking_time = (time.time() - start_time) * 1000
            return RerankingResult(
                results=search_results,
                reranking_time_ms=reranking_time,
                model_used=self.model_name,
                total_scored=len(search_results),
                fallback_used=True
            )
        
        # Create query-document pairs
        pairs = self._create_query_document_pairs(query, search_results)
        
        # Score pairs in batch
        scores = self._score_pairs_batch(pairs)
        
        # Update results with reranking scores and sort
        for result, score in zip(search_results, scores):
            result.metadata['reranking_score'] = score
        
        # Sort by reranking score (descending)
        reranked_results = sorted(search_results, key=lambda x: x.metadata.get('reranking_score', 0.0), reverse=True)
        
        reranking_time = (time.time() - start_time) * 1000
        
        # Check timeout
        if reranking_time > self.config.reranking_timeout_ms:
            logger.warning(f"Reranking took {reranking_time:.2f}ms, exceeds timeout of {self.config.reranking_timeout_ms}ms")
        
        return RerankingResult(
            results=reranked_results,
            reranking_time_ms=reranking_time,
            model_used=self.model_name,
            total_scored=len(pairs),
            fallback_used=False
        )
    
    def rerank_simple(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        Simple reranking interface for documents.
        
        Args:
            query: The search query
            documents: List of document strings
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not self.is_available():
            return [(doc, 0.0) for doc in documents]
        
        pairs = [(query, doc) for doc in documents]
        scores = self._score_pairs_batch(pairs)
        
        # Combine documents with scores and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores


# Global reranker instance (initialized lazily)
_reranker: Optional[ResultReranker] = None


def get_reranker() -> ResultReranker:
    """
    Get the global reranker instance.
    
    Returns:
        ResultReranker instance with the configured model
    """
    global _reranker
    if _reranker is None:
        _reranker = ResultReranker()
    return _reranker


def is_reranking_available() -> bool:
    """Check if reranking functionality is available."""
    return get_reranker().is_available()


def get_reranking_status() -> Dict[str, Any]:
    """Get detailed status of reranking functionality."""
    return get_reranker().get_status()


# Convenience function for quick reranking
def rerank_search_results(query: str, results: List[Dict[str, Any]]) -> RerankingResult:
    """
    Convenience function to rerank search results.
    
    Args:
        query: The search query
        results: List of search result dictionaries
        
    Returns:
        RerankingResult with reranked results
    """
    return get_reranker().rerank_results(query, results)