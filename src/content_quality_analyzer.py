#!/usr/bin/env python3
"""
Advanced Content Quality Metrics and Analysis System

This module implements Subtask 18.2 by building on the existing quality validation
infrastructure (content_quality.py and crawler_quality_validation.py) to provide
enhanced metrics for measuring content quality improvements.

Features:
- Content-to-navigation ratio with advanced classification
- Semantic coherence measurement using embedding similarity
- Link preservation tracking for internal documentation links
- Duplicate content detection using text similarity algorithms
- Quality score aggregation and comprehensive reporting
- Integration with existing ContentQualityMetrics and QualityValidationResult
"""

import re
import time
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import difflib
from pathlib import Path

# Optional dependencies for advanced features
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Import existing quality systems
try:
    from content_quality import (
        ContentQualityMetrics,
        ContentQualityAnalyzer,
        calculate_content_quality
    )
    from crawler_quality_validation import (
        QualityValidationResult, 
        ContentQualityValidator,
        validate_crawler_output
    )
    QUALITY_SYSTEMS_AVAILABLE = True
except ImportError:
    QUALITY_SYSTEMS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SemanticCoherenceMetrics:
    """Metrics for semantic coherence analysis."""
    
    chunk_count: int = 0
    avg_inter_chunk_similarity: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    coherence_variance: float = 0.0
    
    # Semantic clusters
    distinct_topics_detected: int = 0
    topic_distribution_entropy: float = 0.0
    
    # Calculation metadata
    embedding_model_used: Optional[str] = None
    calculation_time_ms: float = 0.0


@dataclass
class LinkPreservationMetrics:
    """Metrics for link preservation analysis."""
    
    total_links_found: int = 0
    internal_documentation_links: int = 0
    external_links: int = 0
    broken_or_malformed_links: int = 0
    
    # Link categories
    code_reference_links: int = 0
    glossary_definition_links: int = 0
    cross_section_links: int = 0
    
    # Quality assessment
    link_preservation_ratio: float = 0.0  # Preserved / Expected
    link_context_quality: float = 0.0     # How well links are integrated in context
    
    # Preserved link examples for validation
    preserved_link_samples: List[str] = field(default_factory=list)


@dataclass
class DuplicateContentMetrics:
    """Metrics for duplicate content detection."""
    
    total_text_blocks: int = 0
    duplicate_blocks_found: int = 0
    duplication_ratio: float = 0.0
    
    # Duplicate categories
    navigation_duplicates: int = 0
    footer_header_duplicates: int = 0
    sidebar_duplicates: int = 0
    content_duplicates: int = 0
    
    # Similarity analysis
    avg_similarity_score: float = 0.0
    high_similarity_pairs: int = 0
    
    # Examples for manual review
    duplicate_examples: List[Tuple[str, str, float]] = field(default_factory=list)


@dataclass
class EnhancedQualityMetrics:
    """
    Comprehensive quality metrics building on existing systems.
    
    This combines metrics from the existing quality systems with new
    advanced analysis for semantic coherence, link preservation, and
    duplicate content detection.
    """
    
    # Integration with existing systems
    base_content_quality: Optional[ContentQualityMetrics] = None
    base_validation_result: Optional[QualityValidationResult] = None
    
    # New advanced metrics (Subtask 18.2)
    semantic_coherence: Optional[SemanticCoherenceMetrics] = None
    link_preservation: Optional[LinkPreservationMetrics] = None
    duplicate_content: Optional[DuplicateContentMetrics] = None
    
    # Unified quality assessment
    combined_quality_score: float = 0.0
    quality_improvement_score: float = 0.0  # vs baseline
    meets_enhanced_standards: bool = False
    
    # Analysis metadata
    analysis_timestamp: str = ""
    analysis_time_ms: float = 0.0
    url_analyzed: str = ""
    page_type: str = ""
    
    # Recommendations
    priority_improvements: List[str] = field(default_factory=list)
    technical_recommendations: List[str] = field(default_factory=list)


class AdvancedContentQualityAnalyzer:
    """
    Advanced content quality analyzer that builds on existing systems.
    
    This analyzer implements Subtask 18.2 by providing enhanced metrics
    while maintaining compatibility with existing quality validation
    infrastructure from content_quality.py and crawler_quality_validation.py.
    """
    
    def __init__(self, 
                 enable_embeddings: bool = True,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.8):
        """
        Initialize the advanced quality analyzer.
        
        Args:
            enable_embeddings: Whether to use embedding-based semantic analysis
            embedding_model: SentenceTransformer model name for embeddings
            similarity_threshold: Threshold for considering content as duplicate
        """
        self.enable_embeddings = enable_embeddings and EMBEDDINGS_AVAILABLE
        self.similarity_threshold = similarity_threshold
        
        # Initialize embedding model if available
        self.embedding_model = None
        self.embedding_model_name = embedding_model
        
        if self.enable_embeddings:
            try:
                logger.info(f"Loading embedding model: {embedding_model}")
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info("✅ Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.enable_embeddings = False
        
        # Initialize existing quality systems
        self.base_analyzer = ContentQualityAnalyzer() if QUALITY_SYSTEMS_AVAILABLE else None
        self.base_validator = ContentQualityValidator() if QUALITY_SYSTEMS_AVAILABLE else None
        
        # Internal link detection patterns
        self.internal_link_patterns = [
            r'\[([^\]]+)\]\((/[^)]*|#[^)]*)\)',  # Markdown internal links
            r'\[([^\]]+)\]\((?:\.\.?/[^)]*)\)',   # Relative links
            r'\[([^\]]+)\]\((?:[^:)]*\.html?[^)]*)\)'  # HTML file links
        ]
        
        # Code reference patterns
        self.code_reference_patterns = [
            r'\[([^\]]*(?:function|method|class|api)[^\]]*)\]',
            r'\[`([^`]+)`\]',
            r'\[([^\]]*\.(?:js|py|ts|json|yaml|yml)[^\]]*)\]'
        ]
        
        # Glossary definition patterns
        self.glossary_patterns = [
            r'\[([^\]]*(?:definition|glossary|term)[^\]]*)\]',
            r'\[([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\]\(.*glossary.*\)'
        ]
    
    def analyze_content(self, markdown_content: str, 
                       url: str = "", 
                       page_type: str = "",
                       expected_links: Optional[List[str]] = None) -> EnhancedQualityMetrics:
        """
        Perform comprehensive quality analysis with enhanced metrics.
        
        Args:
            markdown_content: The markdown content to analyze
            url: Source URL for context
            page_type: Type of page (glossary, guide, api, etc.)
            expected_links: Optional list of links that should be preserved
            
        Returns:
            EnhancedQualityMetrics with comprehensive analysis
        """
        start_time = time.time()
        
        logger.info(f"Starting enhanced quality analysis for {url or 'unknown URL'}")
        
        # Get base quality metrics from existing systems
        base_content_quality = None
        base_validation_result = None
        
        if QUALITY_SYSTEMS_AVAILABLE:
            # Use existing ContentQualityAnalyzer
            base_content_quality = calculate_content_quality(markdown_content)
            
            # Use existing ContentQualityValidator
            base_validation_result = validate_crawler_output(markdown_content, url)
        
        # Perform enhanced analysis
        semantic_coherence = self._analyze_semantic_coherence(markdown_content)
        link_preservation = self._analyze_link_preservation(markdown_content, expected_links)
        duplicate_content = self._analyze_duplicate_content(markdown_content)
        
        # Calculate combined quality score
        combined_score = self._calculate_combined_quality_score(
            base_content_quality, 
            base_validation_result,
            semantic_coherence,
            link_preservation,
            duplicate_content
        )
        
        # Determine if content meets enhanced standards
        meets_enhanced = self._meets_enhanced_quality_standards(
            combined_score,
            semantic_coherence,
            link_preservation,
            duplicate_content
        )
        
        # Generate recommendations
        priority_improvements, technical_recommendations = self._generate_recommendations(
            base_content_quality,
            base_validation_result,
            semantic_coherence,
            link_preservation,
            duplicate_content
        )
        
        analysis_time = (time.time() - start_time) * 1000
        
        result = EnhancedQualityMetrics(
            base_content_quality=base_content_quality,
            base_validation_result=base_validation_result,
            semantic_coherence=semantic_coherence,
            link_preservation=link_preservation,
            duplicate_content=duplicate_content,
            combined_quality_score=combined_score,
            quality_improvement_score=self._calculate_improvement_score(combined_score),
            meets_enhanced_standards=meets_enhanced,
            analysis_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            analysis_time_ms=analysis_time,
            url_analyzed=url,
            page_type=page_type,
            priority_improvements=priority_improvements,
            technical_recommendations=technical_recommendations
        )
        
        logger.info(f"Enhanced analysis completed: {combined_score:.3f} score, {meets_enhanced} enhanced standards")
        
        return result
    
    def _analyze_semantic_coherence(self, markdown_content: str) -> SemanticCoherenceMetrics:
        """Analyze semantic coherence using embedding similarity."""
        
        start_time = time.time()
        
        if not self.enable_embeddings or not self.embedding_model:
            logger.info("Semantic coherence analysis skipped: embeddings not available")
            return SemanticCoherenceMetrics(
                embedding_model_used="none",
                calculation_time_ms=(time.time() - start_time) * 1000
            )
        
        # Split content into semantic chunks
        chunks = self._split_into_semantic_chunks(markdown_content)
        
        if len(chunks) < 2:
            return SemanticCoherenceMetrics(
                chunk_count=len(chunks),
                embedding_model_used=self.embedding_model_name,
                calculation_time_ms=(time.time() - start_time) * 1000
            )
        
        try:
            # Generate embeddings for each chunk
            embeddings = self.embedding_model.encode(chunks)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append(similarity)
            
            # Calculate coherence metrics
            avg_similarity = np.mean(similarities) if similarities else 0.0
            min_similarity = np.min(similarities) if similarities else 0.0
            max_similarity = np.max(similarities) if similarities else 0.0
            coherence_variance = np.var(similarities) if similarities else 0.0
            
            # Estimate topic diversity (simplified clustering)
            distinct_topics = self._estimate_topic_count(embeddings)
            topic_entropy = self._calculate_topic_entropy(embeddings)
            
            return SemanticCoherenceMetrics(
                chunk_count=len(chunks),
                avg_inter_chunk_similarity=float(avg_similarity),
                min_similarity=float(min_similarity),
                max_similarity=float(max_similarity),
                coherence_variance=float(coherence_variance),
                distinct_topics_detected=distinct_topics,
                topic_distribution_entropy=float(topic_entropy),
                embedding_model_used=self.embedding_model_name,
                calculation_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Semantic coherence analysis failed: {e}")
            return SemanticCoherenceMetrics(
                chunk_count=len(chunks),
                embedding_model_used=f"{self.embedding_model_name} (failed)",
                calculation_time_ms=(time.time() - start_time) * 1000
            )
    
    def _analyze_link_preservation(self, markdown_content: str, 
                                 expected_links: Optional[List[str]] = None) -> LinkPreservationMetrics:
        """Analyze link preservation and categorization."""
        
        # Extract all links from content
        all_links = self._extract_all_links(markdown_content)
        
        # Categorize links
        internal_links = []
        external_links = []
        broken_links = []
        
        for link_text, link_url in all_links:
            if self._is_internal_documentation_link(link_url):
                internal_links.append((link_text, link_url))
            elif self._is_external_link(link_url):
                external_links.append((link_text, link_url))
            else:
                # Check for malformed links
                if self._is_broken_or_malformed(link_url):
                    broken_links.append((link_text, link_url))
        
        # Categorize by link type
        code_reference_links = self._count_code_reference_links(all_links)
        glossary_definition_links = self._count_glossary_links(all_links)
        cross_section_links = self._count_cross_section_links(all_links)
        
        # Calculate preservation ratio
        preservation_ratio = 1.0  # Default to perfect if no expected links
        if expected_links:
            preserved_count = self._count_preserved_links(all_links, expected_links)
            preservation_ratio = preserved_count / len(expected_links) if expected_links else 0.0
        
        # Assess link context quality
        context_quality = self._assess_link_context_quality(markdown_content, all_links)
        
        # Get sample preserved links for validation
        preserved_samples = [f"{text} -> {url}" for text, url in internal_links[:5]]
        
        return LinkPreservationMetrics(
            total_links_found=len(all_links),
            internal_documentation_links=len(internal_links),
            external_links=len(external_links),
            broken_or_malformed_links=len(broken_links),
            code_reference_links=code_reference_links,
            glossary_definition_links=glossary_definition_links,
            cross_section_links=cross_section_links,
            link_preservation_ratio=preservation_ratio,
            link_context_quality=context_quality,
            preserved_link_samples=preserved_samples
        )
    
    def _analyze_duplicate_content(self, markdown_content: str) -> DuplicateContentMetrics:
        """Analyze duplicate content patterns."""
        
        # Split content into text blocks
        text_blocks = self._split_into_text_blocks(markdown_content)
        
        if len(text_blocks) < 2:
            return DuplicateContentMetrics(
                total_text_blocks=len(text_blocks)
            )
        
        # Find similar blocks
        duplicate_pairs = []
        similarities = []
        
        for i in range(len(text_blocks)):
            for j in range(i + 1, len(text_blocks)):
                similarity = self._calculate_text_similarity(text_blocks[i], text_blocks[j])
                similarities.append(similarity)
                
                if similarity >= self.similarity_threshold:
                    duplicate_pairs.append((text_blocks[i], text_blocks[j], similarity))
        
        # Categorize duplicates
        nav_duplicates = self._count_navigation_duplicates(duplicate_pairs)
        footer_header_duplicates = self._count_footer_header_duplicates(duplicate_pairs)
        sidebar_duplicates = self._count_sidebar_duplicates(duplicate_pairs)
        content_duplicates = len(duplicate_pairs) - nav_duplicates - footer_header_duplicates - sidebar_duplicates
        
        # Calculate metrics
        duplication_ratio = len(duplicate_pairs) / max(1, len(text_blocks))
        avg_similarity = np.mean(similarities) if similarities else 0.0
        high_similarity_count = sum(1 for s in similarities if s >= 0.9)
        
        return DuplicateContentMetrics(
            total_text_blocks=len(text_blocks),
            duplicate_blocks_found=len(duplicate_pairs),
            duplication_ratio=duplication_ratio,
            navigation_duplicates=nav_duplicates,
            footer_header_duplicates=footer_header_duplicates,
            sidebar_duplicates=sidebar_duplicates,
            content_duplicates=content_duplicates,
            avg_similarity_score=float(avg_similarity),
            high_similarity_pairs=high_similarity_count,
            duplicate_examples=duplicate_pairs[:3]  # Keep top 3 examples
        )
    
    def _calculate_combined_quality_score(self,
                                        base_content: Optional[ContentQualityMetrics],
                                        base_validation: Optional[QualityValidationResult],
                                        semantic: Optional[SemanticCoherenceMetrics],
                                        links: Optional[LinkPreservationMetrics],
                                        duplicates: Optional[DuplicateContentMetrics]) -> float:
        """Calculate combined quality score from all metrics."""
        
        score = 0.0
        weight_sum = 0.0
        
        # Base content quality (40% weight)
        if base_content:
            score += base_content.overall_quality_score * 0.4
            weight_sum += 0.4
        
        # Base validation (20% weight)
        if base_validation:
            score += base_validation.score * 0.2
            weight_sum += 0.2
        
        # Semantic coherence (15% weight)
        if semantic and semantic.chunk_count > 1:
            semantic_score = min(1.0, semantic.avg_inter_chunk_similarity + 0.2)
            score += semantic_score * 0.15
            weight_sum += 0.15
        
        # Link preservation (15% weight)
        if links:
            link_score = (links.link_preservation_ratio + links.link_context_quality) / 2
            score += link_score * 0.15
            weight_sum += 0.15
        
        # Duplicate content penalty (10% weight)
        if duplicates:
            duplicate_penalty = max(0.0, 1.0 - duplicates.duplication_ratio)
            score += duplicate_penalty * 0.1
            weight_sum += 0.1
        
        # Normalize by actual weights used
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _meets_enhanced_quality_standards(self,
                                        combined_score: float,
                                        semantic: Optional[SemanticCoherenceMetrics],
                                        links: Optional[LinkPreservationMetrics],
                                        duplicates: Optional[DuplicateContentMetrics]) -> bool:
        """Determine if content meets enhanced quality standards."""
        
        # Base threshold
        if combined_score < 0.75:
            return False
        
        # Semantic coherence check
        if semantic and semantic.chunk_count > 1:
            if semantic.avg_inter_chunk_similarity < 0.3:
                return False
        
        # Link preservation check
        if links:
            if links.link_preservation_ratio < 0.8 or links.link_context_quality < 0.7:
                return False
        
        # Duplicate content check
        if duplicates:
            if duplicates.duplication_ratio > 0.3:
                return False
        
        return True
    
    def _generate_recommendations(self,
                                base_content: Optional[ContentQualityMetrics],
                                base_validation: Optional[QualityValidationResult],
                                semantic: Optional[SemanticCoherenceMetrics],
                                links: Optional[LinkPreservationMetrics],
                                duplicates: Optional[DuplicateContentMetrics]) -> Tuple[List[str], List[str]]:
        """Generate priority improvements and technical recommendations."""
        
        priority_improvements = []
        technical_recommendations = []
        
        # Base quality recommendations
        if base_content and base_content.overall_quality_score < 0.7:
            priority_improvements.extend(base_content.improvement_suggestions[:2])
        
        if base_validation and not base_validation.passed:
            priority_improvements.extend(base_validation.recommendations[:2])
        
        # Semantic coherence recommendations
        if semantic and semantic.chunk_count > 1:
            if semantic.avg_inter_chunk_similarity < 0.4:
                priority_improvements.append("Low semantic coherence detected - review content structure and topic flow")
                technical_recommendations.append("Consider using more specific CSS selectors to target coherent content sections")
        
        # Link preservation recommendations
        if links:
            if links.link_preservation_ratio < 0.8:
                priority_improvements.append("Important documentation links are being lost during extraction")
                technical_recommendations.append("Review excluded selectors to ensure internal links are preserved")
            
            if links.broken_or_malformed_links > 0:
                technical_recommendations.append(f"Fix {links.broken_or_malformed_links} broken or malformed links")
        
        # Duplicate content recommendations
        if duplicates and duplicates.duplication_ratio > 0.2:
            priority_improvements.append("High duplicate content detected - likely navigation contamination")
            technical_recommendations.append("Add more aggressive navigation and sidebar exclusion selectors")
        
        return priority_improvements[:5], technical_recommendations[:5]
    
    def _calculate_improvement_score(self, combined_score: float) -> float:
        """Calculate improvement score vs baseline (placeholder)."""
        # This would compare against historical baseline scores
        # For now, return a simple calculation
        baseline_threshold = 0.6
        return max(0.0, (combined_score - baseline_threshold) / (1.0 - baseline_threshold))
    
    # Helper methods for analysis
    
    def _split_into_semantic_chunks(self, content: str) -> List[str]:
        """Split content into semantic chunks for coherence analysis."""
        # Split by double newlines (paragraphs) and headers
        chunks = []
        
        # Split by headers first
        header_pattern = r'^#+\s+.+$'
        sections = re.split(header_pattern, content, flags=re.MULTILINE)
        
        for section in sections:
            # Further split by double newlines
            paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
            
            # Combine small paragraphs, split large ones
            current_chunk = ""
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) < 300:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.split()) >= 10]  # Filter very short chunks
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        if not NUMPY_AVAILABLE:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _estimate_topic_count(self, embeddings) -> int:
        """Estimate number of distinct topics using simple clustering."""
        if len(embeddings) < 2:
            return len(embeddings)
        
        # Simple approach: count embeddings that are dissimilar to all others
        distinct_topics = 1
        threshold = 0.5
        
        for i in range(1, len(embeddings)):
            is_similar_to_existing = False
            for j in range(i):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                if similarity > threshold:
                    is_similar_to_existing = True
                    break
            
            if not is_similar_to_existing:
                distinct_topics += 1
        
        return distinct_topics
    
    def _calculate_topic_entropy(self, embeddings) -> float:
        """Calculate topic distribution entropy (simplified)."""
        if len(embeddings) < 2:
            return 0.0
        
        # Simple approximation based on similarity distribution
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarities.append(self._cosine_similarity(embeddings[i], embeddings[j]))
        
        # Bin similarities and calculate entropy
        high_sim = sum(1 for s in similarities if s > 0.7)
        med_sim = sum(1 for s in similarities if 0.3 < s <= 0.7)
        low_sim = len(similarities) - high_sim - med_sim
        
        total = len(similarities)
        if total == 0:
            return 0.0
        
        # Simple entropy calculation
        entropy = 0.0
        for count in [high_sim, med_sim, low_sim]:
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _extract_all_links(self, content: str) -> List[Tuple[str, str]]:
        """Extract all links from markdown content."""
        links = []
        
        # Markdown links [text](url)
        markdown_pattern = r'\[([^\]]*)\]\(([^)]*)\)'
        links.extend(re.findall(markdown_pattern, content))
        
        # Reference links [text][ref] - simplified
        reference_pattern = r'\[([^\]]*)\]\[([^\]]*)\]'
        links.extend(re.findall(reference_pattern, content))
        
        return links
    
    def _is_internal_documentation_link(self, url: str) -> bool:
        """Check if link is internal documentation."""
        if not url:
            return False
        
        return (url.startswith('/') or 
                url.startswith('#') or 
                url.startswith('../') or
                url.startswith('./') or
                'docs.' in url or
                url.endswith('.html') or
                url.endswith('.md'))
    
    def _is_external_link(self, url: str) -> bool:
        """Check if link is external."""
        return url.startswith('http://') or url.startswith('https://')
    
    def _is_broken_or_malformed(self, url: str) -> bool:
        """Check if link appears broken or malformed."""
        if not url or url.strip() == "":
            return True
        
        # Check for obvious malformations
        malformed_patterns = [
            r'^\s*$',           # Empty or whitespace only
            r'^[^a-zA-Z/#.]',   # Starts with invalid character
            r'\s+',             # Contains whitespace
            r'[<>"]',           # Contains HTML characters
        ]
        
        for pattern in malformed_patterns:
            if re.search(pattern, url):
                return True
        
        return False
    
    def _count_code_reference_links(self, links: List[Tuple[str, str]]) -> int:
        """Count links that reference code."""
        count = 0
        for text, url in links:
            for pattern in self.code_reference_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    count += 1
                    break
        return count
    
    def _count_glossary_links(self, links: List[Tuple[str, str]]) -> int:
        """Count links to glossary definitions."""
        count = 0
        for text, url in links:
            if 'glossary' in url.lower():
                count += 1
            else:
                for pattern in self.glossary_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        count += 1
                        break
        return count
    
    def _count_cross_section_links(self, links: List[Tuple[str, str]]) -> int:
        """Count links between documentation sections."""
        count = 0
        for text, url in links:
            if (url.startswith('#') or 
                '/' in url and not url.startswith('http') or
                url.startswith('../')):
                count += 1
        return count
    
    def _count_preserved_links(self, found_links: List[Tuple[str, str]], 
                             expected_links: List[str]) -> int:
        """Count how many expected links were preserved."""
        found_urls = [url for text, url in found_links]
        preserved = 0
        
        for expected in expected_links:
            if any(expected in found_url for found_url in found_urls):
                preserved += 1
        
        return preserved
    
    def _assess_link_context_quality(self, content: str, 
                                   links: List[Tuple[str, str]]) -> float:
        """Assess how well links are integrated in their context."""
        if not links:
            return 1.0
        
        context_scores = []
        
        for text, url in links:
            # Find the link in content and examine surrounding context
            link_pattern = re.escape(f"[{text}]({url})")
            match = re.search(link_pattern, content)
            
            if match:
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                
                # Score based on context quality
                score = self._score_link_context(context, text)
                context_scores.append(score)
        
        return np.mean(context_scores) if context_scores else 0.0
    
    def _score_link_context(self, context: str, link_text: str) -> float:
        """Score the quality of link context."""
        score = 0.5  # Base score
        
        # Positive indicators
        if any(word in context.lower() for word in ['see', 'refer', 'read', 'documentation', 'guide']):
            score += 0.2
        
        if len(context.split()) > 20:  # Substantial context
            score += 0.2
        
        if link_text.lower() in context.lower():  # Link text appears in context
            score += 0.1
        
        # Negative indicators
        if context.count('[') > 5:  # Too many links nearby
            score -= 0.2
        
        if len(context.split()) < 10:  # Very little context
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _split_into_text_blocks(self, content: str) -> List[str]:
        """Split content into text blocks for duplicate analysis."""
        # Split by double newlines and filter
        blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
        
        # Filter out very short blocks and code blocks
        text_blocks = []
        for block in blocks:
            if (len(block.split()) >= 5 and 
                not block.startswith('```') and 
                not block.startswith('#')):
                text_blocks.append(block)
        
        return text_blocks
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text blocks."""
        # Use difflib for basic similarity
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        return similarity
    
    def _count_navigation_duplicates(self, duplicate_pairs: List[Tuple[str, str, float]]) -> int:
        """Count duplicate pairs that appear to be navigation."""
        nav_indicators = ['menu', 'navigation', 'nav', 'home', 'back', 'next', 'previous']
        count = 0
        
        for text1, text2, score in duplicate_pairs:
            combined_text = (text1 + " " + text2).lower()
            if any(indicator in combined_text for indicator in nav_indicators):
                count += 1
        
        return count
    
    def _count_footer_header_duplicates(self, duplicate_pairs: List[Tuple[str, str, float]]) -> int:
        """Count duplicates from headers/footers."""
        indicators = ['footer', 'header', 'copyright', '©', 'all rights reserved']
        count = 0
        
        for text1, text2, score in duplicate_pairs:
            combined_text = (text1 + " " + text2).lower()
            if any(indicator in combined_text for indicator in indicators):
                count += 1
        
        return count
    
    def _count_sidebar_duplicates(self, duplicate_pairs: List[Tuple[str, str, float]]) -> int:
        """Count duplicates from sidebars."""
        indicators = ['sidebar', 'table of contents', 'toc', 'on this page']
        count = 0
        
        for text1, text2, score in duplicate_pairs:
            combined_text = (text1 + " " + text2).lower()
            if any(indicator in combined_text for indicator in indicators):
                count += 1
        
        return count


# Convenience functions for integration

def analyze_content_quality_enhanced(markdown_content: str, 
                                   url: str = "",
                                   page_type: str = "",
                                   expected_links: Optional[List[str]] = None) -> EnhancedQualityMetrics:
    """
    Convenience function for enhanced content quality analysis.
    
    Args:
        markdown_content: Markdown content to analyze
        url: Source URL
        page_type: Type of page
        expected_links: Expected links for preservation analysis
        
    Returns:
        EnhancedQualityMetrics with comprehensive analysis
    """
    analyzer = AdvancedContentQualityAnalyzer()
    return analyzer.analyze_content(markdown_content, url, page_type, expected_links)


def create_quality_comparison_report(results: List[EnhancedQualityMetrics], 
                                   output_path: Optional[Path] = None) -> str:
    """
    Create a comparison report from multiple enhanced quality analyses.
    
    Args:
        results: List of EnhancedQualityMetrics
        output_path: Optional path to save report
        
    Returns:
        Report content as string
    """
    if not results:
        return "No results to analyze."
    
    # Calculate aggregate statistics
    total_analyses = len(results)
    avg_combined_score = np.mean([r.combined_quality_score for r in results])
    meets_enhanced_count = sum(1 for r in results if r.meets_enhanced_standards)
    
    # Page type analysis
    page_type_stats = defaultdict(list)
    for result in results:
        if result.page_type:
            page_type_stats[result.page_type].append(result.combined_quality_score)
    
    report = f"""# Enhanced Content Quality Analysis Report

## Summary
- **Total Analyses**: {total_analyses}
- **Average Combined Score**: {avg_combined_score:.3f}
- **Enhanced Standards Met**: {meets_enhanced_count} ({meets_enhanced_count/total_analyses*100:.1f}%)

## Quality Metrics Comparison

### Base Quality Systems Integration
- **Content Quality System**: Available
- **Validation System**: Available  
- **Enhanced Metrics**: Semantic coherence, link preservation, duplicate detection

### Page Type Performance
"""
    
    for page_type, scores in page_type_stats.items():
        avg_score = np.mean(scores)
        count = len(scores)
        report += f"- **{page_type.title()}**: {count} pages, avg score {avg_score:.3f}\n"
    
    # Add detailed analysis sections
    report += "\n## Detailed Analysis\n\n"
    
    # Semantic coherence summary
    semantic_results = [r for r in results if r.semantic_coherence]
    if semantic_results:
        avg_coherence = np.mean([r.semantic_coherence.avg_inter_chunk_similarity for r in semantic_results])
        report += f"### Semantic Coherence\n"
        report += f"- **Average Inter-chunk Similarity**: {avg_coherence:.3f}\n"
        report += f"- **Analyses with Embeddings**: {len(semantic_results)}\n\n"
    
    # Link preservation summary
    link_results = [r for r in results if r.link_preservation]
    if link_results:
        avg_preservation = np.mean([r.link_preservation.link_preservation_ratio for r in link_results])
        total_links = sum([r.link_preservation.total_links_found for r in link_results])
        report += f"### Link Preservation\n"
        report += f"- **Average Preservation Ratio**: {avg_preservation:.3f}\n"
        report += f"- **Total Links Analyzed**: {total_links}\n\n"
    
    # Duplicate content summary
    duplicate_results = [r for r in results if r.duplicate_content]
    if duplicate_results:
        avg_duplication = np.mean([r.duplicate_content.duplication_ratio for r in duplicate_results])
        report += f"### Duplicate Content\n"
        report += f"- **Average Duplication Ratio**: {avg_duplication:.3f}\n"
        report += f"- **Analyses Performed**: {len(duplicate_results)}\n\n"
    
    # Recommendations summary
    all_priority_improvements = []
    all_technical_recommendations = []
    
    for result in results:
        all_priority_improvements.extend(result.priority_improvements)
        all_technical_recommendations.extend(result.technical_recommendations)
    
    # Count most common recommendations
    priority_counts = Counter(all_priority_improvements)
    technical_counts = Counter(all_technical_recommendations)
    
    report += "## Common Recommendations\n\n"
    report += "### Priority Improvements\n"
    for rec, count in priority_counts.most_common(5):
        report += f"- {rec} ({count} pages)\n"
    
    report += "\n### Technical Recommendations\n"
    for rec, count in technical_counts.most_common(5):
        report += f"- {rec} ({count} pages)\n"
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Enhanced quality report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    # Example usage and testing
    sample_markdown = """# Test Document

## Introduction

This is a sample document for testing the enhanced quality analyzer.
It contains [internal links](/docs/guide) and [external links](https://example.com).

### Code Examples

Here's some code:

```python
def example_function():
    return "Hello, World!"
```

## Conclusion

This document tests various quality metrics including:
- Content structure
- Link preservation  
- Semantic coherence
- Duplicate detection

[Back to top](#test-document)
"""
    
    print("🧪 Testing Enhanced Content Quality Analyzer")
    print("=" * 50)
    
    analyzer = AdvancedContentQualityAnalyzer()
    result = analyzer.analyze_content(
        sample_markdown, 
        url="https://test.example.com/doc",
        page_type="guide"
    )
    
    print(f"Combined Quality Score: {result.combined_quality_score:.3f}")
    print(f"Meets Enhanced Standards: {result.meets_enhanced_standards}")
    print(f"Analysis Time: {result.analysis_time_ms:.1f}ms")
    
    if result.semantic_coherence:
        print(f"Semantic Coherence: {result.semantic_coherence.avg_inter_chunk_similarity:.3f}")
    
    if result.link_preservation:
        print(f"Link Preservation: {result.link_preservation.link_preservation_ratio:.3f}")
    
    if result.duplicate_content:
        print(f"Duplication Ratio: {result.duplicate_content.duplication_ratio:.3f}")
    
    print("\nPriority Improvements:")
    for improvement in result.priority_improvements:
        print(f"- {improvement}")