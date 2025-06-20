#!/usr/bin/env python3
"""
Content quality validation for documentation site extraction.

This module provides quality metrics to assess the effectiveness of content extraction
and determine when fallback extraction strategies should be triggered.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re
import logging
from collections import Counter
import time

logger = logging.getLogger(__name__)


@dataclass
class ContentQualityMetrics:
    """Metrics for assessing content extraction quality."""
    
    # Core quality indicators
    content_to_navigation_ratio: float
    link_density: float  # Links per word
    text_coherence_score: float
    word_count: int
    
    # Navigation indicators
    navigation_element_count: int
    unique_link_count: int
    repeated_link_count: int
    
    # Content structure indicators
    paragraph_count: int
    code_block_count: int
    heading_count: int
    
    # Quality assessment
    overall_quality_score: float
    quality_category: str  # "excellent", "good", "fair", "poor"
    
    # Performance metrics
    calculation_time_ms: float
    
    # Recommendations
    should_retry_with_fallback: bool
    improvement_suggestions: List[str]


class ContentQualityAnalyzer:
    """Analyzer for assessing extracted content quality."""
    
    def __init__(self):
        # Quality thresholds
        self.excellent_threshold = 0.8
        self.good_threshold = 0.7
        self.fair_threshold = 0.5
        self.poor_threshold = 0.3
        
        # Navigation detection patterns
        self.navigation_patterns = [
            r'\b(home|back|next|previous|prev|skip|menu|navigation|nav)\b',
            r'\b(table of contents|toc|contents|index)\b',
            r'\b(breadcrumb|breadcrumbs)\b',
            r'\b(sidebar|footer|header)\b',
            r'\b(edit|edit page|edit this page)\b',
            r'\b(search|search docs|search documentation)\b'
        ]
        
        # Link patterns that indicate navigation
        self.nav_link_patterns = [
            r'^(#|\.\.?/)',  # Internal links
            r'\b(edit|view|source|raw|blame)\b',
            r'\b(print|share|bookmark)\b'
        ]
    
    def calculate_content_quality(self, markdown_content: str) -> ContentQualityMetrics:
        """
        Calculate comprehensive quality metrics for extracted content.
        
        Args:
            markdown_content: The extracted markdown content to analyze
            
        Returns:
            ContentQualityMetrics with all quality indicators
        """
        start_time = time.time()
        
        # Basic content analysis
        word_count = self._count_words(markdown_content)
        paragraph_count = self._count_paragraphs(markdown_content)
        code_block_count = self._count_code_blocks(markdown_content)
        heading_count = self._count_headings(markdown_content)
        
        # Link analysis
        links = self._extract_links(markdown_content)
        unique_link_count = len(set(links))
        repeated_link_count = len(links) - unique_link_count
        link_density = len(links) / max(word_count, 1)
        
        # Navigation content detection
        navigation_element_count = self._count_navigation_elements(markdown_content)
        
        # Content-to-navigation ratio
        content_to_navigation_ratio = self._calculate_content_navigation_ratio(
            markdown_content, navigation_element_count, len(links)
        )
        
        # Text coherence assessment
        text_coherence_score = self._assess_text_coherence(markdown_content)
        
        # Overall quality score calculation
        overall_quality_score = self._calculate_overall_quality_score(
            content_to_navigation_ratio,
            link_density,
            text_coherence_score,
            word_count,
            code_block_count,
            paragraph_count
        )
        
        # Quality categorization
        quality_category = self._categorize_quality(overall_quality_score)
        
        # Recommendations
        should_retry = overall_quality_score < self.poor_threshold
        suggestions = self._generate_improvement_suggestions(
            content_to_navigation_ratio, link_density, word_count, navigation_element_count
        )
        
        calculation_time_ms = (time.time() - start_time) * 1000
        
        return ContentQualityMetrics(
            content_to_navigation_ratio=content_to_navigation_ratio,
            link_density=link_density,
            text_coherence_score=text_coherence_score,
            word_count=word_count,
            navigation_element_count=navigation_element_count,
            unique_link_count=unique_link_count,
            repeated_link_count=repeated_link_count,
            paragraph_count=paragraph_count,
            code_block_count=code_block_count,
            heading_count=heading_count,
            overall_quality_score=overall_quality_score,
            quality_category=quality_category,
            calculation_time_ms=calculation_time_ms,
            should_retry_with_fallback=should_retry,
            improvement_suggestions=suggestions
        )
    
    def _count_words(self, text: str) -> int:
        """Count words in text, excluding markdown syntax."""
        # Remove markdown syntax and count words
        clean_text = re.sub(r'[#*`\[\]()_~]', '', text)
        words = re.findall(r'\b\w+\b', clean_text)
        return len(words)
    
    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs in markdown text."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        # Filter out single-line items that are likely navigation
        substantial_paragraphs = [p for p in paragraphs if len(p.split()) > 10]
        return len(substantial_paragraphs)
    
    def _count_code_blocks(self, text: str) -> int:
        """Count code blocks in markdown."""
        fenced_blocks = len(re.findall(r'```[\s\S]*?```', text))
        inline_code = len(re.findall(r'`[^`\n]+`', text))
        return fenced_blocks + (inline_code // 5)  # Group inline code
    
    def _count_headings(self, text: str) -> int:
        """Count headings in markdown."""
        return len(re.findall(r'^#+\s+.+$', text, re.MULTILINE))
    
    def _extract_links(self, text: str) -> List[str]:
        """Extract all links from markdown text."""
        # Markdown links [text](url)
        markdown_links = re.findall(r'\[([^\]]*)\]\(([^)]*)\)', text)
        # Reference links [text][ref]
        reference_links = re.findall(r'\[([^\]]*)\]\[([^\]]*)\]', text)
        # Bare URLs
        bare_urls = re.findall(r'https?://[^\s\]]+', text)
        
        all_links = []
        all_links.extend([link[1] for link in markdown_links])  # URL part
        all_links.extend([link[1] for link in reference_links])  # Ref part
        all_links.extend(bare_urls)
        
        return all_links
    
    def _count_navigation_elements(self, text: str) -> int:
        """Count elements that appear to be navigation."""
        nav_count = 0
        text_lower = text.lower()
        
        for pattern in self.navigation_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            nav_count += len(matches)
        
        # Count list items that are likely navigation
        list_items = re.findall(r'^\s*[-*+]\s+(.+)$', text, re.MULTILINE)
        nav_list_items = 0
        
        for item in list_items:
            item_lower = item.lower().strip()
            # Short items with links are likely navigation
            if len(item_lower.split()) <= 5 and ('[' in item or 'http' in item):
                nav_list_items += 1
        
        nav_count += nav_list_items
        return nav_count
    
    def _calculate_content_navigation_ratio(self, text: str, nav_elements: int, link_count: int) -> float:
        """Calculate the ratio of content to navigation elements."""
        lines = text.split('\n')
        content_lines = 0
        nav_lines = 0
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Classify line as content or navigation
            is_navigation = False
            
            # Check for navigation patterns
            for pattern in self.navigation_patterns:
                if re.search(pattern, line_clean, re.IGNORECASE):
                    is_navigation = True
                    break
            
            # Check for list items with links (likely navigation)
            if re.match(r'^\s*[-*+]\s+.*\[.*\]', line_clean):
                word_count_in_line = len(line_clean.split())
                if word_count_in_line <= 8:  # Short list items with links
                    is_navigation = True
            
            # Check for repeated short lines (breadcrumbs, navigation)
            if len(line_clean.split()) <= 3 and ('>' in line_clean or '/' in line_clean):
                is_navigation = True
            
            if is_navigation:
                nav_lines += 1
            else:
                content_lines += 1
        
        total_lines = content_lines + nav_lines
        if total_lines == 0:
            return 0.0
            
        return content_lines / total_lines
    
    def _assess_text_coherence(self, text: str) -> float:
        """Assess text coherence based on structure and content patterns."""
        coherence_score = 0.0
        
        # Check for well-structured content
        has_headings = bool(re.search(r'^#+\s+.+$', text, re.MULTILINE))
        has_paragraphs = '\n\n' in text
        has_code_examples = '```' in text or '`' in text
        
        if has_headings:
            coherence_score += 0.3
        if has_paragraphs:
            coherence_score += 0.3
        if has_code_examples:
            coherence_score += 0.2
        
        # Check for technical content indicators
        technical_words = [
            'function', 'method', 'class', 'variable', 'parameter', 'return',
            'example', 'usage', 'configuration', 'installation', 'api',
            'endpoint', 'request', 'response', 'error', 'exception'
        ]
        
        text_lower = text.lower()
        technical_word_count = sum(1 for word in technical_words if word in text_lower)
        
        if technical_word_count >= 3:
            coherence_score += 0.2
        
        return min(coherence_score, 1.0)
    
    def _calculate_overall_quality_score(self, content_ratio: float, link_density: float, 
                                       coherence: float, word_count: int, code_blocks: int, 
                                       paragraphs: int) -> float:
        """Calculate overall quality score weighted across multiple factors."""
        
        # Weighted scoring
        score = 0.0
        
        # Content-to-navigation ratio (40% weight)
        score += content_ratio * 0.4
        
        # Link density penalty (20% weight)
        # Lower link density is better (more content, less navigation)
        link_density_score = max(0, 1.0 - (link_density * 2))  # Penalty for high link density
        score += link_density_score * 0.2
        
        # Text coherence (25% weight)
        score += coherence * 0.25
        
        # Content substance (15% weight)
        # Reward substantial content
        substance_score = 0.0
        if word_count >= 200:
            substance_score += 0.4
        elif word_count >= 100:
            substance_score += 0.2
        elif word_count >= 50:
            substance_score += 0.1
            
        if paragraphs >= 3:
            substance_score += 0.3
        elif paragraphs >= 2:
            substance_score += 0.2
        elif paragraphs >= 1:
            substance_score += 0.1
            
        if code_blocks > 0:
            substance_score += 0.3
            
        score += min(substance_score, 1.0) * 0.15
        
        return min(score, 1.0)
    
    def _categorize_quality(self, score: float) -> str:
        """Categorize quality score into human-readable categories."""
        if score >= self.excellent_threshold:
            return "excellent"
        elif score >= self.good_threshold:
            return "good"
        elif score >= self.fair_threshold:
            return "fair"
        else:
            return "poor"
    
    def _generate_improvement_suggestions(self, content_ratio: float, link_density: float, 
                                        word_count: int, nav_elements: int) -> List[str]:
        """Generate specific suggestions for improving extraction quality."""
        suggestions = []
        
        if content_ratio < 0.5:
            suggestions.append("Content-to-navigation ratio is low. Consider using more specific CSS selectors to target main content areas.")
        
        if link_density > 0.3:
            suggestions.append("High link density detected. Try excluding navigation elements and sidebars.")
        
        if nav_elements > 20:
            suggestions.append(f"High navigation element count ({nav_elements}). Add navigation-specific exclusion selectors.")
        
        if word_count < 100:
            suggestions.append("Low word count. Check if main content areas are being targeted correctly.")
        
        if not suggestions:
            suggestions.append("Quality metrics look good. Consider fine-tuning thresholds if needed.")
        
        return suggestions


# Global analyzer instance
quality_analyzer = ContentQualityAnalyzer()


def calculate_content_quality(markdown_content: str) -> ContentQualityMetrics:
    """Convenience function for calculating content quality metrics."""
    return quality_analyzer.calculate_content_quality(markdown_content)


def is_high_quality_content(metrics: ContentQualityMetrics) -> bool:
    """Check if content meets high quality standards."""
    return metrics.overall_quality_score >= quality_analyzer.good_threshold


def should_retry_extraction(metrics: ContentQualityMetrics) -> bool:
    """Check if extraction should be retried with fallback strategy."""
    return metrics.should_retry_with_fallback


def log_quality_metrics(metrics: ContentQualityMetrics, url: str = "", framework: str = ""):
    """Log quality metrics for monitoring and debugging."""
    logger.info(f"Content quality analysis for {url or 'unknown URL'}")
    logger.info(f"Framework: {framework}")
    quality_category = getattr(metrics, 'quality_category', 'unknown')
    logger.info(f"Overall quality: {quality_category} ({metrics.overall_quality_score:.3f})")
    logger.info(f"Content/Nav ratio: {metrics.content_to_navigation_ratio:.3f}")
    logger.info(f"Link density: {metrics.link_density:.3f}")
    logger.info(f"Word count: {metrics.word_count}")
    logger.info(f"Navigation elements: {metrics.navigation_element_count}")
    logger.info(f"Analysis time: {metrics.calculation_time_ms:.1f}ms")
    
    if metrics.should_retry_with_fallback:
        logger.warning("Quality is below threshold - retry recommended")
        for suggestion in metrics.improvement_suggestions:
            logger.info(f"Suggestion: {suggestion}")