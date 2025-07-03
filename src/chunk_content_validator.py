#!/usr/bin/env python3
"""
Chunk Content Quality Validator

This module implements Subtask 18.3 by analyzing generated chunks to ensure they contain
substantive content rather than navigation elements, verifying preservation of glossary
definitions, code examples, and explanatory text while maintaining internal documentation links.

Features:
- Validates chunks contain >100 characters of actual content
- Detects and flags navigation patterns (breadcrumbs, menu items, footer links)
- Confirms preservation of glossary definitions and code examples using pattern matching
- Validates internal documentation links are maintained within content
- Generates detailed content quality reports with examples
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import time
import json
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ChunkContentMetrics:
    """Detailed metrics for a single chunk's content quality."""
    
    # Basic content metrics
    chunk_id: str = ""
    character_count: int = 0
    word_count: int = 0
    line_count: int = 0
    
    # Content type classification
    is_substantive_content: bool = False
    content_type: str = "unknown"  # 'content', 'navigation', 'header', 'footer', 'mixed'
    
    # Content structure
    has_headings: bool = False
    heading_count: int = 0
    has_paragraphs: bool = False
    paragraph_count: int = 0
    has_code_examples: bool = False
    code_block_count: int = 0
    
    # Link analysis
    internal_links_count: int = 0
    external_links_count: int = 0
    broken_links_count: int = 0
    links_preserved: List[str] = field(default_factory=list)
    
    # Navigation detection
    navigation_indicators_found: List[str] = field(default_factory=list)
    breadcrumb_detected: bool = False
    menu_items_detected: bool = False
    footer_content_detected: bool = False
    
    # Content quality indicators
    has_glossary_definitions: bool = False
    glossary_terms_found: List[str] = field(default_factory=list)
    has_explanatory_text: bool = False
    technical_content_score: float = 0.0
    
    # Quality assessment
    meets_minimum_standards: bool = False
    quality_category: str = "poor"  # 'excellent', 'good', 'fair', 'poor'
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ChunkValidationResult:
    """Complete validation result for a set of chunks."""
    
    # Overall statistics
    total_chunks: int = 0
    valid_chunks: int = 0
    invalid_chunks: int = 0
    validation_timestamp: str = ""
    
    # Content quality breakdown
    substantive_content_chunks: int = 0
    navigation_chunks: int = 0
    mixed_content_chunks: int = 0
    
    # Detailed metrics per chunk
    chunk_metrics: List[ChunkContentMetrics] = field(default_factory=list)
    
    # Aggregate quality indicators
    total_glossary_definitions: int = 0
    total_code_examples: int = 0
    total_preserved_links: int = 0
    
    # Quality distribution
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Common issues and recommendations
    common_issues: List[Tuple[str, int]] = field(default_factory=list)  # (issue, count)
    priority_recommendations: List[str] = field(default_factory=list)
    
    # Performance metrics
    validation_time_ms: float = 0.0
    
    # Summary assessment
    overall_quality_score: float = 0.0
    meets_quality_standards: bool = False
    improvement_needed: bool = True


class ChunkContentValidator:
    """
    Validator for analyzing chunk content quality and ensuring meaningful data extraction.
    
    This validator implements comprehensive analysis to ensure that generated chunks:
    1. Contain substantive content (>100 characters)
    2. Are free from navigation contamination
    3. Preserve important content like glossary definitions and code examples
    4. Maintain internal documentation links within content
    """
    
    def __init__(self):
        """Initialize the chunk content validator."""
        
        # Minimum content standards
        self.min_character_count = 100
        self.min_word_count = 15
        self.min_substantive_ratio = 0.7
        
        # Navigation detection patterns
        self.navigation_patterns = {
            'breadcrumbs': [
                r'\bhome\s*[>\/]\s*\w+',
                r'\w+\s*[>\/]\s*\w+\s*[>\/]\s*\w+',
                r'you\s+are\s+here',
                r'current\s+page',
                r'path:?\s*home'
            ],
            'menu_items': [
                r'\b(menu|navigation|nav)\b',
                r'\b(skip\s+to|jump\s+to)\b',
                r'\b(toggle|hamburger|mobile\s+menu)\b',
                r'\b(search|filter|sort\s+by)\b',
                r'\b(previous|next|back|forward)\b\s*(page|section)?'
            ],
            'footer_content': [
                r'\b(copyright|©|\(c\))\b',
                r'\ball\s+rights\s+reserved\b',
                r'\bprivacy\s+policy\b',
                r'\bterms\s+of\s+(service|use)\b',
                r'\bcontact\s+us\b',
                r'\bfooter\b'
            ],
            'sidebar': [
                r'\btable\s+of\s+contents\b',
                r'\bon\s+this\s+page\b',
                r'\bin\s+this\s+(section|chapter)\b',
                r'\brelated\s+(topics|articles|links)\b',
                r'\bsidebar\b'
            ]
        }
        
        # Glossary definition patterns
        self.glossary_patterns = [
            r'(?:^|\n)([A-Z][a-zA-Z\s]+):\s+([^.\n]+\.)',  # "Term: Definition."
            r'(?:^|\n)\*\*([A-Z][a-zA-Z\s]+)\*\*:?\s+([^.\n]+\.)',  # "**Term**: Definition."
            r'(?:^|\n)###?\s+([A-Z][a-zA-Z\s]+)\n([^#\n]+)',  # Heading followed by definition
            r'\[([A-Z][a-zA-Z\s]+)\]\([^)]*glossary[^)]*\)',  # Link to glossary
        ]
        
        # Code example patterns
        self.code_patterns = [
            r'```[\w]*\n.*?\n```',  # Fenced code blocks
            r'`[^`\n]+`',  # Inline code
            r'(?:^|\n)    \w+.*$',  # Indented code (4 spaces)
            r'<code>.*?</code>',  # HTML code tags
            r'(?:function|class|def|const|let|var)\s+\w+',  # Code keywords
        ]
        
        # Technical content indicators
        self.technical_indicators = [
            'api', 'endpoint', 'method', 'function', 'parameter', 'argument',
            'return', 'response', 'request', 'header', 'payload', 'schema',
            'configuration', 'setting', 'option', 'property', 'attribute',
            'example', 'tutorial', 'guide', 'documentation', 'reference',
            'install', 'setup', 'configure', 'initialize', 'implement',
            'workflow', 'node', 'trigger', 'action', 'execution', 'pipeline'
        ]
        
        # Link extraction patterns
        self.link_patterns = [
            r'\[([^\]]+)\]\(([^)]+)\)',  # Markdown links
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>',  # HTML links
            r'https?://[^\s\)]+',  # Bare URLs
        ]
    
    def validate_chunks(self, chunks: List[str], 
                       source_url: str = "",
                       chunk_metadata: Optional[List[Dict[str, Any]]] = None) -> ChunkValidationResult:
        """
        Validate a list of content chunks for quality and meaningfulness.
        
        Args:
            chunks: List of chunk content strings
            source_url: Source URL for context
            chunk_metadata: Optional metadata for each chunk
            
        Returns:
            ChunkValidationResult with comprehensive analysis
        """
        start_time = time.time()
        
        logger.info(f"Validating {len(chunks)} chunks from {source_url or 'unknown source'}")
        
        # Initialize result
        result = ChunkValidationResult(
            total_chunks=len(chunks),
            validation_timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Validate each chunk
        for i, chunk_content in enumerate(chunks):
            chunk_id = f"chunk_{i+1:03d}"
            
            # Get metadata if available
            metadata = chunk_metadata[i] if chunk_metadata and i < len(chunk_metadata) else {}
            
            # Analyze chunk
            chunk_metrics = self._analyze_single_chunk(chunk_content, chunk_id, metadata)
            result.chunk_metrics.append(chunk_metrics)
            
            # Update aggregate statistics
            if chunk_metrics.meets_minimum_standards:
                result.valid_chunks += 1
            else:
                result.invalid_chunks += 1
            
            # Update content type counts
            if chunk_metrics.content_type == 'content':
                result.substantive_content_chunks += 1
            elif chunk_metrics.content_type == 'navigation':
                result.navigation_chunks += 1
            elif chunk_metrics.content_type == 'mixed':
                result.mixed_content_chunks += 1
            
            # Aggregate quality indicators
            if chunk_metrics.has_glossary_definitions:
                result.total_glossary_definitions += len(chunk_metrics.glossary_terms_found)
            
            if chunk_metrics.has_code_examples:
                result.total_code_examples += chunk_metrics.code_block_count
            
            result.total_preserved_links += len(chunk_metrics.links_preserved)
        
        # Calculate overall metrics
        result.validation_time_ms = (time.time() - start_time) * 1000
        result.overall_quality_score = self._calculate_overall_quality_score(result)
        result.meets_quality_standards = result.overall_quality_score >= 0.7
        result.improvement_needed = result.overall_quality_score < 0.8
        
        # Analyze quality distribution
        result.quality_distribution = self._calculate_quality_distribution(result.chunk_metrics)
        
        # Identify common issues and recommendations
        result.common_issues = self._identify_common_issues(result.chunk_metrics)
        result.priority_recommendations = self._generate_priority_recommendations(result)
        
        logger.info(f"Validation completed: {result.valid_chunks}/{result.total_chunks} chunks valid, "
                   f"score: {result.overall_quality_score:.3f}")
        
        return result
    
    def _analyze_single_chunk(self, content: str, chunk_id: str, 
                            metadata: Dict[str, Any]) -> ChunkContentMetrics:
        """Analyze a single chunk for content quality."""
        
        metrics = ChunkContentMetrics(chunk_id=chunk_id)
        
        # Basic content metrics
        metrics.character_count = len(content)
        metrics.word_count = len(content.split())
        metrics.line_count = len(content.split('\n'))
        
        # Content structure analysis
        metrics.has_headings, metrics.heading_count = self._analyze_headings(content)
        metrics.has_paragraphs, metrics.paragraph_count = self._analyze_paragraphs(content)
        metrics.has_code_examples, metrics.code_block_count = self._analyze_code_content(content)
        
        # Link analysis
        links = self._extract_links(content)
        metrics.internal_links_count = sum(1 for _, url in links if self._is_internal_link(url))
        metrics.external_links_count = sum(1 for _, url in links if self._is_external_link(url))
        metrics.broken_links_count = sum(1 for _, url in links if self._is_broken_link(url))
        metrics.links_preserved = [f"{text} -> {url}" for text, url in links[:3]]  # Sample
        
        # Navigation detection
        metrics.navigation_indicators_found = self._detect_navigation_patterns(content)
        metrics.breadcrumb_detected = self._has_breadcrumbs(content)
        metrics.menu_items_detected = self._has_menu_items(content)
        metrics.footer_content_detected = self._has_footer_content(content)
        
        # Content quality indicators
        metrics.has_glossary_definitions, metrics.glossary_terms_found = self._detect_glossary_content(content)
        metrics.has_explanatory_text = self._has_explanatory_text(content)
        metrics.technical_content_score = self._calculate_technical_content_score(content)
        
        # Content type classification
        metrics.content_type = self._classify_content_type(metrics)
        
        # Quality assessment
        metrics.is_substantive_content = self._is_substantive_content(metrics)
        metrics.meets_minimum_standards = self._meets_minimum_standards(metrics)
        metrics.quality_category = self._determine_quality_category(metrics)
        
        # Generate issues and recommendations
        metrics.issues_found = self._identify_chunk_issues(metrics)
        metrics.recommendations = self._generate_chunk_recommendations(metrics)
        
        return metrics
    
    def _analyze_headings(self, content: str) -> Tuple[bool, int]:
        """Analyze heading structure in content."""
        heading_pattern = r'^#+\s+.+$'
        headings = re.findall(heading_pattern, content, re.MULTILINE)
        return len(headings) > 0, len(headings)
    
    def _analyze_paragraphs(self, content: str) -> Tuple[bool, int]:
        """Analyze paragraph structure in content."""
        # Split by double newlines and filter substantive paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        substantive_paragraphs = [p for p in paragraphs if len(p.split()) >= 10]
        return len(substantive_paragraphs) > 0, len(substantive_paragraphs)
    
    def _analyze_code_content(self, content: str) -> Tuple[bool, int]:
        """Analyze code examples in content."""
        code_blocks = 0
        
        for pattern in self.code_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
            code_blocks += len(matches)
        
        return code_blocks > 0, code_blocks
    
    def _extract_links(self, content: str) -> List[Tuple[str, str]]:
        """Extract all links from content."""
        links = []
        
        for pattern in self.link_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        links.append((match[0], match[1]))  # (text, url)
                    else:
                        links.append((match[1], match[0]))  # HTML links (url, text)
                else:
                    links.append(("", match))  # Bare URLs
        
        return links
    
    def _is_internal_link(self, url: str) -> bool:
        """Check if link is internal documentation."""
        if not url:
            return False
        return (url.startswith('/') or url.startswith('#') or 
                url.startswith('../') or url.startswith('./'))
    
    def _is_external_link(self, url: str) -> bool:
        """Check if link is external."""
        return url.startswith('http://') or url.startswith('https://')
    
    def _is_broken_link(self, url: str) -> bool:
        """Check if link appears broken."""
        if not url or url.strip() == "":
            return True
        
        # Check for malformed patterns
        broken_patterns = [r'^\s*$', r'[<>"]', r'\s+']
        return any(re.search(pattern, url) for pattern in broken_patterns)
    
    def _detect_navigation_patterns(self, content: str) -> List[str]:
        """Detect navigation patterns in content."""
        found_patterns = []
        content_lower = content.lower()
        
        for category, patterns in self.navigation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    found_patterns.append(f"{category}: {pattern}")
                    break  # Only count each category once
        
        return found_patterns
    
    def _has_breadcrumbs(self, content: str) -> bool:
        """Check for breadcrumb navigation."""
        for pattern in self.navigation_patterns['breadcrumbs']:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _has_menu_items(self, content: str) -> bool:
        """Check for menu items."""
        for pattern in self.navigation_patterns['menu_items']:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _has_footer_content(self, content: str) -> bool:
        """Check for footer content."""
        for pattern in self.navigation_patterns['footer_content']:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _detect_glossary_content(self, content: str) -> Tuple[bool, List[str]]:
        """Detect glossary definitions."""
        terms_found = []
        
        for pattern in self.glossary_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 1:
                    term = match[0].strip()
                    if len(term) > 2 and term not in terms_found:
                        terms_found.append(term)
        
        return len(terms_found) > 0, terms_found
    
    def _has_explanatory_text(self, content: str) -> bool:
        """Check for explanatory/instructional text patterns."""
        explanatory_patterns = [
            r'\b(for example|such as|this means|in other words)\b',
            r'\b(to understand|to learn|to configure|to setup)\b',
            r'\b(step \d+|first|second|third|finally|next)\b',
            r'\b(note:|important:|warning:|tip:)\b',
            r'\b(explanation|description|overview|introduction)\b'
        ]
        
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in explanatory_patterns)
    
    def _calculate_technical_content_score(self, content: str) -> float:
        """Calculate technical content density score."""
        if not content:
            return 0.0
        
        content_lower = content.lower()
        words = content_lower.split()
        
        if len(words) == 0:
            return 0.0
        
        technical_word_count = sum(1 for word in words if word in self.technical_indicators)
        technical_density = technical_word_count / len(words)
        
        # Bonus for code examples
        code_bonus = 0.1 if any(re.search(pattern, content) for pattern in self.code_patterns) else 0.0
        
        # Bonus for structured content
        structure_bonus = 0.05 if re.search(r'^#+\s+', content, re.MULTILINE) else 0.0
        
        return min(1.0, technical_density + code_bonus + structure_bonus)
    
    def _classify_content_type(self, metrics: ChunkContentMetrics) -> str:
        """Classify the type of content in the chunk."""
        
        # Check for navigation content
        if (metrics.breadcrumb_detected or 
            metrics.menu_items_detected or 
            len(metrics.navigation_indicators_found) >= 2):
            return 'navigation'
        
        # Check for footer content
        if metrics.footer_content_detected:
            return 'footer'
        
        # Check for substantive content
        if (metrics.has_paragraphs and 
            metrics.word_count >= self.min_word_count and
            metrics.technical_content_score > 0.1):
            
            # Mixed content if navigation indicators present
            if len(metrics.navigation_indicators_found) > 0:
                return 'mixed'
            else:
                return 'content'
        
        # Check for header content
        if metrics.has_headings and metrics.paragraph_count == 0:
            return 'header'
        
        return 'unknown'
    
    def _is_substantive_content(self, metrics: ChunkContentMetrics) -> bool:
        """Determine if chunk contains substantive content."""
        return (metrics.character_count >= self.min_character_count and
                metrics.word_count >= self.min_word_count and
                metrics.content_type in ['content', 'mixed'] and
                (metrics.has_paragraphs or metrics.has_code_examples))
    
    def _meets_minimum_standards(self, metrics: ChunkContentMetrics) -> bool:
        """Check if chunk meets minimum quality standards."""
        return (metrics.is_substantive_content and
                metrics.content_type != 'navigation' and
                not metrics.footer_content_detected and
                len(metrics.navigation_indicators_found) <= 2)
    
    def _determine_quality_category(self, metrics: ChunkContentMetrics) -> str:
        """Determine quality category for the chunk."""
        
        if not metrics.meets_minimum_standards:
            return 'poor'
        
        score = 0.0
        
        # Content quality factors
        if metrics.has_paragraphs:
            score += 0.2
        if metrics.has_code_examples:
            score += 0.2
        if metrics.has_glossary_definitions:
            score += 0.2
        if metrics.has_explanatory_text:
            score += 0.1
        
        # Technical content bonus
        score += metrics.technical_content_score * 0.2
        
        # Penalties
        if len(metrics.navigation_indicators_found) > 0:
            score -= 0.1 * len(metrics.navigation_indicators_found)
        
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _identify_chunk_issues(self, metrics: ChunkContentMetrics) -> List[str]:
        """Identify issues with the chunk."""
        issues = []
        
        if metrics.character_count < self.min_character_count:
            issues.append(f"Below minimum character count ({metrics.character_count} < {self.min_character_count})")
        
        if metrics.word_count < self.min_word_count:
            issues.append(f"Below minimum word count ({metrics.word_count} < {self.min_word_count})")
        
        if metrics.content_type == 'navigation':
            issues.append("Content appears to be navigation elements")
        
        if metrics.breadcrumb_detected:
            issues.append("Breadcrumb navigation detected")
        
        if metrics.menu_items_detected:
            issues.append("Menu items detected")
        
        if metrics.footer_content_detected:
            issues.append("Footer content detected")
        
        if not metrics.has_paragraphs and not metrics.has_code_examples:
            issues.append("No substantial paragraphs or code examples")
        
        if metrics.broken_links_count > 0:
            issues.append(f"{metrics.broken_links_count} broken links found")
        
        return issues
    
    def _generate_chunk_recommendations(self, metrics: ChunkContentMetrics) -> List[str]:
        """Generate recommendations for improving the chunk."""
        recommendations = []
        
        if metrics.content_type == 'navigation':
            recommendations.append("Exclude this chunk - contains only navigation elements")
        
        if len(metrics.navigation_indicators_found) > 2:
            recommendations.append("Review CSS selectors to better exclude navigation elements")
        
        if metrics.character_count < self.min_character_count:
            recommendations.append("Combine with adjacent chunks to meet minimum content requirements")
        
        if not metrics.has_paragraphs and not metrics.has_code_examples:
            recommendations.append("Verify this chunk contains meaningful content")
        
        if metrics.broken_links_count > 0:
            recommendations.append("Fix broken or malformed links in the source content")
        
        if metrics.technical_content_score < 0.1:
            recommendations.append("Consider if this chunk provides valuable technical information")
        
        return recommendations
    
    def _calculate_overall_quality_score(self, result: ChunkValidationResult) -> float:
        """Calculate overall quality score for all chunks."""
        if result.total_chunks == 0:
            return 0.0
        
        # Base score from valid chunks ratio
        valid_ratio = result.valid_chunks / result.total_chunks
        
        # Content type distribution score
        content_ratio = result.substantive_content_chunks / result.total_chunks
        navigation_penalty = result.navigation_chunks / result.total_chunks * 0.5
        
        # Quality indicators bonus
        indicators_bonus = 0.0
        if result.total_chunks > 0:
            glossary_bonus = min(0.1, result.total_glossary_definitions / result.total_chunks * 0.5)
            code_bonus = min(0.1, result.total_code_examples / result.total_chunks * 0.3)
            links_bonus = min(0.1, result.total_preserved_links / result.total_chunks * 0.2)
            indicators_bonus = glossary_bonus + code_bonus + links_bonus
        
        # Calculate final score
        score = (valid_ratio * 0.5 + 
                content_ratio * 0.3 - 
                navigation_penalty + 
                indicators_bonus)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_quality_distribution(self, chunk_metrics: List[ChunkContentMetrics]) -> Dict[str, int]:
        """Calculate distribution of quality categories."""
        distribution = defaultdict(int)
        for metrics in chunk_metrics:
            distribution[metrics.quality_category] += 1
        return dict(distribution)
    
    def _identify_common_issues(self, chunk_metrics: List[ChunkContentMetrics]) -> List[Tuple[str, int]]:
        """Identify most common issues across chunks."""
        issue_counter = Counter()
        
        for metrics in chunk_metrics:
            for issue in metrics.issues_found:
                issue_counter[issue] += 1
        
        return issue_counter.most_common(10)
    
    def _generate_priority_recommendations(self, result: ChunkValidationResult) -> List[str]:
        """Generate priority recommendations for improving chunk quality."""
        recommendations = []
        
        if result.navigation_chunks > result.total_chunks * 0.2:
            recommendations.append("High navigation contamination detected - review CSS exclusion selectors")
        
        if result.valid_chunks < result.total_chunks * 0.7:
            recommendations.append("Many chunks below quality standards - review chunking strategy")
        
        if result.total_code_examples == 0 and result.total_chunks > 5:
            recommendations.append("No code examples found - verify technical content is being preserved")
        
        if result.total_glossary_definitions == 0:
            recommendations.append("No glossary definitions detected - check if definitions are being preserved")
        
        if result.overall_quality_score < 0.6:
            recommendations.append("Overall quality below acceptable threshold - comprehensive review needed")
        
        return recommendations[:5]  # Return top 5 recommendations


def validate_chunk_content(chunks: List[str], 
                         source_url: str = "",
                         chunk_metadata: Optional[List[Dict[str, Any]]] = None) -> ChunkValidationResult:
    """
    Convenience function for validating chunk content quality.
    
    Args:
        chunks: List of chunk content strings
        source_url: Source URL for context  
        chunk_metadata: Optional metadata for each chunk
        
    Returns:
        ChunkValidationResult with comprehensive analysis
    """
    validator = ChunkContentValidator()
    return validator.validate_chunks(chunks, source_url, chunk_metadata)


def create_chunk_validation_report(result: ChunkValidationResult, 
                                 output_path: Optional[Path] = None) -> str:
    """
    Create a detailed chunk validation report.
    
    Args:
        result: ChunkValidationResult to analyze
        output_path: Optional path to save report
        
    Returns:
        Report content as string
    """
    
    report = f"""# Chunk Content Validation Report

## Summary
- **Total Chunks Analyzed**: {result.total_chunks}
- **Valid Chunks**: {result.valid_chunks} ({result.valid_chunks/result.total_chunks*100:.1f}%)
- **Invalid Chunks**: {result.invalid_chunks} ({result.invalid_chunks/result.total_chunks*100:.1f}%)
- **Overall Quality Score**: {result.overall_quality_score:.3f}
- **Meets Quality Standards**: {'✅ Yes' if result.meets_quality_standards else '❌ No'}

## Content Type Distribution
- **Substantive Content**: {result.substantive_content_chunks} chunks
- **Navigation Content**: {result.navigation_chunks} chunks  
- **Mixed Content**: {result.mixed_content_chunks} chunks

## Quality Indicators
- **Glossary Definitions Found**: {result.total_glossary_definitions}
- **Code Examples Found**: {result.total_code_examples}
- **Preserved Links**: {result.total_preserved_links}

## Quality Distribution
"""
    
    for category, count in result.quality_distribution.items():
        percentage = (count / result.total_chunks) * 100 if result.total_chunks > 0 else 0
        report += f"- **{category.title()}**: {count} chunks ({percentage:.1f}%)\n"
    
    report += f"""
## Common Issues
"""
    
    for issue, count in result.common_issues[:10]:
        report += f"- {issue}: {count} chunks\n"
    
    report += f"""
## Priority Recommendations
"""
    
    for recommendation in result.priority_recommendations:
        report += f"- {recommendation}\n"
    
    # Add sample chunk analysis for detailed review
    report += f"""
## Sample Chunk Analysis

### Excellent Quality Chunks
"""
    
    excellent_chunks = [m for m in result.chunk_metrics if m.quality_category == 'excellent']
    for chunk in excellent_chunks[:3]:
        report += f"""
#### {chunk.chunk_id}
- **Character Count**: {chunk.character_count}
- **Content Type**: {chunk.content_type}
- **Has Code Examples**: {'✅' if chunk.has_code_examples else '❌'}
- **Has Glossary Definitions**: {'✅' if chunk.has_glossary_definitions else '❌'}
- **Technical Score**: {chunk.technical_content_score:.3f}
"""
    
    if result.navigation_chunks > 0:
        report += f"""
### Navigation Contamination Examples
"""
        
        nav_chunks = [m for m in result.chunk_metrics if m.content_type == 'navigation']
        for chunk in nav_chunks[:3]:
            report += f"""
#### {chunk.chunk_id}
- **Navigation Indicators**: {', '.join(chunk.navigation_indicators_found[:3])}
- **Issues**: {', '.join(chunk.issues_found[:3])}
"""
    
    report += f"""
## Validation Metadata
- **Validation Time**: {result.validation_time_ms:.1f}ms
- **Timestamp**: {result.validation_timestamp}
- **Improvement Needed**: {'Yes' if result.improvement_needed else 'No'}
"""
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Chunk validation report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    # Example usage and testing
    sample_chunks = [
        # Good content chunk
        """# API Authentication

        To authenticate with the n8n API, you need to provide an API key in the Authorization header.

        ## Example Request

        ```bash
        curl -X GET "https://api.n8n.io/workflows" \\
          -H "Authorization: Bearer YOUR_API_KEY"
        ```

        The API key can be generated in your n8n settings under the API section.
        """,
        
        # Navigation chunk (should be flagged)
        """Home > Documentation > API > Authentication
        
        Menu:
        - Home
        - Getting Started  
        - API Reference
        - Tutorials
        
        Search documentation...
        """,
        
        # Mixed content chunk
        """## Workflow Configuration

        Configure your workflow settings in the workflow editor.

        Navigation: Previous | Next | Home
        
        ### Settings Options
        
        - **Name**: Give your workflow a descriptive name
        - **Active**: Enable or disable the workflow
        - **Tags**: Add tags for organization
        """,
        
        # Code example chunk
        """```javascript
        // Create a new workflow
        const workflow = {
          name: "My Workflow",
          active: true,
          nodes: [
            {
              type: "n8n-nodes-base.start",
              parameters: {},
              position: [250, 300]
            }
          ]
        };
        ```"""
    ]
    
    print("🧪 Testing Chunk Content Validator")
    print("=" * 50)
    
    validator = ChunkContentValidator()
    result = validator.validate_chunks(
        sample_chunks, 
        source_url="https://docs.n8n.io/api/authentication"
    )
    
    print(f"Total Chunks: {result.total_chunks}")
    print(f"Valid Chunks: {result.valid_chunks}")
    print(f"Quality Score: {result.overall_quality_score:.3f}")
    print(f"Meets Standards: {result.meets_quality_standards}")
    
    print(f"\nContent Distribution:")
    print(f"- Substantive: {result.substantive_content_chunks}")
    print(f"- Navigation: {result.navigation_chunks}") 
    print(f"- Mixed: {result.mixed_content_chunks}")
    
    print(f"\nQuality Distribution:")
    for category, count in result.quality_distribution.items():
        print(f"- {category.title()}: {count}")
    
    print(f"\nCommon Issues:")
    for issue, count in result.common_issues[:5]:
        print(f"- {issue}: {count}")
    
    # Generate and display report
    report = create_chunk_validation_report(result)
    print(f"\n📋 Generated validation report ({len(report)} characters)")