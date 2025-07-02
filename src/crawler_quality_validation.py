#!/usr/bin/env python3
"""
Automated Content Quality Validation Suite for AdvancedWebCrawler

This module provides comprehensive quality validation for markdown content produced
by the AdvancedWebCrawler, ensuring it meets the standards required for the
DocumentIngestionPipeline and SemanticChunker processing.

Features:
- HTML artifact detection and removal validation
- Script block contamination checks  
- Whitespace and formatting validation
- Content-to-navigation ratio assessment
- Link density and semantic structure analysis
- Golden set comparison for quality benchmarking
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class QualityValidationResult:
    """Result from content quality validation."""
    
    passed: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    warnings: List[str]
    
    # Detailed metrics
    html_artifacts_found: int
    script_contamination: bool
    excessive_whitespace: bool
    content_to_navigation_ratio: float
    link_density: float
    word_count: int
    
    # Quality categories
    category: str  # excellent, good, fair, poor
    recommendations: List[str]


class ContentQualityValidator:
    """
    Automated content quality validation suite for crawler output.
    
    This validator ensures markdown content meets the quality standards
    required for the DocumentIngestionPipeline and downstream processing.
    """
    
    def __init__(self):
        # Blacklisted strings that indicate poor extraction
        self.html_blacklist = [
            '</script>',
            '<footer>',
            '<nav>',
            '<header>',
            '<aside>',
            '</div>',
            '</span>',
            '<iframe',
            'onclick=',
            'javascript:',
            '&nbsp;',
            '&amp;',
            '&lt;',
            '&gt;'
        ]
        
        # Script contamination indicators
        self.script_indicators = [
            'function(',
            'var ',
            'let ',
            'const ',
            'document.',
            'window.',
            'alert(',
            'console.',
            'addEventListener',
            'getElementById',
            '$(', 
            'jQuery'
        ]
        
        # Navigation noise indicators
        self.navigation_indicators = [
            'menu',
            'navigation', 
            'sidebar',
            'breadcrumb',
            'toc',
            'table of contents',
            'next page',
            'previous page',
            'home',
            'back to top',
            'skip to',
            'toggle',
            'hamburger'
        ]
        
        # Quality thresholds
        self.thresholds = {
            'min_word_count': 50,
            'max_html_artifacts': 5,
            'min_content_ratio': 0.6,
            'max_link_density': 0.3,
            'max_whitespace_ratio': 0.15
        }
        
        # Golden set patterns for comparison
        self.golden_patterns = {
            'good_structure': [
                r'^# .+',  # Main heading
                r'^## .+', # Section headings
                r'\[.+\]\(.+\)',  # Proper links
                r'```\w*\n.*?\n```',  # Code blocks
            ],
            'bad_patterns': [
                r'<\w+[^>]*>',  # HTML tags
                r'javascript:',  # JavaScript
                r'onclick=',   # Event handlers
                r'\n\s*\n\s*\n\s*\n',  # Excessive newlines
            ]
        }
    
    def validate_content(self, markdown: str, url: str = "", expected_word_count: int = None) -> QualityValidationResult:
        """
        Perform comprehensive quality validation on markdown content.
        
        Args:
            markdown: The markdown content to validate
            url: Optional URL for context-specific validation
            expected_word_count: Optional expected word count for comparison
            
        Returns:
            QualityValidationResult with detailed analysis
        """
        
        issues = []
        warnings = []
        recommendations = []
        
        # Basic metrics
        word_count = len(markdown.split())
        char_count = len(markdown)
        
        if word_count == 0:
            return QualityValidationResult(
                passed=False,
                score=0.0,
                issues=["Empty content - no words extracted"],
                warnings=[],
                html_artifacts_found=0,
                script_contamination=False,
                excessive_whitespace=False,
                content_to_navigation_ratio=0.0,
                link_density=0.0,
                word_count=0,
                category="poor",
                recommendations=["Check CSS selectors and extraction configuration"]
            )
        
        # 1. HTML artifacts detection
        html_artifacts = self._detect_html_artifacts(markdown)
        if html_artifacts > self.thresholds['max_html_artifacts']:
            issues.append(f"HTML artifacts detected: {html_artifacts} instances")
            recommendations.append("Review Html2TextConverter configuration")
        
        # 2. Script contamination check
        script_contamination = self._detect_script_contamination(markdown)
        if script_contamination:
            issues.append("JavaScript code contamination detected")
            recommendations.append("Improve CSS selectors to exclude script blocks")
        
        # 3. Whitespace analysis
        excessive_whitespace = self._detect_excessive_whitespace(markdown)
        if excessive_whitespace:
            warnings.append("Excessive whitespace patterns detected")
            recommendations.append("Consider markdown cleanup post-processing")
        
        # 4. Content-to-navigation ratio
        content_ratio = self._calculate_content_ratio(markdown)
        if content_ratio < self.thresholds['min_content_ratio']:
            issues.append(f"Low content ratio: {content_ratio:.2f} (expected ≥{self.thresholds['min_content_ratio']})")
            recommendations.append("Refine CSS selectors to exclude more navigation elements")
        
        # 5. Link density analysis
        link_density = self._calculate_link_density(markdown)
        if link_density > self.thresholds['max_link_density']:
            warnings.append(f"High link density: {link_density:.2f}")
            recommendations.append("Consider if excessive links indicate navigation contamination")
        
        # 6. Word count validation
        if word_count < self.thresholds['min_word_count']:
            issues.append(f"Low word count: {word_count} (expected ≥{self.thresholds['min_word_count']})")
            recommendations.append("Check if main content is being properly targeted")
        
        # 7. Expected word count comparison
        if expected_word_count and abs(word_count - expected_word_count) / expected_word_count > 0.5:
            warnings.append(f"Word count deviation: got {word_count}, expected ~{expected_word_count}")
        
        # 8. Golden set pattern validation
        golden_score = self._validate_against_golden_patterns(markdown)
        if golden_score < 0.7:
            warnings.append(f"Golden pattern match: {golden_score:.2f}")
            recommendations.append("Content structure may not match expected documentation patterns")
        
        # Calculate overall score and category
        score = self._calculate_overall_score(
            word_count, html_artifacts, script_contamination, 
            content_ratio, link_density, excessive_whitespace, golden_score
        )
        
        category = self._determine_quality_category(score)
        passed = len(issues) == 0 and score >= 0.6
        
        logger.info(f"Quality validation: {category} (score: {score:.2f}, issues: {len(issues)}, warnings: {len(warnings)})")
        
        return QualityValidationResult(
            passed=passed,
            score=score,
            issues=issues,
            warnings=warnings,
            html_artifacts_found=html_artifacts,
            script_contamination=script_contamination,
            excessive_whitespace=excessive_whitespace,
            content_to_navigation_ratio=content_ratio,
            link_density=link_density,
            word_count=word_count,
            category=category,
            recommendations=recommendations
        )
    
    def _detect_html_artifacts(self, markdown: str) -> int:
        """Detect HTML artifacts that should not be in clean markdown."""
        count = 0
        markdown_lower = markdown.lower()
        
        for artifact in self.html_blacklist:
            count += markdown_lower.count(artifact.lower())
        
        # Additional regex-based detection
        html_tag_pattern = r'<[^>]+>'
        html_tags = re.findall(html_tag_pattern, markdown)
        count += len(html_tags)
        
        return count
    
    def _detect_script_contamination(self, markdown: str) -> bool:
        """Detect JavaScript code contamination."""
        markdown_lower = markdown.lower()
        
        for indicator in self.script_indicators:
            if indicator.lower() in markdown_lower:
                return True
        
        # Check for script-like patterns
        script_patterns = [
            r'function\s*\([^)]*\)\s*{',
            r'var\s+\w+\s*=',
            r'document\.\w+',
            r'window\.\w+',
        ]
        
        for pattern in script_patterns:
            if re.search(pattern, markdown, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_excessive_whitespace(self, markdown: str) -> bool:
        """Detect excessive whitespace patterns."""
        # Count consecutive newlines
        excessive_newlines = re.findall(r'\n\s*\n\s*\n\s*\n', markdown)
        if len(excessive_newlines) > 3:
            return True
        
        # Calculate whitespace ratio
        total_chars = len(markdown)
        whitespace_chars = len(re.findall(r'\s', markdown))
        
        if total_chars > 0:
            whitespace_ratio = whitespace_chars / total_chars
            return whitespace_ratio > self.thresholds['max_whitespace_ratio']
        
        return False
    
    def _calculate_content_ratio(self, markdown: str) -> float:
        """Calculate content-to-navigation ratio."""
        total_words = len(markdown.split())
        
        if total_words == 0:
            return 0.0
        
        # Count navigation-related words
        markdown_lower = markdown.lower()
        nav_word_count = 0
        
        for indicator in self.navigation_indicators:
            nav_word_count += markdown_lower.count(indicator.lower())
        
        # Rough heuristic for content ratio
        nav_ratio = min(nav_word_count / total_words, 1.0)
        content_ratio = max(0.0, 1.0 - nav_ratio * 2)  # Penalize navigation words
        
        return content_ratio
    
    def _calculate_link_density(self, markdown: str) -> float:
        """Calculate link density (links per word)."""
        total_words = len(markdown.split())
        
        if total_words == 0:
            return 0.0
        
        # Count markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, markdown)
        
        # Count bare URLs  
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, markdown)
        
        total_links = len(links) + len(urls)
        return total_links / total_words
    
    def _validate_against_golden_patterns(self, markdown: str) -> float:
        """Validate content against golden set patterns."""
        good_score = 0
        bad_score = 0
        
        # Check for good patterns
        for pattern in self.golden_patterns['good_structure']:
            if re.search(pattern, markdown, re.MULTILINE):
                good_score += 1
        
        # Check for bad patterns
        for pattern in self.golden_patterns['bad_patterns']:
            if re.search(pattern, markdown, re.MULTILINE | re.DOTALL):
                bad_score += 1
        
        # Normalize score
        total_good_patterns = len(self.golden_patterns['good_structure'])
        total_bad_patterns = len(self.golden_patterns['bad_patterns'])
        
        good_ratio = good_score / total_good_patterns if total_good_patterns > 0 else 0
        bad_penalty = bad_score / total_bad_patterns if total_bad_patterns > 0 else 0
        
        return max(0.0, good_ratio - bad_penalty)
    
    def _calculate_overall_score(self, word_count: int, html_artifacts: int, 
                               script_contamination: bool, content_ratio: float,
                               link_density: float, excessive_whitespace: bool,
                               golden_score: float) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        
        score = 1.0
        
        # Word count score (0.2 weight)
        if word_count >= self.thresholds['min_word_count']:
            word_score = min(1.0, word_count / 200)  # Optimal around 200 words
        else:
            word_score = word_count / self.thresholds['min_word_count']
        score *= (0.8 + 0.2 * word_score)
        
        # HTML artifacts penalty (0.2 weight) 
        if html_artifacts > 0:
            artifact_penalty = min(0.5, html_artifacts / 10)
            score *= (1.0 - artifact_penalty)
        
        # Script contamination penalty (0.3 weight)
        if script_contamination:
            score *= 0.7
        
        # Content ratio score (0.3 weight)
        score *= (0.7 + 0.3 * content_ratio)
        
        # Link density penalty (0.1 weight)
        if link_density > self.thresholds['max_link_density']:
            link_penalty = min(0.2, (link_density - self.thresholds['max_link_density']) * 2)
            score *= (1.0 - link_penalty)
        
        # Whitespace penalty (0.1 weight)
        if excessive_whitespace:
            score *= 0.9
        
        # Golden pattern score (0.2 weight)
        score *= (0.8 + 0.2 * golden_score)
        
        return max(0.0, min(1.0, score))
    
    def _determine_quality_category(self, score: float) -> str:
        """Determine quality category based on score."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.75:
            return "good"
        elif score >= 0.6:
            return "fair"
        else:
            return "poor"


class GoldenSetManager:
    """
    Manager for golden set comparison and benchmarking.
    
    Maintains a collection of high-quality markdown examples for
    automated comparison and quality benchmarking.
    """
    
    def __init__(self, golden_set_path: Optional[Path] = None):
        self.golden_set_path = golden_set_path or Path("golden_set")
        self.validator = ContentQualityValidator()
        
    def create_golden_example(self, url: str, markdown: str, 
                            description: str = "") -> bool:
        """
        Add a markdown example to the golden set.
        
        Args:
            url: Source URL
            markdown: High-quality markdown content
            description: Optional description of why this is a good example
            
        Returns:
            True if successfully added
        """
        
        # Validate the content first
        result = self.validator.validate_content(markdown, url)
        
        if result.score < 0.8:
            logger.warning(f"Content quality too low for golden set: {result.score}")
            return False
        
        # Create golden set directory if needed
        self.golden_set_path.mkdir(exist_ok=True)
        
        # Generate filename from URL
        domain = urlparse(url).netloc.replace('.', '_')
        path_safe = re.sub(r'[^\w\-_]', '_', urlparse(url).path)
        filename = f"{domain}_{path_safe}.md"
        
        # Save markdown with metadata
        file_path = self.golden_set_path / filename
        
        metadata = f"""<!-- Golden Set Example
URL: {url}
Description: {description}
Quality Score: {result.score:.3f}
Word Count: {result.word_count}
Created: {Path(__file__).stat().st_mtime}
-->

"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(metadata + markdown)
        
        logger.info(f"Added golden set example: {filename}")
        return True
    
    def compare_against_golden_set(self, markdown: str) -> Dict[str, float]:
        """
        Compare markdown against golden set examples.
        
        Args:
            markdown: Markdown content to compare
            
        Returns:
            Dictionary of similarity scores against golden examples
        """
        
        if not self.golden_set_path.exists():
            return {}
        
        similarities = {}
        
        for golden_file in self.golden_set_path.glob("*.md"):
            with open(golden_file, 'r', encoding='utf-8') as f:
                golden_content = f.read()
            
            # Extract just the markdown (skip metadata)
            lines = golden_content.split('\n')
            content_start = 0
            for i, line in enumerate(lines):
                if line.strip() == '-->' and i > 0:
                    content_start = i + 1
                    break
            
            golden_markdown = '\n'.join(lines[content_start:])
            
            # Simple similarity calculation
            similarity = self._calculate_similarity(markdown, golden_markdown)
            similarities[golden_file.stem] = similarity
        
        return similarities
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple similarity between two markdown contents."""
        
        # Normalize content
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


# Convenience functions for integration

def validate_crawler_output(markdown: str, url: str = "", 
                          expected_word_count: int = None) -> QualityValidationResult:
    """
    Convenience function for validating crawler output.
    
    Args:
        markdown: Markdown content from crawler
        url: Source URL for context
        expected_word_count: Expected word count for comparison
        
    Returns:
        QualityValidationResult
    """
    validator = ContentQualityValidator()
    return validator.validate_content(markdown, url, expected_word_count)


def create_quality_report(results: List[QualityValidationResult], 
                         output_path: Optional[Path] = None) -> str:
    """
    Create a comprehensive quality report from validation results.
    
    Args:
        results: List of validation results
        output_path: Optional path to save report
        
    Returns:
        Report content as string
    """
    
    if not results:
        return "No validation results to report."
    
    # Calculate aggregate statistics
    total_results = len(results)
    passed_count = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / total_results
    
    categories = {}
    for result in results:
        categories[result.category] = categories.get(result.category, 0) + 1
    
    # Generate report
    report = f"""# Content Quality Validation Report

## Summary
- Total validations: {total_results}
- Passed: {passed_count} ({passed_count/total_results*100:.1f}%)
- Average score: {avg_score:.3f}

## Quality Distribution
"""
    
    for category, count in sorted(categories.items()):
        percentage = count / total_results * 100
        report += f"- {category.title()}: {count} ({percentage:.1f}%)\n"
    
    report += "\n## Common Issues\n"
    
    # Aggregate common issues
    issue_counts = {}
    for result in results:
        for issue in result.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        report += f"- {issue}: {count} occurrences\n"
    
    report += "\n## Recommendations\n"
    
    # Aggregate recommendations
    rec_counts = {}
    for result in results:
        for rec in result.recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
    
    for rec, count in sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        report += f"- {rec} (needed for {count} cases)\n"
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Quality report saved to {output_path}")
    
    return report