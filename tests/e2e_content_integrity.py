#!/usr/bin/env python3
"""
End-to-End Content Integrity Testing Framework

This comprehensive test suite validates that the crawler system extracts actual content
rather than navigation elements, with specific assertions for content presence and
navigation absence across different page types and documentation frameworks.

Key Features:
- Granular content presence assertions (specific text, headings, definitions)
- Navigation absence verification (menus, sidebars, breadcrumbs)
- Framework detection validation for different documentation sites
- Quality score range validation for different content types
- Performance baseline verification
- Content structure validation (headings hierarchy, links, code blocks)
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import sys
import os
from datetime import datetime

# Add src directory to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from advanced_web_crawler import (
        AdvancedWebCrawler, 
        AdvancedCrawlResult,
        crawl_single_page_advanced
    )
    from content_quality import (
        ContentQualityAnalyzer,
        ContentQualityMetrics,
        calculate_content_quality
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Ensure you're running from the project root or that the src module is in your Python path")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ContentAssertion:
    """Defines what content should be present or absent."""
    
    # Required content (must be present)
    required_headings: List[str] = None  # Specific headings that must exist
    required_text_fragments: List[str] = None  # Key text that must be present
    required_keywords: List[str] = None  # Important keywords
    required_code_blocks: int = 0  # Minimum number of code blocks
    required_links: List[str] = None  # Specific links that should exist
    
    # Forbidden content (must be absent)
    forbidden_navigation_terms: List[str] = None  # Navigation-specific terms
    forbidden_ui_elements: List[str] = None  # UI elements like "menu", "sidebar"
    max_navigation_density: float = 0.3  # Maximum allowed navigation density
    
    # Quality expectations
    min_quality_score: float = 0.7
    max_quality_score: float = 1.0
    min_word_count: int = 100
    min_content_to_nav_ratio: float = 0.6
    
    # Performance expectations
    max_extraction_time_ms: float = 10000  # 10 seconds
    expected_framework: Optional[str] = None


@dataclass
class E2ETestCase:
    """Defines a complete end-to-end test case."""
    
    test_id: str
    url: str
    description: str
    page_type: str  # 'glossary', 'api', 'guide', 'reference'
    site_framework: str  # 'material_design', 'gitbook', 'sphinx', etc.
    content_assertions: ContentAssertion
    
    # Test metadata
    priority: str = "normal"  # 'critical', 'high', 'normal', 'low'
    timeout_ms: int = 30000
    retry_attempts: int = 2


@dataclass
class E2ETestResult:
    """Result of running a single E2E test case."""
    
    test_case: E2ETestCase
    overall_passed: bool = False
    test_timestamp: str = ""
    
    # Crawl results
    crawl_success: bool = False
    crawl_result: Optional[AdvancedCrawlResult] = None
    quality_metrics: Optional[ContentQualityMetrics] = None
    
    # Content assertion results
    content_assertions_passed: List[str] = None
    content_assertions_failed: List[str] = None
    
    # Specific validation results
    required_headings_found: List[str] = None
    required_headings_missing: List[str] = None
    required_text_found: List[str] = None
    required_text_missing: List[str] = None
    forbidden_content_found: List[str] = None
    
    # Quality validation
    quality_score_in_range: bool = False
    word_count_sufficient: bool = False
    content_nav_ratio_sufficient: bool = False
    framework_detected_correctly: bool = False
    
    # Performance validation
    extraction_time_acceptable: bool = False
    extraction_time_ms: float = 0.0
    
    # Detailed analysis
    extracted_headings: List[str] = None
    detected_navigation_elements: List[str] = None
    link_analysis: Dict[str, Any] = None
    
    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = None


class E2EContentIntegrityTester:
    """
    Comprehensive E2E testing framework for content integrity validation.
    
    Tests that the crawler extracts meaningful content rather than navigation
    elements, with specific assertions for different types of documentation pages.
    """
    
    def __init__(self):
        """Initialize the E2E content integrity tester."""
        self.test_results: List[E2ETestResult] = []
        
        # Define comprehensive test cases for different scenarios
        self.test_cases = self._define_test_cases()
    
    def _define_test_cases(self) -> List[E2ETestCase]:
        """Define comprehensive test cases covering various scenarios."""
        
        test_cases = []
        
        # N8N.io Glossary Page - Critical test case
        test_cases.append(E2ETestCase(
            test_id="n8n_glossary_main",
            url="https://docs.n8n.io/glossary/",
            description="N8N.io main glossary page with comprehensive definitions",
            page_type="glossary",
            site_framework="material_design",
            priority="critical",
            content_assertions=ContentAssertion(
                required_headings=[
                    "Glossary",
                    "Workflow",
                    "Node", 
                    "Execution",
                    "Connection",
                    "Trigger"
                ],
                required_text_fragments=[
                    "A workflow is a collection of nodes",
                    "Nodes are the building blocks",
                    "execution of a workflow",
                    "trigger node starts",
                    "connection links nodes"
                ],
                required_keywords=[
                    "workflow", "node", "execution", "trigger", 
                    "connection", "data", "automation"
                ],
                forbidden_navigation_terms=[
                    "Skip to content", "Main navigation", "Table of contents",
                    "Previous page", "Next page", "Edit on GitHub"
                ],
                forbidden_ui_elements=[
                    "sidebar", "menu", "breadcrumb", "navigation",
                    "header", "footer", "skip", "toggle"
                ],
                min_quality_score=0.8,
                min_word_count=1000,
                min_content_to_nav_ratio=0.8,
                max_navigation_density=0.2,
                expected_framework="material_design",
                max_extraction_time_ms=5000
            )
        ))
        
        # N8N.io API Documentation - High priority
        test_cases.append(E2ETestCase(
            test_id="n8n_api_auth",
            url="https://docs.n8n.io/api/authentication/",
            description="N8N.io API authentication documentation",
            page_type="api",
            site_framework="material_design", 
            priority="high",
            content_assertions=ContentAssertion(
                required_headings=[
                    "Authentication",
                    "API Key",
                    "Bearer Token"
                ],
                required_text_fragments=[
                    "authentication method",
                    "API key",
                    "bearer token",
                    "Authorization header"
                ],
                required_keywords=[
                    "authentication", "api", "key", "token",
                    "authorization", "header", "security"
                ],
                required_code_blocks=1,
                forbidden_navigation_terms=[
                    "Skip to content", "Main navigation",
                    "Previous", "Next", "Edit on"
                ],
                min_quality_score=0.7,
                min_word_count=200,
                min_content_to_nav_ratio=0.7,
                expected_framework="material_design"
            )
        ))
        
        # N8N.io Getting Started Guide - High priority  
        test_cases.append(E2ETestCase(
            test_id="n8n_getting_started",
            url="https://docs.n8n.io/getting-started/",
            description="N8N.io getting started guide",
            page_type="guide",
            site_framework="material_design",
            priority="high",
            content_assertions=ContentAssertion(
                required_headings=[
                    "Getting started",
                    "Installation", 
                    "First workflow"
                ],
                required_text_fragments=[
                    "getting started with n8n",
                    "install n8n",
                    "create your first workflow"
                ],
                required_keywords=[
                    "getting started", "installation", "workflow",
                    "tutorial", "guide", "first"
                ],
                forbidden_navigation_terms=[
                    "Skip to content", "Table of contents",
                    "Edit on GitHub", "Previous page"
                ],
                min_quality_score=0.6,
                min_word_count=150,
                min_content_to_nav_ratio=0.6,
                expected_framework="material_design"
            )
        ))
        
        # N8N.io Workflow Components - Normal priority
        test_cases.append(E2ETestCase(
            test_id="n8n_workflow_components",
            url="https://docs.n8n.io/workflows/components/",
            description="N8N.io workflow components documentation",
            page_type="reference",
            site_framework="material_design",
            priority="normal",
            content_assertions=ContentAssertion(
                required_headings=[
                    "Components",
                    "Nodes",
                    "Connections"
                ],
                required_text_fragments=[
                    "workflow components",
                    "nodes and connections",
                    "building workflows"
                ],
                required_keywords=[
                    "components", "nodes", "connections",
                    "workflow", "building"
                ],
                forbidden_navigation_terms=[
                    "Skip to content", "Navigation menu",
                    "Edit on GitHub"
                ],
                min_quality_score=0.5,
                min_word_count=100,
                min_content_to_nav_ratio=0.5,
                expected_framework="material_design"
            )
        ))
        
        return test_cases
    
    async def run_comprehensive_e2e_tests(self, max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Run all E2E content integrity tests.
        
        Args:
            max_concurrent: Maximum concurrent test execution
            
        Returns:
            Comprehensive test results and analysis
        """
        
        logger.info(f"🧪 Starting E2E content integrity tests")
        logger.info(f"Testing {len(self.test_cases)} test cases across multiple priority levels")
        
        start_time = time.time()
        
        # Group tests by priority and run critical/high priority tests first
        critical_tests = [tc for tc in self.test_cases if tc.priority == "critical"]
        high_tests = [tc for tc in self.test_cases if tc.priority == "high"] 
        normal_tests = [tc for tc in self.test_cases if tc.priority == "normal"]
        low_tests = [tc for tc in self.test_cases if tc.priority == "low"]
        
        # Run tests in priority order
        all_results = []
        
        for test_group, group_name in [
            (critical_tests, "CRITICAL"),
            (high_tests, "HIGH"),
            (normal_tests, "NORMAL"), 
            (low_tests, "LOW")
        ]:
            if test_group:
                logger.info(f"Running {len(test_group)} {group_name} priority tests")
                group_results = await self._run_test_group(test_group, max_concurrent)
                all_results.extend(group_results)
                
                # Stop if critical tests fail
                if group_name == "CRITICAL":
                    critical_failed = [r for r in group_results if not r.overall_passed]
                    if critical_failed:
                        logger.error(f"❌ {len(critical_failed)} critical tests failed - stopping execution")
                        break
        
        self.test_results = all_results
        total_time = time.time() - start_time
        
        # Analyze results
        analysis = self._analyze_e2e_results()
        
        logger.info(f"✅ E2E testing completed in {total_time:.1f}s")
        logger.info(f"Results: {analysis['passed_tests']}/{analysis['total_tests']} tests passed")
        
        return {
            'test_results': [asdict(result) for result in self.test_results],
            'analysis': analysis,
            'total_time_seconds': total_time,
            'test_summary_by_priority': self._get_priority_summary()
        }
    
    async def _run_test_group(self, test_cases: List[E2ETestCase], max_concurrent: int) -> List[E2ETestResult]:
        """Run a group of test cases with concurrency control."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_test(test_case: E2ETestCase) -> E2ETestResult:
            async with semaphore:
                return await self._run_single_e2e_test(test_case)
        
        # Execute tests concurrently
        tasks = [run_single_test(tc) for tc in test_cases]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def _run_single_e2e_test(self, test_case: E2ETestCase) -> E2ETestResult:
        """Run a single E2E content integrity test."""
        
        logger.info(f"🧪 Running {test_case.test_id}: {test_case.description}")
        
        result = E2ETestResult(
            test_case=test_case,
            test_timestamp=datetime.now().isoformat(),
            content_assertions_passed=[],
            content_assertions_failed=[],
            required_headings_found=[],
            required_headings_missing=[],
            required_text_found=[],
            required_text_missing=[],
            forbidden_content_found=[],
            extracted_headings=[],
            detected_navigation_elements=[],
            warnings=[]
        )
        
        try:
            # Crawl the page
            start_time = time.time()
            
            async with AdvancedWebCrawler(
                headless=True,
                timeout_ms=test_case.timeout_ms,
                enable_quality_validation=True,
                max_fallback_attempts=test_case.retry_attempts
            ) as crawler:
                crawl_result = await crawler.crawl_single_page(test_case.url)
            
            extraction_time = (time.time() - start_time) * 1000
            result.extraction_time_ms = extraction_time
            result.crawl_result = crawl_result
            result.crawl_success = crawl_result.success
            
            if not crawl_result.success:
                result.error_message = f"Crawl failed: {crawl_result.error_message}"
                return result
            
            # Calculate quality metrics
            result.quality_metrics = calculate_content_quality(crawl_result.markdown)
            
            # Run content integrity validations
            await self._validate_content_assertions(result)
            await self._validate_quality_metrics(result)
            await self._validate_performance(result)
            await self._validate_framework_detection(result)
            
            # Extract detailed content analysis
            self._analyze_content_structure(result)
            
            # Determine overall pass/fail
            result.overall_passed = (
                result.crawl_success and
                len(result.content_assertions_failed) == 0 and
                result.quality_score_in_range and
                result.word_count_sufficient and
                result.content_nav_ratio_sufficient and
                result.extraction_time_acceptable
            )
            
            status = "✅ PASSED" if result.overall_passed else "❌ FAILED"
            logger.info(f"{status} {test_case.test_id}: quality={result.quality_metrics.overall_quality_score:.3f}")
            
        except Exception as e:
            result.error_message = f"Test error: {str(e)}"
            logger.error(f"❌ Test {test_case.test_id} failed with error: {e}")
        
        return result
    
    async def _validate_content_assertions(self, result: E2ETestResult):
        """Validate content presence and absence assertions."""
        
        content = result.crawl_result.markdown.lower()
        assertions = result.test_case.content_assertions
        
        # Check required headings
        if assertions.required_headings:
            for heading in assertions.required_headings:
                # Look for heading patterns like "# Heading", "## Heading", etc.
                heading_pattern = rf'^#+\s+.*{re.escape(heading.lower())}.*$'
                if re.search(heading_pattern, content, re.MULTILINE | re.IGNORECASE):
                    result.required_headings_found.append(heading)
                    result.content_assertions_passed.append(f"Required heading found: {heading}")
                else:
                    result.required_headings_missing.append(heading)
                    result.content_assertions_failed.append(f"Required heading missing: {heading}")
        
        # Check required text fragments
        if assertions.required_text_fragments:
            for text_fragment in assertions.required_text_fragments:
                if text_fragment.lower() in content:
                    result.required_text_found.append(text_fragment)
                    result.content_assertions_passed.append(f"Required text found: {text_fragment}")
                else:
                    result.required_text_missing.append(text_fragment)
                    result.content_assertions_failed.append(f"Required text missing: {text_fragment}")
        
        # Check required keywords
        if assertions.required_keywords:
            for keyword in assertions.required_keywords:
                if keyword.lower() in content:
                    result.content_assertions_passed.append(f"Required keyword found: {keyword}")
                else:
                    result.content_assertions_failed.append(f"Required keyword missing: {keyword}")
        
        # Check forbidden navigation terms
        if assertions.forbidden_navigation_terms:
            for nav_term in assertions.forbidden_navigation_terms:
                if nav_term.lower() in content:
                    result.forbidden_content_found.append(nav_term)
                    result.content_assertions_failed.append(f"Forbidden navigation term found: {nav_term}")
        
        # Check forbidden UI elements
        if assertions.forbidden_ui_elements:
            for ui_element in assertions.forbidden_ui_elements:
                if ui_element.lower() in content:
                    result.forbidden_content_found.append(ui_element)
                    result.content_assertions_failed.append(f"Forbidden UI element found: {ui_element}")
        
        # Check code blocks if required
        if assertions.required_code_blocks > 0:
            code_block_count = content.count('```') // 2  # Count pairs of triple backticks
            if code_block_count >= assertions.required_code_blocks:
                result.content_assertions_passed.append(f"Required code blocks found: {code_block_count}")
            else:
                result.content_assertions_failed.append(
                    f"Insufficient code blocks: {code_block_count} < {assertions.required_code_blocks}"
                )
    
    async def _validate_quality_metrics(self, result: E2ETestResult):
        """Validate quality score and ratio metrics."""
        
        metrics = result.quality_metrics
        assertions = result.test_case.content_assertions
        
        # Quality score range
        result.quality_score_in_range = (
            assertions.min_quality_score <= metrics.overall_quality_score <= assertions.max_quality_score
        )
        
        if not result.quality_score_in_range:
            result.content_assertions_failed.append(
                f"Quality score {metrics.overall_quality_score:.3f} outside range "
                f"[{assertions.min_quality_score:.3f}, {assertions.max_quality_score:.3f}]"
            )
        
        # Word count
        result.word_count_sufficient = metrics.word_count >= assertions.min_word_count
        
        if not result.word_count_sufficient:
            result.content_assertions_failed.append(
                f"Word count {metrics.word_count} below minimum {assertions.min_word_count}"
            )
        
        # Content-to-navigation ratio
        result.content_nav_ratio_sufficient = (
            metrics.content_to_navigation_ratio >= assertions.min_content_to_nav_ratio
        )
        
        if not result.content_nav_ratio_sufficient:
            result.content_assertions_failed.append(
                f"Content-to-nav ratio {metrics.content_to_navigation_ratio:.3f} below minimum "
                f"{assertions.min_content_to_nav_ratio:.3f}"
            )
        
        # Navigation density check
        if metrics.link_density > assertions.max_navigation_density:
            result.content_assertions_failed.append(
                f"Navigation density {metrics.link_density:.3f} exceeds maximum "
                f"{assertions.max_navigation_density:.3f}"
            )
    
    async def _validate_performance(self, result: E2ETestResult):
        """Validate performance metrics."""
        
        assertions = result.test_case.content_assertions
        
        result.extraction_time_acceptable = (
            result.extraction_time_ms <= assertions.max_extraction_time_ms
        )
        
        if not result.extraction_time_acceptable:
            result.content_assertions_failed.append(
                f"Extraction time {result.extraction_time_ms:.1f}ms exceeds maximum "
                f"{assertions.max_extraction_time_ms:.1f}ms"
            )
    
    async def _validate_framework_detection(self, result: E2ETestResult):
        """Validate framework detection."""
        
        expected_framework = result.test_case.content_assertions.expected_framework
        detected_framework = result.crawl_result.framework_detected
        
        result.framework_detected_correctly = (
            expected_framework is None or detected_framework == expected_framework
        )
        
        if not result.framework_detected_correctly:
            result.content_assertions_failed.append(
                f"Framework detection mismatch: expected {expected_framework}, got {detected_framework}"
            )
    
    def _analyze_content_structure(self, result: E2ETestResult):
        """Analyze the structure of extracted content."""
        
        content = result.crawl_result.markdown
        
        # Extract headings
        heading_pattern = r'^(#+)\s+(.+)$'
        headings = re.findall(heading_pattern, content, re.MULTILINE)
        result.extracted_headings = [heading[1] for heading in headings]
        
        # Detect potential navigation elements
        nav_patterns = [
            r'skip to',
            r'table of contents',
            r'previous\s+page',
            r'next\s+page',
            r'edit on github',
            r'main navigation',
            r'sidebar',
            r'breadcrumb'
        ]
        
        for pattern in nav_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            result.detected_navigation_elements.extend(matches)
        
        # Analyze links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, content)
        
        result.link_analysis = {
            'total_links': len(links),
            'internal_links': len([link for link in links if not link[1].startswith('http')]),
            'external_links': len([link for link in links if link[1].startswith('http')]),
            'documentation_links': len([link for link in links if 'docs' in link[1]]),
        }
    
    def _analyze_e2e_results(self) -> Dict[str, Any]:
        """Analyze E2E test results and generate statistics."""
        
        if not self.test_results:
            return {'error': 'No test results to analyze'}
        
        analysis = {
            'total_tests': len(self.test_results),
            'passed_tests': sum(1 for r in self.test_results if r.overall_passed),
            'failed_tests': sum(1 for r in self.test_results if not r.overall_passed),
            'crawl_failures': sum(1 for r in self.test_results if not r.crawl_success),
            'content_assertion_failures': sum(1 for r in self.test_results if r.content_assertions_failed),
            'quality_failures': sum(1 for r in self.test_results if not r.quality_score_in_range),
            'performance_failures': sum(1 for r in self.test_results if not r.extraction_time_acceptable),
            'framework_detection_failures': sum(1 for r in self.test_results if not r.framework_detected_correctly),
            'tests_by_priority': {},
            'tests_by_page_type': {},
            'common_failures': [],
            'performance_stats': {},
            'quality_stats': {}
        }
        
        # Group by priority
        for result in self.test_results:
            priority = result.test_case.priority
            if priority not in analysis['tests_by_priority']:
                analysis['tests_by_priority'][priority] = {
                    'total': 0, 'passed': 0, 'failed': 0
                }
            
            analysis['tests_by_priority'][priority]['total'] += 1
            if result.overall_passed:
                analysis['tests_by_priority'][priority]['passed'] += 1
            else:
                analysis['tests_by_priority'][priority]['failed'] += 1
        
        # Group by page type
        for result in self.test_results:
            page_type = result.test_case.page_type
            if page_type not in analysis['tests_by_page_type']:
                analysis['tests_by_page_type'][page_type] = {
                    'total': 0, 'passed': 0, 'failed': 0
                }
            
            analysis['tests_by_page_type'][page_type]['total'] += 1
            if result.overall_passed:
                analysis['tests_by_page_type'][page_type]['passed'] += 1
            else:
                analysis['tests_by_page_type'][page_type]['failed'] += 1
        
        # Collect common failures
        all_failures = []
        for result in self.test_results:
            all_failures.extend(result.content_assertions_failed)
        
        failure_counts = {}
        for failure in all_failures:
            # Normalize failure message for grouping
            normalized = failure.split(':')[0]
            failure_counts[normalized] = failure_counts.get(normalized, 0) + 1
        
        analysis['common_failures'] = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Performance statistics
        successful_results = [r for r in self.test_results if r.crawl_success]
        if successful_results:
            extraction_times = [r.extraction_time_ms for r in successful_results]
            quality_scores = [r.quality_metrics.overall_quality_score for r in successful_results]
            
            analysis['performance_stats'] = {
                'avg_extraction_time_ms': sum(extraction_times) / len(extraction_times),
                'max_extraction_time_ms': max(extraction_times),
                'min_extraction_time_ms': min(extraction_times),
            }
            
            analysis['quality_stats'] = {
                'avg_quality_score': sum(quality_scores) / len(quality_scores),
                'max_quality_score': max(quality_scores),
                'min_quality_score': min(quality_scores),
            }
        
        return analysis
    
    def _get_priority_summary(self) -> Dict[str, Any]:
        """Get summary of test results by priority level."""
        
        summary = {}
        for result in self.test_results:
            priority = result.test_case.priority
            if priority not in summary:
                summary[priority] = {
                    'tests': [],
                    'passed': 0,
                    'failed': 0,
                    'critical_issues': []
                }
            
            summary[priority]['tests'].append({
                'test_id': result.test_case.test_id,
                'passed': result.overall_passed,
                'url': result.test_case.url,
                'quality_score': result.quality_metrics.overall_quality_score if result.quality_metrics else 0.0
            })
            
            if result.overall_passed:
                summary[priority]['passed'] += 1
            else:
                summary[priority]['failed'] += 1
                if priority == 'critical':
                    summary[priority]['critical_issues'].extend(result.content_assertions_failed)
        
        return summary
    
    def generate_e2e_report(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive E2E test report."""
        
        report = f"""# End-to-End Content Integrity Test Report

## Test Summary
- **Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Tests**: {analysis['total_tests']}
- **Passed Tests**: {analysis['passed_tests']} ({analysis['passed_tests']/analysis['total_tests']*100:.1f}%)
- **Failed Tests**: {analysis['failed_tests']} ({analysis['failed_tests']/analysis['total_tests']*100:.1f}%)

## Test Results by Priority
"""
        
        for priority, stats in analysis['tests_by_priority'].items():
            pass_rate = (stats['passed'] / stats['total']) * 100 if stats['total'] > 0 else 0
            status_icon = "✅" if pass_rate >= 90 else "⚠️" if pass_rate >= 70 else "❌"
            report += f"- **{priority.upper()}**: {status_icon} {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)\n"
        
        report += f"""
## Test Results by Page Type
"""
        
        for page_type, stats in analysis['tests_by_page_type'].items():
            pass_rate = (stats['passed'] / stats['total']) * 100 if stats['total'] > 0 else 0
            report += f"- **{page_type.title()}**: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)\n"
        
        report += f"""
## Failure Analysis

### Content Assertion Failures
- **Tests with Content Issues**: {analysis['content_assertion_failures']}
- **Quality Score Failures**: {analysis['quality_failures']}
- **Performance Failures**: {analysis['performance_failures']}
- **Framework Detection Failures**: {analysis['framework_detection_failures']}

### Most Common Failure Types
"""
        
        for failure_type, count in analysis['common_failures']:
            report += f"- **{failure_type}**: {count} occurrences\n"
        
        if 'performance_stats' in analysis and analysis['performance_stats']:
            report += f"""
## Performance Analysis
- **Average Extraction Time**: {analysis['performance_stats']['avg_extraction_time_ms']:.1f}ms
- **Fastest Extraction**: {analysis['performance_stats']['min_extraction_time_ms']:.1f}ms  
- **Slowest Extraction**: {analysis['performance_stats']['max_extraction_time_ms']:.1f}ms
"""
        
        if 'quality_stats' in analysis and analysis['quality_stats']:
            report += f"""
## Quality Analysis
- **Average Quality Score**: {analysis['quality_stats']['avg_quality_score']:.3f}
- **Highest Quality Score**: {analysis['quality_stats']['max_quality_score']:.3f}
- **Lowest Quality Score**: {analysis['quality_stats']['min_quality_score']:.3f}
"""
        
        report += f"""
## Recommendations

### Critical Issues
"""
        
        critical_pass_rate = analysis['tests_by_priority'].get('critical', {}).get('passed', 0) / max(1, analysis['tests_by_priority'].get('critical', {}).get('total', 1)) * 100
        
        if critical_pass_rate < 100:
            report += "- 🚨 Critical tests are failing - immediate attention required\n"
        
        if analysis['content_assertion_failures'] > analysis['total_tests'] * 0.3:
            report += "- 📝 High rate of content assertion failures - review content extraction logic\n"
        
        if analysis['quality_failures'] > analysis['total_tests'] * 0.2:
            report += "- 📊 Multiple quality score failures - review quality metrics\n"
        
        if analysis['performance_failures'] > 0:
            report += "- ⚡ Performance issues detected - optimize extraction speed\n"
        
        report += f"""
### Quality Improvements
"""
        
        if analysis['passed_tests'] / analysis['total_tests'] >= 0.9:
            report += "- ✅ Excellent overall pass rate - system is performing well\n"
        
        if analysis['crawl_failures'] == 0:
            report += "- 🌐 All crawl attempts successful - good stability\n"
        
        report += f"""
## Conclusion

The E2E content integrity test {'**PASSED**' if analysis['passed_tests'] / analysis['total_tests'] >= 0.8 else '**FAILED**'} with a {analysis['passed_tests']/analysis['total_tests']*100:.1f}% success rate.

{'🎯 The crawler system successfully extracts meaningful content and avoids navigation elements.' if analysis['passed_tests'] / analysis['total_tests'] >= 0.9 else '⚠️ Some content integrity issues detected - review failed assertions and extraction logic.'}
"""
        
        return report


# Main execution functions

async def run_e2e_content_integrity_test():
    """Main function to run E2E content integrity tests."""
    
    print("🧪 End-to-End Content Integrity Testing Framework")
    print("=" * 60)
    print("Testing crawler content extraction vs navigation detection")
    print("with granular assertions for content presence and quality.\n")
    
    # Initialize tester
    tester = E2EContentIntegrityTester()
    
    print(f"📋 Configured {len(tester.test_cases)} comprehensive test cases:")
    for tc in tester.test_cases:
        print(f"  - {tc.test_id} ({tc.priority}): {tc.description}")
    
    print()
    
    try:
        # Run comprehensive E2E tests
        results = await tester.run_comprehensive_e2e_tests(max_concurrent=2)
        
        # Generate and display report
        report = tester.generate_e2e_report(results['analysis'])
        
        print("\n" + "=" * 60)
        print("📋 E2E CONTENT INTEGRITY TEST REPORT")
        print("=" * 60)
        print(report)
        
        # Save results to files
        results_dir = project_root / "test_results" / "e2e_content_integrity"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"e2e_results_{timestamp}.json"
        report_file = results_dir / f"e2e_report_{timestamp}.md"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📁 Results saved to:")
        print(f"   JSON: {results_file}")
        print(f"   Report: {report_file}")
        
        # Return overall success/failure
        overall_success = results['analysis']['passed_tests'] / results['analysis']['total_tests'] >= 0.8
        print(f"\n{'🎉 E2E CONTENT INTEGRITY TESTS PASSED' if overall_success else '❌ E2E CONTENT INTEGRITY TESTS FAILED'}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ E2E content integrity test failed: {e}")
        logger.error(f"E2E test error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Run the E2E content integrity tests
    asyncio.run(run_e2e_content_integrity_test())