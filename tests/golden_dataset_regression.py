#!/usr/bin/env python3
"""
Golden Dataset Regression Testing Framework for Crawler Quality Validation

This framework implements a comprehensive regression testing system that:
1. Maintains a curated "golden dataset" of high-quality crawl results
2. Compares current crawler outputs against golden standards
3. Detects quality regressions and content drift
4. Provides automated validation for continuous integration

Features:
- Golden dataset management with versioning
- Semantic comparison using embeddings for resilience
- Performance baseline tracking  
- Automated golden dataset updates with approval workflows
- Comprehensive diff reporting with visual comparisons
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import sys
import os
from datetime import datetime, timedelta

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
    # Optional: sentence-transformers for semantic comparison
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        
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
class GoldenDatasetEntry:
    """A single entry in the golden dataset."""
    
    # Identification
    url: str
    page_type: str  # 'documentation', 'api', 'glossary', 'reference'
    test_site: str  # 'n8n.io', 'docs.python.org', etc.
    entry_id: str   # Unique identifier
    
    # Golden content
    golden_markdown: str
    golden_quality_metrics: ContentQualityMetrics
    
    # Metadata
    created_at: str
    last_updated: str
    created_by: str  # Tool/person that created this entry
    approval_status: str  # 'pending', 'approved', 'deprecated'
    
    # Performance baselines
    expected_extraction_time_ms: float
    expected_word_count_range: Tuple[int, int]  # (min, max)
    expected_quality_score_range: Tuple[float, float]  # (min, max)
    
    # Validation settings
    strict_comparison: bool = False  # Whether to use exact string matching
    tolerance_threshold: float = 0.95  # Similarity threshold for semantic comparison
    
    # Test configuration
    framework_expected: Optional[str] = None
    css_selectors_expected: List[str] = None
    


@dataclass 
class RegressionTestResult:
    """Result of comparing current crawl against golden dataset."""
    
    # Basic info
    entry_id: str
    url: str
    test_timestamp: str
    
    # Comparison results
    overall_passed: bool = False
    similarity_score: float = 0.0
    quality_score_diff: float = 0.0
    word_count_diff: int = 0
    extraction_time_diff_ms: float = 0.0
    
    # Detailed comparisons
    content_similarity: float = 0.0
    structure_similarity: float = 0.0
    quality_metrics_comparison: Dict[str, Any] = None
    
    # Issues and warnings
    critical_issues: List[str] = None
    warnings: List[str] = None
    performance_issues: List[str] = None
    
    # Current crawl result
    current_result: Optional[AdvancedCrawlResult] = None
    current_quality_metrics: Optional[ContentQualityMetrics] = None
    
    # Detailed diff information
    content_diff_summary: str = ""
    missing_headings: List[str] = None
    extra_headings: List[str] = None
    missing_links: List[str] = None
    extra_links: List[str] = None


class GoldenDatasetManager:
    """
    Manages the golden dataset for regression testing.
    
    Provides functionality to:
    - Load and save golden datasets
    - Create new entries from successful crawls
    - Update existing entries with approval workflows
    - Validate dataset integrity
    """
    
    def __init__(self, dataset_path: Path = None):
        """Initialize the golden dataset manager."""
        
        if dataset_path is None:
            dataset_path = project_root / "tests" / "golden_dataset"
        
        self.dataset_path = dataset_path
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset files
        self.entries_file = self.dataset_path / "golden_entries.json"
        self.metadata_file = self.dataset_path / "dataset_metadata.json"
        self.approvals_file = self.dataset_path / "pending_approvals.json"
        
        # Load existing dataset
        self.entries: Dict[str, GoldenDatasetEntry] = {}
        self.load_dataset()
        
        # Optional semantic similarity model
        self.similarity_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Could not load sentence transformer model: {e}")
    
    def load_dataset(self):
        """Load the golden dataset from disk."""
        
        if not self.entries_file.exists():
            logger.info("No existing golden dataset found. Starting with empty dataset.")
            return
        
        try:
            with open(self.entries_file, 'r', encoding='utf-8') as f:
                entries_data = json.load(f)
            
            self.entries = {}
            for entry_id, entry_dict in entries_data.items():
                # Convert quality metrics back to dataclass
                if 'golden_quality_metrics' in entry_dict:
                    metrics_dict = entry_dict['golden_quality_metrics']
                    entry_dict['golden_quality_metrics'] = ContentQualityMetrics(**metrics_dict)
                
                # Convert tuples back from lists
                if 'expected_word_count_range' in entry_dict:
                    entry_dict['expected_word_count_range'] = tuple(entry_dict['expected_word_count_range'])
                if 'expected_quality_score_range' in entry_dict:
                    entry_dict['expected_quality_score_range'] = tuple(entry_dict['expected_quality_score_range'])
                
                self.entries[entry_id] = GoldenDatasetEntry(**entry_dict)
            
            logger.info(f"Loaded {len(self.entries)} golden dataset entries")
            
        except Exception as e:
            logger.error(f"Error loading golden dataset: {e}")
            self.entries = {}
    
    def save_dataset(self):
        """Save the golden dataset to disk."""
        
        try:
            # Convert entries to serializable format
            entries_data = {}
            for entry_id, entry in self.entries.items():
                entry_dict = asdict(entry)
                
                # Convert quality metrics to dict
                if hasattr(entry_dict['golden_quality_metrics'], '__dict__'):
                    entry_dict['golden_quality_metrics'] = asdict(entry.golden_quality_metrics)
                
                entries_data[entry_id] = entry_dict
            
            with open(self.entries_file, 'w', encoding='utf-8') as f:
                json.dump(entries_data, f, indent=2, default=str)
            
            # Update metadata
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'total_entries': len(self.entries),
                'entries_by_site': {},
                'entries_by_type': {}
            }
            
            # Calculate statistics
            for entry in self.entries.values():
                site = entry.test_site
                page_type = entry.page_type
                
                metadata['entries_by_site'][site] = metadata['entries_by_site'].get(site, 0) + 1
                metadata['entries_by_type'][page_type] = metadata['entries_by_type'].get(page_type, 0) + 1
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved golden dataset with {len(self.entries)} entries")
            
        except Exception as e:
            logger.error(f"Error saving golden dataset: {e}")
            raise
    
    def create_entry_from_crawl(
        self,
        url: str,
        crawl_result: AdvancedCrawlResult,
        quality_metrics: ContentQualityMetrics,
        page_type: str,
        test_site: str,
        created_by: str = "automated",
        auto_approve: bool = False
    ) -> str:
        """
        Create a new golden dataset entry from a successful crawl.
        
        Args:
            url: The URL that was crawled
            crawl_result: The successful crawl result
            quality_metrics: Quality metrics for the content
            page_type: Type of page (documentation, api, etc.)
            test_site: Site being tested (n8n.io, etc.)
            created_by: Who/what created this entry
            auto_approve: Whether to auto-approve (use carefully!)
            
        Returns:
            entry_id: Unique identifier for the new entry
        """
        
        # Generate unique ID
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        entry_id = f"{test_site}_{page_type}_{url_hash}_{timestamp}"
        
        # Calculate performance baselines with some tolerance
        word_count = quality_metrics.word_count
        word_count_range = (
            max(1, int(word_count * 0.9)),  # Allow 10% decrease
            int(word_count * 1.2)           # Allow 20% increase
        )
        
        quality_score = quality_metrics.overall_quality_score
        quality_score_range = (
            max(0.0, quality_score - 0.1),  # Allow 0.1 decrease
            min(1.0, quality_score + 0.05)  # Allow small increase
        )
        
        # Create entry
        entry = GoldenDatasetEntry(
            url=url,
            page_type=page_type,
            test_site=test_site,
            entry_id=entry_id,
            golden_markdown=crawl_result.markdown,
            golden_quality_metrics=quality_metrics,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            created_by=created_by,
            approval_status="approved" if auto_approve else "pending",
            expected_extraction_time_ms=crawl_result.extraction_time_ms * 1.5,  # Allow 50% slower
            expected_word_count_range=word_count_range,
            expected_quality_score_range=quality_score_range,
            framework_expected=crawl_result.framework_detected,
            css_selectors_expected=getattr(crawl_result, 'css_selectors_used', [])
        )
        
        # Add to dataset
        self.entries[entry_id] = entry
        
        # Save dataset
        self.save_dataset()
        
        logger.info(f"Created golden dataset entry {entry_id} for {url}")
        logger.info(f"  - Approval status: {entry.approval_status}")
        logger.info(f"  - Word count range: {word_count_range}")
        logger.info(f"  - Quality score range: {quality_score_range}")
        
        return entry_id
    
    def get_entries_for_testing(self, include_pending: bool = False) -> List[GoldenDatasetEntry]:
        """Get all entries suitable for regression testing."""
        
        entries = []
        for entry in self.entries.values():
            if entry.approval_status == "approved":
                entries.append(entry)
            elif include_pending and entry.approval_status == "pending":
                entries.append(entry)
        
        return entries
    
    def calculate_content_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text contents.
        
        Uses sentence transformers if available, otherwise falls back to
        simple word overlap similarity.
        """
        
        if self.similarity_model is not None:
            try:
                # Use semantic similarity
                embeddings = self.similarity_model.encode([text1, text2])
                similarity = float(embeddings[0] @ embeddings[1] / 
                                 (embeddings[0].norm() * embeddings[1].norm()))
                return similarity
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
        
        # Fallback to simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


class GoldenDatasetRegressionTester:
    """
    Runs regression tests against the golden dataset.
    
    Compares current crawler outputs against golden standards and
    identifies regressions, improvements, and drift.
    """
    
    def __init__(self, dataset_manager: GoldenDatasetManager = None):
        """Initialize the regression tester."""
        
        self.dataset_manager = dataset_manager or GoldenDatasetManager()
        self.test_results: List[RegressionTestResult] = []
    
    async def run_full_regression_test(self, max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Run regression tests against all approved golden dataset entries.
        
        Args:
            max_concurrent: Maximum concurrent crawling sessions
            
        Returns:
            Comprehensive test results and analysis
        """
        
        entries = self.dataset_manager.get_entries_for_testing()
        
        if not entries:
            logger.warning("No approved golden dataset entries found for testing")
            return {
                'error': 'No golden dataset entries available',
                'entries_available': len(self.dataset_manager.entries),
                'approved_entries': 0
            }
        
        logger.info(f"🧪 Starting regression test against {len(entries)} golden dataset entries")
        
        start_time = time.time()
        
        # Run tests with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def test_single_entry(entry: GoldenDatasetEntry) -> RegressionTestResult:
            async with semaphore:
                return await self._test_against_golden_entry(entry)
        
        # Execute all tests
        tasks = [test_single_entry(entry) for entry in entries]
        self.test_results = await asyncio.gather(*tasks, return_exceptions=False)
        
        total_time = time.time() - start_time
        
        # Analyze results
        analysis = self._analyze_regression_results()
        
        logger.info(f"✅ Regression test completed in {total_time:.1f}s")
        logger.info(f"Results: {analysis['passed_tests']}/{analysis['total_tests']} tests passed")
        
        return {
            'test_results': [asdict(result) for result in self.test_results],
            'analysis': analysis,
            'total_time_seconds': total_time,
            'dataset_info': {
                'total_entries': len(self.dataset_manager.entries),
                'tested_entries': len(entries),
                'last_updated': datetime.now().isoformat()
            }
        }
    
    async def _test_against_golden_entry(self, entry: GoldenDatasetEntry) -> RegressionTestResult:
        """Test current crawler against a single golden dataset entry."""
        
        logger.info(f"Testing {entry.url} against golden entry {entry.entry_id}")
        
        result = RegressionTestResult(
            entry_id=entry.entry_id,
            url=entry.url,
            test_timestamp=datetime.now().isoformat(),
            critical_issues=[],
            warnings=[],
            performance_issues=[],
            missing_headings=[],
            extra_headings=[],
            missing_links=[],
            extra_links=[]
        )
        
        try:
            # Crawl the current page
            async with AdvancedWebCrawler(
                headless=True,
                timeout_ms=30000,
                enable_quality_validation=True
            ) as crawler:
                crawl_result = await crawler.crawl_single_page(entry.url)
            
            if not crawl_result.success:
                result.critical_issues.append(f"Crawl failed: {crawl_result.error_message}")
                return result
            
            result.current_result = crawl_result
            result.current_quality_metrics = calculate_content_quality(crawl_result.markdown)
            
            # Compare content similarity
            result.content_similarity = self.dataset_manager.calculate_content_similarity(
                entry.golden_markdown,
                crawl_result.markdown
            )
            
            # Compare quality metrics
            golden_quality = entry.golden_quality_metrics
            current_quality = result.current_quality_metrics
            
            result.quality_score_diff = current_quality.overall_quality_score - golden_quality.overall_quality_score
            result.word_count_diff = current_quality.word_count - golden_quality.word_count
            result.extraction_time_diff_ms = crawl_result.extraction_time_ms - entry.expected_extraction_time_ms
            
            # Check quality score range
            min_quality, max_quality = entry.expected_quality_score_range
            if not (min_quality <= current_quality.overall_quality_score <= max_quality):
                result.critical_issues.append(
                    f"Quality score {current_quality.overall_quality_score:.3f} outside expected range "
                    f"[{min_quality:.3f}, {max_quality:.3f}]"
                )
            
            # Check word count range
            min_words, max_words = entry.expected_word_count_range
            if not (min_words <= current_quality.word_count <= max_words):
                result.warnings.append(
                    f"Word count {current_quality.word_count} outside expected range [{min_words}, {max_words}]"
                )
            
            # Check performance
            if crawl_result.extraction_time_ms > entry.expected_extraction_time_ms:
                result.performance_issues.append(
                    f"Extraction time {crawl_result.extraction_time_ms:.1f}ms exceeds baseline "
                    f"{entry.expected_extraction_time_ms:.1f}ms"
                )
            
            # Content similarity check
            if result.content_similarity < entry.tolerance_threshold:
                result.critical_issues.append(
                    f"Content similarity {result.content_similarity:.3f} below threshold "
                    f"{entry.tolerance_threshold:.3f}"
                )
            
            # Framework detection check
            if entry.framework_expected and crawl_result.framework_detected != entry.framework_expected:
                result.warnings.append(
                    f"Framework detection changed: expected {entry.framework_expected}, "
                    f"got {crawl_result.framework_detected}"
                )
            
            # Calculate overall similarity and pass/fail
            result.similarity_score = (result.content_similarity + 
                                     (1.0 - abs(result.quality_score_diff)) +
                                     (1.0 if not result.critical_issues else 0.0)) / 3.0
            
            result.overall_passed = (
                result.content_similarity >= entry.tolerance_threshold and
                len(result.critical_issues) == 0
            )
            
            # Create detailed quality comparison
            result.quality_metrics_comparison = {
                'golden_score': golden_quality.overall_quality_score,
                'current_score': current_quality.overall_quality_score,
                'score_diff': result.quality_score_diff,
                'golden_word_count': golden_quality.word_count,
                'current_word_count': current_quality.word_count,
                'word_count_diff': result.word_count_diff,
                'golden_category': golden_quality.quality_category,
                'current_category': current_quality.quality_category,
                'content_similarity': result.content_similarity
            }
            
            logger.info(f"✅ Tested {entry.url}: similarity={result.content_similarity:.3f}, passed={result.overall_passed}")
            
        except Exception as e:
            result.critical_issues.append(f"Test error: {str(e)}")
            logger.error(f"❌ Test failed for {entry.url}: {e}")
        
        return result
    
    def _analyze_regression_results(self) -> Dict[str, Any]:
        """Analyze regression test results and generate statistics."""
        
        if not self.test_results:
            return {'error': 'No test results to analyze'}
        
        analysis = {
            'total_tests': len(self.test_results),
            'passed_tests': sum(1 for r in self.test_results if r.overall_passed),
            'failed_tests': sum(1 for r in self.test_results if not r.overall_passed),
            'average_similarity': sum(r.content_similarity for r in self.test_results) / len(self.test_results),
            'average_quality_diff': sum(r.quality_score_diff for r in self.test_results) / len(self.test_results),
            'performance_regressions': sum(1 for r in self.test_results if r.performance_issues),
            'critical_failures': sum(1 for r in self.test_results if r.critical_issues),
            'entries_with_warnings': sum(1 for r in self.test_results if r.warnings),
            'similarity_distribution': {},
            'quality_score_changes': {},
            'common_issues': [],
        }
        
        # Categorize similarity scores
        for result in self.test_results:
            if result.content_similarity >= 0.95:
                category = 'excellent'
            elif result.content_similarity >= 0.90:
                category = 'good'
            elif result.content_similarity >= 0.80:
                category = 'fair'
            else:
                category = 'poor'
            
            analysis['similarity_distribution'][category] = analysis['similarity_distribution'].get(category, 0) + 1
        
        # Categorize quality score changes
        for result in self.test_results:
            if result.quality_score_diff > 0.1:
                category = 'improved'
            elif result.quality_score_diff < -0.1:
                category = 'degraded'
            else:
                category = 'stable'
            
            analysis['quality_score_changes'][category] = analysis['quality_score_changes'].get(category, 0) + 1
        
        # Collect common issues
        all_issues = []
        for result in self.test_results:
            all_issues.extend(result.critical_issues)
            all_issues.extend(result.warnings)
        
        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            # Normalize issue text for grouping
            normalized = issue.lower().split(':')[0]  # Take first part before colon
            issue_counts[normalized] = issue_counts.get(normalized, 0) + 1
        
        # Top 5 most common issues
        analysis['common_issues'] = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return analysis
    
    def generate_regression_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive regression test report."""
        
        report = f"""# Golden Dataset Regression Test Report

## Test Summary
- **Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Tests**: {analysis['total_tests']}
- **Passed Tests**: {analysis['passed_tests']} ({analysis['passed_tests']/analysis['total_tests']*100:.1f}%)
- **Failed Tests**: {analysis['failed_tests']} ({analysis['failed_tests']/analysis['total_tests']*100:.1f}%)
- **Average Content Similarity**: {analysis['average_similarity']:.3f}
- **Average Quality Score Change**: {analysis['average_quality_diff']:+.3f}

## Quality Assessment

### Content Similarity Distribution
"""
        
        thresholds = {'excellent': 0.95, 'good': 0.90, 'fair': 0.80, 'poor': 0.0}
        for category, count in analysis['similarity_distribution'].items():
            percentage = (count / analysis['total_tests']) * 100
            threshold = thresholds.get(category, 0.0)
            report += f"- **{category.title()}** (≥{threshold:.2f}): {count} tests ({percentage:.1f}%)\n"
        
        report += f"""
### Quality Score Changes
"""
        
        for category, count in analysis['quality_score_changes'].items():
            percentage = (count / analysis['total_tests']) * 100
            report += f"- **{category.title()}**: {count} tests ({percentage:.1f}%)\n"
        
        report += f"""
## Issue Analysis

### Critical Failures
- **Tests with Critical Issues**: {analysis['critical_failures']}
- **Performance Regressions**: {analysis['performance_regressions']}
- **Tests with Warnings**: {analysis['entries_with_warnings']}

### Most Common Issues
"""
        
        for issue, count in analysis['common_issues']:
            report += f"- **{issue}**: {count} occurrences\n"
        
        report += f"""
## Recommendations

### Immediate Action Required
"""
        if analysis['critical_failures'] > 0:
            report += f"- 🚨 {analysis['critical_failures']} tests have critical failures requiring immediate attention\n"
        
        if analysis['performance_regressions'] > 0:
            report += f"- ⚠️ {analysis['performance_regressions']} tests show performance regressions\n"
        
        if analysis['average_quality_diff'] < -0.05:
            report += f"- 📉 Average quality score decreased by {abs(analysis['average_quality_diff']):.3f}\n"
        
        if analysis['passed_tests'] / analysis['total_tests'] < 0.8:
            report += "- 🔍 Overall pass rate below 80% - investigate common failure patterns\n"
        
        report += f"""
### Quality Improvements
"""
        if analysis['average_quality_diff'] > 0.05:
            report += f"- ✅ Average quality score improved by {analysis['average_quality_diff']:.3f}\n"
        
        if analysis['passed_tests'] / analysis['total_tests'] >= 0.95:
            report += "- 🎉 Excellent pass rate (≥95%) - system is stable\n"
        
        report += f"""
## Conclusion

The regression test {'**PASSED**' if analysis['passed_tests'] / analysis['total_tests'] >= 0.8 else '**FAILED**'} with a {analysis['passed_tests']/analysis['total_tests']*100:.1f}% success rate.

{'🚀 The crawler system is performing well against the golden dataset.' if analysis['passed_tests'] / analysis['total_tests'] >= 0.9 else '⚠️ Some regressions detected - review failed tests and common issues.'}
"""
        
        return report


# Utility functions for golden dataset management

async def create_n8n_golden_dataset():
    """Create a comprehensive golden dataset from known-good n8n.io pages."""
    
    dataset_manager = GoldenDatasetManager()
    
    # Define high-quality n8n.io pages for golden dataset
    golden_urls = [
        {
            'url': 'https://docs.n8n.io/glossary/',
            'page_type': 'glossary',
            'description': 'Main glossary page with comprehensive terms'
        },
        {
            'url': 'https://docs.n8n.io/getting-started/',
            'page_type': 'documentation',
            'description': 'Getting started guide with clear structure'
        },
        {
            'url': 'https://docs.n8n.io/workflows/components/',
            'page_type': 'documentation', 
            'description': 'Workflow components documentation'
        },
        {
            'url': 'https://docs.n8n.io/api/authentication/',
            'page_type': 'api',
            'description': 'API authentication documentation'
        }
    ]
    
    logger.info(f"🏗️ Creating golden dataset from {len(golden_urls)} n8n.io pages")
    
    created_entries = []
    
    for url_info in golden_urls:
        try:
            logger.info(f"Processing {url_info['url']}")
            
            # Crawl the page
            async with AdvancedWebCrawler(
                headless=True,
                timeout_ms=30000,
                enable_quality_validation=True
            ) as crawler:
                crawl_result = await crawler.crawl_single_page(url_info['url'])
            
            if not crawl_result.success:
                logger.warning(f"Failed to crawl {url_info['url']}: {crawl_result.error_message}")
                continue
            
            # Calculate quality metrics
            quality_metrics = calculate_content_quality(crawl_result.markdown)
            
            # Only add high-quality results to golden dataset
            if quality_metrics.overall_quality_score >= 0.7:
                entry_id = dataset_manager.create_entry_from_crawl(
                    url=url_info['url'],
                    crawl_result=crawl_result,
                    quality_metrics=quality_metrics,
                    page_type=url_info['page_type'],
                    test_site='n8n.io',
                    created_by='create_n8n_golden_dataset',
                    auto_approve=True  # Auto-approve for initial dataset creation
                )
                
                created_entries.append({
                    'entry_id': entry_id,
                    'url': url_info['url'],
                    'quality_score': quality_metrics.overall_quality_score,
                    'word_count': quality_metrics.word_count
                })
                
                logger.info(f"✅ Created golden entry {entry_id}")
            else:
                logger.warning(f"❌ Page quality too low ({quality_metrics.overall_quality_score:.3f}) for golden dataset")
        
        except Exception as e:
            logger.error(f"Error processing {url_info['url']}: {e}")
    
    logger.info(f"✅ Created {len(created_entries)} golden dataset entries")
    
    return created_entries


# Main execution functions

async def run_golden_dataset_regression_test():
    """Main function to run golden dataset regression tests."""
    
    print("🧪 Golden Dataset Regression Testing Framework")
    print("=" * 60)
    print("Testing current crawler against curated golden dataset")
    print("to detect regressions and quality drift.\n")
    
    # Initialize components
    dataset_manager = GoldenDatasetManager()
    tester = GoldenDatasetRegressionTester(dataset_manager)
    
    # Check if we have golden dataset entries
    approved_entries = dataset_manager.get_entries_for_testing()
    
    if not approved_entries:
        print("⚠️ No approved golden dataset entries found.")
        print("Creating initial golden dataset from n8n.io...")
        
        # Create initial golden dataset
        await create_n8n_golden_dataset()
        approved_entries = dataset_manager.get_entries_for_testing()
    
    if not approved_entries:
        print("❌ Failed to create golden dataset entries.")
        return None
    
    print(f"📊 Testing against {len(approved_entries)} golden dataset entries")
    
    try:
        # Run regression tests
        results = await tester.run_full_regression_test(max_concurrent=3)
        
        # Generate and display report
        report = tester.generate_regression_report(results['analysis'])
        
        print("\n" + "=" * 60)
        print("📋 REGRESSION TEST REPORT")
        print("=" * 60)
        print(report)
        
        # Save results to file
        results_dir = project_root / "test_results" / "regression"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"regression_test_{timestamp}.json"
        report_file = results_dir / f"regression_report_{timestamp}.md"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📁 Results saved to:")
        print(f"   JSON: {results_file}")
        print(f"   Report: {report_file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Regression test failed: {e}")
        logger.error(f"Regression test error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Run the regression test
    asyncio.run(run_golden_dataset_regression_test())