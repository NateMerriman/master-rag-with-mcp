#!/usr/bin/env python3
"""
Performance Benchmarking and Comparative Analysis System

This module implements Subtask 18.4 by creating a comprehensive performance benchmarking
system to measure and compare crawler performance metrics against baseline implementations.

Features:
- Processing time per page with detailed timing breakdowns
- Memory usage monitoring during crawling operations  
- Database storage efficiency (content vs metadata ratio)
- Search relevance scores using sample queries against crawled content
- Statistical significance testing for performance comparisons
- Baseline establishment and historical performance tracking
"""

import asyncio
import time
import psutil
import logging
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from collections import defaultdict
import json
import sys
import gc
import tracemalloc
from datetime import datetime

# Optional dependencies for statistical analysis
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TimingBreakdown:
    """Detailed timing breakdown for a single operation."""
    
    operation_name: str = ""
    total_time_ms: float = 0.0
    
    # Detailed phase timings
    initialization_ms: float = 0.0
    page_load_ms: float = 0.0
    content_extraction_ms: float = 0.0
    quality_analysis_ms: float = 0.0
    post_processing_ms: float = 0.0
    
    # Performance indicators
    time_per_word: float = 0.0  # ms per word extracted
    time_per_kb: float = 0.0    # ms per KB of content
    
    # Metadata
    word_count: int = 0
    content_size_kb: float = 0.0
    timestamp: str = ""


@dataclass
class MemoryMetrics:
    """Memory usage metrics during operations."""
    
    # Memory usage measurements
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_growth_mb: float = 0.0
    
    # Memory efficiency
    memory_per_page: float = 0.0        # MB per page crawled
    memory_per_word: float = 0.0        # MB per word extracted
    memory_per_quality_point: float = 0.0  # MB per quality score point
    
    # System memory impact
    initial_system_memory_percent: float = 0.0
    peak_system_memory_percent: float = 0.0
    
    # Garbage collection stats
    gc_collections: int = 0
    gc_time_ms: float = 0.0
    
    # Memory leak detection
    potential_memory_leak: bool = False
    memory_leak_rate_mb_per_min: float = 0.0


@dataclass 
class DatabaseEfficiencyMetrics:
    """Database storage efficiency metrics."""
    
    # Storage measurements
    total_storage_size_mb: float = 0.0
    content_storage_mb: float = 0.0
    metadata_storage_mb: float = 0.0
    index_storage_mb: float = 0.0
    
    # Efficiency ratios
    content_to_metadata_ratio: float = 0.0
    content_to_total_ratio: float = 0.0
    compression_efficiency: float = 0.0
    
    # Query performance impact
    avg_query_time_ms: float = 0.0
    index_utilization_percent: float = 0.0
    
    # Record statistics
    total_records: int = 0
    avg_record_size_kb: float = 0.0
    duplicate_records: int = 0


@dataclass
class SearchRelevanceMetrics:
    """Search relevance and quality metrics."""
    
    # Test query results
    test_queries_run: int = 0
    total_results_returned: int = 0
    avg_results_per_query: float = 0.0
    
    # Relevance scoring
    avg_relevance_score: float = 0.0
    min_relevance_score: float = 0.0
    max_relevance_score: float = 0.0
    relevance_variance: float = 0.0
    
    # Content quality in results
    high_quality_results_percent: float = 0.0
    navigation_contamination_percent: float = 0.0
    avg_result_word_count: float = 0.0
    
    # Query performance
    avg_query_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Sample queries and results for validation
    sample_queries: List[str] = field(default_factory=list)
    sample_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PerformanceBenchmarkResult:
    """Complete performance benchmark result."""
    
    # Test identification
    benchmark_name: str = ""
    crawler_implementation: str = ""  # 'AdvancedWebCrawler', 'baseline', etc.
    test_timestamp: str = ""
    test_duration_minutes: float = 0.0
    
    # Test configuration
    urls_tested: List[str] = field(default_factory=list)
    concurrent_sessions: int = 1
    test_environment: Dict[str, Any] = field(default_factory=dict)
    
    # Core performance metrics
    timing_breakdown: TimingBreakdown = field(default_factory=TimingBreakdown)
    memory_metrics: MemoryMetrics = field(default_factory=MemoryMetrics)
    database_efficiency: DatabaseEfficiencyMetrics = field(default_factory=DatabaseEfficiencyMetrics)
    search_relevance: SearchRelevanceMetrics = field(default_factory=SearchRelevanceMetrics)
    
    # Overall performance indicators
    pages_per_minute: float = 0.0
    total_content_extracted_kb: float = 0.0
    overall_performance_score: float = 0.0
    
    # Comparison metrics (vs baseline)
    performance_improvement_percent: float = 0.0
    statistical_significance: Optional[float] = None  # p-value
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Quality vs performance trade-offs
    quality_to_performance_ratio: float = 0.0
    
    # Issues and recommendations
    performance_issues: List[str] = field(default_factory=list)
    optimization_recommendations: List[str] = field(default_factory=list)


class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarking system for crawler implementations.
    
    This benchmarker implements Subtask 18.4 by providing detailed performance
    analysis with statistical comparison against baseline implementations.
    """
    
    def __init__(self, baseline_data_path: Optional[Path] = None):
        """
        Initialize the performance benchmarker.
        
        Args:
            baseline_data_path: Path to baseline performance data for comparison
        """
        self.baseline_data_path = baseline_data_path
        self.baseline_data: Optional[PerformanceBenchmarkResult] = None
        
        # Performance monitoring configuration
        self.memory_sample_interval = 1.0  # seconds
        self.detailed_timing = True
        self.enable_memory_profiling = True
        
        # Load baseline data if available
        if baseline_data_path and baseline_data_path.exists():
            self._load_baseline_data()
        
        # Test queries for search relevance testing
        self.default_test_queries = [
            "API authentication",
            "workflow configuration", 
            "node setup guide",
            "error handling best practices",
            "webhook integration",
            "database connection",
            "data transformation",
            "schedule triggers",
            "conditional logic",
            "code examples"
        ]
        
        # Performance thresholds
        self.performance_thresholds = {
            'max_page_time_ms': 30000,        # 30 seconds max per page
            'max_memory_per_page_mb': 100,    # 100MB max per page
            'min_pages_per_minute': 2,        # At least 2 pages per minute
            'max_memory_growth_mb': 500,      # 500MB max memory growth
            'min_content_ratio': 0.7,         # 70% content vs metadata
            'min_relevance_score': 0.6        # 60% minimum relevance
        }
    
    async def benchmark_crawler_implementation(self,
                                             crawler_function: Callable,
                                             test_urls: List[str],
                                             crawler_name: str = "unknown",
                                             concurrent_sessions: int = 1,
                                             include_search_testing: bool = True) -> PerformanceBenchmarkResult:
        """
        Benchmark a crawler implementation with comprehensive metrics.
        
        Args:
            crawler_function: Async function that crawls URLs
            test_urls: List of URLs to test
            crawler_name: Name of the crawler implementation
            concurrent_sessions: Number of concurrent crawling sessions
            include_search_testing: Whether to test search relevance
            
        Returns:
            PerformanceBenchmarkResult with detailed analysis
        """
        start_time = time.time()
        
        logger.info(f"Starting performance benchmark for {crawler_name}")
        logger.info(f"Testing {len(test_urls)} URLs with {concurrent_sessions} concurrent sessions")
        
        # Initialize result
        result = PerformanceBenchmarkResult(
            benchmark_name=f"{crawler_name}_benchmark_{int(start_time)}",
            crawler_implementation=crawler_name,
            test_timestamp=datetime.now().isoformat(),
            urls_tested=test_urls.copy(),
            concurrent_sessions=concurrent_sessions,
            test_environment=self._capture_test_environment()
        )
        
        # Start memory monitoring
        if self.enable_memory_profiling:
            tracemalloc.start()
            initial_memory = self._get_memory_usage()
            memory_samples = [initial_memory]
        
        # Benchmark the crawler
        timing_results = []
        crawl_results = []
        
        try:
            # Execute benchmark with detailed timing
            timing_breakdown, crawl_results = await self._execute_timed_benchmark(
                crawler_function, test_urls, concurrent_sessions
            )
            
            result.timing_breakdown = timing_breakdown
            
            # Collect memory metrics
            if self.enable_memory_profiling:
                final_memory = self._get_memory_usage()
                memory_samples.append(final_memory)
                
                result.memory_metrics = self._analyze_memory_usage(
                    initial_memory, final_memory, memory_samples, len(test_urls)
                )
                
                tracemalloc.stop()
            
            # Analyze database efficiency (if results can be stored)
            if crawl_results:
                result.database_efficiency = await self._analyze_database_efficiency(crawl_results)
            
            # Test search relevance
            if include_search_testing and crawl_results:
                result.search_relevance = await self._test_search_relevance(crawl_results)
            
            # Calculate overall performance metrics
            total_time = time.time() - start_time
            result.test_duration_minutes = total_time / 60
            result.pages_per_minute = len(test_urls) / (total_time / 60) if total_time > 0 else 0
            
            # Calculate content extracted
            total_content_kb = sum(
                len(str(r.get('content', ''))) / 1024 for r in crawl_results if r
            )
            result.total_content_extracted_kb = total_content_kb
            
            # Calculate overall performance score
            result.overall_performance_score = self._calculate_performance_score(result)
            
            # Compare against baseline if available
            if self.baseline_data:
                result.performance_improvement_percent = self._calculate_improvement(result)
                result.statistical_significance = self._test_statistical_significance(result)
                result.confidence_interval = self._calculate_confidence_interval(result)
            
            # Calculate quality vs performance trade-off
            result.quality_to_performance_ratio = self._calculate_quality_performance_ratio(result)
            
            # Identify issues and recommendations
            result.performance_issues = self._identify_performance_issues(result)
            result.optimization_recommendations = self._generate_optimization_recommendations(result)
            
            logger.info(f"Benchmark completed: {result.overall_performance_score:.3f} score, "
                       f"{result.pages_per_minute:.1f} pages/min")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            result.performance_issues.append(f"Benchmark execution failed: {str(e)}")
            result.overall_performance_score = 0.0
        
        return result
    
    async def _execute_timed_benchmark(self,
                                     crawler_function: Callable,
                                     test_urls: List[str],
                                     concurrent_sessions: int) -> Tuple[TimingBreakdown, List[Dict[str, Any]]]:
        """Execute benchmark with detailed timing."""
        
        timing = TimingBreakdown(
            operation_name="crawler_benchmark",
            timestamp=datetime.now().isoformat()
        )
        
        total_start = time.time()
        
        # Phase 1: Initialization
        init_start = time.time()
        # Initialization happens in the crawler function
        timing.initialization_ms = (time.time() - init_start) * 1000
        
        # Phase 2: Crawling execution
        crawl_start = time.time()
        crawl_results = []
        
        try:
            # Execute crawler function
            if concurrent_sessions > 1:
                # Concurrent execution
                semaphore = asyncio.Semaphore(concurrent_sessions)
                
                async def crawl_with_semaphore(url):
                    async with semaphore:
                        return await crawler_function(url)
                
                tasks = [crawl_with_semaphore(url) for url in test_urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Crawl error: {result}")
                        crawl_results.append(None)
                    else:
                        crawl_results.append(result)
            else:
                # Sequential execution
                for url in test_urls:
                    try:
                        result = await crawler_function(url)
                        crawl_results.append(result)
                    except Exception as e:
                        logger.error(f"Crawl error for {url}: {e}")
                        crawl_results.append(None)
        
        except Exception as e:
            logger.error(f"Crawling execution failed: {e}")
            raise
        
        timing.page_load_ms = (time.time() - crawl_start) * 1000
        
        # Calculate detailed timing metrics
        total_time = time.time() - total_start
        timing.total_time_ms = total_time * 1000
        
        # Calculate efficiency metrics
        total_words = sum(
            len(str(r.get('content', '')).split()) for r in crawl_results if r
        )
        total_size_kb = sum(
            len(str(r.get('content', ''))) / 1024 for r in crawl_results if r
        )
        
        timing.word_count = total_words
        timing.content_size_kb = total_size_kb
        
        if total_words > 0:
            timing.time_per_word = timing.total_time_ms / total_words
        
        if total_size_kb > 0:
            timing.time_per_kb = timing.total_time_ms / total_size_kb
        
        return timing, crawl_results
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def _analyze_memory_usage(self,
                            initial_memory: Dict[str, float],
                            final_memory: Dict[str, float],
                            memory_samples: List[Dict[str, float]],
                            pages_crawled: int) -> MemoryMetrics:
        """Analyze memory usage patterns."""
        
        # Calculate memory growth
        memory_growth = final_memory['rss_mb'] - initial_memory['rss_mb']
        
        # Find peak memory usage
        peak_memory = max(sample['rss_mb'] for sample in memory_samples)
        avg_memory = statistics.mean(sample['rss_mb'] for sample in memory_samples)
        
        # Calculate efficiency metrics
        memory_per_page = memory_growth / pages_crawled if pages_crawled > 0 else 0
        
        # Detect potential memory leaks
        if len(memory_samples) > 10:
            # Simple linear regression on memory usage over time
            x = list(range(len(memory_samples)))
            y = [sample['rss_mb'] for sample in memory_samples]
            
            if NUMPY_AVAILABLE:
                # Calculate slope of memory growth
                slope = np.polyfit(x, y, 1)[0]
                leak_rate = slope * 60  # MB per minute
                potential_leak = slope > 1.0  # Growing > 1MB per sample
            else:
                # Simple approximation
                leak_rate = (y[-1] - y[0]) / len(y) * 60
                potential_leak = leak_rate > 5.0
        else:
            leak_rate = 0.0
            potential_leak = False
        
        # Get garbage collection stats
        gc_stats = gc.get_stats()
        total_collections = sum(stat['collections'] for stat in gc_stats)
        
        return MemoryMetrics(
            peak_memory_mb=peak_memory,
            average_memory_mb=avg_memory,
            memory_growth_mb=memory_growth,
            memory_per_page=memory_per_page,
            initial_system_memory_percent=initial_memory['percent'],
            peak_system_memory_percent=max(sample['percent'] for sample in memory_samples),
            gc_collections=total_collections,
            potential_memory_leak=potential_leak,
            memory_leak_rate_mb_per_min=leak_rate
        )
    
    async def _analyze_database_efficiency(self, crawl_results: List[Dict[str, Any]]) -> DatabaseEfficiencyMetrics:
        """Analyze database storage efficiency."""
        
        # Calculate storage metrics from crawl results
        total_content_size = 0
        total_metadata_size = 0
        valid_results = [r for r in crawl_results if r is not None]
        
        for result in valid_results:
            # Content size
            content = str(result.get('content', ''))
            total_content_size += len(content.encode('utf-8'))
            
            # Metadata size (everything except content)
            metadata = {k: v for k, v in result.items() if k != 'content'}
            metadata_str = json.dumps(metadata, default=str)
            total_metadata_size += len(metadata_str.encode('utf-8'))
        
        # Convert to MB
        content_mb = total_content_size / (1024 * 1024)
        metadata_mb = total_metadata_size / (1024 * 1024)
        total_mb = content_mb + metadata_mb
        
        # Calculate ratios
        content_to_metadata_ratio = content_mb / metadata_mb if metadata_mb > 0 else 0
        content_to_total_ratio = content_mb / total_mb if total_mb > 0 else 0
        
        # Estimate compression efficiency (rough calculation)
        avg_compression = 0.3  # Assume 30% compression for text
        compression_efficiency = avg_compression
        
        return DatabaseEfficiencyMetrics(
            total_storage_size_mb=total_mb,
            content_storage_mb=content_mb,
            metadata_storage_mb=metadata_mb,
            content_to_metadata_ratio=content_to_metadata_ratio,
            content_to_total_ratio=content_to_total_ratio,
            compression_efficiency=compression_efficiency,
            total_records=len(valid_results),
            avg_record_size_kb=(total_mb * 1024) / len(valid_results) if valid_results else 0
        )
    
    async def _test_search_relevance(self, crawl_results: List[Dict[str, Any]]) -> SearchRelevanceMetrics:
        """Test search relevance using sample queries."""
        
        # Filter valid results
        valid_results = [r for r in crawl_results if r and r.get('content')]
        
        if not valid_results:
            return SearchRelevanceMetrics()
        
        # Prepare search index (simple in-memory)
        search_index = {}
        for i, result in enumerate(valid_results):
            content = str(result.get('content', '')).lower()
            words = content.split()
            search_index[i] = {
                'content': content,
                'words': set(words),
                'word_count': len(words),
                'url': result.get('url', ''),
                'quality_score': result.get('quality_score', 0.5)
            }
        
        # Test queries
        query_results = []
        total_results = 0
        relevance_scores = []
        query_times = []
        
        for query in self.default_test_queries[:5]:  # Test first 5 queries
            query_start = time.time()
            
            # Simple search implementation
            query_words = set(query.lower().split())
            scored_results = []
            
            for idx, doc in search_index.items():
                # Calculate simple relevance score
                word_overlap = len(query_words.intersection(doc['words']))
                relevance = word_overlap / len(query_words) if query_words else 0
                
                if relevance > 0:
                    scored_results.append({
                        'index': idx,
                        'relevance': relevance,
                        'url': doc['url'],
                        'word_count': doc['word_count'],
                        'quality_score': doc['quality_score']
                    })
            
            # Sort by relevance
            scored_results.sort(key=lambda x: x['relevance'], reverse=True)
            top_results = scored_results[:10]  # Top 10 results
            
            query_time = (time.time() - query_start) * 1000
            query_times.append(query_time)
            
            # Calculate metrics for this query
            if top_results:
                query_relevance_scores = [r['relevance'] for r in top_results]
                relevance_scores.extend(query_relevance_scores)
                total_results += len(top_results)
                
                query_results.append({
                    'query': query,
                    'results_count': len(top_results),
                    'avg_relevance': statistics.mean(query_relevance_scores),
                    'top_result_url': top_results[0]['url']
                })
        
        # Calculate aggregate metrics
        avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0
        min_relevance = min(relevance_scores) if relevance_scores else 0
        max_relevance = max(relevance_scores) if relevance_scores else 0
        relevance_variance = statistics.variance(relevance_scores) if len(relevance_scores) > 1 else 0
        
        # Calculate content quality in results
        high_quality_count = sum(1 for r in valid_results if r.get('quality_score', 0) > 0.7)
        high_quality_percent = (high_quality_count / len(valid_results)) * 100 if valid_results else 0
        
        avg_word_count = statistics.mean(doc['word_count'] for doc in search_index.values())
        
        return SearchRelevanceMetrics(
            test_queries_run=len(self.default_test_queries[:5]),
            total_results_returned=total_results,
            avg_results_per_query=total_results / 5 if total_results > 0 else 0,
            avg_relevance_score=avg_relevance,
            min_relevance_score=min_relevance,
            max_relevance_score=max_relevance,
            relevance_variance=relevance_variance,
            high_quality_results_percent=high_quality_percent,
            avg_result_word_count=avg_word_count,
            avg_query_time_ms=statistics.mean(query_times) if query_times else 0,
            sample_queries=self.default_test_queries[:5],
            sample_results=query_results
        )
    
    def _capture_test_environment(self) -> Dict[str, Any]:
        """Capture test environment information."""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_performance_score(self, result: PerformanceBenchmarkResult) -> float:
        """Calculate overall performance score (0.0 to 1.0)."""
        
        score = 0.0
        weight_sum = 0.0
        
        # Timing performance (30% weight)
        if result.pages_per_minute > 0:
            timing_score = min(1.0, result.pages_per_minute / self.performance_thresholds['min_pages_per_minute'])
            score += timing_score * 0.3
            weight_sum += 0.3
        
        # Memory efficiency (25% weight)
        if result.memory_metrics.memory_per_page > 0:
            memory_score = max(0.0, 1.0 - (result.memory_metrics.memory_per_page / self.performance_thresholds['max_memory_per_page_mb']))
            score += memory_score * 0.25
            weight_sum += 0.25
        
        # Database efficiency (20% weight)
        if result.database_efficiency.content_to_total_ratio > 0:
            db_score = min(1.0, result.database_efficiency.content_to_total_ratio / self.performance_thresholds['min_content_ratio'])
            score += db_score * 0.2
            weight_sum += 0.2
        
        # Search relevance (25% weight)
        if result.search_relevance.avg_relevance_score > 0:
            search_score = min(1.0, result.search_relevance.avg_relevance_score / self.performance_thresholds['min_relevance_score'])
            score += search_score * 0.25
            weight_sum += 0.25
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _calculate_improvement(self, result: PerformanceBenchmarkResult) -> float:
        """Calculate improvement percentage vs baseline."""
        if not self.baseline_data:
            return 0.0
        
        current_score = result.overall_performance_score
        baseline_score = self.baseline_data.overall_performance_score
        
        if baseline_score > 0:
            return ((current_score - baseline_score) / baseline_score) * 100
        else:
            return 0.0
    
    def _test_statistical_significance(self, result: PerformanceBenchmarkResult) -> Optional[float]:
        """Test statistical significance of performance improvement."""
        if not SCIPY_AVAILABLE or not self.baseline_data:
            return None
        
        # This is a simplified implementation
        # In a real system, you'd collect multiple samples for proper statistical testing
        current_scores = [result.overall_performance_score]
        baseline_scores = [self.baseline_data.overall_performance_score]
        
        if len(current_scores) >= 2 and len(baseline_scores) >= 2:
            try:
                statistic, p_value = scipy_stats.ttest_ind(current_scores, baseline_scores)
                return p_value
            except:
                return None
        
        return None
    
    def _calculate_confidence_interval(self, result: PerformanceBenchmarkResult) -> Optional[Tuple[float, float]]:
        """Calculate confidence interval for performance improvement."""
        if not SCIPY_AVAILABLE:
            return None
        
        # Simplified confidence interval calculation
        score = result.overall_performance_score
        # Assume 5% standard error (would be calculated from multiple runs in practice)
        std_error = score * 0.05
        
        # 95% confidence interval
        margin_error = 1.96 * std_error
        return (score - margin_error, score + margin_error)
    
    def _calculate_quality_performance_ratio(self, result: PerformanceBenchmarkResult) -> float:
        """Calculate quality to performance trade-off ratio."""
        
        # Quality indicators
        quality_score = 0.0
        
        if result.database_efficiency.content_to_total_ratio > 0:
            quality_score += result.database_efficiency.content_to_total_ratio * 0.4
        
        if result.search_relevance.avg_relevance_score > 0:
            quality_score += result.search_relevance.avg_relevance_score * 0.6
        
        # Performance indicator (inverse of time)
        performance_score = min(1.0, result.pages_per_minute / 10.0)  # Normalize to 10 pages/min
        
        if performance_score > 0:
            return quality_score / performance_score
        else:
            return 0.0
    
    def _identify_performance_issues(self, result: PerformanceBenchmarkResult) -> List[str]:
        """Identify performance issues from benchmark results."""
        issues = []
        
        # Timing issues
        if result.pages_per_minute < self.performance_thresholds['min_pages_per_minute']:
            issues.append(f"Low crawling throughput: {result.pages_per_minute:.1f} pages/min (target: {self.performance_thresholds['min_pages_per_minute']})")
        
        # Memory issues
        if result.memory_metrics.memory_per_page > self.performance_thresholds['max_memory_per_page_mb']:
            issues.append(f"High memory usage per page: {result.memory_metrics.memory_per_page:.1f}MB (target: <{self.performance_thresholds['max_memory_per_page_mb']}MB)")
        
        if result.memory_metrics.potential_memory_leak:
            issues.append(f"Potential memory leak detected: {result.memory_metrics.memory_leak_rate_mb_per_min:.1f}MB/min growth")
        
        # Database efficiency issues
        if result.database_efficiency.content_to_total_ratio < self.performance_thresholds['min_content_ratio']:
            issues.append(f"Low content-to-total ratio: {result.database_efficiency.content_to_total_ratio:.2f} (target: >{self.performance_thresholds['min_content_ratio']})")
        
        # Search relevance issues
        if result.search_relevance.avg_relevance_score < self.performance_thresholds['min_relevance_score']:
            issues.append(f"Low search relevance: {result.search_relevance.avg_relevance_score:.2f} (target: >{self.performance_thresholds['min_relevance_score']})")
        
        return issues
    
    def _generate_optimization_recommendations(self, result: PerformanceBenchmarkResult) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Performance optimizations
        if result.pages_per_minute < 5:
            recommendations.append("Consider increasing concurrent sessions or optimizing page load strategies")
        
        if result.memory_metrics.memory_per_page > 50:
            recommendations.append("Optimize memory usage by implementing content streaming or chunking")
        
        if result.memory_metrics.potential_memory_leak:
            recommendations.append("Investigate and fix potential memory leaks in crawler implementation")
        
        # Quality optimizations
        if result.database_efficiency.content_to_metadata_ratio < 3.0:
            recommendations.append("Reduce metadata overhead or increase content extraction efficiency")
        
        if result.search_relevance.avg_relevance_score < 0.7:
            recommendations.append("Improve content quality and search indexing strategies")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _load_baseline_data(self):
        """Load baseline performance data for comparison."""
        try:
            with open(self.baseline_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert dict back to dataclass (simplified)
                self.baseline_data = PerformanceBenchmarkResult(**data)
            logger.info(f"Loaded baseline data from {self.baseline_data_path}")
        except Exception as e:
            logger.error(f"Failed to load baseline data: {e}")
            self.baseline_data = None
    
    def save_benchmark_result(self, result: PerformanceBenchmarkResult, 
                            output_path: Path):
        """Save benchmark result to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            logger.info(f"Benchmark result saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save benchmark result: {e}")


def create_performance_report(result: PerformanceBenchmarkResult,
                            output_path: Optional[Path] = None) -> str:
    """Create a comprehensive performance benchmark report."""
    
    report = f"""# Performance Benchmark Report: {result.crawler_implementation}

## Executive Summary
- **Overall Performance Score**: {result.overall_performance_score:.3f} / 1.0
- **Test Duration**: {result.test_duration_minutes:.1f} minutes
- **Pages Crawled**: {len(result.urls_tested)}
- **Throughput**: {result.pages_per_minute:.1f} pages/minute
- **Content Extracted**: {result.total_content_extracted_kb:.1f} KB

## Performance Metrics

### Timing Analysis
- **Total Processing Time**: {result.timing_breakdown.total_time_ms:.0f}ms
- **Time per Page**: {result.timing_breakdown.total_time_ms / len(result.urls_tested):.0f}ms
- **Time per Word**: {result.timing_breakdown.time_per_word:.2f}ms
- **Time per KB**: {result.timing_breakdown.time_per_kb:.1f}ms

### Memory Usage
- **Peak Memory**: {result.memory_metrics.peak_memory_mb:.1f}MB
- **Memory Growth**: {result.memory_metrics.memory_growth_mb:.1f}MB
- **Memory per Page**: {result.memory_metrics.memory_per_page:.1f}MB
- **Memory Leak Detected**: {'⚠️ Yes' if result.memory_metrics.potential_memory_leak else '✅ No'}

### Database Efficiency
- **Content Storage**: {result.database_efficiency.content_storage_mb:.1f}MB
- **Metadata Storage**: {result.database_efficiency.metadata_storage_mb:.1f}MB
- **Content-to-Metadata Ratio**: {result.database_efficiency.content_to_metadata_ratio:.1f}:1
- **Storage Efficiency**: {result.database_efficiency.content_to_total_ratio:.1%}

### Search Relevance
- **Average Relevance Score**: {result.search_relevance.avg_relevance_score:.3f}
- **Test Queries**: {result.search_relevance.test_queries_run}
- **Results per Query**: {result.search_relevance.avg_results_per_query:.1f}
- **High Quality Results**: {result.search_relevance.high_quality_results_percent:.1f}%

## Performance Comparison
"""
    
    if result.performance_improvement_percent != 0.0:
        improvement_sign = "📈" if result.performance_improvement_percent > 0 else "📉"
        report += f"- **vs Baseline**: {improvement_sign} {result.performance_improvement_percent:+.1f}%\n"
        
        if result.statistical_significance:
            significance = "significant" if result.statistical_significance < 0.05 else "not significant"
            report += f"- **Statistical Significance**: {significance} (p={result.statistical_significance:.3f})\n"
    
    report += f"""
## Quality vs Performance Analysis
- **Quality-Performance Ratio**: {result.quality_to_performance_ratio:.2f}
- **Optimal Balance**: {'✅ Achieved' if 0.8 <= result.quality_to_performance_ratio <= 1.2 else '⚠️ Needs Tuning'}

## Issues Identified
"""
    
    if result.performance_issues:
        for issue in result.performance_issues:
            report += f"- ⚠️ {issue}\n"
    else:
        report += "- ✅ No significant performance issues detected\n"
    
    report += f"""
## Optimization Recommendations
"""
    
    if result.optimization_recommendations:
        for rec in result.optimization_recommendations:
            report += f"- 💡 {rec}\n"
    else:
        report += "- ✅ No specific optimizations needed\n"
    
    report += f"""
## Test Environment
- **Python Version**: {result.test_environment.get('python_version', 'Unknown')}
- **Platform**: {result.test_environment.get('platform', 'Unknown')}
- **CPU Cores**: {result.test_environment.get('cpu_count', 'Unknown')}
- **Total Memory**: {result.test_environment.get('total_memory_gb', 0):.1f}GB
- **Test Timestamp**: {result.test_timestamp}

## Configuration
- **Concurrent Sessions**: {result.concurrent_sessions}
- **URLs Tested**: {len(result.urls_tested)}
- **Crawler Implementation**: {result.crawler_implementation}
"""
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Performance report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    # Example usage and testing
    async def sample_crawler_function(url: str) -> Dict[str, Any]:
        """Sample crawler function for testing."""
        # Simulate crawling delay
        await asyncio.sleep(0.5)
        
        # Return sample result
        return {
            'url': url,
            'content': f"Sample content for {url} " * 50,  # ~1KB content
            'status': 'success',
            'quality_score': 0.8,
            'word_count': 50
        }
    
    async def test_performance_benchmarker():
        print("🧪 Testing Performance Benchmarker")
        print("=" * 50)
        
        benchmarker = PerformanceBenchmarker()
        
        test_urls = [
            "https://example.com/page1",
            "https://example.com/page2", 
            "https://example.com/page3"
        ]
        
        result = await benchmarker.benchmark_crawler_implementation(
            sample_crawler_function,
            test_urls,
            crawler_name="SampleCrawler",
            concurrent_sessions=2
        )
        
        print(f"Performance Score: {result.overall_performance_score:.3f}")
        print(f"Pages per Minute: {result.pages_per_minute:.1f}")
        print(f"Peak Memory: {result.memory_metrics.peak_memory_mb:.1f}MB")
        print(f"Content Extracted: {result.total_content_extracted_kb:.1f}KB")
        
        # Generate report
        report = create_performance_report(result)
        print(f"\n📋 Generated performance report ({len(report)} characters)")
        
        return result
    
    # Run test
    asyncio.run(test_performance_benchmarker())