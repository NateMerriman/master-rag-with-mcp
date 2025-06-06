#!/usr/bin/env python3
"""
Automated performance regression tests for Crawl4AI RAG MCP Server.

This module provides automated tests to detect performance regressions
during development and CI/CD.
"""

import os
import sys
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Try to import dotenv, fall back to os.environ if not available
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass
from supabase import create_client
from performance_monitor import PerformanceMonitor, validate_against_baseline

# Load environment variables
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path, override=True)

# Override SUPABASE_URL for local testing (temporary for performance tests)
# This preserves the original .env for n8n while allowing local testing
if os.getenv("SUPABASE_URL") == "http://host.docker.internal:54321":
    print("🔧 Temporarily overriding SUPABASE_URL for local testing...")
    os.environ["SUPABASE_URL"] = "http://localhost:54321"


class PerformanceRegressionTests(unittest.TestCase):
    """Test suite for performance regression detection."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Check if we have required environment variables
        cls.supabase_url = os.getenv("SUPABASE_URL")
        cls.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not cls.supabase_url or not cls.supabase_key:
            raise unittest.SkipTest("SUPABASE_URL and SUPABASE_SERVICE_KEY required for performance tests")
        
        # Initialize Supabase client
        cls.supabase_client = create_client(cls.supabase_url, cls.supabase_key)
        
        # Initialize performance monitor
        cls.monitor = PerformanceMonitor()
        
        # Check if baseline exists
        if not cls.monitor.baseline_data:
            raise unittest.SkipTest("Performance baseline not found. Run performance_baseline.py first.")
    
    def test_simple_search_performance(self):
        """Test performance of simple search queries."""
        query = "installation"
        metrics = self.monitor.measure_search_performance(self.supabase_client, query)
        comparison = self.monitor.compare_to_baseline(metrics)
        
        # Assert no major regressions
        self.assertNotEqual(comparison["status"], "regression_detected", 
                          f"Performance regression detected for query '{query}': {comparison.get('regressions', [])}")
        
        # Assert reasonable response time (should be under 5 seconds)
        self.assertLess(metrics["total_time_ms"], 5000, 
                       f"Query '{query}' took too long: {metrics['total_time_ms']}ms")
        
        # Assert we get some results
        self.assertGreater(metrics["result_count"], 0, 
                          f"Query '{query}' returned no results")
    
    def test_complex_search_performance(self):
        """Test performance of complex search queries."""
        query = "API endpoint documentation example"
        metrics = self.monitor.measure_search_performance(self.supabase_client, query)
        comparison = self.monitor.compare_to_baseline(metrics)
        
        # Assert no major regressions
        self.assertNotEqual(comparison["status"], "regression_detected", 
                          f"Performance regression detected for query '{query}': {comparison.get('regressions', [])}")
        
        # Assert reasonable response time (complex queries can take a bit longer)
        self.assertLess(metrics["total_time_ms"], 8000, 
                       f"Complex query '{query}' took too long: {metrics['total_time_ms']}ms")
    
    def test_memory_usage_regression(self):
        """Test that memory usage hasn't regressed significantly."""
        query = "python code example"
        metrics = self.monitor.measure_search_performance(self.supabase_client, query)
        comparison = self.monitor.compare_to_baseline(metrics)
        
        # Check for memory regressions specifically
        memory_regressions = [r for r in comparison.get("regressions", []) if "memory" in r.lower()]
        self.assertEqual(len(memory_regressions), 0, 
                        f"Memory usage regression detected: {memory_regressions}")
        
        # Assert memory usage is reasonable (under 100MB delta)
        self.assertLess(abs(metrics["memory_delta_mb"]), 100, 
                       f"Memory usage too high: {metrics['memory_delta_mb']}MB")
    
    def test_batch_performance_validation(self):
        """Test performance across multiple queries."""
        test_queries = [
            "installation",
            "configuration",
            "error handling",
            "documentation"
        ]
        
        validation_results = self.monitor.validate_performance(self.supabase_client, test_queries)
        
        # Assert no regressions detected
        self.assertFalse(validation_results["regressions_detected"], 
                        f"Performance regressions detected: {validation_results['summary'].get('unique_regressions', [])}")
        
        # Assert all queries completed successfully
        successful_tests = [r for r in validation_results["test_results"] if "error" not in r]
        self.assertEqual(len(successful_tests), len(test_queries), 
                        "Not all test queries completed successfully")
    
    def test_embedding_performance(self):
        """Test that embedding generation performance is acceptable."""
        from utils import create_embedding
        import time
        
        query = "test embedding performance"
        
        start_time = time.time()
        embedding = create_embedding(query)
        embedding_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Assert embedding generation is fast (under 2 seconds)
        self.assertLess(embedding_time, 2000, 
                       f"Embedding generation too slow: {embedding_time}ms")
        
        # Assert embedding is correct size
        self.assertEqual(len(embedding), 1536, 
                        f"Embedding size incorrect: {len(embedding)}")
    
    def test_baseline_integrity(self):
        """Test that baseline data is valid and complete."""
        baseline = self.monitor.baseline_data
        
        # Check required sections exist
        required_sections = ["search_performance", "database_stats", "system_info"]
        for section in required_sections:
            self.assertIn(section, baseline, f"Baseline missing section: {section}")
        
        # Check search performance has required metrics
        search_perf = baseline["search_performance"]
        required_metrics = ["avg_response_time_ms", "avg_result_count", "successful_queries"]
        for metric in required_metrics:
            self.assertIn(metric, search_perf, f"Baseline missing metric: {metric}")
            self.assertGreater(search_perf[metric], 0, f"Baseline metric {metric} is zero or negative")


class QuickRegressionTest(unittest.TestCase):
    """Quick regression test for CI/CD pipelines."""
    
    @classmethod
    def setUpClass(cls):
        """Set up quick test environment."""
        cls.supabase_url = os.getenv("SUPABASE_URL")
        cls.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not cls.supabase_url or not cls.supabase_key:
            raise unittest.SkipTest("SUPABASE_URL and SUPABASE_SERVICE_KEY required")
        
        cls.supabase_client = create_client(cls.supabase_url, cls.supabase_key)
    
    def test_quick_performance_check(self):
        """Quick performance validation using helper function."""
        # This is a fast test that can be run in CI/CD
        result = validate_against_baseline(self.supabase_client, verbose=False)
        self.assertTrue(result, "Performance regression detected in quick validation")


def run_performance_tests(quick_only: bool = False):
    """
    Run performance regression tests.
    
    Args:
        quick_only: If True, run only quick tests suitable for CI/CD
    """
    if quick_only:
        suite = unittest.TestLoader().loadTestsFromTestCase(QuickRegressionTest)
        print("🏃 Running quick performance regression tests...")
    else:
        suite = unittest.TestLoader().loadTestsFromTestCase(PerformanceRegressionTests)
        print("🔍 Running full performance regression test suite...")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ All performance tests passed!")
        return True
    else:
        print(f"❌ Performance tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run performance regression tests")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only quick tests suitable for CI/CD")
    args = parser.parse_args()
    
    success = run_performance_tests(quick_only=args.quick)
    sys.exit(0 if success else 1)