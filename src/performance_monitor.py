#!/usr/bin/env python3
"""
Performance monitoring framework for ongoing performance validation.

This module provides utilities to compare current performance against baseline
and detect performance regressions during development.
"""

import json
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import psutil

from utils import search_documents, create_embedding

# Try to import dotenv, fall back to os.environ if not available
try:
    from dotenv import load_dotenv
    import os
    # Load environment variables
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path, override=True)
    
    # Override SUPABASE_URL for local testing (temporary for performance monitoring)
    # This preserves the original .env for n8n while allowing local testing
    if os.getenv("SUPABASE_URL") == "http://host.docker.internal:54321":
        print("🔧 Temporarily overriding SUPABASE_URL for local testing...")
        os.environ["SUPABASE_URL"] = "http://localhost:54321"
except ImportError:
    pass


class PerformanceMonitor:
    """Performance monitoring for ongoing validation against baseline."""
    
    def __init__(self, baseline_file: Optional[Path] = None):
        """
        Initialize performance monitor.
        
        Args:
            baseline_file: Path to baseline performance file
        """
        self.project_root = Path(__file__).resolve().parent.parent
        self.baseline_file = baseline_file or (self.project_root / "performance_baseline.json")
        self.baseline_data = self._load_baseline()
    
    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline performance data."""
        try:
            if self.baseline_file.exists():
                with open(self.baseline_file) as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load baseline data: {e}")
        return None
    
    def measure_search_performance(self, supabase_client, query: str, match_count: int = 10) -> Dict[str, Any]:
        """
        Measure search performance for a single query.
        
        Args:
            supabase_client: Supabase client instance
            query: Search query
            match_count: Number of results to return
            
        Returns:
            Performance metrics dictionary
        """
        memory_before = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        # Measure embedding time
        embedding_start = time.time()
        embedding = create_embedding(query)
        embedding_time = time.time() - embedding_start
        
        # Measure search time
        search_start = time.time()
        results = search_documents(supabase_client, query, match_count=match_count)
        search_time = time.time() - search_start
        
        memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
        total_time = embedding_time + search_time
        
        return {
            "query": query,
            "total_time_ms": round(total_time * 1000, 2),
            "embedding_time_ms": round(embedding_time * 1000, 2),
            "search_time_ms": round(search_time * 1000, 2),
            "memory_delta_mb": round(memory_after - memory_before, 2),
            "result_count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    
    def compare_to_baseline(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current metrics to baseline.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Comparison results with warnings for regressions
        """
        if not self.baseline_data or "search_performance" not in self.baseline_data:
            return {"status": "no_baseline", "message": "No baseline data available for comparison"}
        
        baseline_perf = self.baseline_data["search_performance"]
        
        # Define regression thresholds (% increase over baseline)
        REGRESSION_THRESHOLDS = {
            "response_time": 25,  # 25% slower is a regression
            "memory": 50,         # 50% more memory is a regression
            "result_count": -20   # 20% fewer results is a regression
        }
        
        comparison = {
            "baseline_avg_response_time_ms": baseline_perf.get("avg_response_time_ms", 0),
            "current_response_time_ms": current_metrics.get("total_time_ms", 0),
            "baseline_avg_memory_mb": baseline_perf.get("avg_memory_delta_mb", 0),
            "current_memory_mb": current_metrics.get("memory_delta_mb", 0),
            "baseline_avg_results": baseline_perf.get("avg_result_count", 0),
            "current_results": current_metrics.get("result_count", 0),
            "regressions": [],
            "improvements": []
        }
        
        # Check response time
        if baseline_perf.get("avg_response_time_ms", 0) > 0:
            response_change = (
                (current_metrics.get("total_time_ms", 0) - baseline_perf["avg_response_time_ms"]) /
                baseline_perf["avg_response_time_ms"] * 100
            )
            comparison["response_time_change_percent"] = round(response_change, 1)
            
            if response_change > REGRESSION_THRESHOLDS["response_time"]:
                comparison["regressions"].append(
                    f"Response time regression: {response_change:.1f}% slower than baseline"
                )
            elif response_change < -10:  # 10% improvement threshold
                comparison["improvements"].append(
                    f"Response time improvement: {abs(response_change):.1f}% faster than baseline"
                )
        
        # Check memory usage
        baseline_memory = baseline_perf.get("avg_memory_delta_mb", 0)
        current_memory = current_metrics.get("memory_delta_mb", 0)
        if baseline_memory > 0:
            memory_change = (current_memory - baseline_memory) / baseline_memory * 100
            comparison["memory_change_percent"] = round(memory_change, 1)
            
            if memory_change > REGRESSION_THRESHOLDS["memory"]:
                comparison["regressions"].append(
                    f"Memory usage regression: {memory_change:.1f}% more memory than baseline"
                )
        
        # Check result count
        baseline_results = baseline_perf.get("avg_result_count", 0)
        current_results = current_metrics.get("result_count", 0)
        if baseline_results > 0:
            result_change = (current_results - baseline_results) / baseline_results * 100
            comparison["result_count_change_percent"] = round(result_change, 1)
            
            if result_change < REGRESSION_THRESHOLDS["result_count"]:
                comparison["regressions"].append(
                    f"Result count regression: {abs(result_change):.1f}% fewer results than baseline"
                )
        
        # Overall status
        if comparison["regressions"]:
            comparison["status"] = "regression_detected"
        elif comparison["improvements"]:
            comparison["status"] = "improvement_detected"
        else:
            comparison["status"] = "stable"
        
        return comparison
    
    def validate_performance(self, supabase_client, test_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run performance validation against baseline.
        
        Args:
            supabase_client: Supabase client instance
            test_queries: Optional list of test queries. Uses defaults if None.
            
        Returns:
            Validation results with regression warnings
        """
        if test_queries is None:
            test_queries = [
                "installation",
                "how to configure settings", 
                "API endpoint documentation example"
            ]
        
        print("🔍 Running performance validation...")
        
        results = {
            "validation_timestamp": datetime.now().isoformat(),
            "test_results": [],
            "summary": {},
            "regressions_detected": False
        }
        
        all_regressions = []
        all_response_times = []
        
        for query in test_queries:
            print(f"  Testing: {query}")
            try:
                metrics = self.measure_search_performance(supabase_client, query)
                comparison = self.compare_to_baseline(metrics)
                
                result = {
                    "query": query,
                    "metrics": metrics,
                    "comparison": comparison
                }
                results["test_results"].append(result)
                
                all_response_times.append(metrics["total_time_ms"])
                
                if comparison.get("regressions"):
                    all_regressions.extend(comparison["regressions"])
                    results["regressions_detected"] = True
                
            except Exception as e:
                print(f"    Error testing query '{query}': {e}")
                results["test_results"].append({
                    "query": query,
                    "error": str(e)
                })
        
        # Summary statistics
        if all_response_times:
            results["summary"] = {
                "avg_response_time_ms": round(statistics.mean(all_response_times), 2),
                "total_regressions": len(all_regressions),
                "unique_regressions": list(set(all_regressions))
            }
        
        if results["regressions_detected"]:
            print(f"⚠️  Performance regressions detected: {len(all_regressions)}")
            for regression in set(all_regressions):
                print(f"    - {regression}")
        else:
            print("✅ No performance regressions detected")
        
        return results


def validate_against_baseline(supabase_client, verbose: bool = True) -> bool:
    """
    Quick validation function against baseline.
    
    Args:
        supabase_client: Supabase client instance
        verbose: Whether to print detailed results
        
    Returns:
        True if no regressions detected, False otherwise
    """
    monitor = PerformanceMonitor()
    results = monitor.validate_performance(supabase_client)
    
    if verbose and results["summary"]:
        print(f"\n📊 Performance Summary:")
        print(f"  • Average response time: {results['summary']['avg_response_time_ms']}ms")
        print(f"  • Total regressions: {results['summary']['total_regressions']}")
    
    return not results["regressions_detected"]