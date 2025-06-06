#!/usr/bin/env python3
"""
Performance baseline establishment script for Crawl4AI RAG MCP Server.

This script captures current search performance metrics to establish a baseline
before implementing enhancements.
"""

import os
import json
import time
import statistics
import traceback
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

import psutil
from supabase import create_client

# Try to import dotenv, fall back to os.environ if not available
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    def load_dotenv(*args, **kwargs):
        pass

from utils import search_documents, create_embedding

# Load environment variables
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path, override=True)

# Override SUPABASE_URL for local testing (temporary for performance baseline)
# This preserves the original .env for n8n while allowing local testing
if os.getenv("SUPABASE_URL") == "http://host.docker.internal:54321":
    print("🔧 Temporarily overriding SUPABASE_URL for local testing...")
    os.environ["SUPABASE_URL"] = "http://localhost:54321"


class PerformanceBaseline:
    """Class to capture and analyze current search performance."""
    
    def __init__(self):
        self.supabase_client = self._get_supabase_client()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "search_performance": {},
            "system_info": {},
            "database_stats": {},
            "test_queries": []
        }
    
    def _get_supabase_client(self):
        """Get Supabase client."""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        
        return create_client(url, key)
    
    def capture_system_info(self):
        """Capture system information."""
        print("📊 Capturing system information...")
        
        self.results["system_info"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.name
        }
    
    def capture_database_stats(self):
        """Capture database statistics."""
        print("📊 Capturing database statistics...")
        
        try:
            # Get total document count
            result = self.supabase_client.table("crawled_pages").select("*", count="exact").execute()
            total_docs = result.count if hasattr(result, 'count') else len(result.data)
            
            # Get unique sources
            sources_result = (
                self.supabase_client.table("crawled_pages")
                .select("metadata")
                .not_.is_("metadata->>source", "null")
                .execute()
            )
            
            unique_sources = set()
            if sources_result.data:
                for item in sources_result.data:
                    source = item.get("metadata", {}).get("source")
                    if source:
                        unique_sources.add(source)
            
            # Calculate average content length
            if result.data:
                content_lengths = [len(item.get("content", "")) for item in result.data[:100]]  # Sample first 100
                avg_content_length = statistics.mean(content_lengths) if content_lengths else 0
            else:
                avg_content_length = 0
            
            self.results["database_stats"] = {
                "total_documents": total_docs,
                "unique_sources": len(unique_sources),
                "sources_list": sorted(list(unique_sources)),
                "avg_content_length_chars": round(avg_content_length, 2)
            }
            
        except Exception as e:
            print(f"⚠️  Error capturing database stats: {e}")
            self.results["database_stats"] = {"error": str(e)}
    
    def test_search_performance(self):
        """Test search performance with various queries."""
        print("🚀 Testing search performance...")
        
        # Define test queries of varying complexity
        test_queries = [
            {"query": "installation", "description": "Simple term search"},
            {"query": "how to configure settings", "description": "Natural language query"},
            {"query": "API endpoint documentation example", "description": "Multi-term technical query"},
            {"query": "error handling best practices", "description": "Complex conceptual query"},
            {"query": "python code example function", "description": "Code-related query"}
        ]
        
        performance_metrics = {
            "response_times": [],
            "memory_usage": [],
            "result_counts": [],
            "embedding_times": [],
            "detailed_results": []
        }
        
        # Warm up - run one query to initialize connections
        try:
            search_documents(self.supabase_client, "test", match_count=1)
        except:
            pass
        
        for i, test_case in enumerate(test_queries):
            print(f"  Testing query {i+1}/5: {test_case['description']}")
            
            try:
                # Measure memory before
                memory_before = psutil.Process().memory_info().rss / (1024**2)  # MB
                
                # Measure embedding creation time
                embedding_start = time.time()
                embedding = create_embedding(test_case["query"])
                embedding_time = time.time() - embedding_start
                
                # Measure search time
                search_start = time.time()
                results = search_documents(
                    self.supabase_client,
                    test_case["query"],
                    match_count=10
                )
                search_time = time.time() - search_start
                
                # Measure memory after
                memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
                
                # Calculate total response time
                total_time = embedding_time + search_time
                
                # Store metrics
                performance_metrics["response_times"].append(total_time)
                performance_metrics["embedding_times"].append(embedding_time)
                performance_metrics["memory_usage"].append(memory_after - memory_before)
                performance_metrics["result_counts"].append(len(results))
                
                # Store detailed results
                performance_metrics["detailed_results"].append({
                    "query": test_case["query"],
                    "description": test_case["description"],
                    "total_time_ms": round(total_time * 1000, 2),
                    "embedding_time_ms": round(embedding_time * 1000, 2),
                    "search_time_ms": round(search_time * 1000, 2),
                    "memory_delta_mb": round(memory_after - memory_before, 2),
                    "result_count": len(results),
                    "has_results": len(results) > 0
                })
                
                self.results["test_queries"].append({
                    "query": test_case["query"],
                    "result_count": len(results),
                    "response_time_ms": round(total_time * 1000, 2)
                })
                
            except Exception as e:
                print(f"    ❌ Error testing query '{test_case['query']}': {e}")
                performance_metrics["detailed_results"].append({
                    "query": test_case["query"],
                    "description": test_case["description"],
                    "error": str(e)
                })
        
        # Calculate summary statistics
        if performance_metrics["response_times"]:
            self.results["search_performance"] = {
                "avg_response_time_ms": round(statistics.mean(performance_metrics["response_times"]) * 1000, 2),
                "median_response_time_ms": round(statistics.median(performance_metrics["response_times"]) * 1000, 2),
                "min_response_time_ms": round(min(performance_metrics["response_times"]) * 1000, 2),
                "max_response_time_ms": round(max(performance_metrics["response_times"]) * 1000, 2),
                "avg_embedding_time_ms": round(statistics.mean(performance_metrics["embedding_times"]) * 1000, 2),
                "avg_memory_delta_mb": round(statistics.mean(performance_metrics["memory_usage"]), 2),
                "avg_result_count": round(statistics.mean(performance_metrics["result_counts"]), 1),
                "successful_queries": len([r for r in performance_metrics["detailed_results"] if "error" not in r]),
                "detailed_metrics": performance_metrics["detailed_results"]
            }
        else:
            self.results["search_performance"] = {"error": "No successful queries"}
    
    def run_baseline_capture(self):
        """Run complete baseline capture."""
        print("🎯 Starting performance baseline capture...")
        print("=" * 50)
        
        try:
            self.capture_system_info()
            self.capture_database_stats()
            self.test_search_performance()
            
            # Save results
            baseline_file = project_root / "performance_baseline.json"
            with open(baseline_file, "w") as f:
                json.dump(self.results, f, indent=2)
            
            print("\n📋 Baseline Summary:")
            print(f"  • Total documents: {self.results['database_stats'].get('total_documents', 'N/A')}")
            print(f"  • Unique sources: {self.results['database_stats'].get('unique_sources', 'N/A')}")
            
            if "avg_response_time_ms" in self.results["search_performance"]:
                print(f"  • Avg response time: {self.results['search_performance']['avg_response_time_ms']}ms")
                print(f"  • Avg result count: {self.results['search_performance']['avg_result_count']}")
                print(f"  • Successful queries: {self.results['search_performance']['successful_queries']}/5")
            
            print(f"\n✅ Baseline saved to: {baseline_file}")
            print("=" * 50)
            
            return True
            
        except Exception as e:
            print(f"❌ Error during baseline capture: {e}")
            traceback.print_exc()
            return False


def main():
    """Main function to run baseline capture."""
    try:
        baseline = PerformanceBaseline()
        success = baseline.run_baseline_capture()
        
        if success:
            print("🎉 Performance baseline capture completed successfully!")
        else:
            print("💥 Performance baseline capture failed!")
            return 1
            
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())