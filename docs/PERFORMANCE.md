# Performance Monitoring and Baseline Documentation

This document describes the performance monitoring tools and baseline establishment process for the Crawl4AI RAG MCP Server enhancement project.

## Overview

Task 1.0 has been completed with the implementation of a comprehensive performance monitoring framework that includes:

1. **Performance Baseline Capture** - Establishes current system performance metrics
2. **Continuous Performance Monitoring** - Tracks performance during development
3. **Automated Regression Testing** - Detects performance regressions in CI/CD

## Prerequisites

Before running performance tests, ensure your Supabase Docker stack is running:

```bash
# Start the full Supabase stack with hybrid search edge functions
./start-supabase-all.sh --hybrid-search-crawled-pages

# Verify the stack is running
curl http://localhost:54323  # Should return Supabase Studio
```

Also ensure you have some crawled data in your database for meaningful performance tests.

## Performance Tools

### 1. Performance Baseline Capture (`src/performance_baseline.py`)

Captures comprehensive baseline metrics including:
- System information (CPU, memory, Python version)
- Database statistics (document count, sources, content length)
- Search performance across 5 test query types
- Memory usage patterns during operations

**Usage:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Run baseline capture
python src/performance_baseline.py
```

**Output:** Creates `performance_baseline.json` with complete metrics.

### 2. Performance Monitor (`src/performance_monitor.py`)

Provides ongoing performance monitoring and regression detection:
- Real-time performance measurement
- Comparison against baseline metrics
- Regression threshold detection (25% response time, 50% memory, -20% results)
- Detailed performance analysis

**Usage:**
```python
from performance_monitor import PerformanceMonitor, validate_against_baseline
from utils import get_supabase_client

# Quick validation
client = get_supabase_client()
no_regressions = validate_against_baseline(client)

# Detailed monitoring
monitor = PerformanceMonitor()
metrics = monitor.measure_search_performance(client, "test query")
comparison = monitor.compare_to_baseline(metrics)
```

### 3. Automated Regression Tests (`tests/test_performance_regression.py`)

Comprehensive test suite for automated regression detection:
- Unit tests for individual performance metrics
- Integration tests for multiple query types
- Memory usage regression detection
- Batch performance validation

**Usage:**
```bash
# Run full performance test suite
python tests/test_performance_regression.py

# Run quick tests for CI/CD
python tests/test_performance_regression.py --quick

# Run via unittest
python -m unittest tests.test_performance_regression
```

## Performance Thresholds

The monitoring system uses the following regression thresholds:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Response Time | +25% | 25% slower than baseline triggers regression warning |
| Memory Usage | +50% | 50% more memory usage triggers regression warning |
| Result Count | -20% | 20% fewer results triggers regression warning |
| Absolute Limits | 5s simple, 8s complex | Hard limits for query response times |

## Integration with Enhancement Project

### Baseline Requirements (TASKS.md Task 1.0)

✅ **Completed Tasks:**
- [x] Capture current search performance metrics
- [x] Document baseline performance in metrics file
- [x] Set up performance monitoring framework for comparison
- [x] Create automated performance regression tests

### Usage During Development

1. **Before Phase 1:** Run baseline capture to establish current metrics
2. **During Development:** Use performance monitor for real-time validation
3. **After Each Phase:** Run regression tests to validate no performance loss
4. **In CI/CD:** Use quick regression tests for automated validation

### Integration Points

The performance tools integrate with:
- **Supabase Docker Stack:** Requires running stack for database operations
- **Edge Functions:** Tests the `hybrid_search_crawled_pages` RPC function
- **n8n Workflows:** Performance changes may affect workflow response times
- **Enhancement Strategies:** Each new strategy is performance-validated

## Example Workflow

```bash
# 1. Ensure Supabase stack is running
./start-supabase-all.sh

# 2. Capture initial baseline (run once)
python src/performance_baseline.py

# 3. During development - validate changes
python -c "
from src.performance_monitor import validate_against_baseline
from src.utils import get_supabase_client
result = validate_against_baseline(get_supabase_client())
print('✅ No regressions' if result else '⚠️ Regressions detected')
"

# 4. Run full regression tests
python tests/test_performance_regression.py
```

## Baseline Results Structure

The `performance_baseline.json` contains:

```json
{
  "timestamp": "2025-01-07T...",
  "system_info": {
    "cpu_count": 8,
    "memory_total_gb": 16.0,
    "python_version": "3.12.10"
  },
  "database_stats": {
    "total_documents": 1250,
    "unique_sources": 15,
    "avg_content_length_chars": 2847.3
  },
  "search_performance": {
    "avg_response_time_ms": 342.5,
    "median_response_time_ms": 315.2,
    "avg_result_count": 8.4,
    "successful_queries": 5
  }
}
```

## Troubleshooting

### Common Issues

1. **DNS Resolution Error:** Ensure Supabase Docker stack is running on correct ports
2. **No Baseline Data:** Run `performance_baseline.py` first before other tools
3. **Empty Results:** Ensure database has crawled content for meaningful tests
4. **Memory Errors:** Check available system memory before running tests

### Environment Variables Required

```bash
SUPABASE_URL=http://localhost:54321  # Your Supabase instance
SUPABASE_SERVICE_KEY=your_service_key
OPENAI_API_KEY=your_openai_key
```

## Next Steps

With Task 1.0 completed, the performance monitoring framework is ready for:
- **Task 1.1:** Strategy Configuration System development
- **Task 1.2:** Sentence Transformers Integration
- **Phase 2:** Database Architecture Enhancements

All future enhancements will be validated against this baseline to ensure no performance regressions.