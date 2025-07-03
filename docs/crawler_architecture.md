# Advanced Web Crawler Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [Unified AdvancedWebCrawler Design](#unified-advancedwebcrawler-design)
3. [Framework Detection System](#framework-detection-system)
4. [Quality Validation Infrastructure](#quality-validation-infrastructure)
5. [Performance and Monitoring](#performance-and-monitoring)
6. [Adding New Documentation Sites](#adding-new-documentation-sites)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Performance Optimization](#performance-optimization)
9. [API Reference](#api-reference)

## Overview

The Advanced Web Crawler system has been consolidated (Task 17) into a unified architecture that provides high-quality content extraction from modern, JavaScript-heavy documentation websites. The system is designed as the first stage of a Document Ingestion Pipeline, optimized for producing clean markdown suitable for semantic chunking and RAG applications.

### Key Design Principles

1. **Unified Architecture**: Single `AdvancedWebCrawler` class replaces multiple redundant crawler implementations
2. **Quality-First Approach**: Built-in quality validation with automatic fallback mechanisms
3. **Framework-Aware Extraction**: Intelligent detection and targeting of documentation frameworks
4. **Pipeline Optimization**: Designed specifically for DocumentIngestionPipeline compatibility
5. **Comprehensive Monitoring**: Integrated performance and quality metrics

### Architecture Benefits

- **Simplified Maintenance**: Single codebase instead of multiple crawler variants
- **Consistent Quality**: Unified quality validation across all crawling operations
- **Better Performance**: Optimized extraction with framework-specific targeting
- **Enhanced Reliability**: Multi-attempt extraction with progressive fallback strategies
- **Comprehensive Testing**: End-to-end validation with real-world documentation sites

## Unified AdvancedWebCrawler Design

### Core Components

```
AdvancedWebCrawler
├── Browser Engine (Playwright)
├── Configuration System
│   ├── Enhanced Framework Detection
│   ├── Simple Domain-Based Configuration
│   └── Fallback CSS Selectors
├── Quality Validation
│   ├── Enhanced Quality Analyzer
│   ├── Legacy Validator
│   └── Content Quality Metrics
├── Extraction Pipeline
│   ├── NoExtractionStrategy
│   ├── DefaultMarkdownGenerator
│   └── Post-Processing
└── Monitoring & Analytics
    ├── Performance Benchmarker
    ├── Chunk Content Validator
    └── Quality Metrics Collection
```

### Class Hierarchy

```python
class AdvancedWebCrawler:
    """
    Unified web crawler optimized for DocumentIngestionPipeline.
    
    Replaces: smart_crawler_factory.EnhancedCrawler, basic crawlers
    """
    
    def __init__(self,
                 headless: bool = True,
                 timeout_ms: int = 30000,
                 enable_quality_validation: bool = True,
                 max_fallback_attempts: int = 3):
        """Initialize with quality validation and fallback support."""
        
    async def crawl_single_page(self, url: str) -> AdvancedCrawlResult:
        """
        Main crawling method with multi-attempt extraction.
        
        Process:
        1. Framework detection and configuration
        2. Primary extraction attempt
        3. Quality validation
        4. Fallback attempts if quality is poor
        5. Enhanced quality metrics collection
        """
```

### Integration Points

#### With Quality Systems
```python
# Enhanced quality integration (from smart_crawler_factory)
from content_quality import (
    ContentQualityAnalyzer,
    ContentQualityMetrics,
    calculate_content_quality,
    should_retry_extraction
)

# Legacy quality validation
from crawler_quality_validation import (
    ContentQualityValidator,
    QualityValidationResult,
    validate_crawler_output
)
```

#### With Configuration Systems
```python
# Enhanced framework detection (if available)
from enhanced_crawler_config import (
    DocumentationFramework,
    config_manager,
    detect_framework
)

# Simple domain-based configuration (Task 16)
from documentation_site_config import (
    get_config_by_domain,
    extract_domain_from_url
)
```

### Key Architectural Changes (Task 17)

#### Before Consolidation
```
Multiple Crawler Implementations:
├── AsyncWebCrawler (baseline)
├── smart_crawler_factory.EnhancedCrawler
├── Various specialized crawlers
└── Conditional USE_ENHANCED_CRAWLING flag
```

#### After Consolidation
```
Unified AdvancedWebCrawler:
├── Built-in enhanced functionality
├── Integrated quality validation
├── Framework detection
├── Multi-attempt extraction
└── No feature flags - always enhanced
```

## Framework Detection System

### Supported Documentation Frameworks

| Framework | Detection Method | CSS Selectors | Special Handling |
|-----------|-----------------|---------------|------------------|
| **Material Design** (n8n.io) | Domain + UI patterns | `main.md-main, article.md-content__inner` | Navigation exclusion |
| **ReadMe.io** | Domain pattern | `.rm-Guides, .rm-Article` | API doc optimization |
| **GitBook** | Domain + structure | `.gitbook-content` | Sidebar handling |
| **Docusaurus** | Meta tags + DOM | `main article, .theme-doc-wrapper` | React navigation |
| **MkDocs** | Generator meta | `main.md-main, .md-content` | Theme variations |
| **GitHub Pages** | Domain pattern | `main, .main-content` | Jekyll support |
| **Generic** | Fallback | `main article` | Universal selectors |

### Framework Detection Logic

```python
def _detect_framework_from_content(self, markdown: str, url: str) -> Optional[str]:
    """
    Multi-step framework detection process.
    
    1. Domain-based detection (fastest)
    2. Meta tag analysis
    3. DOM structure analysis
    4. Content pattern matching
    5. Fallback to generic
    """
    domain = urlparse(url).netloc.lower()
    
    # Domain-based detection
    if 'n8n.io' in domain:
        return 'material_design'
    elif 'readme.io' in domain:
        return 'readme_io'
    # ... additional patterns
    
    # Content-based detection
    if 'docusaurus' in markdown.lower():
        return 'docusaurus'
    
    return 'generic'
```

### Configuration Priority System

1. **Simple Domain Configuration** (Task 16) - Highest priority
2. **Enhanced Framework Configuration** - Framework-specific rules
3. **Fallback CSS Selectors** - Progressive fallback attempts
4. **Generic Selectors** - Universal fallback

## Quality Validation Infrastructure

### Dual Quality System

The unified crawler integrates two complementary quality validation systems:

#### Enhanced Quality System (from smart_crawler_factory)
```python
class ContentQualityMetrics:
    content_to_navigation_ratio: float
    link_density: float
    text_coherence_score: float
    overall_quality_score: float
    should_retry_with_fallback: bool
```

**Features:**
- Advanced content-to-navigation ratio calculation
- Link density analysis
- Text coherence assessment
- Automatic retry recommendations

#### Legacy Quality Validator
```python
class QualityValidationResult:
    passed: bool
    score: float
    issues: List[str]
    recommendations: List[str]
    category: str  # excellent, good, fair, poor
```

**Features:**
- HTML artifact detection
- Script contamination checks
- Golden set pattern validation
- Detailed issue reporting

### Quality-Driven Fallback System

```python
async def crawl_single_page(self, url: str) -> AdvancedCrawlResult:
    """Multi-attempt extraction with quality validation."""
    
    for attempt in range(1, self.max_fallback_attempts + 2):
        if attempt == 1:
            # Primary extraction with framework-specific config
            config = self._create_optimized_run_config(url)
        else:
            # Fallback extraction with alternative selectors
            config = self._create_fallback_config(attempt - 1)
        
        result = await self.crawler.arun(url=url, config=config)
        
        # Quality validation
        quality_metrics = calculate_content_quality(result.markdown)
        
        if not should_retry_extraction(quality_metrics):
            break  # Quality acceptable
        
        # Continue to next fallback attempt
```

## Performance and Monitoring

### Performance Benchmarking System

The system includes comprehensive performance monitoring (Subtask 18.4):

```python
class PerformanceBenchmarker:
    """Comprehensive performance analysis system."""
    
    async def benchmark_crawler_implementation(
        self,
        crawler_function: Callable,
        test_urls: List[str]
    ) -> PerformanceBenchmarkResult:
        """
        Benchmark with detailed metrics:
        - Processing time per page
        - Memory usage monitoring
        - Database storage efficiency
        - Search relevance scores
        """
```

**Key Metrics:**
- **Timing**: Processing time, time per word, time per KB
- **Memory**: Peak usage, growth patterns, leak detection
- **Database**: Content-to-metadata ratio, storage efficiency
- **Quality**: Search relevance, content preservation

### Chunk Content Validation

Validates that generated chunks contain meaningful content (Subtask 18.3):

```python
class ChunkContentValidator:
    """Validates chunk quality and meaningful data extraction."""
    
    def validate_chunks(self, chunks: List[str]) -> ChunkValidationResult:
        """
        Validates chunks for:
        - Substantive content (>100 characters)
        - Navigation pattern detection
        - Glossary definition preservation
        - Code example preservation
        - Internal link maintenance
        """
```

## Adding New Documentation Sites

### Step-by-Step Guide

#### 1. Framework Analysis

Before adding support for a new documentation site, analyze its structure:

```bash
# Inspect the site structure
curl -s "https://docs.newsite.com" | grep -E "(generator|framework|theme)"

# Check for common patterns
curl -s "https://docs.newsite.com" | grep -E "(main|article|content|docs)"
```

#### 2. Simple Configuration (Recommended)

Add domain-specific configuration to `documentation_site_config.py`:

```python
DOCUMENTATION_CONFIGS = {
    "docs.newsite.com": {
        "name": "NewSite Documentation",
        "framework": "custom",
        "content_selectors": [
            "main.docs-content",
            ".documentation-main",
            "article.main-content"
        ],
        "excluded_selectors": [
            "nav.docs-nav",
            ".sidebar",
            "footer.docs-footer"
        ],
        "wait_for_selector": "main.docs-content",
        "additional_config": {
            "word_count_threshold": 20,
            "preserve_code_blocks": True
        }
    }
}
```

#### 3. Enhanced Framework Configuration (Advanced)

For complex sites, add to `enhanced_crawler_config.py`:

```python
class NewSiteFramework(DocumentationFramework):
    """Custom framework for NewSite documentation."""
    
    def __init__(self):
        super().__init__(
            name="newsite",
            target_elements=[
                "main.docs-content",
                "article.documentation",
                ".content-wrapper"
            ],
            excluded_selectors=[
                "nav.primary-nav",
                ".sidebar-navigation",
                "footer.site-footer",
                ".breadcrumb-nav"
            ],
            wait_for_selector="main.docs-content",
            word_count_threshold=25
        )
```

#### 4. Testing Configuration

Test the new configuration:

```python
# Test with the n8n validation suite
from tests.integration.test_n8n_validation import N8nDocumentationTestSuite

# Create custom test suite
test_urls = [
    "https://docs.newsite.com/getting-started",
    "https://docs.newsite.com/api-reference",
    "https://docs.newsite.com/guides/tutorial"
]

test_suite = N8nDocumentationTestSuite()
# Replace n8n URLs with your test URLs
test_suite.test_urls = {'newsite': test_urls}
results = await test_suite.run_comprehensive_test_suite()
```

#### 5. Quality Validation

Ensure the configuration produces high-quality results:

```python
from chunk_content_validator import validate_chunk_content
from content_quality_analyzer import analyze_content_quality_enhanced

# Validate extracted content
chunks = extract_chunks_from_crawl_results(results)
validation_result = validate_chunk_content(chunks, "https://docs.newsite.com")

# Analyze quality metrics
quality_metrics = analyze_content_quality_enhanced(
    content, 
    url="https://docs.newsite.com",
    page_type="guide"
)
```

### Configuration Templates

#### Basic Documentation Site
```python
"docs.example.com": {
    "name": "Example Docs",
    "framework": "generic",
    "content_selectors": ["main", "article", ".content"],
    "excluded_selectors": ["nav", "footer", ".sidebar"]
}
```

#### React-Based Documentation
```python
"react-docs.example.com": {
    "name": "React Documentation",
    "framework": "react",
    "content_selectors": [
        "[data-content='main']",
        ".docs-content",
        "main article"
    ],
    "excluded_selectors": [
        "[data-nav]",
        ".navigation",
        ".theme-sidebar"
    ],
    "wait_for_selector": "[data-content='main']",
    "additional_config": {
        "wait_time_ms": 2000,  # Wait for React hydration
        "dynamic_content": True
    }
}
```

#### API Documentation Site
```python
"api.example.com": {
    "name": "API Documentation",
    "framework": "api",
    "content_selectors": [
        ".api-content",
        "main.documentation",
        ".endpoint-docs"
    ],
    "excluded_selectors": [
        ".api-nav",
        ".method-list",
        ".version-selector"
    ],
    "additional_config": {
        "preserve_code_blocks": True,
        "preserve_json_examples": True,
        "word_count_threshold": 15
    }
}
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Low Content Quality Scores

**Symptoms:**
- Content-to-navigation ratio < 0.5
- High number of navigation elements in extracted content
- Quality validation failures

**Diagnosis:**
```python
# Check quality metrics
quality_metrics = calculate_content_quality(extracted_content)
print(f"Content ratio: {quality_metrics.content_to_navigation_ratio}")
print(f"Navigation elements: {quality_metrics.navigation_element_count}")
print(f"Suggestions: {quality_metrics.improvement_suggestions}")
```

**Solutions:**
1. **Review CSS Selectors**: Make selectors more specific
   ```python
   # Instead of generic selectors
   content_selectors = ["main", "article"]
   
   # Use site-specific selectors
   content_selectors = ["main.docs-content", "article.documentation"]
   ```

2. **Add Navigation Exclusions**:
   ```python
   excluded_selectors = [
       "nav.primary-navigation",
       ".sidebar-menu",
       "footer.site-footer",
       ".breadcrumb",
       ".table-of-contents"
   ]
   ```

3. **Adjust Word Count Threshold**:
   ```python
   word_count_threshold = 25  # Increase to filter short navigation items
   ```

#### 2. JavaScript Rendering Issues

**Symptoms:**
- Empty or minimal content extraction
- "Loading..." text in extracted content
- Missing dynamic content

**Diagnosis:**
```python
# Check for dynamic content indicators
dynamic_indicators = [
    'loading', 'please wait', 'javascript required'
]

content_lower = extracted_content.lower()
has_dynamic_issues = any(indicator in content_lower for indicator in dynamic_indicators)
```

**Solutions:**
1. **Increase Wait Time**:
   ```python
   wait_for_selector = "main.content"
   wait_time_ms = 3000  # Increase wait time
   ```

2. **Add Specific Wait Conditions**:
   ```python
   # Wait for specific content to load
   wait_for_selector = "[data-loaded='true']"
   ```

3. **Use Alternative Selectors**:
   ```python
   # Target content that loads after JavaScript
   content_selectors = [
       "[data-hydrated] .content",
       ".dynamic-content.loaded"
   ]
   ```

#### 3. Memory Usage Issues

**Symptoms:**
- High memory consumption per page
- Memory growth during crawling
- Out of memory errors

**Diagnosis:**
```python
# Use performance benchmarker
benchmarker = PerformanceBenchmarker()
result = await benchmarker.benchmark_crawler_implementation(
    crawler_function, test_urls
)

print(f"Memory per page: {result.memory_metrics.memory_per_page:.1f}MB")
print(f"Memory leak detected: {result.memory_metrics.potential_memory_leak}")
```

**Solutions:**
1. **Reduce Concurrent Sessions**:
   ```python
   max_concurrent = 3  # Reduce from higher values
   ```

2. **Implement Content Streaming**:
   ```python
   # Process content in chunks
   chunk_size = 1000  # characters
   for chunk in process_content_in_chunks(content, chunk_size):
       process_chunk(chunk)
   ```

3. **Clean Up Resources**:
   ```python
   async with AdvancedWebCrawler() as crawler:
       # Automatically handles cleanup
       result = await crawler.crawl_single_page(url)
   ```

#### 4. Performance Issues

**Symptoms:**
- Slow crawling speed (< 2 pages/minute)
- High processing time per page
- Timeouts

**Diagnosis:**
```python
# Check timing breakdown
timing = result.timing_breakdown
print(f"Total time: {timing.total_time_ms}ms")
print(f"Time per word: {timing.time_per_word:.2f}ms")
print(f"Time per KB: {timing.time_per_kb:.1f}ms")
```

**Solutions:**
1. **Optimize CSS Selectors**:
   ```python
   # Use more specific selectors to reduce processing
   content_selectors = ["#main-content"]  # ID selector (fastest)
   ```

2. **Adjust Timeout Settings**:
   ```python
   timeout_ms = 45000  # Increase for slow sites
   ```

3. **Increase Concurrency**:
   ```python
   max_concurrent = 5  # Increase if memory allows
   ```

#### 5. Framework Detection Failures

**Symptoms:**
- Generic framework detected instead of specific
- Incorrect CSS selectors applied
- Poor extraction quality

**Diagnosis:**
```python
# Check framework detection
framework = crawler._detect_framework_from_content("", url)
print(f"Detected framework: {framework}")

# Check domain configuration
domain = extract_domain_from_url(url)
config = get_config_by_domain(domain)
print(f"Domain config: {config}")
```

**Solutions:**
1. **Add Domain-Specific Configuration**:
   ```python
   # Add to documentation_site_config.py
   DOCUMENTATION_CONFIGS["new-domain.com"] = {
       "framework": "custom",
       "content_selectors": ["specific-selector"]
   }
   ```

2. **Improve Detection Logic**:
   ```python
   # Add patterns to framework detection
   if 'specific-pattern' in url or 'framework-indicator' in content:
       return 'specific_framework'
   ```

### Debugging Tools

#### 1. Quality Analysis
```python
from content_quality_analyzer import analyze_content_quality_enhanced

# Comprehensive quality analysis
quality_result = analyze_content_quality_enhanced(
    content, url, page_type, expected_links
)

print(f"Combined score: {quality_result.combined_quality_score}")
print(f"Issues: {quality_result.priority_improvements}")
```

#### 2. Chunk Validation
```python
from chunk_content_validator import validate_chunk_content

# Validate chunk quality
validation_result = validate_chunk_content(chunks, source_url)
print(f"Valid chunks: {validation_result.valid_chunks}/{validation_result.total_chunks}")
print(f"Common issues: {validation_result.common_issues}")
```

#### 3. Performance Profiling
```python
from performance_benchmarker import PerformanceBenchmarker

# Profile crawler performance
benchmarker = PerformanceBenchmarker()
result = await benchmarker.benchmark_crawler_implementation(
    crawler_function, test_urls
)

# Generate detailed report
report = create_performance_report(result)
print(report)
```

## Performance Optimization

### Best Practices

#### 1. Configuration Optimization

**CSS Selector Hierarchy (Performance Order):**
1. ID selectors: `#main-content` (fastest)
2. Class selectors: `.docs-content` 
3. Element + class: `main.content`
4. Descendant selectors: `main .content`
5. Complex selectors: `main > .content > article` (slowest)

**Recommended Pattern:**
```python
content_selectors = [
    "#docs-main",              # Primary (ID)
    ".documentation-content",   # Secondary (class)
    "main article",            # Fallback (element)
    "body .content"            # Last resort
]
```

#### 2. Memory Optimization

```python
# Efficient crawler configuration
crawler = AdvancedWebCrawler(
    headless=True,                    # Reduce memory usage
    timeout_ms=30000,                 # Reasonable timeout
    enable_quality_validation=True,   # But limit validation scope
    max_fallback_attempts=2           # Reduce fallback attempts
)
```

#### 3. Concurrency Tuning

```python
# Balanced concurrency for different site types
CONCURRENCY_CONFIGS = {
    'large_sites': 2,      # Conservative for large documentation sites
    'api_docs': 3,         # Moderate for API documentation
    'small_sites': 5,      # Aggressive for smaller sites
    'localhost': 10        # Maximum for local testing
}
```

#### 4. Quality vs Performance Trade-offs

```python
# High-quality extraction (slower)
high_quality_config = {
    'max_fallback_attempts': 3,
    'enable_quality_validation': True,
    'detailed_analysis': True
}

# Fast extraction (lower quality)
fast_config = {
    'max_fallback_attempts': 1,
    'enable_quality_validation': False,
    'timeout_ms': 15000
}

# Balanced approach (recommended)
balanced_config = {
    'max_fallback_attempts': 2,
    'enable_quality_validation': True,
    'timeout_ms': 30000
}
```

### Performance Monitoring

#### Setting Up Continuous Monitoring

```python
# Establish baseline performance
baseline_urls = [
    "https://docs.example.com/guide",
    "https://docs.example.com/api",
    "https://docs.example.com/reference"
]

# Run baseline benchmark
benchmarker = PerformanceBenchmarker()
baseline_result = await benchmarker.benchmark_crawler_implementation(
    crawler_function, baseline_urls, "baseline_crawler"
)

# Save baseline for future comparisons
benchmarker.save_benchmark_result(baseline_result, "baseline_performance.json")
```

#### Performance Regression Testing

```python
# Compare against baseline
new_benchmarker = PerformanceBenchmarker(baseline_data_path="baseline_performance.json")
new_result = await new_benchmarker.benchmark_crawler_implementation(
    improved_crawler_function, baseline_urls, "improved_crawler"
)

# Check for regressions
if new_result.performance_improvement_percent < -10:
    print("⚠️ Performance regression detected!")
    print(f"Performance change: {new_result.performance_improvement_percent:.1f}%")
```

## API Reference

### Core Classes

#### AdvancedWebCrawler

```python
class AdvancedWebCrawler:
    def __init__(self,
                 headless: bool = True,
                 timeout_ms: int = 30000,
                 custom_css_selectors: Optional[List[str]] = None,
                 enable_quality_validation: bool = True,
                 max_fallback_attempts: int = 3)
    
    async def crawl_single_page(self, url: str) -> AdvancedCrawlResult
    async def __aenter__(self) -> 'AdvancedWebCrawler'
    async def __aexit__(self, exc_type, exc_val, exc_tb)
```

#### Convenience Functions

```python
# Single page crawling
async def crawl_single_page_advanced(url: str, **kwargs) -> AdvancedCrawlResult

# Batch crawling
async def batch_crawl_advanced(urls: List[str], 
                             max_concurrent: int = 5,
                             **kwargs) -> List[AdvancedCrawlResult]

# Smart URL handling
async def smart_crawl_url_advanced(url: str, **kwargs) -> List[AdvancedCrawlResult]

# Recursive crawling
async def crawl_recursive_internal_links_advanced(
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
    **kwargs
) -> List[AdvancedCrawlResult]
```

### Quality Analysis

```python
# Enhanced quality analysis
from content_quality_analyzer import analyze_content_quality_enhanced

result = analyze_content_quality_enhanced(
    markdown_content: str,
    url: str = "",
    page_type: str = "",
    expected_links: Optional[List[str]] = None
) -> EnhancedQualityMetrics

# Chunk validation
from chunk_content_validator import validate_chunk_content

result = validate_chunk_content(
    chunks: List[str],
    source_url: str = "",
    chunk_metadata: Optional[List[Dict[str, Any]]] = None
) -> ChunkValidationResult

# Performance benchmarking
from performance_benchmarker import PerformanceBenchmarker

benchmarker = PerformanceBenchmarker()
result = await benchmarker.benchmark_crawler_implementation(
    crawler_function: Callable,
    test_urls: List[str],
    crawler_name: str = "unknown"
) -> PerformanceBenchmarkResult
```

### Configuration

```python
# Simple domain configuration
from documentation_site_config import get_config_by_domain, extract_domain_from_url

domain = extract_domain_from_url(url)
config = get_config_by_domain(domain)

# Enhanced framework configuration
from enhanced_crawler_config import detect_framework, config_manager

framework = detect_framework(url)
framework_config = config_manager.get_framework_config(framework)
```

---

## Conclusion

The unified AdvancedWebCrawler architecture provides a robust, scalable solution for extracting high-quality content from modern documentation websites. By consolidating multiple crawler implementations into a single, well-tested system with comprehensive quality validation and performance monitoring, the architecture ensures reliable content extraction for RAG applications while maintaining excellent performance characteristics.

The modular design allows for easy extension to new documentation sites while the comprehensive monitoring and validation systems ensure consistent quality across all crawling operations. The troubleshooting guide and performance optimization recommendations provide actionable guidance for maintaining and improving the system over time.