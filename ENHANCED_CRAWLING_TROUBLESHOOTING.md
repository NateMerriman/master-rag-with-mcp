# Enhanced Crawling Troubleshooting Guide

This guide helps diagnose and resolve issues with the enhanced crawling functionality for documentation sites.

## Quick Diagnosis Commands

### Check Enhanced Crawling Status
```bash
# View current strategy configuration
curl -X POST http://localhost:8051/tools/get_strategy_status

# Check if enhanced crawling is enabled
echo $USE_ENHANCED_CRAWLING
```

### Test Framework Detection
```bash
# Analyze a site's framework
curl -X POST http://localhost:8051/tools/analyze_site_framework \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"url": "https://docs.n8n.io/workflows/"}}'
```

### Run Validation Tests
```bash
# Run comprehensive validation
cd tests
python validate_enhanced_crawling.py

# Run unit tests
pytest test_enhanced_crawling.py -v
```

## Common Issues and Solutions

### 1. Enhanced Crawling Tools Not Available

**Symptoms:**
- Tools `crawl_single_page_enhanced`, `smart_crawl_url_enhanced`, or `analyze_site_framework` not appearing in tool list
- Error: "Enhanced crawling requires USE_ENHANCED_CRAWLING=true"

**Solutions:**

**Step 1: Check Environment Variable**
```bash
echo $USE_ENHANCED_CRAWLING
# Should output: true
```

**Step 2: Set Environment Variable**
```bash
# In .env file
USE_ENHANCED_CRAWLING=true

# Or export directly
export USE_ENHANCED_CRAWLING=true
```

**Step 3: Restart Server**
```bash
# Docker
docker restart your_container_name

# Direct Python
# Stop and restart the MCP server
```

**Step 4: Verify Module Loading**
Check server logs for:
```
✅ Enhanced crawling modules loaded successfully
```

If you see:
```
❌ Enhanced crawling modules not available: ImportError
```

Then install missing dependencies:
```bash
uv pip install crawl4ai asyncio aiohttp
```

### 2. Framework Detection Not Working

**Symptoms:**
- All sites detected as "generic" framework
- Poor extraction quality despite enhanced crawling enabled
- Framework detection confidence always "low"

**Diagnostic Steps:**

**Step 1: Test Framework Detection**
```bash
# Test with known frameworks
curl -X POST http://localhost:8051/tools/analyze_site_framework \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"url": "https://docs.n8n.io/workflows/"}}'

# Should detect "material_design" with high confidence
```

**Step 2: Check Domain Patterns**
Common framework domains:
- **ReadMe.io**: `*.readme.io`, `*.readme.com`
- **GitBook**: `*.gitbook.io`, `*.gitbook.com`
- **Material Design**: Detected via HTML analysis

**Step 3: Manual Override**
If framework detection fails, create a custom configuration:
```python
# In enhanced_crawler_config.py
# Add your domain to _initialize_domain_patterns()

self._domain_patterns[DocumentationFramework.MATERIAL_DESIGN] = [
    r".*docs\.yoursite\.com$"
]
```

**Step 4: HTML Analysis Issues**
If HTML analysis fails:
- Check if site blocks automated requests
- Verify HTML contains expected CSS classes
- Use `analyze_site_framework` tool to debug HTML content

### 3. Poor Content Quality Scores

**Symptoms:**
- Quality scores consistently below 0.5
- High navigation-to-content ratios
- Extraction falls back to generic methods

**Diagnostic Steps:**

**Step 1: Check Quality Metrics**
```bash
# Look for quality metrics in crawl response
{
  "quality_metrics": {
    "overall_score": 0.3,           # Should be > 0.5
    "content_to_navigation_ratio": 0.2,  # Should be > 0.6
    "link_density": 0.8,           # Should be < 0.3
    "category": "poor"             # Should be "good" or "excellent"
  }
}
```

**Step 2: Analyze Framework Configuration**
```bash
# Check recommended configuration for detected framework
curl -X POST http://localhost:8051/tools/analyze_site_framework \
  -H "Content-Type: application/json" \
  -d '{"arguments": {"url": "YOUR_PROBLEMATIC_URL"}}'
```

**Step 3: Tune Quality Thresholds**
Lower thresholds temporarily for testing:
```bash
# In .env file
ENHANCED_CRAWLING_MIN_CONTENT_RATIO=0.4  # Default: 0.6
ENHANCED_CRAWLING_MAX_LINK_DENSITY=0.5   # Default: 0.3
ENHANCED_CRAWLING_MIN_QUALITY_SCORE=0.3  # Default: 0.5
```

**Step 4: Custom CSS Selectors**
For sites with unique structures, add custom configuration:
```python
# In enhanced_crawler_config.py
# Modify framework configs or add new ones

configs[DocumentationFramework.CUSTOM_SITE] = FrameworkConfig(
    target_elements=["#main-content", ".docs-wrapper"],
    excluded_selectors=[".sidebar", ".navigation"],
    excluded_tags=["nav", "header", "footer"],
    word_count_threshold=10,
    min_content_ratio=0.4  # Lower threshold for difficult sites
)
```

### 4. Slow Performance Issues

**Symptoms:**
- Extraction takes longer than 5 seconds per page
- Framework detection timeout
- Quality analysis timeout

**Diagnostic Steps:**

**Step 1: Check Performance Metrics**
```bash
# Look for performance data in crawl response
{
  "performance": {
    "total_time_seconds": 7.5,      # Should be < 5.0
    "framework_detection_ms": 200,  # Should be < 100ms (cached)
    "quality_analysis_ms": 150      # Should be < 100ms
  }
}
```

**Step 2: Tune Performance Settings**
```bash
# In .env file
ENHANCED_CRAWLING_MAX_EXTRACTION_TIME=10.0      # Increase timeout
ENHANCED_CRAWLING_MAX_FALLBACK_ATTEMPTS=2       # Reduce attempts
ENHANCED_CRAWLING_CACHE_FRAMEWORK_DETECTION=true # Enable caching
```

**Step 3: Reduce Concurrency**
```bash
# When calling smart_crawl_url_enhanced
{
  "arguments": {
    "url": "YOUR_URL",
    "max_concurrent": 3  # Reduce from default 5
  }
}
```

**Step 4: Disable Heavy Features**
For speed-critical scenarios:
```python
# Custom configuration with minimal overhead
config = FrameworkConfig(
    target_elements=["main", "article"],
    excluded_tags=["nav", "header", "footer"],
    word_count_threshold=5,
    min_content_ratio=0.3  # Lower standard for speed
)
```

### 5. Fallback Mechanisms Not Working

**Symptoms:**
- Extraction fails completely instead of falling back
- No alternative extraction attempts
- Error messages about CSS selectors

**Diagnostic Steps:**

**Step 1: Check Fallback Configuration**
```bash
# Verify fallback attempts setting
echo $ENHANCED_CRAWLING_MAX_FALLBACK_ATTEMPTS
# Should be 2 or 3
```

**Step 2: Test Fallback Manually**
```python
from smart_crawler_factory import SmartCrawlerFactory

factory = SmartCrawlerFactory()

# Test fallback configs
for i in range(1, 4):
    config = factory.create_fallback_config(i)
    print(f"Fallback {i}: {config.css_selector}")
```

**Step 3: Enable Fallback Logging**
Add logging to see fallback attempts:
```python
import logging
logging.basicConfig(level=logging.INFO)

# Should see logs like:
# "Primary extraction attempt for URL"
# "Fallback extraction attempt 1 for URL"
```

**Step 4: Check Error Handling**
Ensure proper exception handling in crawler:
```python
try:
    result = await crawler.crawl_single_page_enhanced(url)
except Exception as e:
    print(f"Extraction failed: {e}")
    # Should see detailed error information
```

### 6. Import Errors and Dependencies

**Symptoms:**
- ModuleNotFoundError for enhanced crawling modules
- ImportError when trying to use enhanced tools
- "Enhanced crawling modules not available" in logs

**Solutions:**

**Step 1: Install Dependencies**
```bash
# Core dependencies
uv pip install crawl4ai asyncio aiohttp xml.etree.ElementTree

# Optional dependencies for analysis
uv pip install pytest beautifulsoup4
```

**Step 2: Check Python Path**
```python
import sys
sys.path.append('./src')

# Test imports
from enhanced_crawler_config import detect_framework
from content_quality import calculate_content_quality
from smart_crawler_factory import EnhancedCrawler
```

**Step 3: Verify File Structure**
```
src/
├── enhanced_crawler_config.py
├── content_quality.py
├── smart_crawler_factory.py
└── crawl4ai_mcp.py
```

**Step 4: Check for Circular Imports**
If you see circular import errors:
```python
# Use absolute imports
from src.enhanced_crawler_config import ...

# Or restructure imports to avoid cycles
```

## Configuration Tuning Guidelines

### For High-Traffic Sites
```bash
USE_ENHANCED_CRAWLING=true
ENHANCED_CRAWLING_MAX_EXTRACTION_TIME=3.0
ENHANCED_CRAWLING_MAX_FALLBACK_ATTEMPTS=2
ENHANCED_CRAWLING_CACHE_FRAMEWORK_DETECTION=true
```

### For Complex Documentation Sites
```bash
USE_ENHANCED_CRAWLING=true
ENHANCED_CRAWLING_MIN_CONTENT_RATIO=0.4
ENHANCED_CRAWLING_MAX_LINK_DENSITY=0.4
ENHANCED_CRAWLING_MAX_FALLBACK_ATTEMPTS=3
```

### For Sites with Unique Structures
```bash
USE_ENHANCED_CRAWLING=true
ENHANCED_CRAWLING_MIN_QUALITY_SCORE=0.3
ENHANCED_CRAWLING_ENABLE_HTML_ANALYSIS=true
ENHANCED_CRAWLING_ENABLE_META_TAG_DETECTION=true
```

## Advanced Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Will show detailed framework detection and quality analysis
```

### Manual Framework Detection Test
```python
from enhanced_crawler_config import DocumentationSiteConfigManager

config_manager = DocumentationSiteConfigManager()

# Test specific site
html_content = "<html><main class='md-main'>Content</main></html>"
framework = config_manager.detect_documentation_framework(
    "https://test.com", 
    html_content
)
print(f"Detected: {framework}")
```

### Quality Analysis Test
```python
from content_quality import calculate_content_quality

test_content = """
# Documentation
This is real content with useful information.
Links: [Home](/) [About](/about)
More substantial content here.
"""

metrics = calculate_content_quality(test_content)
print(f"Quality: {metrics.quality_category}")
print(f"Score: {metrics.overall_quality_score}")
print(f"Suggestions: {metrics.improvement_suggestions}")
```

### Performance Profiling
```python
import time
import cProfile

def profile_extraction():
    # Your extraction code here
    pass

cProfile.run('profile_extraction()')
```

## Getting Help

### Check Documentation
- Main README.md for configuration examples
- .env.example for all available settings
- Source code comments for implementation details

### Run Validation Tests
```bash
# Comprehensive testing
python tests/validate_enhanced_crawling.py

# Unit tests
pytest tests/test_enhanced_crawling.py -v

# Performance tests
python tests/test_performance_regression.py
```

### Generate Debug Report
```bash
# Create comprehensive debug information
python -c "
from enhanced_crawler_config import config_manager
from content_quality import quality_analyzer
import json

debug_info = {
    'framework_configs': len(config_manager._framework_configs),
    'domain_patterns': len(config_manager._domain_patterns),
    'cache_size': len(config_manager._detection_cache),
    'quality_thresholds': {
        'excellent': quality_analyzer.excellent_threshold,
        'good': quality_analyzer.good_threshold,
        'fair': quality_analyzer.fair_threshold,
        'poor': quality_analyzer.poor_threshold
    }
}

print(json.dumps(debug_info, indent=2))
"
```

### Contact and Issues
- Create issues with detailed error logs and configuration
- Include output from `analyze_site_framework` tool
- Provide quality metrics and performance data
- Include browser developer tools network tab if relevant

## Known Limitations

1. **JavaScript-Heavy Sites**: Enhanced crawling works best with server-rendered content
2. **Dynamic Content**: Sites that load content via JavaScript may need special handling
3. **Authentication**: Sites requiring login are not currently supported
4. **Rate Limiting**: Some sites may block rapid automated requests
5. **Framework Updates**: Documentation frameworks may change CSS classes, requiring configuration updates

## Future Enhancements

1. **JavaScript Rendering**: Support for single-page applications
2. **Authentication**: Login handling for private documentation
3. **Dynamic Configuration**: Runtime configuration updates without restart
4. **Machine Learning**: Automatic CSS selector discovery
5. **Performance Optimization**: Parallel framework detection and quality analysis