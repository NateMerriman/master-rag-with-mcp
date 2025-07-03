#!/usr/bin/env python3
"""
Advanced Web Crawler for Document Ingestion Pipeline

This module implements an AdvancedWebCrawler specifically designed to be the first stage
of a Document Ingestion Pipeline, inspired by the agentic-rag-knowledge-graph reference.
The primary goal is to reliably extract clean, high-quality markdown from modern,
JavaScript-heavy websites suitable for downstream processing by a SemanticChunker.

Key Features:
- Playwright browser automation for full JavaScript rendering  
- NoExtractionStrategy for clean content extraction
- Optimized DefaultMarkdownGenerator for semantic chunking compatibility
- Robust wait strategies for dynamic content
- Framework-specific CSS targeting
"""

from typing import Dict, List, Optional, Any
import asyncio
import time
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig, 
    CrawlerRunConfig,
    CacheMode
)
from crawl4ai.extraction_strategy import NoExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Import existing framework detection and configuration system
try:
    from enhanced_crawler_config import (
        DocumentationFramework,
        config_manager,
        detect_framework
    )
    ENHANCED_CONFIG_AVAILABLE = True
except ImportError:
    # Fallback if enhanced config not available
    ENHANCED_CONFIG_AVAILABLE = False

# Import the new Task 16 simple configuration system
try:
    from documentation_site_config import (
        get_config_by_domain,
        extract_domain_from_url
    )
    SIMPLE_CONFIG_AVAILABLE = True
except ImportError:
    SIMPLE_CONFIG_AVAILABLE = False

# Import quality validation system
try:
    from crawler_quality_validation import (
        ContentQualityValidator,
        QualityValidationResult,
        validate_crawler_output
    )
    QUALITY_VALIDATION_AVAILABLE = True
except ImportError:
    # Fallback if quality validation not available
    QUALITY_VALIDATION_AVAILABLE = False

# Import enhanced content quality system from smart_crawler_factory
try:
    from content_quality import (
        ContentQualityAnalyzer,
        ContentQualityMetrics,
        calculate_content_quality,
        should_retry_extraction,
        log_quality_metrics
    )
    ENHANCED_QUALITY_AVAILABLE = True
except ImportError:
    ENHANCED_QUALITY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass 
class AdvancedCrawlResult:
    """Result from AdvancedWebCrawler optimized for DocumentIngestionPipeline."""
    
    url: str
    markdown: str
    success: bool
    error_message: Optional[str] = None
    
    # Metadata for pipeline processing
    title: Optional[str] = None
    word_count: int = 0
    extraction_time_ms: float = 0.0
    framework_detected: Optional[str] = None
    
    # Quality indicators
    content_to_navigation_ratio: float = 0.0
    has_dynamic_content: bool = False
    
    # Quality validation results
    quality_validation: Optional['QualityValidationResult'] = None
    quality_passed: bool = False
    quality_score: float = 0.0


class AdvancedWebCrawler:
    """
    Advanced web crawler optimized for producing clean markdown for DocumentIngestionPipeline.
    
    This crawler implements the target architecture pattern:
    URL → AdvancedWebCrawler → Clean Markdown → DocumentIngestionPipeline
    
    Based on the reference implementation with these key configurations:
    - Playwright browser engine for JavaScript execution
    - NoExtractionStrategy for clean content filtering  
    - DefaultMarkdownGenerator optimized for semantic chunking
    - CSS selectors targeting main content areas
    - Robust wait strategies for dynamic content
    """
    
    def __init__(self, 
                 headless: bool = True,
                 timeout_ms: int = 30000,
                 custom_css_selectors: Optional[List[str]] = None,
                 enable_quality_validation: bool = True,
                 max_fallback_attempts: int = 3):
        """
        Initialize the AdvancedWebCrawler.
        
        Args:
            headless: Run browser in headless mode
            timeout_ms: Timeout for page loading and content waiting
            custom_css_selectors: Override default CSS selectors for content targeting
            enable_quality_validation: Enable automated quality validation
            max_fallback_attempts: Maximum number of fallback extraction attempts
        """
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.crawler: Optional[AsyncWebCrawler] = None
        self.max_fallback_attempts = max_fallback_attempts
        
        # Enhanced quality validation system
        self.enable_quality_validation = enable_quality_validation and QUALITY_VALIDATION_AVAILABLE
        self.enable_enhanced_quality = enable_quality_validation and ENHANCED_QUALITY_AVAILABLE
        
        # Initialize quality validator if available
        if self.enable_quality_validation:
            self.quality_validator = ContentQualityValidator()
        else:
            self.quality_validator = None
            
        # Initialize enhanced quality analyzer if available
        if self.enable_enhanced_quality:
            self.enhanced_quality_analyzer = ContentQualityAnalyzer()
        else:
            self.enhanced_quality_analyzer = None
        
        # CSS selectors for main content areas (inspired by reference implementation)
        self.css_selectors = custom_css_selectors or [
            "main article",  # Primary target for most documentation sites
            "main",          # Fallback to main element
            "article",       # Fallback to article element  
            ".content",      # Common content class
            "[role='main']"  # Semantic main role
        ]
        
        # Fallback CSS selectors for when primary extraction fails (from smart_crawler_factory)
        self.fallback_css_selectors = [
            # Primary content containers
            "main, article, .content, .main-content, .documentation",
            # Secondary content areas
            ".docs-content, .page-content, [role='main']",
            # Basic content divs
            ".container .content, .wrapper .main, .page .content",
            # Last resort - exclude obvious navigation
            "body > :not(nav):not(header):not(footer):not(aside)"
        ]
        
        # Elements to exclude (navigation, headers, footers, etc.)
        self.excluded_selectors = [
            "nav.navbar",
            "footer.footer", 
            "aside.theme-doc-sidebar-container",
            ".theme-doc-toc",
            ".theme-edit-this-page", 
            ".pagination-nav",
            "header",
            ".navigation",
            ".sidebar",
            ".menu"
        ]
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_crawler()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
            
    async def _initialize_crawler(self):
        """Initialize the Playwright-based crawler with optimized configuration."""
        
        # Configure browser for JavaScript execution (following reference pattern)
        browser_config = BrowserConfig(
            headless=self.headless,
            verbose=False,
            # Enable Playwright for JavaScript rendering
            browser_type="playwright",
            extra_args=[f"--timeout={self.timeout_ms}"]
        )
        
        self.crawler = AsyncWebCrawler(config=browser_config)
        await self.crawler.__aenter__()
        
    def _create_optimized_run_config(self, url: str) -> CrawlerRunConfig:
        """
        Create optimized CrawlerRunConfig following reference implementation pattern.
        
        This integrates with both the enhanced crawler configuration system and
        the new Task 16 simple configuration system for precise framework-specific 
        targeting and surgical content extraction.
        
        The configuration targets:
        - NoExtractionStrategy for clean content filtering
        - DefaultMarkdownGenerator optimized for semantic chunking  
        - Framework-specific CSS selectors for main content areas
        - Comprehensive excluded selectors for navigation removal
        """
        
        # Task 16.4: Extract domain and look up configuration
        domain = extract_domain_from_url(url) if SIMPLE_CONFIG_AVAILABLE else ""
        simple_config = get_config_by_domain(domain) if SIMPLE_CONFIG_AVAILABLE and domain else None
        
        # Use enhanced framework detection if available
        if ENHANCED_CONFIG_AVAILABLE:
            framework = detect_framework(url)
            framework_config = config_manager.get_framework_config(framework)
            
            # Get comprehensive exclusion selectors 
            exclude_selectors = framework_config.excluded_selectors + self.excluded_selectors
            
            # Use framework-specific word threshold
            word_threshold = framework_config.word_count_threshold
            
            logger.info(f"Using framework-specific config for {framework.value}: {len(exclude_selectors)} exclusions")
            
        else:
            # Fallback to domain-based detection
            exclude_selectors = self.excluded_selectors
            word_threshold = 10
        
        # Get target CSS selector for content areas
        # Task 16.4: Prioritize simple config selectors if available
        if simple_config and simple_config.content_selectors:
            # Use the Task 16 simple configuration selectors
            target_selector = ", ".join(simple_config.content_selectors)
            logger.info(f"Using Task 16 simple config selectors for {domain}: {target_selector}")
        elif ENHANCED_CONFIG_AVAILABLE:
            target_selector = ", ".join(framework_config.target_elements)
        else:
            target_selector = self._get_framework_css_selector_fallback(url)
        
        return CrawlerRunConfig(
            # Use NoExtractionStrategy for raw content extraction
            extraction_strategy=NoExtractionStrategy(),
            
            # Optimize DefaultMarkdownGenerator for DocumentIngestionPipeline compatibility
            markdown_generator=DefaultMarkdownGenerator(
                options={
                    'ignore_links': False,      # Preserve semantic link information for chunking
                    'ignore_images': True,      # Remove images for clean text flow
                    'protect_links': True,      # Protect link formatting for downstream processing
                    'bodywidth': 0,            # No line wrapping to preserve chunking boundaries
                    'escape_all': True,         # Escape HTML entities for clean markdown
                    'strip_html_tags': True,    # Attempt to strip all HTML tags not converted to markdown
                    'strip_js': True,           # Attempt to strip JavaScript code
                    'output_format': 'markdown', # Ensure output is markdown
                    'include_comments': False,  # Exclude HTML comments
                    'no_fallback': True         # Prevent fallback to less precise extraction
                }
            ),
            
            # Target specific content areas (this was missing!)
            css_selector=target_selector,
            
            # Surgically exclude navigation and boilerplate elements
            excluded_selector=", ".join(exclude_selectors),
            
            # Cache configuration for fresh content
            cache_mode=CacheMode.BYPASS,
            
            # Framework-optimized settings
            word_count_threshold=word_threshold,
            exclude_external_links=False,  # Keep external links for semantic context
            exclude_social_media_links=True,  # Remove social media noise
            process_iframes=False         # Skip iframes for performance
        )
        
    def _get_framework_css_selector_fallback(self, url: str) -> str:
        """
        Fallback framework-specific CSS selector when enhanced config is not available.
        
        This implements basic framework detection for common documentation sites.
        """
        
        domain = urlparse(url).netloc.lower()
        
        # Framework-specific CSS selectors (simplified fallback patterns)
        if any(pattern in domain for pattern in ['n8n.io', 'mkdocs']):
            return "main.md-main, article.md-content__inner, .md-content, main article"
        elif any(pattern in domain for pattern in ['readme.io', 'docs.']):  
            return ".rm-Guides, .rm-Article, main article, .content"
        elif 'gitbook' in domain:
            return ".gitbook-content, main article"
        elif 'github.io' in domain:
            return "main, .main-content, article"
        else:
            # Default selector targeting most documentation sites
            return "main article"
    
    def _create_fallback_config(self, attempt_number: int) -> CrawlerRunConfig:
        """
        Create a fallback configuration for when primary extraction fails.
        
        Args:
            attempt_number: The attempt number (1-based)
            
        Returns:
            Fallback CrawlerRunConfig
        """
        if attempt_number <= len(self.fallback_css_selectors):
            css_selector = self.fallback_css_selectors[attempt_number - 1]
        else:
            # Last resort - basic content extraction
            css_selector = "body"
        
        return CrawlerRunConfig(
            # Use NoExtractionStrategy for consistent extraction
            extraction_strategy=NoExtractionStrategy(),
            
            # Basic markdown generation for fallback
            markdown_generator=DefaultMarkdownGenerator(
                options={
                    'ignore_links': False,
                    'ignore_images': True,
                    'protect_links': True,
                    'bodywidth': 0,
                    'escape_all': True,
                    'strip_html_tags': True,
                    'strip_js': True,
                    'output_format': 'markdown',
                    'include_comments': False,
                    'no_fallback': True
                }
            ),
            
            css_selector=css_selector,
            excluded_tags=["nav", "header", "footer", "aside", "script", "style"],
            excluded_selector=", ".join(self.excluded_selectors),
            word_count_threshold=10,  # Lower threshold for fallback
            exclude_external_links=True,
            exclude_social_media_links=True,
            process_iframes=False,
            cache_mode=CacheMode.BYPASS
        )
            
    async def crawl_single_page(self, url: str) -> AdvancedCrawlResult:
        """
        Crawl a single page with enhanced extraction and quality validation.
        
        Uses multi-attempt extraction with progressive fallback strategies:
        1. Primary extraction with framework-specific configuration
        2. Fallback extractions with alternative CSS selectors if quality is poor
        3. Enhanced quality validation with retry logic
        
        Args:
            url: URL to crawl
            
        Returns:
            AdvancedCrawlResult with clean markdown optimized for semantic chunking
        """
        
        if not self.crawler:
            await self._initialize_crawler()
            
        start_time = time.time()
        
        try:
            # Framework detection for primary configuration
            framework_detected = self._detect_framework_from_content("", url)
            
            # Multi-attempt extraction with quality validation
            best_result = None
            extraction_attempts = 0
            used_fallback = False
            enhanced_quality_metrics = None
            
            for attempt in range(1, self.max_fallback_attempts + 2):  # +1 for primary attempt
                extraction_attempts = attempt
                
                try:
                    if attempt == 1:
                        # Primary extraction attempt
                        current_config = self._create_optimized_run_config(url)
                        logger.info(f"Primary extraction attempt for {url}")
                    else:
                        # Fallback extraction attempts
                        current_config = self._create_fallback_config(attempt - 1)
                        used_fallback = True
                        logger.info(f"Fallback extraction attempt {attempt-1} for {url}")
                    
                    # Perform the crawl
                    result = await self.crawler.arun(url=url, config=current_config)
                    
                    if not result.success:
                        logger.warning(f"Crawl failed for {url}: {getattr(result, 'status_code', 'unknown')}")
                        continue
                    
                    # Post-process the markdown to remove unwanted elements
                    cleaned_markdown = self._post_process_markdown(result.markdown)
                    
                    # Enhanced quality validation if available
                    if self.enable_enhanced_quality and self.enhanced_quality_analyzer and ENHANCED_QUALITY_AVAILABLE:
                        enhanced_quality_metrics = calculate_content_quality(cleaned_markdown)
                        
                        # Log quality metrics
                        log_quality_metrics(enhanced_quality_metrics, url, framework_detected)
                        
                        # Check if quality is acceptable
                        if not should_retry_extraction(enhanced_quality_metrics) or attempt >= self.max_fallback_attempts + 1:
                            # Quality is acceptable or we've exhausted attempts
                            break
                        else:
                            logger.info(f"Quality too low for {url}, trying fallback approach")
                            # Store this result in case fallbacks also fail
                            if best_result is None:
                                best_result = (result, cleaned_markdown, enhanced_quality_metrics)
                            continue
                    else:
                        # Use basic quality validation or accept first successful result
                        break
                        
                except Exception as e:
                    logger.error(f"Extraction attempt {attempt} failed for {url}: {str(e)}")
                    continue
            
            extraction_time_ms = (time.time() - start_time) * 1000
            
            if result and result.success and cleaned_markdown:
                # Extract metadata for pipeline processing
                word_count = len(cleaned_markdown.split())
                title = self._extract_title_from_markdown(cleaned_markdown)
                
                # Calculate quality indicators
                content_ratio = self._calculate_content_ratio(cleaned_markdown)
                has_dynamic = self._detect_dynamic_content_indicators(cleaned_markdown)
                
                # Legacy quality validation if enabled
                quality_validation = None
                quality_passed = True
                quality_score = 1.0
                
                if self.enable_quality_validation and self.quality_validator:
                    logger.info(f"Running legacy quality validation for {url}")
                    quality_validation = self.quality_validator.validate_content(cleaned_markdown, url)
                    quality_passed = quality_validation.passed
                    quality_score = quality_validation.score
                    
                    if not quality_passed:
                        logger.warning(f"Legacy quality validation failed for {url}: {', '.join(quality_validation.issues)}")
                    else:
                        logger.info(f"Legacy quality validation passed for {url}: {quality_validation.category} ({quality_score:.3f})")
                
                # Use enhanced quality score if available
                if enhanced_quality_metrics:
                    quality_score = enhanced_quality_metrics.overall_quality_score
                    quality_passed = not enhanced_quality_metrics.should_retry_with_fallback
                
                logger.info(f"Successfully extracted {word_count} words from {url} in {extraction_time_ms:.1f}ms (attempts: {extraction_attempts}, fallback: {used_fallback})")
                
                return AdvancedCrawlResult(
                    url=url,
                    markdown=cleaned_markdown,
                    success=True,
                    title=title,
                    word_count=word_count,
                    extraction_time_ms=extraction_time_ms,
                    framework_detected=framework_detected,
                    content_to_navigation_ratio=content_ratio,
                    has_dynamic_content=has_dynamic,
                    quality_validation=quality_validation,
                    quality_passed=quality_passed,
                    quality_score=quality_score
                )
            else:
                # All attempts failed or no suitable result
                error_msg = "Failed to extract content with acceptable quality"
                if best_result:
                    # Use best available result if we have one
                    result, cleaned_markdown, enhanced_quality_metrics = best_result
                    word_count = len(cleaned_markdown.split())
                    title = self._extract_title_from_markdown(cleaned_markdown)
                    content_ratio = self._calculate_content_ratio(cleaned_markdown)
                    has_dynamic = self._detect_dynamic_content_indicators(cleaned_markdown)
                    
                    logger.warning(f"Using best available result for {url} with low quality")
                    return AdvancedCrawlResult(
                        url=url,
                        markdown=cleaned_markdown,
                        success=True,
                        title=title,
                        word_count=word_count,
                        extraction_time_ms=extraction_time_ms,
                        framework_detected=framework_detected,
                        content_to_navigation_ratio=content_ratio,
                        has_dynamic_content=has_dynamic,
                        quality_validation=None,
                        quality_passed=False,
                        quality_score=enhanced_quality_metrics.overall_quality_score if enhanced_quality_metrics else 0.0
                    )
                else:
                    logger.warning(f"Failed to crawl {url}: {error_msg}")
                    return AdvancedCrawlResult(
                        url=url,
                        markdown="",
                        success=False,
                        error_message=error_msg,
                        extraction_time_ms=extraction_time_ms
                    )
                
        except Exception as e:
            extraction_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Crawling error: {str(e)}"
            logger.error(f"Error crawling {url}: {error_msg}")
            
            return AdvancedCrawlResult(
                url=url,
                markdown="",
                success=False,
                error_message=error_msg,
                extraction_time_ms=extraction_time_ms
            )
            
    def _extract_title_from_markdown(self, markdown: str) -> Optional[str]:
        """Extract title from markdown content."""
        lines = markdown.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return None
        
    def _detect_framework_from_content(self, markdown: str, url: str) -> Optional[str]:
        """Detect documentation framework from content and URL."""
        domain = urlparse(url).netloc.lower()
        
        if 'n8n.io' in domain:
            return 'material_design'
        elif 'readme.io' in domain:
            return 'readme_io'
        elif 'gitbook' in domain:
            return 'gitbook'
        elif 'github.io' in domain:
            return 'github_pages'
        else:
            return 'generic'
            
    def _calculate_content_ratio(self, markdown: str) -> float:
        """Calculate estimated content-to-navigation ratio."""
        total_words = len(markdown.split())
        
        # Rough heuristic: count navigation-like words
        nav_indicators = ['menu', 'navigation', 'sidebar', 'toc', 'breadcrumb']
        nav_word_count = sum(markdown.lower().count(indicator) for indicator in nav_indicators)
        
        if total_words == 0:
            return 0.0
            
        # Estimate content ratio (higher is better)
        nav_ratio = nav_word_count / total_words
        content_ratio = max(0.0, 1.0 - nav_ratio * 2)  # Simple heuristic
        
        return content_ratio
        
    def _detect_dynamic_content_indicators(self, markdown: str) -> bool:
        """Detect if the page likely had dynamic content."""
        dynamic_indicators = [
            'loading',
            'please wait', 
            'javascript',
            'enable js',
            'dynamic content'
        ]
        
        markdown_lower = markdown.lower()
        return any(indicator in markdown_lower for indicator in dynamic_indicators)

    def _post_process_markdown(self, markdown: str) -> str:
        """
        Performs structure-aware cleaning on the extracted markdown content.
        
        This function removes entire HTML blocks that represent navigation, sidebars, 
        footers, and other boilerplate elements while preserving inline content links
        within paragraphs and legitimate content areas.
        
        The approach targets structural HTML elements rather than individual links,
        ensuring that valuable content like glossary definitions with embedded links
        are preserved while removing navigational noise.
        """
        
        # Structure-aware regex patterns targeting entire HTML blocks
        # These patterns remove navigation/boilerplate elements while preserving content
        patterns_to_remove = [
            # Remove entire nav blocks (covers most navigation)
            r'<nav\b[^>]*>.*?</nav>',
            
            # Remove entire footer blocks  
            r'<footer\b[^>]*>.*?</footer>',
            
            # Remove sidebar and menu blocks
            r'<div\b[^>]*class="[^"]*(?:sidebar|menu|navigation)[^"]*"[^>]*>.*?</div>',
            
            # Remove header blocks that contain only navigation
            r'<header\b[^>]*>.*?</header>',
            
            # Remove breadcrumb navigation specifically
            r'<[^>]*class="[^"]*(?:breadcrumb|path|crumb)[^"]*"[^>]*>.*?</[^>]*>',
            
            # Remove edit buttons and action buttons
            r'<a\b[^>]*class="[^"]*(?:button|btn|edit|action)[^"]*"[^>]*>.*?</a>',
            
            # Remove standalone header anchor links (#) 
            r'<a\b[^>]*class="[^"]*headerlink[^"]*"[^>]*>\s*#\s*</a>',
            
            # Remove table of contents blocks
            r'<[^>]*class="[^"]*(?:toc|table-of-contents)[^"]*"[^>]*>.*?</[^>]*>',
        ]
        
        # Apply each pattern to remove unwanted HTML blocks
        cleaned_markdown = markdown
        for pattern in patterns_to_remove:
            cleaned_markdown = re.sub(pattern, '', cleaned_markdown, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up any multiple consecutive newlines created by removing blocks
        cleaned_markdown = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_markdown)
        
        # Remove any leading/trailing whitespace
        return cleaned_markdown.strip()


# Convenience functions for integration with existing codebase

async def crawl_single_page_advanced(url: str, **kwargs) -> AdvancedCrawlResult:
    """
    Convenience function for single page crawling.
    
    Args:
        url: URL to crawl
        **kwargs: Additional arguments for AdvancedWebCrawler
        
    Returns:
        AdvancedCrawlResult with clean markdown
    """
    async with AdvancedWebCrawler(**kwargs) as crawler:
        return await crawler.crawl_single_page(url)


async def batch_crawl_advanced(urls: List[str], 
                             max_concurrent: int = 5,
                             **kwargs) -> List[AdvancedCrawlResult]:
    """
    Batch crawl multiple URLs with concurrency control.
    
    Args:
        urls: List of URLs to crawl
        max_concurrent: Maximum concurrent crawling sessions
        **kwargs: Additional arguments for AdvancedWebCrawler
        
    Returns:
        List of AdvancedCrawlResult
    """
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def crawl_with_semaphore(url: str) -> AdvancedCrawlResult:
        async with semaphore:
            return await crawl_single_page_advanced(url, **kwargs)
    
    tasks = [crawl_with_semaphore(url) for url in urls]
    return await asyncio.gather(*tasks, return_exceptions=False)


async def smart_crawl_url_advanced(url: str, **kwargs) -> List[AdvancedCrawlResult]:
    """
    Smart crawl URL with AdvancedWebCrawler - handles sitemaps, text files, and single pages.
    
    This function replaces the deprecated smart_crawl_url_enhanced from smart_crawler_factory.
    
    Args:
        url: The URL to crawl (can be sitemap, text file, or single page)
        **kwargs: Additional arguments for AdvancedWebCrawler
        
    Returns:
        List of AdvancedCrawlResult objects
    """
    import xml.etree.ElementTree as ET
    import aiohttp
    
    # Determine URL type and handle accordingly
    if url.endswith('.xml') or 'sitemap' in url.lower():
        return await _crawl_sitemap_advanced(url, **kwargs)
    elif url.endswith('.txt'):
        return await _crawl_text_file_advanced(url, **kwargs)
    else:
        # Single page crawl
        result = await crawl_single_page_advanced(url, **kwargs)
        return [result]


async def crawl_recursive_internal_links_advanced(
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
    **kwargs
) -> List[AdvancedCrawlResult]:
    """
    Recursively crawl internal links using AdvancedWebCrawler.
    
    This function replaces the deprecated crawl_recursive_internal_links_enhanced 
    from smart_crawler_factory.
    
    Args:
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent crawling sessions
        **kwargs: Additional arguments for AdvancedWebCrawler
        
    Returns:
        List of AdvancedCrawlResult objects
    """
    from urllib.parse import urldefrag
    from crawl4ai import CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
    
    # For now, use batch crawling as a simplified approach
    # In the future, this could be enhanced with actual recursive link following
    return await batch_crawl_advanced(start_urls, max_concurrent, **kwargs)


async def _crawl_sitemap_advanced(sitemap_url: str, **kwargs) -> List[AdvancedCrawlResult]:
    """Crawl all URLs from a sitemap using AdvancedWebCrawler."""
    import xml.etree.ElementTree as ET
    import aiohttp
    
    results = []
    
    try:
        # Download and parse sitemap
        async with aiohttp.ClientSession() as session:
            async with session.get(sitemap_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download sitemap: {sitemap_url}")
                    return results
                
                sitemap_content = await response.text()
        
        # Parse XML
        root = ET.fromstring(sitemap_content)
        
        # Handle different sitemap formats
        urls = []
        if root.tag.endswith('sitemapindex'):
            # Sitemap index - get sub-sitemaps
            for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc is not None:
                    # Recursively crawl sub-sitemap
                    sub_results = await _crawl_sitemap_advanced(loc.text, **kwargs)
                    results.extend(sub_results)
        else:
            # Regular sitemap - get URLs
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc is not None:
                    urls.append(loc.text)
        
        # Crawl all URLs with concurrency control
        if urls:
            max_concurrent = kwargs.get('max_concurrent', 5)
            # Limit to first 50 URLs to prevent overwhelming the system
            limited_urls = urls[:50]
            results = await batch_crawl_advanced(limited_urls, max_concurrent, **kwargs)
        
    except Exception as e:
        logger.error(f"Error crawling sitemap {sitemap_url}: {str(e)}")
    
    return results


async def _crawl_text_file_advanced(text_file_url: str, **kwargs) -> List[AdvancedCrawlResult]:
    """Crawl URLs from a text file using AdvancedWebCrawler."""
    import aiohttp
    
    results = []
    
    try:
        # Download text file
        async with aiohttp.ClientSession() as session:
            async with session.get(text_file_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download text file: {text_file_url}")
                    return results
                
                content = await response.text()
        
        # Extract URLs from text
        urls = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line.startswith('http://') or line.startswith('https://')):
                urls.append(line)
        
        # Crawl all URLs with concurrency control
        if urls:
            max_concurrent = kwargs.get('max_concurrent', 5)
            # Limit to first 50 URLs to prevent overwhelming the system
            limited_urls = urls[:50]
            results = await batch_crawl_advanced(limited_urls, max_concurrent, **kwargs)
        
    except Exception as e:
        logger.error(f"Error crawling text file {text_file_url}: {str(e)}")
    
    return results