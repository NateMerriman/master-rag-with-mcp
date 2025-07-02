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
    from .enhanced_crawler_config import (
        DocumentationFramework,
        config_manager,
        detect_framework
    )
    ENHANCED_CONFIG_AVAILABLE = True
except ImportError:
    # Fallback if enhanced config not available
    ENHANCED_CONFIG_AVAILABLE = False

# Import quality validation system
try:
    from .crawler_quality_validation import (
        ContentQualityValidator,
        QualityValidationResult,
        validate_crawler_output
    )
    QUALITY_VALIDATION_AVAILABLE = True
except ImportError:
    # Fallback if quality validation not available
    QUALITY_VALIDATION_AVAILABLE = False

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
                 enable_quality_validation: bool = True):
        """
        Initialize the AdvancedWebCrawler.
        
        Args:
            headless: Run browser in headless mode
            timeout_ms: Timeout for page loading and content waiting
            custom_css_selectors: Override default CSS selectors for content targeting
            enable_quality_validation: Enable automated quality validation
        """
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.crawler: Optional[AsyncWebCrawler] = None
        self.enable_quality_validation = enable_quality_validation and QUALITY_VALIDATION_AVAILABLE
        
        # Initialize quality validator if available
        if self.enable_quality_validation:
            self.quality_validator = ContentQualityValidator()
        else:
            self.quality_validator = None
        
        # CSS selectors for main content areas (inspired by reference implementation)
        self.css_selectors = custom_css_selectors or [
            "main article",  # Primary target for most documentation sites
            "main",          # Fallback to main element
            "article",       # Fallback to article element  
            ".content",      # Common content class
            "[role='main']"  # Semantic main role
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
        
        This now integrates with the existing enhanced crawler configuration system
        for precise framework-specific targeting and surgical content extraction.
        
        The configuration targets:
        - NoExtractionStrategy for clean content filtering
        - DefaultMarkdownGenerator optimized for semantic chunking  
        - Framework-specific CSS selectors for main content areas
        - Comprehensive excluded selectors for navigation removal
        """
        
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
            
    async def crawl_single_page(self, url: str) -> AdvancedCrawlResult:
        """
        Crawl a single page and extract clean markdown for DocumentIngestionPipeline.
        
        Uses a two-stage approach for optimal extraction:
        1. Initial HTML fetch for framework detection (if enhanced config available)
        2. Optimized crawl with framework-specific configuration
        
        Args:
            url: URL to crawl
            
        Returns:
            AdvancedCrawlResult with clean markdown optimized for semantic chunking
        """
        
        if not self.crawler:
            await self._initialize_crawler()
            
        start_time = time.time()
        
        try:
            # Stage 2: Optimized crawl with framework-specific configuration
            logger.info(f"Stage 2: Optimized extraction from {url}")
            
            # Create optimized configuration based on detected framework
            run_config = self._create_optimized_run_config(url)
            
            # Execute the optimized crawl with Playwright + NoExtractionStrategy
            result = await self.crawler.arun(url=url, config=run_config)
            
            extraction_time_ms = (time.time() - start_time) * 1000
            
            if result.success and result.markdown:
                
                # Post-process the markdown to remove unwanted elements
                cleaned_markdown = self._post_process_markdown(result.markdown)

                # Extract metadata for pipeline processing
                word_count = len(cleaned_markdown.split())
                title = self._extract_title_from_markdown(cleaned_markdown)
                framework_detected = self._detect_framework_from_content(cleaned_markdown, url)
                
                # Calculate quality indicators
                content_ratio = self._calculate_content_ratio(cleaned_markdown)
                has_dynamic = self._detect_dynamic_content_indicators(cleaned_markdown)
                
                # Perform quality validation if enabled
                quality_validation = None
                quality_passed = True
                quality_score = 1.0
                
                if self.enable_quality_validation and self.quality_validator:
                    logger.info(f"Running quality validation for {url}")
                    quality_validation = self.quality_validator.validate_content(cleaned_markdown, url)
                    quality_passed = quality_validation.passed
                    quality_score = quality_validation.score
                    
                    if not quality_passed:
                        logger.warning(f"Quality validation failed for {url}: {', '.join(quality_validation.issues)}")
                    else:
                        logger.info(f"Quality validation passed for {url}: {quality_validation.category} ({quality_score:.3f})")
                
                logger.info(f"Successfully extracted {word_count} words from {url} in {extraction_time_ms:.1f}ms")
                
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
                error_msg = result.error_message or "No content extracted"
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
        Performs additional cleaning on the extracted markdown content.
        This function is designed to remove specific patterns of unwanted links
        or reformat content that trafilatura might not handle perfectly.
        """
        cleaned_lines = []
        for line in markdown.splitlines():
            stripped_line = line.strip()
            # Heuristic: if a line contains only a link or a list of links, remove it
            # This targets lines like '* [ Link Text ](http://example.com)'
            if re.fullmatch(r'^\s*[-*+]\s*\[[^\]]+\]\([^\)]+\)\s*
                continue
            # Remove lines that are just a link
            if re.fullmatch(r"""[^\)]+\)""", stripped_line):
                continue
            # Remove lines that are just a URL
            if re.fullmatch(r'https?://\S+', stripped_line):
                continue
            
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


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
    return await asyncio.gather(*tasks, return_exceptions=False), stripped_line): # Bullet point followed by a link
                continue
            # Remove lines that are just a link
            if re.fullmatch(r"""[^\)]+\)""", stripped_line):
                continue
            # Remove lines that are just a URL
            if re.fullmatch(r'https?://\S+', stripped_line):
                continue
            
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


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