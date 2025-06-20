#!/usr/bin/env python3
"""
Smart crawler factory for documentation site optimization.

This module provides intelligent crawler configuration creation with framework-specific
optimizations, quality validation, and fallback mechanisms.
"""

from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler, 
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from urllib.parse import urldefrag

# Enhanced imports with Docker compatibility
import sys
import os

# Ensure current directory is in Python path for Docker compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from .enhanced_crawler_config import (
    DocumentationFramework, 
    config_manager,
    detect_framework,
    get_optimized_config,
    get_quality_thresholds
)
from .content_quality import (
    ContentQualityMetrics,
    calculate_content_quality,
    should_retry_extraction,
    log_quality_metrics
)

logger = logging.getLogger(__name__)


def safe_get_quality_category(quality_metrics) -> str:
    """
    Safely extract quality category from quality_metrics.
    
    This defensive function handles cases where quality_metrics might be:
    - None
    - A ContentQualityMetrics object (expected)
    - A string (unexpected but should be handled gracefully)
    - Any other type (unexpected)
    
    Returns:
        str: The quality category or a safe default
    """
    if quality_metrics is None:
        return "unknown"
    
    # Check if it has the expected attribute
    if hasattr(quality_metrics, 'quality_category'):
        return quality_metrics.quality_category
    
    # If it's a string, it might be a serialized version - log and return it
    if isinstance(quality_metrics, str):
        logger.warning(f"quality_metrics is unexpectedly a string: {quality_metrics}")
        return "error_string"
    
    # For any other unexpected type
    logger.error(f"quality_metrics has unexpected type {type(quality_metrics)}: {quality_metrics}")
    return "error_type"


@dataclass
class CrawlResult:
    """Enhanced crawl result with quality metrics."""
    
    # Original crawl4ai result data
    url: str
    html: str
    cleaned_html: str
    markdown: str
    extracted_content: str
    success: bool
    status_code: int
    
    # Enhanced extraction data
    framework: DocumentationFramework
    quality_metrics: Optional[ContentQualityMetrics]
    used_fallback: bool
    extraction_attempts: int
    
    # Performance data
    total_time_seconds: float
    framework_detection_time_ms: float
    quality_analysis_time_ms: float


class SmartCrawlerFactory:
    """Factory for creating optimized crawler configurations."""
    
    def __init__(self):
        self.config_manager = config_manager
        self._crawler_cache: Dict[str, AsyncWebCrawler] = {}
        
        # Fallback CSS selectors for when framework-specific configs fail
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
    
    def create_optimized_config(self, url: str, framework: DocumentationFramework, 
                              custom_overrides: Optional[Dict] = None) -> CrawlerRunConfig:
        """
        Create an optimized Crawl4AI configuration for the given framework.
        
        Args:
            url: The URL being crawled
            framework: The detected documentation framework
            custom_overrides: Optional custom configuration overrides
            
        Returns:
            Optimized CrawlerRunConfig
        """
        # Get base framework configuration
        base_config = self.config_manager.create_crawl4ai_config(framework, custom_overrides)
        
        # Add additional optimizations based on URL patterns
        enhanced_config = self._enhance_config_for_url(base_config, url, framework)
        
        return enhanced_config
    
    def _enhance_config_for_url(self, config: CrawlerRunConfig, url: str, 
                               framework: DocumentationFramework) -> CrawlerRunConfig:
        """Enhance configuration based on specific URL patterns."""
        
        domain = urlparse(url).netloc.lower()
        
        # Domain-specific optimizations
        if 'n8n.io' in domain:
            # n8n-specific optimizations
            config.word_count_threshold = 20  # n8n has very verbose navigation
            config.exclude_external_links = True
            
        elif 'readme.io' in domain or 'docs.' in domain:
            # API documentation sites typically have more structured content
            config.word_count_threshold = 15
            config.exclude_external_links = True
            
        elif 'github.io' in domain:
            # GitHub Pages often use Jekyll or other static generators
            config.word_count_threshold = 12
            
        return config
    
    def create_fallback_config(self, attempt_number: int) -> CrawlerRunConfig:
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
            css_selector=css_selector,
            excluded_tags=["nav", "header", "footer", "aside", "script", "style"],
            word_count_threshold=10,  # Lower threshold for fallback
            exclude_external_links=True,
            exclude_social_media_links=True,
            process_iframes=False
        )


class EnhancedCrawler:
    """Enhanced crawler with quality validation and fallback mechanisms."""
    
    def __init__(self, max_fallback_attempts: int = 3):
        self.factory = SmartCrawlerFactory()
        self.max_fallback_attempts = max_fallback_attempts
        self._crawler: Optional[AsyncWebCrawler] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._crawler = AsyncWebCrawler(verbose=False)
        await self._crawler.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._crawler:
            await self._crawler.__aexit__(exc_type, exc_val, exc_tb)
    
    async def crawl_single_page_enhanced(self, url: str, 
                                       custom_config: Optional[Dict] = None) -> CrawlResult:
        """
        Crawl a single page with enhanced extraction and quality validation.
        
        Args:
            url: The URL to crawl
            custom_config: Optional custom configuration overrides
            
        Returns:
            CrawlResult with quality metrics and extraction details
        """
        import time
        start_time = time.time()
        
        if not self._crawler:
            raise RuntimeError("Crawler not initialized. Use async context manager.")
        
        # Step 1: Framework detection
        framework_start = time.time()
        framework = detect_framework(url)
        framework_detection_time = (time.time() - framework_start) * 1000
        
        logger.info(f"Detected framework {framework.value} for {url}")
        
        # Step 2: Create optimized configuration
        config = self.factory.create_optimized_config(url, framework, custom_config)
        
        # Step 3: Attempt extraction with quality validation
        best_result = None
        extraction_attempts = 0
        used_fallback = False
        
        for attempt in range(1, self.max_fallback_attempts + 2):  # +1 for primary attempt
            extraction_attempts = attempt
            
            try:
                if attempt == 1:
                    # Primary extraction attempt
                    current_config = config
                    logger.info(f"Primary extraction attempt for {url}")
                else:
                    # Fallback extraction attempts
                    current_config = self.factory.create_fallback_config(attempt - 1)
                    used_fallback = True
                    logger.info(f"Fallback extraction attempt {attempt-1} for {url}")
                
                # Perform the crawl
                result = await self._crawler.arun(url=url, config=current_config)
                
                if not result.success:
                    logger.warning(f"Crawl failed for {url}: {result.status_code}")
                    continue
                
                # Step 4: Quality validation
                quality_start = time.time()
                quality_metrics = calculate_content_quality(result.markdown)
                quality_analysis_time = (time.time() - quality_start) * 1000
                
                # Defensive check to ensure quality_metrics is the correct type
                if not hasattr(quality_metrics, 'quality_category'):
                    logger.error(f"calculate_content_quality returned unexpected type: {type(quality_metrics)} = {quality_metrics}")
                    # Return a failure result if quality calculation is broken
                    return CrawlResult(
                        url=url,
                        html="",
                        cleaned_html="",
                        markdown="",
                        extracted_content="",
                        success=False,
                        status_code=0,
                        framework=framework,
                        quality_metrics=None,
                        used_fallback=used_fallback,
                        extraction_attempts=extraction_attempts,
                        total_time_seconds=time.time() - start_time,
                        framework_detection_time_ms=framework_detection_time,
                        quality_analysis_time_ms=0
                    )
                
                # Log quality metrics
                log_quality_metrics(quality_metrics, url, framework.value)
                
                # Step 5: Check if quality is acceptable
                if not should_retry_extraction(quality_metrics) or attempt >= self.max_fallback_attempts + 1:
                    # Quality is acceptable or we've exhausted attempts
                    best_result = CrawlResult(
                        url=url,
                        html=result.html,
                        cleaned_html=result.cleaned_html,
                        markdown=result.markdown,
                        extracted_content=result.extracted_content,
                        success=True,
                        status_code=result.status_code,
                        framework=framework,
                        quality_metrics=quality_metrics,
                        used_fallback=used_fallback,
                        extraction_attempts=extraction_attempts,
                        total_time_seconds=time.time() - start_time,
                        framework_detection_time_ms=framework_detection_time,
                        quality_analysis_time_ms=quality_analysis_time
                    )
                    break
                else:
                    logger.info(f"Quality too low for {url}, trying fallback approach")
                    # Store this result in case fallbacks also fail
                    if best_result is None:
                        best_result = CrawlResult(
                            url=url,
                            html=result.html,
                            cleaned_html=result.cleaned_html,
                            markdown=result.markdown,
                            extracted_content=result.extracted_content,
                            success=True,
                            status_code=result.status_code,
                            framework=framework,
                            quality_metrics=quality_metrics,
                            used_fallback=used_fallback,
                            extraction_attempts=extraction_attempts,
                            total_time_seconds=time.time() - start_time,
                            framework_detection_time_ms=framework_detection_time,
                            quality_analysis_time_ms=quality_analysis_time
                        )
                
            except Exception as e:
                logger.error(f"Extraction attempt {attempt} failed for {url}: {str(e)}")
                continue
        
        # Return best result or failure
        if best_result:
            quality_info = safe_get_quality_category(best_result.quality_metrics)
            logger.info(f"Successfully extracted {url} with quality {quality_info}")
            return best_result
        else:
            # All attempts failed
            return CrawlResult(
                url=url,
                html="",
                cleaned_html="",
                markdown="",
                extracted_content="",
                success=False,
                status_code=0,
                framework=framework,
                quality_metrics=None,
                used_fallback=used_fallback,
                extraction_attempts=extraction_attempts,
                total_time_seconds=time.time() - start_time,
                framework_detection_time_ms=framework_detection_time,
                quality_analysis_time_ms=0
            )
    
    async def smart_crawl_url_enhanced(self, url: str, 
                                     custom_config: Optional[Dict] = None) -> List[CrawlResult]:
        """
        Smart crawl URL with enhanced extraction - handles sitemaps, text files, and single pages.
        
        Args:
            url: The URL to crawl (can be sitemap, text file, or single page)
            custom_config: Optional custom configuration overrides
            
        Returns:
            List of CrawlResult objects
        """
        # Determine URL type and handle accordingly
        if url.endswith('.xml') or 'sitemap' in url.lower():
            return await self._crawl_sitemap_enhanced(url, custom_config)
        elif url.endswith('.txt'):
            return await self._crawl_text_file_enhanced(url, custom_config)
        else:
            # Single page crawl
            result = await self.crawl_single_page_enhanced(url, custom_config)
            return [result]
    
    async def _crawl_sitemap_enhanced(self, sitemap_url: str, 
                                    custom_config: Optional[Dict] = None) -> List[CrawlResult]:
        """Crawl all URLs from a sitemap with enhanced extraction."""
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
                        sub_results = await self._crawl_sitemap_enhanced(loc.text, custom_config)
                        results.extend(sub_results)
            else:
                # Regular sitemap - get URLs
                for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                    loc = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc is not None:
                        urls.append(loc.text)
            
            # Crawl all URLs with concurrency control
            if urls:
                semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
                
                async def crawl_with_semaphore(url):
                    async with semaphore:
                        return await self.crawl_single_page_enhanced(url, custom_config)
                
                tasks = [crawl_with_semaphore(url) for url in urls[:50]]  # Limit to first 50 URLs
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out exceptions
                results = [r for r in results if isinstance(r, CrawlResult)]
            
        except Exception as e:
            logger.error(f"Error crawling sitemap {sitemap_url}: {str(e)}")
        
        return results
    
    async def _crawl_text_file_enhanced(self, text_file_url: str, 
                                      custom_config: Optional[Dict] = None) -> List[CrawlResult]:
        """Crawl URLs from a text file with enhanced extraction."""
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
                semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
                
                async def crawl_with_semaphore(url):
                    async with semaphore:
                        return await self.crawl_single_page_enhanced(url, custom_config)
                
                tasks = [crawl_with_semaphore(url) for url in urls[:50]]  # Limit to first 50 URLs
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Filter out exceptions
                results = [r for r in results if isinstance(r, CrawlResult)]
            
        except Exception as e:
            logger.error(f"Error crawling text file {text_file_url}: {str(e)}")
        
        return results

    async def crawl_recursive_internal_links_enhanced(
        self,
        start_urls: List[str],
        max_depth: int = 3,
        max_concurrent: int = 10,
        custom_config: Optional[Dict] = None
    ) -> List[CrawlResult]:
        """
        Recursively crawl internal links from start URLs up to a maximum depth using enhanced crawling.
        Only crawls URLs that start with the same path as the original start URLs.
        
        Args:
            start_urls: List of starting URLs
            max_depth: Maximum recursion depth
            max_concurrent: Maximum number of concurrent browser sessions
            custom_config: Optional custom configuration
            
        Returns:
            List of CrawlResult objects with enhanced content extraction
        """
        if not self._crawler:
            raise RuntimeError("Crawler not initialized. Use 'async with' statement.")
        
        # Base configuration
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Add custom configuration if provided
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(run_config, key):
                    setattr(run_config, key, value)
        
        # Create memory adaptive dispatcher
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent,
        )
        
        visited = set()
        
        def normalize_url(url):
            return urldefrag(url)[0]
        
        def is_within_allowed_paths(url: str, allowed_prefixes: List[str]) -> bool:
            """Check if URL starts with any of the allowed path prefixes."""
            normalized = normalize_url(url)
            return any(normalized.startswith(prefix) for prefix in allowed_prefixes)
        
        # Extract allowed path prefixes from start URLs
        allowed_prefixes = [normalize_url(url) for url in start_urls]
        logger.info(f"🔒 Enhanced recursive crawling restricted to paths: {allowed_prefixes}")
        
        current_urls = set([normalize_url(u) for u in start_urls])
        enhanced_results = []
        
        for depth in range(max_depth):
            urls_to_crawl = [
                normalize_url(url)
                for url in current_urls
                if normalize_url(url) not in visited
            ]
            if not urls_to_crawl:
                break
            
            logger.info(f"Enhanced recursive crawl - Depth {depth + 1}/{max_depth}: {len(urls_to_crawl)} URLs")
            
            # Use arun_many for parallel crawling like the original
            results = await self._crawler.arun_many(
                urls=urls_to_crawl, config=run_config, dispatcher=dispatcher
            )
            next_level_urls = set()
            
            for result in results:
                norm_url = normalize_url(result.url)
                visited.add(norm_url)
                
                if result.success and result.markdown:
                    # Create enhanced result with quality analysis
                    enhanced_result = await self._create_enhanced_result(result, custom_config)
                    enhanced_results.append(enhanced_result)
                    
                    # Extract internal links for next level (only within allowed paths)
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if (next_url not in visited and 
                            is_within_allowed_paths(next_url, allowed_prefixes)):
                            next_level_urls.add(next_url)
            
            current_urls = next_level_urls
        
        logger.info(f"Enhanced recursive crawl complete: {len(enhanced_results)} pages extracted")
        return enhanced_results
    
    async def _create_enhanced_result(self, crawl_result, custom_config: Optional[Dict] = None) -> CrawlResult:
        """Create an enhanced CrawlResult with quality analysis from a basic crawl result."""
        try:
            # Detect framework and get quality metrics
            framework = detect_framework(crawl_result.url, crawl_result.html)
            quality_metrics = calculate_content_quality(crawl_result.markdown)
            
            # Defensive check to ensure quality_metrics is the correct type
            if not hasattr(quality_metrics, 'quality_category'):
                logger.error(f"calculate_content_quality returned unexpected type in _create_enhanced_result: {type(quality_metrics)} = {quality_metrics}")
                quality_metrics = None
            
            # Log quality metrics
            if quality_metrics:
                log_quality_metrics(quality_metrics, crawl_result.url, framework.value)
            
            # Create enhanced result
            enhanced_result = CrawlResult(
                url=crawl_result.url,
                html=crawl_result.html,
                cleaned_html=getattr(crawl_result, 'cleaned_html', ''),
                markdown=crawl_result.markdown,
                extracted_content=getattr(crawl_result, 'extracted_content', crawl_result.markdown),
                success=crawl_result.success,
                status_code=getattr(crawl_result, 'status_code', 200),
                framework=framework,
                quality_metrics=quality_metrics,
                extraction_attempts=1,
                used_fallback=False,
                total_time_seconds=0.0,
                framework_detection_time_ms=0.0,
                quality_analysis_time_ms=0.0
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error creating enhanced result for {crawl_result.url}: {str(e)}")
            # Return a basic enhanced result if analysis fails
            return CrawlResult(
                url=crawl_result.url,
                html=crawl_result.html,
                cleaned_html=getattr(crawl_result, 'cleaned_html', ''),
                markdown=crawl_result.markdown,
                extracted_content=getattr(crawl_result, 'extracted_content', crawl_result.markdown),
                success=crawl_result.success,
                status_code=getattr(crawl_result, 'status_code', 200),
                framework=DocumentationFramework.GENERIC,
                quality_metrics=None,
                extraction_attempts=1,
                used_fallback=False,
                total_time_seconds=0.0,
                framework_detection_time_ms=0.0,
                quality_analysis_time_ms=0.0
            )


# Convenience functions for easy usage

async def crawl_single_page_enhanced(url: str, custom_config: Optional[Dict] = None) -> CrawlResult:
    """Convenience function for enhanced single page crawling."""
    async with EnhancedCrawler() as crawler:
        return await crawler.crawl_single_page_enhanced(url, custom_config)


async def smart_crawl_url_enhanced(url: str, custom_config: Optional[Dict] = None) -> List[CrawlResult]:
    """Convenience function for enhanced smart URL crawling."""
    async with EnhancedCrawler() as crawler:
        return await crawler.smart_crawl_url_enhanced(url, custom_config)


async def crawl_recursive_internal_links_enhanced(
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
    custom_config: Optional[Dict] = None
) -> List[CrawlResult]:
    """Convenience function for enhanced recursive crawling."""
    async with EnhancedCrawler() as crawler:
        return await crawler.crawl_recursive_internal_links_enhanced(
            start_urls, max_depth, max_concurrent, custom_config
        )