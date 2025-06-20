# Detailed Implementation Plan: Enhanced RAG Crawler for Documentation Sites

**Author:** Manus AI  
**Date:** June 17, 2025  
**Version:** 1.0

## Implementation Overview

This comprehensive implementation plan provides step-by-step instructions for enhancing your RAG system's crawler to effectively handle documentation sites. The plan is designed for junior developers and includes detailed code examples, configuration templates, and testing procedures.

The implementation follows a phased approach that delivers immediate improvements while building toward more sophisticated capabilities. Each phase includes specific deliverables, code modifications, and validation steps to ensure successful deployment.

## Phase 1: Enhanced CSS Targeting Implementation

### Step 1: Create Documentation Site Configuration System

The first step involves creating a flexible configuration system that can handle different documentation frameworks. This system will serve as the foundation for all enhanced extraction capabilities.

**1.1 Create the Configuration Module**

Create a new file `enhanced_crawler_config.py` in your project directory:

```python
#!/usr/bin/env python3
"""
Enhanced crawler configuration for documentation sites.

This module provides framework-specific configurations for optimal content extraction
from various documentation platforms.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re
from urllib.parse import urlparse


class DocumentationFramework(Enum):
    """Supported documentation frameworks."""
    MATERIAL_DESIGN = "material_design"
    README_IO = "readme_io"
    GITBOOK = "gitbook"
    DOCUSAURUS = "docusaurus"
    SPHINX = "sphinx"
    GENERIC = "generic"


@dataclass
class ExtractionConfig:
    """Configuration for content extraction from a specific framework."""
    
    # Target elements for main content
    target_elements: List[str] = field(default_factory=list)
    
    # Tags to exclude from extraction
    excluded_tags: List[str] = field(default_factory=list)
    
    # CSS selectors to exclude
    excluded_selectors: List[str] = field(default_factory=list)
    
    # Minimum word count threshold
    word_count_threshold: int = 15
    
    # Framework-specific settings
    exclude_external_links: bool = True
    exclude_social_media_links: bool = True
    process_iframes: bool = False
    
    # Quality validation thresholds
    min_content_ratio: float = 0.6  # Minimum content-to-navigation ratio
    max_link_density: float = 0.3   # Maximum links per word


class DocumentationSiteConfig:
    """Configuration manager for documentation site extraction."""
    
    def __init__(self):
        self.framework_configs = self._initialize_framework_configs()
        self.domain_patterns = self._initialize_domain_patterns()
    
    def _initialize_framework_configs(self) -> Dict[DocumentationFramework, ExtractionConfig]:
        """Initialize framework-specific extraction configurations."""
        
        configs = {}
        
        # Material Design (used by n8n, MkDocs sites)
        configs[DocumentationFramework.MATERIAL_DESIGN] = ExtractionConfig(
            target_elements=[
                "main.md-main",
                "article.md-content__inner",
                ".md-content",
                "main[class*='md-']",
                "article[class*='md-']"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            excluded_selectors=[
                ".md-sidebar",
                ".md-nav",
                ".md-header",
                ".md-footer",
                ".md-search",
                "[class*='navigation']",
                "[class*='breadcrumb']"
            ],
            word_count_threshold=20
        )
        
        # ReadMe.io (used by VirusTotal, many API docs)
        configs[DocumentationFramework.README_IO] = ExtractionConfig(
            target_elements=[
                "main.rm-Guides",
                "article",
                ".rm-Content",
                "main[class*='rm-']",
                ".content-body"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            excluded_selectors=[
                ".rm-Sidebar",
                ".content-toc",
                ".hub-sidebar",
                "[class*='navigation']",
                "[class*='sidebar']",
                ".toc"
            ],
            word_count_threshold=25
        )
        
        # GitBook
        configs[DocumentationFramework.GITBOOK] = ExtractionConfig(
            target_elements=[
                "main",
                "article",
                ".page-content",
                ".gitbook-content",
                "[data-testid='content']"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            excluded_selectors=[
                ".gitbook-sidebar",
                ".page-toc",
                ".navigation",
                ".summary",
                "[class*='sidebar']"
            ],
            word_count_threshold=15
        )
        
        # Docusaurus (Facebook's documentation platform)
        configs[DocumentationFramework.DOCUSAURUS] = ExtractionConfig(
            target_elements=[
                "main",
                "article",
                ".main-wrapper",
                ".docMainContainer",
                "[class*='docItemContainer']"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            excluded_selectors=[
                ".sidebar",
                ".table-of-contents",
                ".navbar",
                ".footer",
                "[class*='sidebar']",
                "[class*='toc']"
            ],
            word_count_threshold=20
        )
        
        # Sphinx (Python documentation standard)
        configs[DocumentationFramework.SPHINX] = ExtractionConfig(
            target_elements=[
                "main",
                ".document",
                ".body",
                ".section",
                "[role='main']"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            excluded_selectors=[
                ".sphinxsidebar",
                ".related",
                ".footer",
                ".header",
                "[class*='sidebar']",
                "[class*='navigation']"
            ],
            word_count_threshold=15
        )
        
        # Generic fallback configuration
        configs[DocumentationFramework.GENERIC] = ExtractionConfig(
            target_elements=[
                "main",
                "article",
                ".content",
                ".main-content",
                ".documentation",
                ".docs-content",
                "[role='main']"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            excluded_selectors=[
                ".sidebar",
                ".navigation",
                ".nav",
                ".toc",
                ".table-of-contents",
                ".breadcrumb",
                "[class*='sidebar']",
                "[class*='nav']"
            ],
            word_count_threshold=10
        )
        
        return configs
    
    def _initialize_domain_patterns(self) -> Dict[str, DocumentationFramework]:
        """Initialize domain-to-framework mapping patterns."""
        
        return {
            # Material Design patterns
            r".*\.readthedocs\.io": DocumentationFramework.MATERIAL_DESIGN,
            r"docs\..*": DocumentationFramework.MATERIAL_DESIGN,
            
            # ReadMe.io patterns
            r".*\.readme\.io": DocumentationFramework.README_IO,
            r"docs\.virustotal\.com": DocumentationFramework.README_IO,
            
            # GitBook patterns
            r".*\.gitbook\.io": DocumentationFramework.GITBOOK,
            r".*\.gitbook\.com": DocumentationFramework.GITBOOK,
            
            # Docusaurus patterns (common with Facebook/Meta projects)
            r".*\.netlify\.app": DocumentationFramework.DOCUSAURUS,
            r".*\.vercel\.app": DocumentationFramework.DOCUSAURUS,
        }
    
    def detect_framework(self, url: str, html_content: str = None) -> DocumentationFramework:
        """
        Detect the documentation framework for a given URL.
        
        Args:
            url: The URL to analyze
            html_content: Optional HTML content for analysis
            
        Returns:
            Detected documentation framework
        """
        
        domain = urlparse(url).netloc.lower()
        
        # Check domain patterns first
        for pattern, framework in self.domain_patterns.items():
            if re.match(pattern, domain):
                return framework
        
        # If HTML content is available, analyze it
        if html_content:
            framework = self._analyze_html_content(html_content)
            if framework != DocumentationFramework.GENERIC:
                return framework
        
        # Special case detection for known sites
        if "docs.n8n.io" in domain:
            return DocumentationFramework.MATERIAL_DESIGN
        elif "docs.virustotal.com" in domain:
            return DocumentationFramework.README_IO
        
        return DocumentationFramework.GENERIC
    
    def _analyze_html_content(self, html_content: str) -> DocumentationFramework:
        """Analyze HTML content to detect framework."""
        
        html_lower = html_content.lower()
        
        # Material Design indicators
        if any(indicator in html_lower for indicator in [
            'class="md-', 'mkdocs', 'material-theme'
        ]):
            return DocumentationFramework.MATERIAL_DESIGN
        
        # ReadMe.io indicators
        if any(indicator in html_lower for indicator in [
            'class="rm-', 'readme.io', 'hub-sidebar'
        ]):
            return DocumentationFramework.README_IO
        
        # GitBook indicators
        if any(indicator in html_lower for indicator in [
            'gitbook', 'class="gitbook-', 'data-testid="content"'
        ]):
            return DocumentationFramework.GITBOOK
        
        # Docusaurus indicators
        if any(indicator in html_lower for indicator in [
            'docusaurus', 'class="docMainContainer"', 'docItemContainer'
        ]):
            return DocumentationFramework.DOCUSAURUS
        
        # Sphinx indicators
        if any(indicator in html_lower for indicator in [
            'sphinx', 'class="sphinxsidebar"', 'class="document"'
        ]):
            return DocumentationFramework.SPHINX
        
        return DocumentationFramework.GENERIC
    
    def get_extraction_config(self, framework: DocumentationFramework) -> ExtractionConfig:
        """Get extraction configuration for a specific framework."""
        return self.framework_configs.get(framework, self.framework_configs[DocumentationFramework.GENERIC])
    
    def get_config_for_url(self, url: str, html_content: str = None) -> Tuple[DocumentationFramework, ExtractionConfig]:
        """
        Get the optimal extraction configuration for a URL.
        
        Args:
            url: The URL to configure for
            html_content: Optional HTML content for analysis
            
        Returns:
            Tuple of (detected_framework, extraction_config)
        """
        
        framework = self.detect_framework(url, html_content)
        config = self.get_extraction_config(framework)
        
        return framework, config


# Global configuration instance
doc_site_config = DocumentationSiteConfig()
```

This configuration system provides the foundation for framework-specific extraction. The system automatically detects documentation frameworks and applies appropriate extraction configurations.

**1.2 Create the Enhanced Crawler Configuration Factory**

Create a new file `smart_crawler_factory.py`:

```python
#!/usr/bin/env python3
"""
Smart crawler configuration factory for enhanced documentation extraction.

This module creates optimized Crawl4AI configurations based on site analysis
and framework detection.
"""

from crawl4ai import CrawlerRunConfig, CacheMode
from typing import Optional, Dict, Any
import logging
from enhanced_crawler_config import doc_site_config, DocumentationFramework, ExtractionConfig

logger = logging.getLogger(__name__)


class SmartCrawlerConfigFactory:
    """Factory for creating optimized crawler configurations."""
    
    def __init__(self):
        self.config_cache = {}
    
    def create_config(
        self,
        url: str,
        html_content: Optional[str] = None,
        cache_mode: CacheMode = CacheMode.BYPASS,
        custom_overrides: Optional[Dict[str, Any]] = None
    ) -> CrawlerRunConfig:
        """
        Create an optimized crawler configuration for a specific URL.
        
        Args:
            url: Target URL for crawling
            html_content: Optional HTML content for framework detection
            cache_mode: Caching mode for the crawler
            custom_overrides: Optional configuration overrides
            
        Returns:
            Optimized CrawlerRunConfig instance
        """
        
        # Get framework-specific configuration
        framework, extraction_config = doc_site_config.get_config_for_url(url, html_content)
        
        logger.info(f"Detected framework {framework.value} for URL: {url}")
        
        # Create base configuration
        config_params = {
            'cache_mode': cache_mode,
            'stream': False,
        }
        
        # Apply framework-specific settings
        if extraction_config.target_elements:
            config_params['target_elements'] = extraction_config.target_elements
            logger.debug(f"Target elements: {extraction_config.target_elements}")
        
        if extraction_config.excluded_tags:
            config_params['excluded_tags'] = extraction_config.excluded_tags
            logger.debug(f"Excluded tags: {extraction_config.excluded_tags}")
        
        if extraction_config.word_count_threshold:
            config_params['word_count_threshold'] = extraction_config.word_count_threshold
        
        # Link filtering
        config_params['exclude_external_links'] = extraction_config.exclude_external_links
        config_params['exclude_social_media_links'] = extraction_config.exclude_social_media_links
        
        # iframe processing
        config_params['process_iframes'] = extraction_config.process_iframes
        
        # Apply custom overrides if provided
        if custom_overrides:
            config_params.update(custom_overrides)
            logger.debug(f"Applied custom overrides: {custom_overrides}")
        
        # Create and return the configuration
        config = CrawlerRunConfig(**config_params)
        
        logger.info(f"Created optimized config for {framework.value} framework")
        return config
    
    def create_config_with_css_exclusions(
        self,
        url: str,
        html_content: Optional[str] = None,
        additional_exclusions: Optional[list] = None
    ) -> CrawlerRunConfig:
        """
        Create configuration with CSS-based exclusions for complex sites.
        
        This method is useful for sites that need additional CSS-based filtering
        beyond the standard tag exclusions.
        
        Args:
            url: Target URL for crawling
            html_content: Optional HTML content for framework detection
            additional_exclusions: Additional CSS selectors to exclude
            
        Returns:
            CrawlerRunConfig with CSS exclusions applied
        """
        
        framework, extraction_config = doc_site_config.get_config_for_url(url, html_content)
        
        # Start with base configuration
        base_config = self.create_config(url, html_content)
        
        # For complex sites, we might need to use css_selector instead of target_elements
        # This is a more aggressive approach that only extracts matching content
        if framework in [DocumentationFramework.MATERIAL_DESIGN, DocumentationFramework.README_IO]:
            
            # Use the first target element as the primary CSS selector
            if extraction_config.target_elements:
                primary_selector = extraction_config.target_elements[0]
                
                config_params = {
                    'css_selector': primary_selector,
                    'cache_mode': CacheMode.BYPASS,
                    'stream': False,
                    'excluded_tags': extraction_config.excluded_tags,
                    'word_count_threshold': extraction_config.word_count_threshold,
                    'exclude_external_links': extraction_config.exclude_external_links,
                    'exclude_social_media_links': extraction_config.exclude_social_media_links,
                }
                
                logger.info(f"Using CSS selector approach with: {primary_selector}")
                return CrawlerRunConfig(**config_params)
        
        return base_config
    
    def get_quality_validation_config(self, extraction_config: ExtractionConfig) -> Dict[str, float]:
        """
        Get quality validation thresholds for an extraction configuration.
        
        Args:
            extraction_config: The extraction configuration
            
        Returns:
            Dictionary of quality validation parameters
        """
        
        return {
            'min_content_ratio': extraction_config.min_content_ratio,
            'max_link_density': extraction_config.max_link_density,
            'min_word_count': extraction_config.word_count_threshold
        }


# Global factory instance
smart_config_factory = SmartCrawlerConfigFactory()
```

This factory creates optimized Crawl4AI configurations based on automatic framework detection. The factory handles the complexity of configuration generation while providing a simple interface for the crawler.

### Step 2: Integrate Enhanced Configuration with Existing Crawler

Now we need to modify your existing `crawl4ai_mcp.py` file to use the enhanced configuration system. This integration maintains backward compatibility while adding enhanced capabilities.

**2.1 Modify the Crawler Implementation**

Add the following imports to the top of your `crawl4ai_mcp.py` file:

```python
# Add these imports after your existing imports
from smart_crawler_factory import smart_config_factory
from enhanced_crawler_config import doc_site_config, DocumentationFramework
```

**2.2 Create Enhanced Crawling Functions**

Add these new functions to your `crawl4ai_mcp.py` file:

```python
async def crawl_single_page_enhanced(ctx: Context, url: str, use_enhanced_config: bool = True) -> str:
    """
    Enhanced version of crawl_single_page with smart configuration.
    
    This tool uses intelligent framework detection and optimized extraction
    configurations to improve content quality from documentation sites.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
        use_enhanced_config: Whether to use enhanced extraction configuration
        
    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # Create enhanced configuration
        if use_enhanced_config:
            run_config = smart_config_factory.create_config(url)
            logger.info(f"Using enhanced configuration for {url}")
        else:
            # Fallback to original configuration
            run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
            logger.info(f"Using standard configuration for {url}")
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            # Validate content quality
            quality_metrics = validate_content_quality(result.markdown, url)
            
            # If quality is poor and we used enhanced config, try CSS selector approach
            if (quality_metrics['content_ratio'] < 0.5 and use_enhanced_config):
                logger.warning(f"Poor content quality detected for {url}, trying CSS selector approach")
                
                css_config = smart_config_factory.create_config_with_css_exclusions(url)
                result = await crawler.arun(url=url, config=css_config)
                
                if result.success and result.markdown:
                    quality_metrics = validate_content_quality(result.markdown, url)
                    logger.info(f"CSS selector approach quality: {quality_metrics['content_ratio']:.2f}")
            
            # Chunk the content using enhanced chunking
            chunks = smart_chunk_markdown(result.markdown)
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata with quality metrics
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                meta["enhanced_extraction"] = use_enhanced_config
                meta["content_quality"] = quality_metrics
                metadatas.append(meta)
            
            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}
            
            # Add to Supabase
            add_documents_to_supabase(
                supabase_client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document,
            )
            
            return json.dumps(
                {
                    "success": True,
                    "url": url,
                    "chunks_stored": len(chunks),
                    "content_length": len(result.markdown),
                    "enhanced_extraction": use_enhanced_config,
                    "quality_metrics": quality_metrics,
                    "links_count": {
                        "internal": len(result.links.get("internal", [])),
                        "external": len(result.links.get("external", [])),
                    },
                },
                indent=2,
            )
        else:
            return json.dumps(
                {"success": False, "url": url, "error": result.error_message}, indent=2
            )
    except Exception as e:
        logger.error(f"Error in enhanced crawling for {url}: {str(e)}")
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)


def validate_content_quality(markdown_content: str, url: str) -> Dict[str, float]:
    """
    Validate the quality of extracted content.
    
    Args:
        markdown_content: The extracted markdown content
        url: The source URL
        
    Returns:
        Dictionary containing quality metrics
    """
    
    if not markdown_content:
        return {
            "content_ratio": 0.0,
            "link_density": 1.0,
            "avg_paragraph_length": 0.0,
            "quality_score": 0.0
        }
    
    # Split into lines and analyze
    lines = markdown_content.split('\n')
    total_lines = len(lines)
    
    # Count different types of content
    link_lines = 0
    content_lines = 0
    short_lines = 0
    
    total_words = 0
    content_words = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        words = len(line.split())
        total_words += words
        
        # Check if line is primarily links
        if line.count('[') > 2 or line.count('](') > 1:
            link_lines += 1
        elif words < 3:
            short_lines += 1
        else:
            content_lines += 1
            content_words += words
    
    # Calculate metrics
    content_ratio = content_lines / max(total_lines, 1)
    link_density = link_lines / max(total_lines, 1)
    avg_paragraph_length = content_words / max(content_lines, 1)
    
    # Calculate overall quality score
    quality_score = (
        content_ratio * 0.5 +
        (1 - link_density) * 0.3 +
        min(avg_paragraph_length / 20, 1.0) * 0.2
    )
    
    return {
        "content_ratio": content_ratio,
        "link_density": link_density,
        "avg_paragraph_length": avg_paragraph_length,
        "quality_score": quality_score,
        "total_words": total_words,
        "content_words": content_words
    }


async def smart_crawl_url_enhanced(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
    use_enhanced_config: bool = True,
) -> str:
    """
    Enhanced version of smart_crawl_url with intelligent configuration.
    
    This tool automatically detects the URL type and applies enhanced extraction
    configurations optimized for documentation sites.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 5000)
        use_enhanced_config: Whether to use enhanced extraction configuration
        
    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Get the crawler and Supabase client from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        crawl_results = []
        crawl_type = "webpage"
        
        # Detect URL type and use appropriate crawl method
        if is_txt(url):
            # For text files, use simple crawl with enhanced config
            crawl_results = await crawl_markdown_file_enhanced(crawler, url, use_enhanced_config)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel with enhanced config
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps(
                    {"success": False, "url": url, "error": "No URLs found in sitemap"},
                    indent=2,
                )
            crawl_results = await crawl_batch_enhanced(
                crawler, sitemap_urls, max_concurrent=max_concurrent, use_enhanced_config=use_enhanced_config
            )
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl with enhanced config
            crawl_results = await crawl_recursive_internal_links_enhanced(
                crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent, use_enhanced_config=use_enhanced_config
            )
            crawl_type = "webpage"
        
        if not crawl_results:
            return json.dumps(
                {"success": False, "url": url, "error": "No content found"}, indent=2
            )
        
        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0
        quality_summary = {"total_pages": 0, "avg_quality": 0.0, "poor_quality_pages": 0}
        
        for doc in crawl_results:
            source_url = doc["url"]
            md = doc["markdown"]
            quality_metrics = doc.get("quality_metrics", {})
            
            # Update quality summary
            quality_summary["total_pages"] += 1
            quality_summary["avg_quality"] += quality_metrics.get("quality_score", 0.0)
            if quality_metrics.get("quality_score", 0.0) < 0.5:
                quality_summary["poor_quality_pages"] += 1
            
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            
            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata with enhanced information
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = urlparse(source_url).netloc
                meta["crawl_type"] = crawl_type
                meta["enhanced_extraction"] = use_enhanced_config
                meta["page_quality"] = quality_metrics
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)
                
                chunk_count += 1
        
        # Calculate final quality summary
        if quality_summary["total_pages"] > 0:
            quality_summary["avg_quality"] /= quality_summary["total_pages"]
        
        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in crawl_results:
            url_to_full_document[doc["url"]] = doc["markdown"]
        
        # Add to Supabase
        batch_size = 20
        add_documents_to_supabase(
            supabase_client,
            urls,
            chunk_numbers,
            contents,
            metadatas,
            url_to_full_document,
            batch_size=batch_size,
        )
        
        return json.dumps(
            {
                "success": True,
                "url": url,
                "crawl_type": crawl_type,
                "enhanced_extraction": use_enhanced_config,
                "pages_crawled": len(crawl_results),
                "chunks_stored": chunk_count,
                "quality_summary": quality_summary,
                "urls_crawled": [doc["url"] for doc in crawl_results][:5]
                + (["..."] if len(crawl_results) > 5 else []),
            },
            indent=2,
        )
    except Exception as e:
        logger.error(f"Error in enhanced smart crawling for {url}: {str(e)}")
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)
```

This integration provides enhanced crawling capabilities while maintaining backward compatibility with your existing system. The enhanced functions include quality validation and automatic fallback mechanisms.

### Step 3: Create Enhanced Batch Processing Functions

Add these enhanced batch processing functions to handle multiple URLs with optimized configurations:

```python
async def crawl_markdown_file_enhanced(
    crawler: AsyncWebCrawler, url: str, use_enhanced_config: bool = True
) -> List[Dict[str, Any]]:
    """
    Enhanced crawl for .txt or markdown files.
    
    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file
        use_enhanced_config: Whether to use enhanced configuration
        
    Returns:
        List of dictionaries with URL, markdown content, and quality metrics
    """
    
    if use_enhanced_config:
        crawl_config = smart_config_factory.create_config(url)
    else:
        crawl_config = CrawlerRunConfig()
    
    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        quality_metrics = validate_content_quality(result.markdown, url)
        return [{
            "url": url, 
            "markdown": result.markdown,
            "quality_metrics": quality_metrics
        }]
    else:
        logger.error(f"Failed to crawl {url}: {result.error_message}")
        return []


async def crawl_batch_enhanced(
    crawler: AsyncWebCrawler, 
    urls: List[str], 
    max_concurrent: int = 10,
    use_enhanced_config: bool = True
) -> List[Dict[str, Any]]:
    """
    Enhanced batch crawl multiple URLs in parallel.
    
    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions
        use_enhanced_config: Whether to use enhanced configuration
        
    Returns:
        List of dictionaries with URL, markdown content, and quality metrics
    """
    
    # Group URLs by likely framework for batch optimization
    url_groups = {}
    for url in urls:
        framework, _ = doc_site_config.get_config_for_url(url)
        if framework not in url_groups:
            url_groups[framework] = []
        url_groups[framework].append(url)
    
    all_results = []
    
    for framework, framework_urls in url_groups.items():
        logger.info(f"Processing {len(framework_urls)} URLs with {framework.value} configuration")
        
        if use_enhanced_config:
            # Use framework-specific configuration for the batch
            sample_url = framework_urls[0]
            crawl_config = smart_config_factory.create_config(sample_url)
        else:
            crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent,
        )
        
        results = await crawler.arun_many(
            urls=framework_urls, config=crawl_config, dispatcher=dispatcher
        )
        
        # Process results with quality validation
        for result in results:
            if result.success and result.markdown:
                quality_metrics = validate_content_quality(result.markdown, result.url)
                all_results.append({
                    "url": result.url,
                    "markdown": result.markdown,
                    "quality_metrics": quality_metrics
                })
            else:
                logger.warning(f"Failed to crawl {result.url}: {result.error_message}")
    
    return all_results


async def crawl_recursive_internal_links_enhanced(
    crawler: AsyncWebCrawler,
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
    use_enhanced_config: bool = True,
) -> List[Dict[str, Any]]:
    """
    Enhanced recursive crawl of internal links with smart configuration.
    
    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions
        use_enhanced_config: Whether to use enhanced configuration
        
    Returns:
        List of dictionaries with URL, markdown content, and quality metrics
    """
    
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )
    
    visited = set()
    
    def normalize_url(url):
        return urldefrag(url)[0]
    
    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []
    
    # Detect framework from the first URL for consistent configuration
    if use_enhanced_config and start_urls:
        framework, extraction_config = doc_site_config.get_config_for_url(start_urls[0])
        run_config = smart_config_factory.create_config(start_urls[0])
        logger.info(f"Using {framework.value} configuration for recursive crawl")
    else:
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    
    for depth in range(max_depth):
        urls_to_crawl = [
            normalize_url(url)
            for url in current_urls
            if normalize_url(url) not in visited
        ]
        if not urls_to_crawl:
            break
        
        logger.info(f"Crawling depth {depth + 1}: {len(urls_to_crawl)} URLs")
        
        results = await crawler.arun_many(
            urls=urls_to_crawl, config=run_config, dispatcher=dispatcher
        )
        next_level_urls = set()
        
        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)
            
            if result.success and result.markdown:
                quality_metrics = validate_content_quality(result.markdown, result.url)
                results_all.append({
                    "url": result.url,
                    "markdown": result.markdown,
                    "quality_metrics": quality_metrics
                })
                
                # Only follow internal links if quality is reasonable
                if quality_metrics.get("quality_score", 0) > 0.3:
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited:
                            next_level_urls.add(next_url)
                else:
                    logger.warning(f"Poor quality content from {result.url}, not following links")
            else:
                logger.warning(f"Failed to crawl {result.url}: {result.error_message}")
        
        current_urls = next_level_urls
        logger.info(f"Found {len(next_level_urls)} new URLs for next depth")
    
    return results_all
```

These enhanced functions provide intelligent batch processing with framework-aware configuration and quality validation. The system automatically optimizes extraction for different documentation frameworks while maintaining performance.

### Step 4: Register Enhanced Tools with MCP

Add the enhanced tools to your MCP server by registering them as new tools. Add this code to your `crawl4ai_mcp.py` file:

```python
@mcp.tool()
async def crawl_single_page_enhanced_tool(ctx: Context, url: str, use_enhanced_config: bool = True) -> str:
    """
    Enhanced single page crawler with intelligent framework detection.
    
    This tool automatically detects documentation frameworks and applies optimized
    extraction configurations to improve content quality. It includes quality
    validation and automatic fallback mechanisms.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
        use_enhanced_config: Whether to use enhanced extraction (default: True)
        
    Returns:
        JSON summary of crawling operation with quality metrics
    """
    return await crawl_single_page_enhanced(ctx, url, use_enhanced_config)


@mcp.tool()
async def smart_crawl_url_enhanced_tool(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
    use_enhanced_config: bool = True,
) -> str:
    """
    Enhanced smart crawler with framework-aware configuration.
    
    This tool provides intelligent crawling with automatic framework detection,
    optimized extraction configurations, and quality validation. It supports
    sitemaps, text files, and recursive webpage crawling with enhanced content
    extraction specifically designed for documentation sites.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for webpages (default: 3)
        max_concurrent: Maximum concurrent browser sessions (default: 10)
        chunk_size: Maximum chunk size in characters (default: 5000)
        use_enhanced_config: Whether to use enhanced extraction (default: True)
        
    Returns:
        JSON summary with crawl results and quality metrics
    """
    return await smart_crawl_url_enhanced(
        ctx, url, max_depth, max_concurrent, chunk_size, use_enhanced_config
    )


@mcp.tool()
async def analyze_site_framework(ctx: Context, url: str) -> str:
    """
    Analyze a documentation site to detect its framework and optimal configuration.
    
    This diagnostic tool helps understand how the enhanced crawler will handle
    a specific documentation site. It provides framework detection results,
    recommended configurations, and extraction previews.
    
    Args:
        ctx: The MCP server provided context
        url: URL to analyze
        
    Returns:
        JSON analysis of the site's framework and recommended configuration
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        
        # First, do a basic crawl to get HTML content
        basic_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=basic_config)
        
        if not result.success:
            return json.dumps({
                "success": False,
                "url": url,
                "error": f"Failed to crawl site: {result.error_message}"
            }, indent=2)
        
        # Detect framework
        framework, extraction_config = doc_site_config.get_config_for_url(url, result.html)
        
        # Create enhanced configuration
        enhanced_config = smart_config_factory.create_config(url, result.html)
        
        # Analyze content quality with both approaches
        basic_quality = validate_content_quality(result.markdown, url)
        
        # Try enhanced extraction
        enhanced_result = await crawler.arun(url=url, config=enhanced_config)
        enhanced_quality = {}
        if enhanced_result.success:
            enhanced_quality = validate_content_quality(enhanced_result.markdown, url)
        
        # Prepare analysis results
        analysis = {
            "success": True,
            "url": url,
            "detected_framework": framework.value,
            "extraction_config": {
                "target_elements": extraction_config.target_elements,
                "excluded_tags": extraction_config.excluded_tags,
                "excluded_selectors": extraction_config.excluded_selectors,
                "word_count_threshold": extraction_config.word_count_threshold,
            },
            "quality_comparison": {
                "basic_extraction": basic_quality,
                "enhanced_extraction": enhanced_quality,
                "improvement": {
                    "content_ratio": enhanced_quality.get("content_ratio", 0) - basic_quality.get("content_ratio", 0),
                    "quality_score": enhanced_quality.get("quality_score", 0) - basic_quality.get("quality_score", 0),
                }
            },
            "recommendations": []
        }
        
        # Add recommendations based on analysis
        if enhanced_quality.get("quality_score", 0) > basic_quality.get("quality_score", 0):
            analysis["recommendations"].append("Enhanced extraction provides better content quality")
        
        if enhanced_quality.get("content_ratio", 0) > 0.7:
            analysis["recommendations"].append("Excellent content extraction ratio achieved")
        elif enhanced_quality.get("content_ratio", 0) > 0.5:
            analysis["recommendations"].append("Good content extraction ratio, consider CSS selector approach for further improvement")
        else:
            analysis["recommendations"].append("Consider using CSS selector approach or custom configuration")
        
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        logger.error(f"Error analyzing site framework for {url}: {str(e)}")
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)
```

These new tools provide enhanced crawling capabilities while maintaining the same interface as your existing tools. Users can choose between enhanced and standard extraction based on their needs.

## Testing and Validation

### Step 5: Create Test Scripts

Create a test script to validate the enhanced extraction capabilities:

```python
#!/usr/bin/env python3
"""
Test script for enhanced crawler functionality.

This script tests the enhanced crawler against known problematic documentation sites
to validate improvements in content quality.
"""

import asyncio
import json
from crawl4ai import AsyncWebCrawler, BrowserConfig
from smart_crawler_factory import smart_config_factory
from enhanced_crawler_config import doc_site_config

async def test_enhanced_extraction():
    """Test enhanced extraction against problematic sites."""
    
    # Test URLs - known problematic documentation sites
    test_urls = [
        "https://docs.n8n.io/",
        "https://docs.virustotal.com/docs/how-it-works",
        "https://docs.python.org/3/tutorial/",  # Sphinx
        "https://reactjs.org/docs/getting-started.html",  # Docusaurus
    ]
    
    browser_config = BrowserConfig(headless=True, verbose=False)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        
        for url in test_urls:
            print(f"\n{'='*60}")
            print(f"Testing: {url}")
            print(f"{'='*60}")
            
            # Test framework detection
            framework, config = doc_site_config.get_config_for_url(url)
            print(f"Detected Framework: {framework.value}")
            print(f"Target Elements: {config.target_elements}")
            print(f"Excluded Tags: {config.excluded_tags}")
            
            # Test basic extraction
            basic_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
            basic_result = await crawler.arun(url=url, config=basic_config)
            
            # Test enhanced extraction
            enhanced_config = smart_config_factory.create_config(url)
            enhanced_result = await crawler.arun(url=url, config=enhanced_config)
            
            if basic_result.success and enhanced_result.success:
                # Compare quality
                from crawl4ai_mcp import validate_content_quality
                
                basic_quality = validate_content_quality(basic_result.markdown, url)
                enhanced_quality = validate_content_quality(enhanced_result.markdown, url)
                
                print(f"\nQuality Comparison:")
                print(f"Basic - Content Ratio: {basic_quality['content_ratio']:.2f}, Quality Score: {basic_quality['quality_score']:.2f}")
                print(f"Enhanced - Content Ratio: {enhanced_quality['content_ratio']:.2f}, Quality Score: {enhanced_quality['quality_score']:.2f}")
                print(f"Improvement: {enhanced_quality['quality_score'] - basic_quality['quality_score']:.2f}")
                
                # Show content samples
                print(f"\nBasic Extraction Sample (first 300 chars):")
                print(basic_result.markdown[:300] + "...")
                
                print(f"\nEnhanced Extraction Sample (first 300 chars):")
                print(enhanced_result.markdown[:300] + "...")
                
            else:
                print(f"Extraction failed - Basic: {basic_result.success}, Enhanced: {enhanced_result.success}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_extraction())
```

Save this as `test_enhanced_crawler.py` and run it to validate your implementation.

### Step 6: Configuration Validation

Create a configuration validation script:

```python
#!/usr/bin/env python3
"""
Configuration validation script for enhanced crawler.

This script validates that all framework configurations are properly set up
and can be applied successfully.
"""

from enhanced_crawler_config import doc_site_config, DocumentationFramework
from smart_crawler_factory import smart_config_factory

def validate_configurations():
    """Validate all framework configurations."""
    
    print("Validating Enhanced Crawler Configurations")
    print("=" * 50)
    
    # Test framework detection
    test_cases = [
        ("https://docs.n8n.io/", DocumentationFramework.MATERIAL_DESIGN),
        ("https://docs.virustotal.com/", DocumentationFramework.README_IO),
        ("https://docs.python.org/", DocumentationFramework.SPHINX),
        ("https://example.gitbook.io/", DocumentationFramework.GITBOOK),
        ("https://unknown-site.com/docs/", DocumentationFramework.GENERIC),
    ]
    
    for url, expected_framework in test_cases:
        detected_framework, config = doc_site_config.get_config_for_url(url)
        
        print(f"\nURL: {url}")
        print(f"Expected: {expected_framework.value}")
        print(f"Detected: {detected_framework.value}")
        print(f"Match: {'✓' if detected_framework == expected_framework else '✗'}")
        
        # Validate configuration
        if config.target_elements:
            print(f"Target Elements: {len(config.target_elements)} configured")
        else:
            print("⚠️  No target elements configured")
        
        if config.excluded_tags:
            print(f"Excluded Tags: {len(config.excluded_tags)} configured")
        else:
            print("⚠️  No excluded tags configured")
    
    # Test configuration factory
    print(f"\n{'='*50}")
    print("Testing Configuration Factory")
    print(f"{'='*50}")
    
    for url, _ in test_cases:
        try:
            crawler_config = smart_config_factory.create_config(url)
            print(f"✓ Successfully created config for {url}")
        except Exception as e:
            print(f"✗ Failed to create config for {url}: {e}")

if __name__ == "__main__":
    validate_configurations()
```

Save this as `validate_configurations.py` and run it to ensure your configurations are working correctly.

This completes Phase 1 of the implementation. The enhanced CSS targeting system provides immediate improvements to content quality while establishing the foundation for more advanced capabilities in future phases.


## Phase 2: Content Validation and Quality Monitoring

### Step 7: Implement Advanced Content Quality Assessment

The second phase focuses on implementing sophisticated content quality assessment and monitoring capabilities. This system provides real-time feedback on extraction effectiveness and enables automatic optimization.

**7.1 Create Content Quality Analyzer**

Create a new file `content_quality_analyzer.py`:

```python
#!/usr/bin/env python3
"""
Advanced content quality analysis for enhanced crawler.

This module provides sophisticated content quality assessment capabilities
including semantic analysis, structure validation, and quality scoring.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ContentMetrics:
    """Comprehensive content quality metrics."""
    
    # Basic metrics
    total_words: int
    total_lines: int
    total_characters: int
    
    # Content classification
    content_words: int
    navigation_words: int
    code_words: int
    
    # Structure metrics
    headings_count: int
    paragraphs_count: int
    links_count: int
    code_blocks_count: int
    
    # Quality ratios
    content_ratio: float
    link_density: float
    code_ratio: float
    
    # Readability metrics
    avg_sentence_length: float
    avg_paragraph_length: float
    complexity_score: float
    
    # Overall quality
    quality_score: float
    quality_grade: str


class ContentQualityAnalyzer:
    """Advanced content quality analysis system."""
    
    def __init__(self):
        self.navigation_patterns = self._compile_navigation_patterns()
        self.content_patterns = self._compile_content_patterns()
        self.quality_thresholds = self._define_quality_thresholds()
    
    def _compile_navigation_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for identifying navigation content."""
        
        patterns = [
            # Link-heavy lines
            re.compile(r'^\s*\[.*?\]\(.*?\)\s*$'),
            re.compile(r'^\s*\*\s*\[.*?\]\(.*?\)\s*$'),
            
            # Breadcrumb patterns
            re.compile(r'.*?>\s*.*?>\s*.*'),
            re.compile(r'.*?/\s*.*?/\s*.*'),
            
            # Navigation menu patterns
            re.compile(r'^\s*(Home|Docs|API|Guide|Tutorial|About)\s*$', re.IGNORECASE),
            re.compile(r'^\s*(Previous|Next|Back|Forward)\s*$', re.IGNORECASE),
            
            # Table of contents patterns
            re.compile(r'^\s*\d+\.\s*\[.*?\]\(.*?\)\s*$'),
            re.compile(r'^\s*-\s*\[.*?\]\(.*?\)\s*$'),
            
            # Short navigation text
            re.compile(r'^\s*[A-Za-z\s]{1,20}\s*$'),
        ]
        
        return patterns
    
    def _compile_content_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for identifying substantive content."""
        
        patterns = [
            # Technical explanations
            re.compile(r'\b(function|method|parameter|argument|return|example)\b', re.IGNORECASE),
            re.compile(r'\b(configure|install|setup|implement|deploy)\b', re.IGNORECASE),
            
            # Documentation language
            re.compile(r'\b(documentation|tutorial|guide|reference|manual)\b', re.IGNORECASE),
            re.compile(r'\b(step|process|procedure|workflow|instruction)\b', re.IGNORECASE),
            
            # Technical terms
            re.compile(r'\b(API|SDK|CLI|HTTP|JSON|XML|database|server)\b', re.IGNORECASE),
            re.compile(r'\b(authentication|authorization|configuration|integration)\b', re.IGNORECASE),
        ]
        
        return patterns
    
    def _define_quality_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Define quality thresholds for different content grades."""
        
        return {
            'excellent': {
                'content_ratio': 0.8,
                'link_density': 0.2,
                'avg_paragraph_length': 50,
                'quality_score': 0.85
            },
            'good': {
                'content_ratio': 0.6,
                'link_density': 0.4,
                'avg_paragraph_length': 30,
                'quality_score': 0.7
            },
            'fair': {
                'content_ratio': 0.4,
                'link_density': 0.6,
                'avg_paragraph_length': 20,
                'quality_score': 0.5
            },
            'poor': {
                'content_ratio': 0.2,
                'link_density': 0.8,
                'avg_paragraph_length': 10,
                'quality_score': 0.3
            }
        }
    
    def analyze_content(self, markdown_content: str, url: str = "") -> ContentMetrics:
        """
        Perform comprehensive content quality analysis.
        
        Args:
            markdown_content: The markdown content to analyze
            url: Optional URL for context
            
        Returns:
            ContentMetrics object with detailed analysis
        """
        
        if not markdown_content:
            return self._create_empty_metrics()
        
        # Basic text analysis
        lines = [line.strip() for line in markdown_content.split('\n') if line.strip()]
        words = markdown_content.split()
        
        # Classify content
        content_classification = self._classify_content_lines(lines)
        structure_analysis = self._analyze_structure(markdown_content)
        readability_metrics = self._calculate_readability(lines, words)
        
        # Calculate ratios and scores
        total_words = len(words)
        content_words = content_classification['content_words']
        navigation_words = content_classification['navigation_words']
        
        content_ratio = content_words / max(total_words, 1)
        link_density = structure_analysis['links_count'] / max(len(lines), 1)
        code_ratio = content_classification['code_words'] / max(total_words, 1)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            content_ratio, link_density, readability_metrics['avg_paragraph_length']
        )
        
        quality_grade = self._determine_quality_grade(quality_score, content_ratio, link_density)
        
        return ContentMetrics(
            total_words=total_words,
            total_lines=len(lines),
            total_characters=len(markdown_content),
            content_words=content_words,
            navigation_words=navigation_words,
            code_words=content_classification['code_words'],
            headings_count=structure_analysis['headings_count'],
            paragraphs_count=structure_analysis['paragraphs_count'],
            links_count=structure_analysis['links_count'],
            code_blocks_count=structure_analysis['code_blocks_count'],
            content_ratio=content_ratio,
            link_density=link_density,
            code_ratio=code_ratio,
            avg_sentence_length=readability_metrics['avg_sentence_length'],
            avg_paragraph_length=readability_metrics['avg_paragraph_length'],
            complexity_score=readability_metrics['complexity_score'],
            quality_score=quality_score,
            quality_grade=quality_grade
        )
    
    def _classify_content_lines(self, lines: List[str]) -> Dict[str, int]:
        """Classify lines as content, navigation, or code."""
        
        content_words = 0
        navigation_words = 0
        code_words = 0
        
        for line in lines:
            words_in_line = len(line.split())
            
            # Check if line is code
            if self._is_code_line(line):
                code_words += words_in_line
                continue
            
            # Check if line is navigation
            if self._is_navigation_line(line):
                navigation_words += words_in_line
                continue
            
            # Check if line contains substantive content
            if self._is_content_line(line):
                content_words += words_in_line
            else:
                # Default to navigation for ambiguous short lines
                if words_in_line < 5:
                    navigation_words += words_in_line
                else:
                    content_words += words_in_line
        
        return {
            'content_words': content_words,
            'navigation_words': navigation_words,
            'code_words': code_words
        }
    
    def _is_navigation_line(self, line: str) -> bool:
        """Check if a line is primarily navigation content."""
        
        # Check against navigation patterns
        for pattern in self.navigation_patterns:
            if pattern.match(line):
                return True
        
        # High link density indicates navigation
        link_count = line.count('[') + line.count('](')
        word_count = len(line.split())
        if word_count > 0 and (link_count / word_count) > 0.5:
            return True
        
        return False
    
    def _is_content_line(self, line: str) -> bool:
        """Check if a line contains substantive content."""
        
        # Check against content patterns
        content_indicators = 0
        for pattern in self.content_patterns:
            if pattern.search(line):
                content_indicators += 1
        
        # Lines with multiple content indicators are likely substantive
        if content_indicators >= 2:
            return True
        
        # Long lines with few links are likely content
        word_count = len(line.split())
        link_count = line.count('[') + line.count('](')
        
        if word_count > 15 and (link_count / max(word_count, 1)) < 0.3:
            return True
        
        return False
    
    def _is_code_line(self, line: str) -> bool:
        """Check if a line is code content."""
        
        # Code block indicators
        if line.startswith('```') or line.startswith('    '):
            return True
        
        # Inline code indicators
        if line.count('`') >= 2:
            return True
        
        # Programming language patterns
        code_patterns = [
            r'^\s*(def|function|class|import|from|var|let|const)\s+',
            r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[=:]\s*',
            r'^\s*[{}();]\s*$',
        ]
        
        for pattern in code_patterns:
            if re.match(pattern, line):
                return True
        
        return False
    
    def _analyze_structure(self, content: str) -> Dict[str, int]:
        """Analyze structural elements of the content."""
        
        # Count headings
        headings_count = len(re.findall(r'^#+\s+', content, re.MULTILINE))
        
        # Count paragraphs (double newlines)
        paragraphs_count = len(re.findall(r'\n\s*\n', content))
        
        # Count links
        links_count = len(re.findall(r'\[.*?\]\(.*?\)', content))
        
        # Count code blocks
        code_blocks_count = len(re.findall(r'```.*?```', content, re.DOTALL))
        
        return {
            'headings_count': headings_count,
            'paragraphs_count': max(paragraphs_count, 1),  # At least 1 paragraph
            'links_count': links_count,
            'code_blocks_count': code_blocks_count
        }
    
    def _calculate_readability(self, lines: List[str], words: List[str]) -> Dict[str, float]:
        """Calculate readability metrics."""
        
        # Calculate average sentence length
        sentences = []
        for line in lines:
            # Split by sentence endings
            line_sentences = re.split(r'[.!?]+', line)
            sentences.extend([s.strip() for s in line_sentences if s.strip()])
        
        if sentences:
            sentence_lengths = [len(sentence.split()) for sentence in sentences]
            avg_sentence_length = statistics.mean(sentence_lengths)
        else:
            avg_sentence_length = 0
        
        # Calculate average paragraph length
        paragraphs = [line for line in lines if len(line.split()) > 5]
        if paragraphs:
            paragraph_lengths = [len(paragraph.split()) for paragraph in paragraphs]
            avg_paragraph_length = statistics.mean(paragraph_lengths)
        else:
            avg_paragraph_length = 0
        
        # Calculate complexity score (based on word length and sentence structure)
        if words:
            avg_word_length = statistics.mean([len(word) for word in words])
            complexity_score = min((avg_word_length * avg_sentence_length) / 100, 1.0)
        else:
            complexity_score = 0
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_paragraph_length': avg_paragraph_length,
            'complexity_score': complexity_score
        }
    
    def _calculate_quality_score(self, content_ratio: float, link_density: float, avg_paragraph_length: float) -> float:
        """Calculate overall quality score."""
        
        # Weight different factors
        content_weight = 0.4
        link_weight = 0.3
        length_weight = 0.3
        
        # Normalize paragraph length (optimal around 30-50 words)
        length_score = min(avg_paragraph_length / 40, 1.0)
        
        # Calculate weighted score
        quality_score = (
            content_ratio * content_weight +
            (1 - link_density) * link_weight +
            length_score * length_weight
        )
        
        return min(quality_score, 1.0)
    
    def _determine_quality_grade(self, quality_score: float, content_ratio: float, link_density: float) -> str:
        """Determine quality grade based on metrics."""
        
        thresholds = self.quality_thresholds
        
        if (quality_score >= thresholds['excellent']['quality_score'] and
            content_ratio >= thresholds['excellent']['content_ratio']):
            return 'excellent'
        elif (quality_score >= thresholds['good']['quality_score'] and
              content_ratio >= thresholds['good']['content_ratio']):
            return 'good'
        elif (quality_score >= thresholds['fair']['quality_score'] and
              content_ratio >= thresholds['fair']['content_ratio']):
            return 'fair'
        else:
            return 'poor'
    
    def _create_empty_metrics(self) -> ContentMetrics:
        """Create empty metrics for invalid content."""
        
        return ContentMetrics(
            total_words=0, total_lines=0, total_characters=0,
            content_words=0, navigation_words=0, code_words=0,
            headings_count=0, paragraphs_count=0, links_count=0, code_blocks_count=0,
            content_ratio=0.0, link_density=1.0, code_ratio=0.0,
            avg_sentence_length=0.0, avg_paragraph_length=0.0, complexity_score=0.0,
            quality_score=0.0, quality_grade='poor'
        )
    
    def generate_quality_report(self, metrics: ContentMetrics, url: str = "") -> str:
        """Generate a human-readable quality report."""
        
        report = f"""
Content Quality Report
{'='*50}
URL: {url}
Quality Grade: {metrics.quality_grade.upper()}
Overall Score: {metrics.quality_score:.2f}/1.00

Content Analysis:
- Total Words: {metrics.total_words:,}
- Content Words: {metrics.content_words:,} ({metrics.content_ratio:.1%})
- Navigation Words: {metrics.navigation_words:,}
- Code Words: {metrics.code_words:,}

Structure Analysis:
- Headings: {metrics.headings_count}
- Paragraphs: {metrics.paragraphs_count}
- Links: {metrics.links_count}
- Code Blocks: {metrics.code_blocks_count}

Quality Metrics:
- Content Ratio: {metrics.content_ratio:.1%}
- Link Density: {metrics.link_density:.1%}
- Average Paragraph Length: {metrics.avg_paragraph_length:.1f} words
- Complexity Score: {metrics.complexity_score:.2f}

Recommendations:
"""
        
        # Add recommendations based on metrics
        if metrics.content_ratio < 0.5:
            report += "- Improve content extraction to reduce navigation noise\n"
        
        if metrics.link_density > 0.5:
            report += "- Consider more aggressive link filtering\n"
        
        if metrics.avg_paragraph_length < 20:
            report += "- Content may be fragmented, check extraction configuration\n"
        
        if metrics.quality_score < 0.5:
            report += "- Consider using CSS selector approach for better targeting\n"
        
        return report


# Global analyzer instance
content_analyzer = ContentQualityAnalyzer()
```

**7.2 Integrate Quality Analysis with Crawler**

Update your `crawl4ai_mcp.py` file to use the advanced quality analyzer:

```python
# Add this import at the top
from content_quality_analyzer import content_analyzer, ContentMetrics

# Replace the existing validate_content_quality function with this enhanced version
def validate_content_quality_enhanced(markdown_content: str, url: str) -> Dict[str, Any]:
    """
    Enhanced content quality validation using advanced analysis.
    
    Args:
        markdown_content: The extracted markdown content
        url: The source URL
        
    Returns:
        Dictionary containing comprehensive quality metrics
    """
    
    # Use the advanced analyzer
    metrics = content_analyzer.analyze_content(markdown_content, url)
    
    # Convert to dictionary for JSON serialization
    return {
        "total_words": metrics.total_words,
        "content_words": metrics.content_words,
        "navigation_words": metrics.navigation_words,
        "content_ratio": metrics.content_ratio,
        "link_density": metrics.link_density,
        "avg_paragraph_length": metrics.avg_paragraph_length,
        "quality_score": metrics.quality_score,
        "quality_grade": metrics.quality_grade,
        "headings_count": metrics.headings_count,
        "paragraphs_count": metrics.paragraphs_count,
        "links_count": metrics.links_count,
        "code_blocks_count": metrics.code_blocks_count,
        "complexity_score": metrics.complexity_score
    }

# Update the enhanced crawling functions to use the new quality validation
# Replace calls to validate_content_quality with validate_content_quality_enhanced
```

### Step 8: Implement Adaptive Configuration System

Create an adaptive configuration system that learns from quality feedback and automatically adjusts extraction parameters.

**8.1 Create Adaptive Configuration Manager**

Create a new file `adaptive_config_manager.py`:

```python
#!/usr/bin/env python3
"""
Adaptive configuration manager for enhanced crawler.

This module implements learning-based configuration optimization that improves
extraction quality over time through feedback analysis.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import statistics
from datetime import datetime, timedelta

from enhanced_crawler_config import DocumentationFramework, ExtractionConfig
from content_quality_analyzer import ContentMetrics

logger = logging.getLogger(__name__)


@dataclass
class ExtractionAttempt:
    """Record of an extraction attempt with results."""
    
    url: str
    framework: DocumentationFramework
    config_used: Dict
    quality_metrics: Dict
    timestamp: datetime
    success: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'url': self.url,
            'framework': self.framework.value,
            'config_used': self.config_used,
            'quality_metrics': self.quality_metrics,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExtractionAttempt':
        """Create from dictionary."""
        return cls(
            url=data['url'],
            framework=DocumentationFramework(data['framework']),
            config_used=data['config_used'],
            quality_metrics=data['quality_metrics'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            success=data['success']
        )


class AdaptiveConfigManager:
    """Manages adaptive configuration optimization."""
    
    def __init__(self, history_file: str = "extraction_history.json", max_history: int = 1000):
        self.history_file = Path(history_file)
        self.max_history = max_history
        self.extraction_history: deque = deque(maxlen=max_history)
        self.framework_stats: Dict[DocumentationFramework, Dict] = defaultdict(dict)
        
        # Load existing history
        self._load_history()
        
        # Configuration optimization parameters
        self.optimization_thresholds = {
            'min_attempts': 5,  # Minimum attempts before optimization
            'quality_threshold': 0.6,  # Target quality score
            'improvement_threshold': 0.1,  # Minimum improvement to apply changes
        }
    
    def record_extraction(
        self,
        url: str,
        framework: DocumentationFramework,
        config_used: ExtractionConfig,
        quality_metrics: ContentMetrics,
        success: bool
    ) -> None:
        """
        Record an extraction attempt for learning.
        
        Args:
            url: The URL that was crawled
            framework: Detected framework
            config_used: Configuration that was used
            quality_metrics: Quality metrics from the extraction
            success: Whether the extraction was successful
        """
        
        attempt = ExtractionAttempt(
            url=url,
            framework=framework,
            config_used=asdict(config_used),
            quality_metrics=asdict(quality_metrics),
            timestamp=datetime.now(),
            success=success
        )
        
        self.extraction_history.append(attempt)
        self._update_framework_stats(framework, quality_metrics, success)
        
        # Periodically save history
        if len(self.extraction_history) % 10 == 0:
            self._save_history()
        
        logger.debug(f"Recorded extraction attempt for {url} with quality score {quality_metrics.quality_score:.2f}")
    
    def get_optimized_config(
        self,
        framework: DocumentationFramework,
        base_config: ExtractionConfig,
        url: str = ""
    ) -> Tuple[ExtractionConfig, bool]:
        """
        Get optimized configuration based on historical performance.
        
        Args:
            framework: The documentation framework
            base_config: Base configuration to optimize
            url: Optional URL for context
            
        Returns:
            Tuple of (optimized_config, was_modified)
        """
        
        # Get recent attempts for this framework
        recent_attempts = self._get_recent_attempts(framework, days=30)
        
        if len(recent_attempts) < self.optimization_thresholds['min_attempts']:
            logger.debug(f"Insufficient data for {framework.value} optimization ({len(recent_attempts)} attempts)")
            return base_config, False
        
        # Analyze performance patterns
        optimization_suggestions = self._analyze_performance_patterns(recent_attempts)
        
        if not optimization_suggestions:
            logger.debug(f"No optimization suggestions for {framework.value}")
            return base_config, False
        
        # Apply optimizations
        optimized_config = self._apply_optimizations(base_config, optimization_suggestions)
        
        logger.info(f"Applied {len(optimization_suggestions)} optimizations to {framework.value} configuration")
        return optimized_config, True
    
    def _get_recent_attempts(self, framework: DocumentationFramework, days: int = 30) -> List[ExtractionAttempt]:
        """Get recent extraction attempts for a framework."""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            attempt for attempt in self.extraction_history
            if attempt.framework == framework and attempt.timestamp > cutoff_date
        ]
    
    def _analyze_performance_patterns(self, attempts: List[ExtractionAttempt]) -> List[Dict]:
        """Analyze performance patterns to identify optimization opportunities."""
        
        suggestions = []
        
        # Analyze quality scores
        quality_scores = [attempt.quality_metrics['quality_score'] for attempt in attempts if attempt.success]
        
        if quality_scores:
            avg_quality = statistics.mean(quality_scores)
            
            if avg_quality < self.optimization_thresholds['quality_threshold']:
                # Quality is below threshold, suggest improvements
                
                # Analyze content ratios
                content_ratios = [attempt.quality_metrics['content_ratio'] for attempt in attempts if attempt.success]
                if content_ratios and statistics.mean(content_ratios) < 0.5:
                    suggestions.append({
                        'type': 'increase_word_threshold',
                        'current_avg': statistics.mean(content_ratios),
                        'target': 0.6,
                        'confidence': 0.8
                    })
                
                # Analyze link density
                link_densities = [attempt.quality_metrics['link_density'] for attempt in attempts if attempt.success]
                if link_densities and statistics.mean(link_densities) > 0.4:
                    suggestions.append({
                        'type': 'add_exclusion_selectors',
                        'current_avg': statistics.mean(link_densities),
                        'target': 0.3,
                        'confidence': 0.7
                    })
                
                # Analyze paragraph lengths
                paragraph_lengths = [attempt.quality_metrics['avg_paragraph_length'] for attempt in attempts if attempt.success]
                if paragraph_lengths and statistics.mean(paragraph_lengths) < 20:
                    suggestions.append({
                        'type': 'adjust_target_elements',
                        'current_avg': statistics.mean(paragraph_lengths),
                        'target': 30,
                        'confidence': 0.6
                    })
        
        return suggestions
    
    def _apply_optimizations(self, base_config: ExtractionConfig, suggestions: List[Dict]) -> ExtractionConfig:
        """Apply optimization suggestions to configuration."""
        
        # Create a copy of the base configuration
        optimized_config = ExtractionConfig(
            target_elements=base_config.target_elements.copy(),
            excluded_tags=base_config.excluded_tags.copy(),
            excluded_selectors=base_config.excluded_selectors.copy(),
            word_count_threshold=base_config.word_count_threshold,
            exclude_external_links=base_config.exclude_external_links,
            exclude_social_media_links=base_config.exclude_social_media_links,
            process_iframes=base_config.process_iframes,
            min_content_ratio=base_config.min_content_ratio,
            max_link_density=base_config.max_link_density
        )
        
        for suggestion in suggestions:
            if suggestion['confidence'] < 0.5:
                continue  # Skip low-confidence suggestions
            
            if suggestion['type'] == 'increase_word_threshold':
                # Increase word count threshold to filter out short navigation text
                new_threshold = min(optimized_config.word_count_threshold + 5, 30)
                optimized_config.word_count_threshold = new_threshold
                logger.debug(f"Increased word threshold to {new_threshold}")
            
            elif suggestion['type'] == 'add_exclusion_selectors':
                # Add common navigation selectors
                additional_selectors = [
                    "[class*='breadcrumb']",
                    "[class*='pagination']",
                    "[class*='menu']",
                    ".toc-item",
                    ".nav-item"
                ]
                
                for selector in additional_selectors:
                    if selector not in optimized_config.excluded_selectors:
                        optimized_config.excluded_selectors.append(selector)
                        logger.debug(f"Added exclusion selector: {selector}")
            
            elif suggestion['type'] == 'adjust_target_elements':
                # Try more specific target elements
                if "main" in optimized_config.target_elements:
                    # Add more specific selectors
                    specific_selectors = [
                        "main article",
                        "main .content",
                        "[role='main'] article"
                    ]
                    
                    for selector in specific_selectors:
                        if selector not in optimized_config.target_elements:
                            optimized_config.target_elements.append(selector)
                            logger.debug(f"Added target element: {selector}")
        
        return optimized_config
    
    def _update_framework_stats(self, framework: DocumentationFramework, metrics: ContentMetrics, success: bool) -> None:
        """Update framework-specific statistics."""
        
        if framework not in self.framework_stats:
            self.framework_stats[framework] = {
                'total_attempts': 0,
                'successful_attempts': 0,
                'avg_quality_score': 0.0,
                'avg_content_ratio': 0.0,
                'last_updated': datetime.now()
            }
        
        stats = self.framework_stats[framework]
        stats['total_attempts'] += 1
        
        if success:
            stats['successful_attempts'] += 1
            
            # Update running averages
            current_avg_quality = stats['avg_quality_score']
            current_avg_content = stats['avg_content_ratio']
            
            # Simple exponential moving average
            alpha = 0.1  # Learning rate
            stats['avg_quality_score'] = (1 - alpha) * current_avg_quality + alpha * metrics.quality_score
            stats['avg_content_ratio'] = (1 - alpha) * current_avg_content + alpha * metrics.content_ratio
        
        stats['last_updated'] = datetime.now()
    
    def get_framework_performance_report(self) -> Dict[str, Dict]:
        """Get performance report for all frameworks."""
        
        report = {}
        
        for framework, stats in self.framework_stats.items():
            success_rate = stats['successful_attempts'] / max(stats['total_attempts'], 1)
            
            report[framework.value] = {
                'total_attempts': stats['total_attempts'],
                'success_rate': success_rate,
                'avg_quality_score': stats['avg_quality_score'],
                'avg_content_ratio': stats['avg_content_ratio'],
                'last_updated': stats['last_updated'].isoformat(),
                'performance_grade': self._calculate_performance_grade(stats)
            }
        
        return report
    
    def _calculate_performance_grade(self, stats: Dict) -> str:
        """Calculate performance grade for a framework."""
        
        if stats['total_attempts'] < 5:
            return 'insufficient_data'
        
        success_rate = stats['successful_attempts'] / stats['total_attempts']
        quality_score = stats['avg_quality_score']
        
        if success_rate > 0.9 and quality_score > 0.8:
            return 'excellent'
        elif success_rate > 0.8 and quality_score > 0.6:
            return 'good'
        elif success_rate > 0.6 and quality_score > 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _save_history(self) -> None:
        """Save extraction history to file."""
        
        try:
            history_data = {
                'extraction_history': [attempt.to_dict() for attempt in self.extraction_history],
                'framework_stats': {
                    framework.value: {
                        **stats,
                        'last_updated': stats['last_updated'].isoformat()
                    }
                    for framework, stats in self.framework_stats.items()
                },
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.debug(f"Saved extraction history to {self.history_file}")
            
        except Exception as e:
            logger.error(f"Failed to save extraction history: {e}")
    
    def _load_history(self) -> None:
        """Load extraction history from file."""
        
        if not self.history_file.exists():
            logger.debug("No existing history file found")
            return
        
        try:
            with open(self.history_file, 'r') as f:
                history_data = json.load(f)
            
            # Load extraction attempts
            for attempt_data in history_data.get('extraction_history', []):
                attempt = ExtractionAttempt.from_dict(attempt_data)
                self.extraction_history.append(attempt)
            
            # Load framework stats
            for framework_name, stats in history_data.get('framework_stats', {}).items():
                framework = DocumentationFramework(framework_name)
                stats['last_updated'] = datetime.fromisoformat(stats['last_updated'])
                self.framework_stats[framework] = stats
            
            logger.info(f"Loaded {len(self.extraction_history)} extraction attempts from history")
            
        except Exception as e:
            logger.error(f"Failed to load extraction history: {e}")


# Global adaptive manager instance
adaptive_manager = AdaptiveConfigManager()
```

**8.2 Integrate Adaptive Management with Smart Factory**

Update your `smart_crawler_factory.py` to use adaptive configuration:

```python
# Add this import at the top
from adaptive_config_manager import adaptive_manager

# Update the SmartCrawlerConfigFactory class
class SmartCrawlerConfigFactory:
    """Factory for creating optimized crawler configurations with adaptive learning."""
    
    def __init__(self):
        self.config_cache = {}
        self.adaptive_manager = adaptive_manager
    
    def create_adaptive_config(
        self,
        url: str,
        html_content: Optional[str] = None,
        cache_mode: CacheMode = CacheMode.BYPASS,
        custom_overrides: Optional[Dict[str, Any]] = None,
        enable_learning: bool = True
    ) -> Tuple[CrawlerRunConfig, DocumentationFramework]:
        """
        Create an adaptive configuration that learns from previous attempts.
        
        Args:
            url: Target URL for crawling
            html_content: Optional HTML content for framework detection
            cache_mode: Caching mode for the crawler
            custom_overrides: Optional configuration overrides
            enable_learning: Whether to use adaptive learning
            
        Returns:
            Tuple of (optimized_config, detected_framework)
        """
        
        # Get framework-specific configuration
        framework, base_extraction_config = doc_site_config.get_config_for_url(url, html_content)
        
        # Apply adaptive optimizations if enabled
        if enable_learning:
            optimized_extraction_config, was_modified = self.adaptive_manager.get_optimized_config(
                framework, base_extraction_config, url
            )
            
            if was_modified:
                logger.info(f"Applied adaptive optimizations for {framework.value}")
            else:
                optimized_extraction_config = base_extraction_config
        else:
            optimized_extraction_config = base_extraction_config
        
        # Create crawler configuration
        config_params = {
            'cache_mode': cache_mode,
            'stream': False,
        }
        
        # Apply optimized settings
        if optimized_extraction_config.target_elements:
            config_params['target_elements'] = optimized_extraction_config.target_elements
        
        if optimized_extraction_config.excluded_tags:
            config_params['excluded_tags'] = optimized_extraction_config.excluded_tags
        
        if optimized_extraction_config.word_count_threshold:
            config_params['word_count_threshold'] = optimized_extraction_config.word_count_threshold
        
        config_params['exclude_external_links'] = optimized_extraction_config.exclude_external_links
        config_params['exclude_social_media_links'] = optimized_extraction_config.exclude_social_media_links
        config_params['process_iframes'] = optimized_extraction_config.process_iframes
        
        # Apply custom overrides
        if custom_overrides:
            config_params.update(custom_overrides)
        
        config = CrawlerRunConfig(**config_params)
        
        logger.info(f"Created adaptive config for {framework.value} framework")
        return config, framework
    
    def record_extraction_result(
        self,
        url: str,
        framework: DocumentationFramework,
        config_used: ExtractionConfig,
        quality_metrics: ContentMetrics,
        success: bool
    ) -> None:
        """
        Record extraction result for adaptive learning.
        
        Args:
            url: The URL that was crawled
            framework: Detected framework
            config_used: Configuration that was used
            quality_metrics: Quality metrics from extraction
            success: Whether extraction was successful
        """
        
        self.adaptive_manager.record_extraction(
            url, framework, config_used, quality_metrics, success
        )

# Update the global factory instance
smart_config_factory = SmartCrawlerConfigFactory()
```

This adaptive system learns from extraction results and automatically optimizes configurations over time. The system tracks quality metrics and adjusts extraction parameters to improve content quality for different documentation frameworks.

### Step 9: Create Monitoring and Reporting Dashboard

Create a monitoring system that provides insights into crawler performance and quality trends.

**9.1 Create Performance Monitor**

Create a new file `performance_monitor.py`:

```python
#!/usr/bin/env python3
"""
Performance monitoring and reporting for enhanced crawler.

This module provides comprehensive monitoring, reporting, and alerting
capabilities for the enhanced crawler system.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import statistics
from collections import defaultdict, Counter

from enhanced_crawler_config import DocumentationFramework
from content_quality_analyzer import ContentMetrics
from adaptive_config_manager import adaptive_manager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitors and reports on crawler performance."""
    
    def __init__(self, report_dir: str = "reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        
        # Performance thresholds
        self.thresholds = {
            'quality_score': {
                'excellent': 0.8,
                'good': 0.6,
                'fair': 0.4,
                'poor': 0.0
            },
            'content_ratio': {
                'excellent': 0.8,
                'good': 0.6,
                'fair': 0.4,
                'poor': 0.0
            },
            'success_rate': {
                'excellent': 0.95,
                'good': 0.85,
                'fair': 0.7,
                'poor': 0.0
            }
        }
    
    def generate_performance_report(self, days: int = 30) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            days: Number of days to include in the report
            
        Returns:
            Dictionary containing performance report
        """
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get recent extraction attempts
        recent_attempts = [
            attempt for attempt in adaptive_manager.extraction_history
            if attempt.timestamp > cutoff_date
        ]
        
        if not recent_attempts:
            return {
                'error': 'No extraction attempts found in the specified period',
                'period_days': days,
                'report_generated': datetime.now().isoformat()
            }
        
        # Generate report sections
        report = {
            'report_period': {
                'days': days,
                'start_date': cutoff_date.isoformat(),
                'end_date': datetime.now().isoformat(),
                'total_attempts': len(recent_attempts)
            },
            'overall_performance': self._analyze_overall_performance(recent_attempts),
            'framework_performance': self._analyze_framework_performance(recent_attempts),
            'quality_trends': self._analyze_quality_trends(recent_attempts),
            'problem_areas': self._identify_problem_areas(recent_attempts),
            'recommendations': self._generate_recommendations(recent_attempts),
            'report_generated': datetime.now().isoformat()
        }
        
        return report
    
    def _analyze_overall_performance(self, attempts: List) -> Dict:
        """Analyze overall system performance."""
        
        successful_attempts = [a for a in attempts if a.success]
        
        if not attempts:
            return {'error': 'No attempts to analyze'}
        
        success_rate = len(successful_attempts) / len(attempts)
        
        # Quality metrics from successful attempts
        quality_scores = [a.quality_metrics['quality_score'] for a in successful_attempts]
        content_ratios = [a.quality_metrics['content_ratio'] for a in successful_attempts]
        
        performance = {
            'total_attempts': len(attempts),
            'successful_attempts': len(successful_attempts),
            'success_rate': success_rate,
            'success_rate_grade': self._grade_metric('success_rate', success_rate)
        }
        
        if quality_scores:
            avg_quality = statistics.mean(quality_scores)
            avg_content_ratio = statistics.mean(content_ratios)
            
            performance.update({
                'average_quality_score': avg_quality,
                'quality_score_grade': self._grade_metric('quality_score', avg_quality),
                'average_content_ratio': avg_content_ratio,
                'content_ratio_grade': self._grade_metric('content_ratio', avg_content_ratio),
                'quality_std_dev': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                'best_quality_score': max(quality_scores),
                'worst_quality_score': min(quality_scores)
            })
        
        return performance
    
    def _analyze_framework_performance(self, attempts: List) -> Dict:
        """Analyze performance by documentation framework."""
        
        framework_data = defaultdict(list)
        
        for attempt in attempts:
            framework_data[attempt.framework].append(attempt)
        
        framework_performance = {}
        
        for framework, framework_attempts in framework_data.items():
            successful = [a for a in framework_attempts if a.success]
            
            performance = {
                'total_attempts': len(framework_attempts),
                'successful_attempts': len(successful),
                'success_rate': len(successful) / len(framework_attempts) if framework_attempts else 0
            }
            
            if successful:
                quality_scores = [a.quality_metrics['quality_score'] for a in successful]
                content_ratios = [a.quality_metrics['content_ratio'] for a in successful]
                
                performance.update({
                    'average_quality_score': statistics.mean(quality_scores),
                    'average_content_ratio': statistics.mean(content_ratios),
                    'quality_consistency': 1 - (statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0),
                    'best_quality_score': max(quality_scores),
                    'worst_quality_score': min(quality_scores)
                })
            
            framework_performance[framework.value] = performance
        
        return framework_performance
    
    def _analyze_quality_trends(self, attempts: List, window_hours: int = 24) -> Dict:
        """Analyze quality trends over time."""
        
        # Group attempts by time windows
        time_windows = defaultdict(list)
        
        for attempt in attempts:
            if attempt.success:
                window_key = attempt.timestamp.replace(
                    minute=0, second=0, microsecond=0
                ).replace(hour=(attempt.timestamp.hour // window_hours) * window_hours)
                time_windows[window_key].append(attempt)
        
        # Calculate trends
        trend_data = []
        
        for window_time in sorted(time_windows.keys()):
            window_attempts = time_windows[window_time]
            quality_scores = [a.quality_metrics['quality_score'] for a in window_attempts]
            
            if quality_scores:
                trend_data.append({
                    'timestamp': window_time.isoformat(),
                    'attempts_count': len(window_attempts),
                    'average_quality': statistics.mean(quality_scores),
                    'quality_std_dev': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
                })
        
        # Calculate trend direction
        trend_direction = 'stable'
        if len(trend_data) >= 3:
            recent_avg = statistics.mean([d['average_quality'] for d in trend_data[-3:]])
            earlier_avg = statistics.mean([d['average_quality'] for d in trend_data[:3]])
            
            if recent_avg > earlier_avg + 0.05:
                trend_direction = 'improving'
            elif recent_avg < earlier_avg - 0.05:
                trend_direction = 'declining'
        
        return {
            'trend_data': trend_data,
            'trend_direction': trend_direction,
            'data_points': len(trend_data)
        }
    
    def _identify_problem_areas(self, attempts: List) -> Dict:
        """Identify areas that need attention."""
        
        problems = {
            'low_quality_sites': [],
            'failing_frameworks': [],
            'configuration_issues': [],
            'performance_bottlenecks': []
        }
        
        # Identify low-quality sites
        site_quality = defaultdict(list)
        for attempt in attempts:
            if attempt.success:
                site_quality[attempt.url].append(attempt.quality_metrics['quality_score'])
        
        for url, scores in site_quality.items():
            avg_score = statistics.mean(scores)
            if avg_score < 0.4 and len(scores) >= 3:
                problems['low_quality_sites'].append({
                    'url': url,
                    'average_quality': avg_score,
                    'attempts': len(scores)
                })
        
        # Identify failing frameworks
        framework_success = defaultdict(lambda: {'total': 0, 'successful': 0})
        for attempt in attempts:
            framework_success[attempt.framework]['total'] += 1
            if attempt.success:
                framework_success[attempt.framework]['successful'] += 1
        
        for framework, stats in framework_success.items():
            success_rate = stats['successful'] / stats['total']
            if success_rate < 0.7 and stats['total'] >= 5:
                problems['failing_frameworks'].append({
                    'framework': framework.value,
                    'success_rate': success_rate,
                    'total_attempts': stats['total']
                })
        
        return problems
    
    def _generate_recommendations(self, attempts: List) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        # Analyze overall performance
        successful_attempts = [a for a in attempts if a.success]
        if not successful_attempts:
            recommendations.append("No successful extractions found. Check crawler configuration and target sites.")
            return recommendations
        
        success_rate = len(successful_attempts) / len(attempts)
        avg_quality = statistics.mean([a.quality_metrics['quality_score'] for a in successful_attempts])
        avg_content_ratio = statistics.mean([a.quality_metrics['content_ratio'] for a in successful_attempts])
        
        # Success rate recommendations
        if success_rate < 0.8:
            recommendations.append(f"Success rate is {success_rate:.1%}. Consider reviewing error logs and improving error handling.")
        
        # Quality recommendations
        if avg_quality < 0.6:
            recommendations.append(f"Average quality score is {avg_quality:.2f}. Consider using more aggressive content filtering.")
        
        if avg_content_ratio < 0.5:
            recommendations.append(f"Content ratio is {avg_content_ratio:.1%}. Review CSS selectors and exclusion rules.")
        
        # Framework-specific recommendations
        framework_performance = self._analyze_framework_performance(attempts)
        for framework, perf in framework_performance.items():
            if perf.get('average_quality_score', 0) < 0.5 and perf['total_attempts'] >= 5:
                recommendations.append(f"Poor performance for {framework} framework. Consider custom configuration.")
        
        # Adaptive learning recommendations
        if len(attempts) >= 20:
            recommendations.append("Sufficient data available for adaptive learning. Enable adaptive configuration optimization.")
        
        return recommendations
    
    def _grade_metric(self, metric_type: str, value: float) -> str:
        """Grade a metric value."""
        
        thresholds = self.thresholds.get(metric_type, {})
        
        if value >= thresholds.get('excellent', 0.8):
            return 'excellent'
        elif value >= thresholds.get('good', 0.6):
            return 'good'
        elif value >= thresholds.get('fair', 0.4):
            return 'fair'
        else:
            return 'poor'
    
    def save_report(self, report: Dict, filename: Optional[str] = None) -> str:
        """Save report to file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report_path = self.report_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {report_path}")
        return str(report_path)
    
    def generate_html_report(self, report: Dict) -> str:
        """Generate HTML version of the report."""
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Crawler Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
        .grade-excellent { color: #28a745; font-weight: bold; }
        .grade-good { color: #17a2b8; font-weight: bold; }
        .grade-fair { color: #ffc107; font-weight: bold; }
        .grade-poor { color: #dc3545; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .recommendation { background-color: #e7f3ff; padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Enhanced Crawler Performance Report</h1>
        <p>Generated: {report_generated}</p>
        <p>Period: {period_days} days ({start_date} to {end_date})</p>
        <p>Total Attempts: {total_attempts}</p>
    </div>
    
    <div class="section">
        <h2>Overall Performance</h2>
        <div class="metric">
            <strong>Success Rate:</strong> {success_rate:.1%} 
            <span class="grade-{success_rate_grade}">({success_rate_grade})</span>
        </div>
        <div class="metric">
            <strong>Average Quality:</strong> {average_quality_score:.2f} 
            <span class="grade-{quality_score_grade}">({quality_score_grade})</span>
        </div>
        <div class="metric">
            <strong>Content Ratio:</strong> {average_content_ratio:.1%} 
            <span class="grade-{content_ratio_grade}">({content_ratio_grade})</span>
        </div>
    </div>
    
    <div class="section">
        <h2>Framework Performance</h2>
        <table>
            <tr>
                <th>Framework</th>
                <th>Attempts</th>
                <th>Success Rate</th>
                <th>Avg Quality</th>
                <th>Avg Content Ratio</th>
            </tr>
            {framework_rows}
        </table>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {recommendations_html}
    </div>
</body>
</html>
        """
        
        # Format framework performance table
        framework_rows = ""
        for framework, perf in report.get('framework_performance', {}).items():
            framework_rows += f"""
            <tr>
                <td>{framework}</td>
                <td>{perf['total_attempts']}</td>
                <td>{perf['success_rate']:.1%}</td>
                <td>{perf.get('average_quality_score', 0):.2f}</td>
                <td>{perf.get('average_content_ratio', 0):.1%}</td>
            </tr>
            """
        
        # Format recommendations
        recommendations_html = ""
        for rec in report.get('recommendations', []):
            recommendations_html += f'<div class="recommendation">{rec}</div>'
        
        # Fill template
        overall_perf = report.get('overall_performance', {})
        period = report.get('report_period', {})
        
        html_content = html_template.format(
            report_generated=report.get('report_generated', ''),
            period_days=period.get('days', 0),
            start_date=period.get('start_date', ''),
            end_date=period.get('end_date', ''),
            total_attempts=period.get('total_attempts', 0),
            success_rate=overall_perf.get('success_rate', 0),
            success_rate_grade=overall_perf.get('success_rate_grade', 'poor'),
            average_quality_score=overall_perf.get('average_quality_score', 0),
            quality_score_grade=overall_perf.get('quality_score_grade', 'poor'),
            average_content_ratio=overall_perf.get('average_content_ratio', 0),
            content_ratio_grade=overall_perf.get('content_ratio_grade', 'poor'),
            framework_rows=framework_rows,
            recommendations_html=recommendations_html
        )
        
        return html_content


# Global monitor instance
performance_monitor = PerformanceMonitor()
```

This comprehensive monitoring system provides detailed insights into crawler performance, identifies problem areas, and generates actionable recommendations for improvement. The system tracks quality trends over time and helps optimize the enhanced extraction capabilities.

This completes Phase 2 of the implementation, providing advanced content quality assessment, adaptive configuration optimization, and comprehensive performance monitoring. The system now learns from extraction results and continuously improves its performance over time.


## Phase 3: Advanced Extraction Techniques

### Step 10: Implement Machine Learning-Based Content Detection

The third phase introduces machine learning capabilities to automatically identify and extract the most relevant content from documentation pages, even when CSS selectors fail.

**10.1 Create ML-Based Content Classifier**

Create a new file `ml_content_classifier.py`:

```python
#!/usr/bin/env python3
"""
Machine learning-based content classification for enhanced crawler.

This module implements lightweight ML techniques to automatically identify
and classify content sections in documentation pages.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import statistics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ContentSection:
    """Represents a classified content section."""
    
    text: str
    section_type: str  # 'content', 'navigation', 'code', 'metadata'
    confidence: float
    features: Dict[str, float]
    start_line: int
    end_line: int


class MLContentClassifier:
    """Machine learning-based content classification system."""
    
    def __init__(self):
        self.feature_extractors = self._initialize_feature_extractors()
        self.classification_rules = self._initialize_classification_rules()
        self.content_keywords = self._load_content_keywords()
        self.navigation_keywords = self._load_navigation_keywords()
        
        # Initialize TF-IDF vectorizer for semantic analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        self.is_trained = False
    
    def _initialize_feature_extractors(self) -> Dict:
        """Initialize feature extraction functions."""
        
        return {
            'word_count': self._extract_word_count,
            'link_density': self._extract_link_density,
            'code_indicators': self._extract_code_indicators,
            'heading_indicators': self._extract_heading_indicators,
            'list_indicators': self._extract_list_indicators,
            'technical_terms': self._extract_technical_terms,
            'navigation_terms': self._extract_navigation_terms,
            'sentence_complexity': self._extract_sentence_complexity,
            'punctuation_density': self._extract_punctuation_density,
            'capitalization_patterns': self._extract_capitalization_patterns
        }
    
    def _initialize_classification_rules(self) -> Dict:
        """Initialize rule-based classification thresholds."""
        
        return {
            'content': {
                'min_word_count': 10,
                'max_link_density': 0.3,
                'min_technical_terms': 0.1,
                'min_sentence_complexity': 0.2
            },
            'navigation': {
                'max_word_count': 50,
                'min_link_density': 0.4,
                'max_technical_terms': 0.1,
                'min_navigation_terms': 0.2
            },
            'code': {
                'min_code_indicators': 0.3,
                'max_sentence_complexity': 0.1
            },
            'metadata': {
                'max_word_count': 20,
                'min_capitalization_patterns': 0.3
            }
        }
    
    def _load_content_keywords(self) -> Set[str]:
        """Load keywords that indicate substantive content."""
        
        return {
            # Technical documentation terms
            'function', 'method', 'parameter', 'argument', 'return', 'example',
            'configure', 'install', 'setup', 'implement', 'deploy', 'usage',
            'tutorial', 'guide', 'documentation', 'reference', 'manual',
            'step', 'process', 'procedure', 'workflow', 'instruction',
            'api', 'sdk', 'cli', 'http', 'json', 'xml', 'database', 'server',
            'authentication', 'authorization', 'configuration', 'integration',
            
            # Explanatory terms
            'explain', 'describe', 'overview', 'introduction', 'concept',
            'understand', 'learn', 'create', 'build', 'develop', 'design',
            'optimize', 'troubleshoot', 'debug', 'test', 'validate',
            
            # Action terms
            'click', 'select', 'choose', 'enter', 'type', 'upload', 'download',
            'save', 'load', 'import', 'export', 'connect', 'disconnect'
        }
    
    def _load_navigation_keywords(self) -> Set[str]:
        """Load keywords that indicate navigation content."""
        
        return {
            # Navigation terms
            'home', 'back', 'next', 'previous', 'menu', 'navigation', 'nav',
            'sidebar', 'breadcrumb', 'toc', 'contents', 'index', 'search',
            'filter', 'sort', 'page', 'section', 'chapter', 'part',
            
            # Link text
            'more', 'details', 'info', 'read', 'view', 'see', 'go', 'visit',
            'link', 'url', 'href', 'click', 'here', 'this', 'that',
            
            # Meta navigation
            'edit', 'share', 'print', 'bookmark', 'favorite', 'subscribe',
            'follow', 'contact', 'about', 'help', 'support', 'faq'
        }
    
    def classify_content_sections(self, markdown_content: str) -> List[ContentSection]:
        """
        Classify content sections using ML techniques.
        
        Args:
            markdown_content: The markdown content to classify
            
        Returns:
            List of classified content sections
        """
        
        if not markdown_content:
            return []
        
        # Split content into logical sections
        sections = self._split_into_sections(markdown_content)
        
        # Extract features for each section
        section_features = []
        for section in sections:
            features = self._extract_section_features(section['text'])
            section_features.append(features)
        
        # Perform rule-based classification
        classified_sections = []
        for i, section in enumerate(sections):
            features = section_features[i]
            section_type, confidence = self._classify_section(features, section['text'])
            
            classified_section = ContentSection(
                text=section['text'],
                section_type=section_type,
                confidence=confidence,
                features=features,
                start_line=section['start_line'],
                end_line=section['end_line']
            )
            
            classified_sections.append(classified_section)
        
        # Apply semantic clustering for refinement
        if len(classified_sections) > 3:
            classified_sections = self._refine_with_clustering(classified_sections)
        
        return classified_sections
    
    def _split_into_sections(self, content: str) -> List[Dict]:
        """Split content into logical sections."""
        
        lines = content.split('\n')
        sections = []
        current_section = []
        current_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Section boundaries: headings, double newlines, or significant content changes
            is_boundary = (
                line.startswith('#') or  # Heading
                (not line and current_section) or  # Empty line after content
                (len(current_section) > 10 and self._is_content_shift(current_section, line))
            )
            
            if is_boundary and current_section:
                # Save current section
                section_text = '\n'.join(current_section)
                if section_text.strip():
                    sections.append({
                        'text': section_text,
                        'start_line': current_start,
                        'end_line': i - 1
                    })
                
                current_section = []
                current_start = i
            
            if line:  # Only add non-empty lines
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section)
            if section_text.strip():
                sections.append({
                    'text': section_text,
                    'start_line': current_start,
                    'end_line': len(lines) - 1
                })
        
        return sections
    
    def _is_content_shift(self, current_section: List[str], new_line: str) -> bool:
        """Detect significant content shifts."""
        
        if not new_line:
            return False
        
        # Analyze recent lines for content type
        recent_lines = current_section[-3:] if len(current_section) >= 3 else current_section
        recent_text = ' '.join(recent_lines)
        
        # Check for shifts between content types
        recent_has_links = recent_text.count('[') > 2
        new_has_links = new_line.count('[') > 0
        
        recent_is_code = any(line.startswith('```') or line.startswith('    ') for line in recent_lines)
        new_is_code = new_line.startswith('```') or new_line.startswith('    ')
        
        # Detect shifts
        if recent_has_links != new_has_links and abs(len(recent_lines) - 1) > 2:
            return True
        
        if recent_is_code != new_is_code:
            return True
        
        return False
    
    def _extract_section_features(self, text: str) -> Dict[str, float]:
        """Extract features from a text section."""
        
        features = {}
        
        for feature_name, extractor in self.feature_extractors.items():
            try:
                features[feature_name] = extractor(text)
            except Exception as e:
                logger.warning(f"Failed to extract feature {feature_name}: {e}")
                features[feature_name] = 0.0
        
        return features
    
    def _extract_word_count(self, text: str) -> float:
        """Extract word count feature."""
        return len(text.split())
    
    def _extract_link_density(self, text: str) -> float:
        """Extract link density feature."""
        words = len(text.split())
        links = text.count('[') + text.count('](')
        return links / max(words, 1)
    
    def _extract_code_indicators(self, text: str) -> float:
        """Extract code indicators feature."""
        
        code_patterns = [
            r'```',  # Code blocks
            r'`[^`]+`',  # Inline code
            r'^\s{4,}',  # Indented code
            r'\b(def|function|class|import|from|var|let|const)\b',  # Programming keywords
            r'[{}();]',  # Programming punctuation
            r'\b[A-Z_][A-Z0-9_]*\b',  # Constants
        ]
        
        total_matches = 0
        for pattern in code_patterns:
            total_matches += len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
        
        words = len(text.split())
        return total_matches / max(words, 1)
    
    def _extract_heading_indicators(self, text: str) -> float:
        """Extract heading indicators feature."""
        
        lines = text.split('\n')
        heading_lines = sum(1 for line in lines if line.strip().startswith('#'))
        return heading_lines / max(len(lines), 1)
    
    def _extract_list_indicators(self, text: str) -> float:
        """Extract list indicators feature."""
        
        lines = text.split('\n')
        list_lines = sum(1 for line in lines if re.match(r'^\s*[-*+]\s+', line.strip()))
        return list_lines / max(len(lines), 1)
    
    def _extract_technical_terms(self, text: str) -> float:
        """Extract technical terms feature."""
        
        words = text.lower().split()
        technical_count = sum(1 for word in words if word in self.content_keywords)
        return technical_count / max(len(words), 1)
    
    def _extract_navigation_terms(self, text: str) -> float:
        """Extract navigation terms feature."""
        
        words = text.lower().split()
        nav_count = sum(1 for word in words if word in self.navigation_keywords)
        return nav_count / max(len(words), 1)
    
    def _extract_sentence_complexity(self, text: str) -> float:
        """Extract sentence complexity feature."""
        
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0.0
        
        complexities = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 3:
                # Simple complexity measure: average word length and sentence length
                avg_word_length = statistics.mean(len(word) for word in words)
                complexity = (len(words) * avg_word_length) / 100
                complexities.append(min(complexity, 1.0))
        
        return statistics.mean(complexities) if complexities else 0.0
    
    def _extract_punctuation_density(self, text: str) -> float:
        """Extract punctuation density feature."""
        
        punctuation_chars = '.,;:!?()[]{}"\'-'
        punct_count = sum(1 for char in text if char in punctuation_chars)
        return punct_count / max(len(text), 1)
    
    def _extract_capitalization_patterns(self, text: str) -> float:
        """Extract capitalization patterns feature."""
        
        words = text.split()
        if not words:
            return 0.0
        
        # Count words that are all caps, title case, etc.
        caps_count = sum(1 for word in words if word.isupper() and len(word) > 1)
        title_count = sum(1 for word in words if word.istitle())
        
        return (caps_count + title_count) / len(words)
    
    def _classify_section(self, features: Dict[str, float], text: str) -> Tuple[str, float]:
        """Classify a section based on its features."""
        
        scores = {}
        
        # Calculate scores for each section type
        for section_type, rules in self.classification_rules.items():
            score = self._calculate_type_score(features, rules, section_type)
            scores[section_type] = score
        
        # Find the best match
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # Apply minimum confidence threshold
        if confidence < 0.3:
            best_type = 'content'  # Default to content for ambiguous sections
            confidence = 0.3
        
        return best_type, confidence
    
    def _calculate_type_score(self, features: Dict[str, float], rules: Dict[str, float], section_type: str) -> float:
        """Calculate score for a specific section type."""
        
        score = 0.0
        rule_count = 0
        
        for rule_name, threshold in rules.items():
            feature_name = rule_name.replace('min_', '').replace('max_', '')
            
            if feature_name in features:
                feature_value = features[feature_name]
                
                if rule_name.startswith('min_'):
                    # Higher values are better
                    if feature_value >= threshold:
                        score += 1.0
                    else:
                        score += feature_value / threshold
                elif rule_name.startswith('max_'):
                    # Lower values are better
                    if feature_value <= threshold:
                        score += 1.0
                    else:
                        score += threshold / max(feature_value, 0.001)
                
                rule_count += 1
        
        return score / max(rule_count, 1)
    
    def _refine_with_clustering(self, sections: List[ContentSection]) -> List[ContentSection]:
        """Refine classification using semantic clustering."""
        
        try:
            # Extract text for clustering
            texts = [section.text for section in sections]
            
            # Create TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Perform clustering
            n_clusters = min(4, len(sections))  # Max 4 clusters: content, navigation, code, metadata
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Analyze clusters to determine their types
            cluster_types = self._analyze_clusters(sections, cluster_labels, tfidf_matrix)
            
            # Update section classifications based on clustering
            for i, section in enumerate(sections):
                cluster_id = cluster_labels[i]
                cluster_type = cluster_types.get(cluster_id, section.section_type)
                
                # Only update if clustering provides higher confidence
                if cluster_type != section.section_type:
                    # Calculate clustering confidence based on cluster cohesion
                    cluster_confidence = self._calculate_cluster_confidence(
                        i, cluster_labels, tfidf_matrix
                    )
                    
                    if cluster_confidence > section.confidence:
                        section.section_type = cluster_type
                        section.confidence = cluster_confidence
            
        except Exception as e:
            logger.warning(f"Clustering refinement failed: {e}")
        
        return sections
    
    def _analyze_clusters(self, sections: List[ContentSection], labels: np.ndarray, tfidf_matrix) -> Dict[int, str]:
        """Analyze clusters to determine their content types."""
        
        cluster_types = {}
        
        for cluster_id in set(labels):
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            cluster_sections = [sections[i] for i in cluster_indices]
            
            # Analyze the predominant features in this cluster
            feature_averages = defaultdict(float)
            for section in cluster_sections:
                for feature, value in section.features.items():
                    feature_averages[feature] += value
            
            # Average the features
            for feature in feature_averages:
                feature_averages[feature] /= len(cluster_sections)
            
            # Determine cluster type based on features
            cluster_type = self._determine_cluster_type(feature_averages)
            cluster_types[cluster_id] = cluster_type
        
        return cluster_types
    
    def _determine_cluster_type(self, feature_averages: Dict[str, float]) -> str:
        """Determine cluster type based on average features."""
        
        # High link density suggests navigation
        if feature_averages.get('link_density', 0) > 0.4:
            return 'navigation'
        
        # High code indicators suggest code
        if feature_averages.get('code_indicators', 0) > 0.3:
            return 'code'
        
        # High technical terms and complexity suggest content
        if (feature_averages.get('technical_terms', 0) > 0.1 and
            feature_averages.get('sentence_complexity', 0) > 0.2):
            return 'content'
        
        # High capitalization patterns and low word count suggest metadata
        if (feature_averages.get('capitalization_patterns', 0) > 0.3 and
            feature_averages.get('word_count', 0) < 20):
            return 'metadata'
        
        return 'content'  # Default
    
    def _calculate_cluster_confidence(self, section_index: int, labels: np.ndarray, tfidf_matrix) -> float:
        """Calculate confidence based on cluster cohesion."""
        
        try:
            section_cluster = labels[section_index]
            cluster_indices = [i for i, label in enumerate(labels) if label == section_cluster]
            
            if len(cluster_indices) < 2:
                return 0.5  # Low confidence for singleton clusters
            
            # Calculate average similarity within cluster
            section_vector = tfidf_matrix[section_index]
            similarities = []
            
            for other_index in cluster_indices:
                if other_index != section_index:
                    other_vector = tfidf_matrix[other_index]
                    similarity = cosine_similarity(section_vector, other_vector)[0][0]
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = statistics.mean(similarities)
                return min(avg_similarity + 0.2, 1.0)  # Boost confidence slightly
            
        except Exception as e:
            logger.warning(f"Failed to calculate cluster confidence: {e}")
        
        return 0.5
    
    def extract_content_sections(self, sections: List[ContentSection]) -> str:
        """Extract only content sections and combine them."""
        
        content_sections = [
            section for section in sections
            if section.section_type == 'content' and section.confidence > 0.4
        ]
        
        # Sort by confidence and position
        content_sections.sort(key=lambda x: (x.start_line, -x.confidence))
        
        # Combine content sections
        combined_content = '\n\n'.join(section.text for section in content_sections)
        
        return combined_content
    
    def get_classification_summary(self, sections: List[ContentSection]) -> Dict:
        """Get summary of classification results."""
        
        type_counts = Counter(section.section_type for section in sections)
        avg_confidence = statistics.mean(section.confidence for section in sections) if sections else 0
        
        content_sections = [s for s in sections if s.section_type == 'content']
        content_words = sum(len(s.text.split()) for s in content_sections)
        total_words = sum(len(s.text.split()) for s in sections)
        
        return {
            'total_sections': len(sections),
            'section_types': dict(type_counts),
            'average_confidence': avg_confidence,
            'content_ratio': content_words / max(total_words, 1),
            'content_sections_count': len(content_sections)
        }


# Global classifier instance
ml_classifier = MLContentClassifier()
```

**10.2 Integrate ML Classification with Enhanced Crawler**

Update your `crawl4ai_mcp.py` to include ML-based content extraction:

```python
# Add this import at the top
from ml_content_classifier import ml_classifier

# Add this new function for ML-enhanced extraction
async def crawl_single_page_ml_enhanced(ctx: Context, url: str, use_ml_classification: bool = True) -> str:
    """
    ML-enhanced version of crawl_single_page with intelligent content classification.
    
    This tool uses machine learning techniques to automatically identify and extract
    the most relevant content sections from documentation pages.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
        use_ml_classification: Whether to use ML-based content classification
        
    Returns:
        Summary of the crawling operation with ML classification results
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client
        
        # First, try enhanced configuration
        framework, enhanced_config = smart_config_factory.create_adaptive_config(url, enable_learning=True)
        
        # Crawl the page
        result = await crawler.arun(url=url, config=enhanced_config)
        
        if result.success and result.markdown:
            original_content = result.markdown
            
            # Apply ML classification if enabled
            if use_ml_classification:
                logger.info(f"Applying ML content classification to {url}")
                
                # Classify content sections
                classified_sections = ml_classifier.classify_content_sections(original_content)
                
                # Extract only high-quality content sections
                ml_extracted_content = ml_classifier.extract_content_sections(classified_sections)
                
                # Get classification summary
                classification_summary = ml_classifier.get_classification_summary(classified_sections)
                
                # Use ML-extracted content if it's significantly better
                if (classification_summary['content_ratio'] > 0.6 and 
                    len(ml_extracted_content.split()) > 50):
                    final_content = ml_extracted_content
                    extraction_method = 'ml_enhanced'
                    logger.info(f"Using ML-extracted content with {classification_summary['content_ratio']:.1%} content ratio")
                else:
                    final_content = original_content
                    extraction_method = 'standard_enhanced'
                    logger.info(f"ML extraction didn't improve quality, using standard enhanced extraction")
            else:
                final_content = original_content
                extraction_method = 'standard_enhanced'
                classification_summary = {}
            
            # Validate final content quality
            quality_metrics = validate_content_quality_enhanced(final_content, url)
            
            # Record extraction result for adaptive learning
            if hasattr(smart_config_factory, 'record_extraction_result'):
                from enhanced_crawler_config import doc_site_config
                _, extraction_config = doc_site_config.get_config_for_url(url)
                from content_quality_analyzer import content_analyzer
                content_metrics = content_analyzer.analyze_content(final_content, url)
                
                smart_config_factory.record_extraction_result(
                    url, framework, extraction_config, content_metrics, True
                )
            
            # Chunk the content using enhanced chunking
            chunks = smart_chunk_markdown(final_content)
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata with ML classification info
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                meta["extraction_method"] = extraction_method
                meta["framework"] = framework.value
                meta["quality_metrics"] = quality_metrics
                meta["ml_classification"] = classification_summary
                metadatas.append(meta)
            
            # Create url_to_full_document mapping
            url_to_full_document = {url: final_content}
            
            # Add to Supabase
            add_documents_to_supabase(
                supabase_client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document,
            )
            
            return json.dumps(
                {
                    "success": True,
                    "url": url,
                    "extraction_method": extraction_method,
                    "framework": framework.value,
                    "chunks_stored": len(chunks),
                    "content_length": len(final_content),
                    "original_content_length": len(original_content),
                    "quality_metrics": quality_metrics,
                    "ml_classification": classification_summary,
                    "links_count": {
                        "internal": len(result.links.get("internal", [])),
                        "external": len(result.links.get("external", [])),
                    },
                },
                indent=2,
            )
        else:
            return json.dumps(
                {"success": False, "url": url, "error": result.error_message}, indent=2
            )
    except Exception as e:
        logger.error(f"Error in ML-enhanced crawling for {url}: {str(e)}")
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)

# Register the new ML-enhanced tool
@mcp.tool()
async def crawl_single_page_ml_enhanced_tool(ctx: Context, url: str, use_ml_classification: bool = True) -> str:
    """
    ML-enhanced single page crawler with intelligent content classification.
    
    This tool uses machine learning techniques to automatically identify and extract
    the most relevant content sections from documentation pages, providing superior
    content quality compared to traditional CSS-based extraction methods.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
        use_ml_classification: Whether to use ML-based content classification (default: True)
        
    Returns:
        JSON summary of crawling operation with ML classification results
    """
    return await crawl_single_page_ml_enhanced(ctx, url, use_ml_classification)
```

### Step 11: Implement Content Validation and Quality Assurance

Create a comprehensive content validation system that ensures extracted content meets quality standards.

**11.1 Create Content Validation System**

Create a new file `content_validator.py`:

```python
#!/usr/bin/env python3
"""
Content validation and quality assurance for enhanced crawler.

This module provides comprehensive validation of extracted content to ensure
quality and relevance for RAG applications.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter
import statistics
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of content validation."""
    
    is_valid: bool
    quality_score: float
    issues: List[str]
    recommendations: List[str]
    metrics: Dict[str, float]
    validation_timestamp: datetime


class ContentValidator:
    """Comprehensive content validation system."""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.quality_thresholds = self._initialize_quality_thresholds()
        self.content_patterns = self._initialize_content_patterns()
    
    def _initialize_validation_rules(self) -> Dict:
        """Initialize content validation rules."""
        
        return {
            'minimum_word_count': 20,
            'maximum_link_density': 0.5,
            'minimum_content_ratio': 0.3,
            'maximum_repetition_ratio': 0.3,
            'minimum_unique_sentences': 3,
            'maximum_navigation_ratio': 0.6,
            'minimum_technical_relevance': 0.1,
            'maximum_boilerplate_ratio': 0.4
        }
    
    def _initialize_quality_thresholds(self) -> Dict:
        """Initialize quality score thresholds."""
        
        return {
            'excellent': 0.85,
            'good': 0.7,
            'acceptable': 0.5,
            'poor': 0.3
        }
    
    def _initialize_content_patterns(self) -> Dict:
        """Initialize content pattern recognition."""
        
        return {
            'boilerplate_patterns': [
                r'copyright\s+\d{4}',
                r'all\s+rights\s+reserved',
                r'privacy\s+policy',
                r'terms\s+of\s+service',
                r'cookie\s+policy',
                r'powered\s+by',
                r'designed\s+by',
                r'follow\s+us\s+on',
                r'subscribe\s+to\s+our',
                r'newsletter\s+signup'
            ],
            'navigation_patterns': [
                r'home\s*>\s*docs',
                r'breadcrumb',
                r'table\s+of\s+contents',
                r'previous\s+page',
                r'next\s+page',
                r'back\s+to\s+top',
                r'edit\s+this\s+page',
                r'improve\s+this\s+doc'
            ],
            'content_patterns': [
                r'\b(function|method|parameter|class|object)\b',
                r'\b(example|tutorial|guide|how\s+to)\b',
                r'\b(configure|install|setup|deploy)\b',
                r'\b(api|endpoint|request|response)\b',
                r'\b(documentation|reference|manual)\b'
            ]
        }
    
    def validate_content(self, content: str, url: str = "", context: Dict = None) -> ValidationResult:
        """
        Perform comprehensive content validation.
        
        Args:
            content: The content to validate
            url: Optional URL for context
            context: Optional additional context information
            
        Returns:
            ValidationResult with detailed validation information
        """
        
        if not content or not content.strip():
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=["Content is empty"],
                recommendations=["Ensure content extraction is working properly"],
                metrics={},
                validation_timestamp=datetime.now()
            )
        
        # Perform validation checks
        metrics = self._calculate_validation_metrics(content)
        issues = self._identify_issues(metrics, content)
        recommendations = self._generate_recommendations(metrics, issues, url)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(metrics)
        
        # Determine if content is valid
        is_valid = (
            quality_score >= self.quality_thresholds['acceptable'] and
            len(issues) <= 2  # Allow minor issues
        )
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations,
            metrics=metrics,
            validation_timestamp=datetime.now()
        )
    
    def _calculate_validation_metrics(self, content: str) -> Dict[str, float]:
        """Calculate comprehensive validation metrics."""
        
        metrics = {}
        
        # Basic metrics
        words = content.split()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        metrics['word_count'] = len(words)
        metrics['line_count'] = len(lines)
        metrics['sentence_count'] = len(sentences)
        metrics['character_count'] = len(content)
        
        # Content quality metrics
        metrics['link_density'] = self._calculate_link_density(content)
        metrics['content_ratio'] = self._calculate_content_ratio(content)
        metrics['repetition_ratio'] = self._calculate_repetition_ratio(content)
        metrics['navigation_ratio'] = self._calculate_navigation_ratio(content)
        metrics['boilerplate_ratio'] = self._calculate_boilerplate_ratio(content)
        metrics['technical_relevance'] = self._calculate_technical_relevance(content)
        
        # Structural metrics
        metrics['unique_sentences'] = len(set(sentences))
        metrics['avg_sentence_length'] = statistics.mean([len(s.split()) for s in sentences]) if sentences else 0
        metrics['heading_count'] = len(re.findall(r'^#+\s+', content, re.MULTILINE))
        metrics['code_block_count'] = len(re.findall(r'```.*?```', content, re.DOTALL))
        
        # Readability metrics
        metrics['readability_score'] = self._calculate_readability_score(content)
        metrics['complexity_score'] = self._calculate_complexity_score(content)
        
        return metrics
    
    def _calculate_link_density(self, content: str) -> float:
        """Calculate link density in content."""
        
        words = len(content.split())
        links = len(re.findall(r'\[.*?\]\(.*?\)', content))
        return links / max(words, 1)
    
    def _calculate_content_ratio(self, content: str) -> float:
        """Calculate ratio of substantive content vs navigation/metadata."""
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        content_lines = 0
        
        for line in lines:
            # Check if line contains substantive content
            if self._is_substantive_content(line):
                content_lines += 1
        
        return content_lines / max(len(lines), 1)
    
    def _is_substantive_content(self, line: str) -> bool:
        """Check if a line contains substantive content."""
        
        # Skip very short lines
        if len(line.split()) < 3:
            return False
        
        # Skip lines that are primarily links
        if line.count('[') > 2 or line.count('](') > 1:
            return False
        
        # Check for content patterns
        for pattern in self.content_patterns['content_patterns']:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        # Check for navigation patterns (negative indicator)
        for pattern in self.content_patterns['navigation_patterns']:
            if re.search(pattern, line, re.IGNORECASE):
                return False
        
        # Default: lines with reasonable length are considered content
        return len(line.split()) >= 5
    
    def _calculate_repetition_ratio(self, content: str) -> float:
        """Calculate ratio of repeated content."""
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return 0.0
        
        line_counts = Counter(lines)
        repeated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
        
        return repeated_lines / len(lines)
    
    def _calculate_navigation_ratio(self, content: str) -> float:
        """Calculate ratio of navigation content."""
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        navigation_lines = 0
        
        for line in lines:
            for pattern in self.content_patterns['navigation_patterns']:
                if re.search(pattern, line, re.IGNORECASE):
                    navigation_lines += 1
                    break
        
        return navigation_lines / max(len(lines), 1)
    
    def _calculate_boilerplate_ratio(self, content: str) -> float:
        """Calculate ratio of boilerplate content."""
        
        boilerplate_matches = 0
        for pattern in self.content_patterns['boilerplate_patterns']:
            boilerplate_matches += len(re.findall(pattern, content, re.IGNORECASE))
        
        words = len(content.split())
        return boilerplate_matches / max(words, 1)
    
    def _calculate_technical_relevance(self, content: str) -> float:
        """Calculate technical relevance score."""
        
        technical_matches = 0
        for pattern in self.content_patterns['content_patterns']:
            technical_matches += len(re.findall(pattern, content, re.IGNORECASE))
        
        words = len(content.split())
        return technical_matches / max(words, 1)
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score (simplified)."""
        
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Simple readability based on sentence length and word complexity
        avg_sentence_length = statistics.mean([len(s.split()) for s in sentences])
        
        words = content.split()
        avg_word_length = statistics.mean([len(word) for word in words]) if words else 0
        
        # Normalize to 0-1 scale (optimal sentence length ~15-20 words, word length ~5-6 chars)
        sentence_score = 1 - abs(avg_sentence_length - 17.5) / 30
        word_score = 1 - abs(avg_word_length - 5.5) / 10
        
        return max(0, min(1, (sentence_score + word_score) / 2))
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate content complexity score."""
        
        # Count complex indicators
        complex_indicators = [
            r'\b(however|therefore|furthermore|moreover|nevertheless)\b',  # Complex connectors
            r'\b(implementation|configuration|optimization|integration)\b',  # Technical terms
            r'\([^)]+\)',  # Parenthetical expressions
            r'[;:]',  # Complex punctuation
        ]
        
        complexity_count = 0
        for pattern in complex_indicators:
            complexity_count += len(re.findall(pattern, content, re.IGNORECASE))
        
        words = len(content.split())
        return complexity_count / max(words, 1)
    
    def _identify_issues(self, metrics: Dict[str, float], content: str) -> List[str]:
        """Identify content quality issues."""
        
        issues = []
        
        # Check against validation rules
        if metrics['word_count'] < self.validation_rules['minimum_word_count']:
            issues.append(f"Content too short ({metrics['word_count']} words, minimum {self.validation_rules['minimum_word_count']})")
        
        if metrics['link_density'] > self.validation_rules['maximum_link_density']:
            issues.append(f"Too many links ({metrics['link_density']:.1%} density, maximum {self.validation_rules['maximum_link_density']:.1%})")
        
        if metrics['content_ratio'] < self.validation_rules['minimum_content_ratio']:
            issues.append(f"Low content ratio ({metrics['content_ratio']:.1%}, minimum {self.validation_rules['minimum_content_ratio']:.1%})")
        
        if metrics['repetition_ratio'] > self.validation_rules['maximum_repetition_ratio']:
            issues.append(f"High repetition ({metrics['repetition_ratio']:.1%}, maximum {self.validation_rules['maximum_repetition_ratio']:.1%})")
        
        if metrics['unique_sentences'] < self.validation_rules['minimum_unique_sentences']:
            issues.append(f"Too few unique sentences ({metrics['unique_sentences']}, minimum {self.validation_rules['minimum_unique_sentences']})")
        
        if metrics['navigation_ratio'] > self.validation_rules['maximum_navigation_ratio']:
            issues.append(f"Too much navigation content ({metrics['navigation_ratio']:.1%}, maximum {self.validation_rules['maximum_navigation_ratio']:.1%})")
        
        if metrics['technical_relevance'] < self.validation_rules['minimum_technical_relevance']:
            issues.append(f"Low technical relevance ({metrics['technical_relevance']:.1%}, minimum {self.validation_rules['minimum_technical_relevance']:.1%})")
        
        if metrics['boilerplate_ratio'] > self.validation_rules['maximum_boilerplate_ratio']:
            issues.append(f"Too much boilerplate content ({metrics['boilerplate_ratio']:.1%}, maximum {self.validation_rules['maximum_boilerplate_ratio']:.1%})")
        
        return issues
    
    def _generate_recommendations(self, metrics: Dict[str, float], issues: List[str], url: str = "") -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        # Recommendations based on issues
        if any("too short" in issue.lower() for issue in issues):
            recommendations.append("Consider using more specific CSS selectors to capture complete content sections")
        
        if any("too many links" in issue.lower() for issue in issues):
            recommendations.append("Apply more aggressive link filtering or exclude navigation sections")
        
        if any("low content ratio" in issue.lower() for issue in issues):
            recommendations.append("Review extraction configuration to better target main content areas")
        
        if any("high repetition" in issue.lower() for issue in issues):
            recommendations.append("Implement deduplication logic to remove repeated content sections")
        
        if any("navigation content" in issue.lower() for issue in issues):
            recommendations.append("Add navigation-specific CSS selectors to exclusion list")
        
        if any("low technical relevance" in issue.lower() for issue in issues):
            recommendations.append("Verify that the correct content sections are being extracted")
        
        if any("boilerplate" in issue.lower() for issue in issues):
            recommendations.append("Add boilerplate patterns to content filtering rules")
        
        # General recommendations based on metrics
        if metrics.get('readability_score', 0) < 0.5:
            recommendations.append("Content may be fragmented; consider adjusting section boundaries")
        
        if metrics.get('heading_count', 0) == 0 and metrics.get('word_count', 0) > 100:
            recommendations.append("Content lacks structure; verify heading extraction is working")
        
        return recommendations
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score."""
        
        # Weight different metrics
        weights = {
            'content_ratio': 0.25,
            'technical_relevance': 0.20,
            'readability_score': 0.15,
            'link_density_inverse': 0.15,  # Lower is better
            'repetition_inverse': 0.10,    # Lower is better
            'navigation_inverse': 0.10,    # Lower is better
            'boilerplate_inverse': 0.05    # Lower is better
        }
        
        # Calculate weighted score
        score = 0.0
        
        score += metrics.get('content_ratio', 0) * weights['content_ratio']
        score += metrics.get('technical_relevance', 0) * weights['technical_relevance']
        score += metrics.get('readability_score', 0) * weights['readability_score']
        
        # Inverse metrics (lower is better)
        score += (1 - min(metrics.get('link_density', 0), 1)) * weights['link_density_inverse']
        score += (1 - min(metrics.get('repetition_ratio', 0), 1)) * weights['repetition_inverse']
        score += (1 - min(metrics.get('navigation_ratio', 0), 1)) * weights['navigation_inverse']
        score += (1 - min(metrics.get('boilerplate_ratio', 0), 1)) * weights['boilerplate_inverse']
        
        return min(score, 1.0)
    
    def get_quality_grade(self, quality_score: float) -> str:
        """Get quality grade based on score."""
        
        if quality_score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif quality_score >= self.quality_thresholds['good']:
            return 'good'
        elif quality_score >= self.quality_thresholds['acceptable']:
            return 'acceptable'
        else:
            return 'poor'
    
    def generate_validation_report(self, validation_result: ValidationResult, url: str = "") -> str:
        """Generate a human-readable validation report."""
        
        grade = self.get_quality_grade(validation_result.quality_score)
        
        report = f"""
Content Validation Report
{'='*50}
URL: {url}
Validation Status: {'PASSED' if validation_result.is_valid else 'FAILED'}
Quality Score: {validation_result.quality_score:.2f}/1.00
Quality Grade: {grade.upper()}
Validation Time: {validation_result.validation_timestamp.isoformat()}

Metrics:
- Word Count: {validation_result.metrics.get('word_count', 0):,}
- Content Ratio: {validation_result.metrics.get('content_ratio', 0):.1%}
- Link Density: {validation_result.metrics.get('link_density', 0):.1%}
- Technical Relevance: {validation_result.metrics.get('technical_relevance', 0):.1%}
- Readability Score: {validation_result.metrics.get('readability_score', 0):.2f}

Issues Found ({len(validation_result.issues)}):
"""
        
        for i, issue in enumerate(validation_result.issues, 1):
            report += f"{i}. {issue}\n"
        
        if not validation_result.issues:
            report += "No issues found.\n"
        
        report += f"\nRecommendations ({len(validation_result.recommendations)}):\n"
        
        for i, rec in enumerate(validation_result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        if not validation_result.recommendations:
            report += "No recommendations.\n"
        
        return report


# Global validator instance
content_validator = ContentValidator()
```

This comprehensive implementation provides advanced extraction techniques using machine learning for content classification and thorough content validation. The system can automatically identify the most relevant content sections and ensure quality standards are met before storing content in your RAG system.

The ML-based approach is particularly effective for documentation sites where traditional CSS selectors fail, as it can learn to recognize content patterns and automatically filter out navigation and boilerplate content.


## Phase 4: Testing, Deployment, and Optimization

### Step 12: Comprehensive Testing Framework

Create a comprehensive testing framework to validate all enhanced crawler capabilities and ensure reliable operation.

**12.1 Create Test Suite**

Create a new file `test_enhanced_crawler.py`:

```python
#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced crawler functionality.

This module provides thorough testing of all enhanced crawler components
including configuration, extraction, quality analysis, and ML classification.
"""

import asyncio
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import unittest
from unittest.mock import Mock, patch
from datetime import datetime

# Import enhanced crawler components
from enhanced_crawler_config import doc_site_config, DocumentationFramework
from smart_crawler_factory import smart_config_factory
from content_quality_analyzer import content_analyzer
from ml_content_classifier import ml_classifier
from content_validator import content_validator
from adaptive_config_manager import adaptive_manager
from performance_monitor import performance_monitor

# Import Crawl4AI components
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

logger = logging.getLogger(__name__)


class TestEnhancedCrawler(unittest.TestCase):
    """Test suite for enhanced crawler functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.test_urls = {
            'n8n_docs': 'https://docs.n8n.io/',
            'virustotal_docs': 'https://docs.virustotal.com/docs/how-it-works',
            'python_docs': 'https://docs.python.org/3/tutorial/',
            'react_docs': 'https://reactjs.org/docs/getting-started.html'
        }
        
        # Sample content for testing
        cls.sample_content = {
            'good_content': """
# API Documentation

This guide explains how to use the API effectively.

## Authentication

To authenticate with the API, you need to include your API key in the request headers:

```python
headers = {
    'Authorization': 'Bearer your-api-key'
}
```

## Making Requests

The API supports standard HTTP methods. Here's an example of making a GET request:

```python
import requests

response = requests.get('https://api.example.com/data', headers=headers)
data = response.json()
```

## Error Handling

The API returns standard HTTP status codes. Common errors include:

- 401: Unauthorized - Invalid API key
- 404: Not Found - Resource doesn't exist
- 500: Internal Server Error - Server issue

For more information, see the complete API reference.
            """,
            
            'poor_content': """
[Home](/) > [Docs](/docs) > [API](/docs/api)

Navigation:
- [Getting Started](/getting-started)
- [API Reference](/api)
- [Examples](/examples)
- [Support](/support)

Footer Links:
- [Privacy Policy](/privacy)
- [Terms of Service](/terms)
- [Contact Us](/contact)

Social Media:
- [Twitter](https://twitter.com/example)
- [GitHub](https://github.com/example)
- [LinkedIn](https://linkedin.com/company/example)

Copyright © 2024 Example Company. All rights reserved.
            """,
            
            'mixed_content': """
# Getting Started

[Home](/) > [Docs](/docs) > Getting Started

Welcome to our platform! This guide will help you get started quickly.

## Quick Links
- [Installation](/install)
- [Configuration](/config)
- [Examples](/examples)

## Installation

To install the software, run the following command:

```bash
npm install example-package
```

## Configuration

Create a configuration file with the following structure:

```json
{
  "apiKey": "your-key-here",
  "endpoint": "https://api.example.com"
}
```

## Next Steps

After installation, you can:
1. Configure your settings
2. Run your first example
3. Explore advanced features

For more help, visit our [support page](/support) or [contact us](/contact).

Footer: Privacy Policy | Terms of Service | © 2024 Example
            """
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    def test_framework_detection(self):
        """Test documentation framework detection."""
        
        # Test domain-based detection
        framework, config = doc_site_config.get_config_for_url('https://docs.n8n.io/')
        self.assertEqual(framework, DocumentationFramework.MATERIAL_DESIGN)
        
        framework, config = doc_site_config.get_config_for_url('https://docs.virustotal.com/')
        self.assertEqual(framework, DocumentationFramework.README_IO)
        
        # Test generic fallback
        framework, config = doc_site_config.get_config_for_url('https://unknown-site.com/docs/')
        self.assertEqual(framework, DocumentationFramework.GENERIC)
        
        # Test HTML content analysis
        html_content = '<div class="md-content">Material Design content</div>'
        framework, config = doc_site_config.get_config_for_url('https://example.com/', html_content)
        self.assertEqual(framework, DocumentationFramework.MATERIAL_DESIGN)
    
    def test_configuration_creation(self):
        """Test smart configuration creation."""
        
        # Test basic configuration creation
        config, framework = smart_config_factory.create_adaptive_config('https://docs.n8n.io/')
        self.assertIsInstance(config, CrawlerRunConfig)
        self.assertEqual(framework, DocumentationFramework.MATERIAL_DESIGN)
        
        # Test configuration with custom overrides
        overrides = {'word_count_threshold': 25}
        config, framework = smart_config_factory.create_adaptive_config(
            'https://docs.n8n.io/', 
            custom_overrides=overrides
        )
        self.assertEqual(config.word_count_threshold, 25)
    
    def test_content_quality_analysis(self):
        """Test content quality analysis."""
        
        # Test good content
        metrics = content_analyzer.analyze_content(self.sample_content['good_content'])
        self.assertGreater(metrics.quality_score, 0.7)
        self.assertGreater(metrics.content_ratio, 0.6)
        self.assertEqual(metrics.quality_grade, 'good')
        
        # Test poor content
        metrics = content_analyzer.analyze_content(self.sample_content['poor_content'])
        self.assertLess(metrics.quality_score, 0.4)
        self.assertLess(metrics.content_ratio, 0.3)
        self.assertEqual(metrics.quality_grade, 'poor')
        
        # Test mixed content
        metrics = content_analyzer.analyze_content(self.sample_content['mixed_content'])
        self.assertGreater(metrics.quality_score, 0.5)
        self.assertGreater(metrics.content_ratio, 0.4)
    
    def test_ml_content_classification(self):
        """Test ML-based content classification."""
        
        # Test content section classification
        sections = ml_classifier.classify_content_sections(self.sample_content['mixed_content'])
        
        self.assertGreater(len(sections), 0)
        
        # Check that we have different section types
        section_types = {section.section_type for section in sections}
        self.assertIn('content', section_types)
        
        # Test content extraction
        extracted_content = ml_classifier.extract_content_sections(sections)
        self.assertGreater(len(extracted_content), 0)
        
        # Test classification summary
        summary = ml_classifier.get_classification_summary(sections)
        self.assertIn('total_sections', summary)
        self.assertIn('content_ratio', summary)
        self.assertGreater(summary['content_ratio'], 0)
    
    def test_content_validation(self):
        """Test content validation."""
        
        # Test good content validation
        result = content_validator.validate_content(self.sample_content['good_content'])
        self.assertTrue(result.is_valid)
        self.assertGreater(result.quality_score, 0.5)
        self.assertLessEqual(len(result.issues), 2)
        
        # Test poor content validation
        result = content_validator.validate_content(self.sample_content['poor_content'])
        self.assertFalse(result.is_valid)
        self.assertLess(result.quality_score, 0.5)
        self.assertGreater(len(result.issues), 0)
        
        # Test empty content
        result = content_validator.validate_content("")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.quality_score, 0.0)
        self.assertIn("Content is empty", result.issues)
    
    def test_adaptive_configuration(self):
        """Test adaptive configuration management."""
        
        # Create mock extraction attempts
        from enhanced_crawler_config import ExtractionConfig
        from content_quality_analyzer import ContentMetrics
        
        base_config = ExtractionConfig()
        
        # Simulate poor quality extraction
        poor_metrics = ContentMetrics(
            total_words=100, total_lines=20, total_characters=500,
            content_words=30, navigation_words=70, code_words=0,
            headings_count=1, paragraphs_count=5, links_count=15, code_blocks_count=0,
            content_ratio=0.3, link_density=0.6, code_ratio=0.0,
            avg_sentence_length=8.0, avg_paragraph_length=15.0, complexity_score=0.2,
            quality_score=0.3, quality_grade='poor'
        )
        
        # Record extraction attempt
        adaptive_manager.record_extraction(
            'https://test.com',
            DocumentationFramework.GENERIC,
            base_config,
            poor_metrics,
            True
        )
        
        # Test optimization
        optimized_config, was_modified = adaptive_manager.get_optimized_config(
            DocumentationFramework.GENERIC,
            base_config
        )
        
        # Should not modify with insufficient data
        self.assertFalse(was_modified)
    
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        
        # Test report generation with no data
        report = performance_monitor.generate_performance_report(days=1)
        self.assertIn('error', report)
        
        # Test HTML report generation
        mock_report = {
            'report_period': {'days': 7, 'start_date': '2024-01-01', 'end_date': '2024-01-07', 'total_attempts': 10},
            'overall_performance': {'success_rate': 0.8, 'success_rate_grade': 'good', 'average_quality_score': 0.7, 'quality_score_grade': 'good', 'average_content_ratio': 0.6, 'content_ratio_grade': 'good'},
            'framework_performance': {},
            'recommendations': ['Test recommendation'],
            'report_generated': '2024-01-07T12:00:00'
        }
        
        html_report = performance_monitor.generate_html_report(mock_report)
        self.assertIn('Performance Report', html_report)
        self.assertIn('Test recommendation', html_report)


class TestIntegration(unittest.TestCase):
    """Integration tests for enhanced crawler."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.browser_config = BrowserConfig(headless=True, verbose=False)
    
    @unittest.skipIf(not hasattr(unittest, 'INTEGRATION_TESTS'), "Integration tests disabled")
    async def test_end_to_end_crawling(self):
        """Test end-to-end crawling with enhanced features."""
        
        test_url = "https://docs.python.org/3/tutorial/introduction.html"
        
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            # Test framework detection
            framework, base_config = doc_site_config.get_config_for_url(test_url)
            
            # Create enhanced configuration
            enhanced_config, detected_framework = smart_config_factory.create_adaptive_config(test_url)
            
            # Crawl with enhanced configuration
            result = await crawler.arun(url=test_url, config=enhanced_config)
            
            self.assertTrue(result.success)
            self.assertIsNotNone(result.markdown)
            
            # Test quality analysis
            quality_metrics = content_analyzer.analyze_content(result.markdown, test_url)
            self.assertGreater(quality_metrics.quality_score, 0.3)
            
            # Test ML classification
            sections = ml_classifier.classify_content_sections(result.markdown)
            self.assertGreater(len(sections), 0)
            
            # Test content validation
            validation_result = content_validator.validate_content(result.markdown, test_url)
            self.assertIsNotNone(validation_result)
    
    @unittest.skipIf(not hasattr(unittest, 'INTEGRATION_TESTS'), "Integration tests disabled")
    async def test_batch_crawling(self):
        """Test batch crawling with enhanced features."""
        
        test_urls = [
            "https://docs.python.org/3/tutorial/introduction.html",
            "https://docs.python.org/3/tutorial/controlflow.html"
        ]
        
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            # Test batch configuration creation
            configs = []
            for url in test_urls:
                config, framework = smart_config_factory.create_adaptive_config(url)
                configs.append(config)
            
            # Use the first config for batch processing
            results = await crawler.arun_many(urls=test_urls, config=configs[0])
            
            self.assertEqual(len(results), len(test_urls))
            
            for result in results:
                if result.success:
                    # Test quality analysis for each result
                    quality_metrics = content_analyzer.analyze_content(result.markdown, result.url)
                    self.assertIsNotNone(quality_metrics)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for enhanced crawler."""
    
    def test_framework_detection_performance(self):
        """Benchmark framework detection performance."""
        
        import time
        
        test_urls = [
            'https://docs.n8n.io/',
            'https://docs.virustotal.com/',
            'https://docs.python.org/',
            'https://reactjs.org/docs/',
            'https://unknown-site.com/docs/'
        ] * 100  # Test with 500 URLs
        
        start_time = time.time()
        
        for url in test_urls:
            framework, config = doc_site_config.get_config_for_url(url)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(test_urls)
        
        # Should process at least 1000 URLs per second
        self.assertLess(avg_time, 0.001)
        
        print(f"Framework detection: {len(test_urls)} URLs in {total_time:.3f}s ({avg_time*1000:.3f}ms per URL)")
    
    def test_content_analysis_performance(self):
        """Benchmark content analysis performance."""
        
        import time
        
        # Create test content of various sizes
        test_contents = []
        base_content = "This is a test sentence with some technical terms like API and configuration. "
        
        for size in [100, 500, 1000, 5000]:  # Different content sizes
            content = base_content * (size // len(base_content))
            test_contents.append(content)
        
        for content in test_contents:
            start_time = time.time()
            
            # Test quality analysis
            metrics = content_analyzer.analyze_content(content)
            
            # Test ML classification
            sections = ml_classifier.classify_content_sections(content)
            
            # Test validation
            validation_result = content_validator.validate_content(content)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process content quickly
            self.assertLess(processing_time, 2.0)  # Max 2 seconds for any content size
            
            print(f"Content analysis: {len(content)} chars in {processing_time:.3f}s")


def create_test_suite():
    """Create comprehensive test suite."""
    
    suite = unittest.TestSuite()
    
    # Add unit tests
    suite.addTest(unittest.makeSuite(TestEnhancedCrawler))
    
    # Add integration tests (if enabled)
    if hasattr(unittest, 'INTEGRATION_TESTS'):
        suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Add performance benchmarks
    suite.addTest(unittest.makeSuite(TestPerformanceBenchmarks))
    
    return suite


def run_tests(include_integration=False, include_benchmarks=True):
    """Run the test suite."""
    
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Enable integration tests if requested
    if include_integration:
        unittest.INTEGRATION_TESTS = True
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced crawler tests')
    parser.add_argument('--integration', action='store_true', help='Include integration tests')
    parser.add_argument('--no-benchmarks', action='store_true', help='Skip performance benchmarks')
    
    args = parser.parse_args()
    
    success = run_tests(
        include_integration=args.integration,
        include_benchmarks=not args.no_benchmarks
    )
    
    exit(0 if success else 1)
```

**12.2 Create Deployment Script**

Create a new file `deploy_enhanced_crawler.py`:

```python
#!/usr/bin/env python3
"""
Deployment script for enhanced crawler system.

This script handles the deployment and configuration of the enhanced crawler
components in your existing RAG system.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import subprocess

logger = logging.getLogger(__name__)


class EnhancedCrawlerDeployer:
    """Handles deployment of enhanced crawler components."""
    
    def __init__(self, target_directory: str, backup_existing: bool = True):
        self.target_dir = Path(target_directory)
        self.backup_existing = backup_existing
        self.backup_dir = self.target_dir / "backup_original"
        
        # Component files to deploy
        self.component_files = [
            'enhanced_crawler_config.py',
            'smart_crawler_factory.py',
            'content_quality_analyzer.py',
            'ml_content_classifier.py',
            'content_validator.py',
            'adaptive_config_manager.py',
            'performance_monitor.py'
        ]
        
        # Dependencies to check
        self.required_dependencies = [
            'crawl4ai',
            'scikit-learn',
            'numpy',
            'supabase'
        ]
    
    def deploy(self) -> bool:
        """
        Deploy enhanced crawler components.
        
        Returns:
            True if deployment successful, False otherwise
        """
        
        try:
            logger.info("Starting enhanced crawler deployment...")
            
            # Check prerequisites
            if not self._check_prerequisites():
                return False
            
            # Create backup if requested
            if self.backup_existing:
                self._create_backup()
            
            # Deploy component files
            self._deploy_components()
            
            # Update existing files
            self._update_existing_files()
            
            # Install dependencies
            self._install_dependencies()
            
            # Run validation tests
            if not self._validate_deployment():
                logger.error("Deployment validation failed")
                return False
            
            logger.info("Enhanced crawler deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _check_prerequisites(self) -> bool:
        """Check deployment prerequisites."""
        
        logger.info("Checking prerequisites...")
        
        # Check target directory exists
        if not self.target_dir.exists():
            logger.error(f"Target directory does not exist: {self.target_dir}")
            return False
        
        # Check for existing crawl4ai_mcp.py
        if not (self.target_dir / "crawl4ai_mcp.py").exists():
            logger.error("crawl4ai_mcp.py not found in target directory")
            return False
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher required")
            return False
        
        logger.info("Prerequisites check passed")
        return True
    
    def _create_backup(self) -> None:
        """Create backup of existing files."""
        
        logger.info("Creating backup of existing files...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        self.backup_dir.mkdir(parents=True)
        
        # Backup existing files that will be modified
        files_to_backup = [
            'crawl4ai_mcp.py',
            'improved_chunking.py',
            'utils.py',
            'config.py'
        ]
        
        for filename in files_to_backup:
            source_file = self.target_dir / filename
            if source_file.exists():
                backup_file = self.backup_dir / filename
                shutil.copy2(source_file, backup_file)
                logger.info(f"Backed up {filename}")
    
    def _deploy_components(self) -> None:
        """Deploy enhanced crawler component files."""
        
        logger.info("Deploying enhanced crawler components...")
        
        # Get the directory containing this script
        script_dir = Path(__file__).parent
        
        for component_file in self.component_files:
            source_file = script_dir / component_file
            target_file = self.target_dir / component_file
            
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                logger.info(f"Deployed {component_file}")
            else:
                logger.warning(f"Component file not found: {component_file}")
    
    def _update_existing_files(self) -> None:
        """Update existing files with enhanced functionality."""
        
        logger.info("Updating existing files...")
        
        # Update crawl4ai_mcp.py with enhanced tools
        self._update_crawl4ai_mcp()
        
        # Update other files as needed
        self._update_config_files()
    
    def _update_crawl4ai_mcp(self) -> None:
        """Update crawl4ai_mcp.py with enhanced functionality."""
        
        mcp_file = self.target_dir / "crawl4ai_mcp.py"
        
        if not mcp_file.exists():
            logger.error("crawl4ai_mcp.py not found")
            return
        
        # Read existing content
        with open(mcp_file, 'r') as f:
            content = f.read()
        
        # Add imports if not present
        imports_to_add = [
            "from smart_crawler_factory import smart_config_factory",
            "from enhanced_crawler_config import doc_site_config, DocumentationFramework",
            "from content_quality_analyzer import content_analyzer",
            "from ml_content_classifier import ml_classifier",
            "from content_validator import content_validator",
            "from adaptive_config_manager import adaptive_manager",
            "from performance_monitor import performance_monitor"
        ]
        
        for import_line in imports_to_add:
            if import_line not in content:
                # Add import after existing imports
                import_section = content.find("import")
                if import_section != -1:
                    # Find end of import section
                    lines = content.split('\n')
                    insert_index = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith(('import', 'from')):
                            insert_index = i + 1
                    
                    lines.insert(insert_index, import_line)
                    content = '\n'.join(lines)
        
        # Add enhanced tool registrations
        tool_registrations = """
# Enhanced crawler tools
@mcp.tool()
async def crawl_single_page_enhanced_tool(ctx: Context, url: str, use_enhanced_config: bool = True) -> str:
    \"\"\"Enhanced single page crawler with intelligent framework detection.\"\"\"
    return await crawl_single_page_enhanced(ctx, url, use_enhanced_config)

@mcp.tool()
async def crawl_single_page_ml_enhanced_tool(ctx: Context, url: str, use_ml_classification: bool = True) -> str:
    \"\"\"ML-enhanced single page crawler with intelligent content classification.\"\"\"
    return await crawl_single_page_ml_enhanced(ctx, url, use_ml_classification)

@mcp.tool()
async def analyze_site_framework_tool(ctx: Context, url: str) -> str:
    \"\"\"Analyze a documentation site to detect its framework and optimal configuration.\"\"\"
    return await analyze_site_framework(ctx, url)

@mcp.tool()
async def generate_performance_report_tool(ctx: Context, days: int = 30) -> str:
    \"\"\"Generate performance report for enhanced crawler.\"\"\"
    try:
        report = performance_monitor.generate_performance_report(days)
        report_path = performance_monitor.save_report(report)
        
        return json.dumps({
            "success": True,
            "report": report,
            "report_file": report_path
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)
"""
        
        # Add tool registrations if not present
        if "@mcp.tool()" not in content or "crawl_single_page_enhanced_tool" not in content:
            content += tool_registrations
        
        # Write updated content
        with open(mcp_file, 'w') as f:
            f.write(content)
        
        logger.info("Updated crawl4ai_mcp.py with enhanced functionality")
    
    def _update_config_files(self) -> None:
        """Update configuration files."""
        
        # Update config.py if it exists
        config_file = self.target_dir / "config.py"
        if config_file.exists():
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Add enhanced crawler configuration
            enhanced_config = """
# Enhanced crawler configuration
ENHANCED_CRAWLER_CONFIG = {
    'enable_adaptive_learning': True,
    'enable_ml_classification': True,
    'enable_content_validation': True,
    'quality_threshold': 0.5,
    'max_extraction_attempts': 3,
    'performance_monitoring': True
}
"""
            
            if "ENHANCED_CRAWLER_CONFIG" not in content:
                content += enhanced_config
                
                with open(config_file, 'w') as f:
                    f.write(content)
                
                logger.info("Updated config.py with enhanced crawler settings")
    
    def _install_dependencies(self) -> None:
        """Install required dependencies."""
        
        logger.info("Installing dependencies...")
        
        for dependency in self.required_dependencies:
            try:
                __import__(dependency.replace('-', '_'))
                logger.info(f"Dependency {dependency} already installed")
            except ImportError:
                logger.info(f"Installing {dependency}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', dependency])
    
    def _validate_deployment(self) -> bool:
        """Validate deployment by running basic tests."""
        
        logger.info("Validating deployment...")
        
        try:
            # Test imports
            sys.path.insert(0, str(self.target_dir))
            
            import enhanced_crawler_config
            import smart_crawler_factory
            import content_quality_analyzer
            import ml_content_classifier
            import content_validator
            import adaptive_config_manager
            import performance_monitor
            
            logger.info("All components imported successfully")
            
            # Test basic functionality
            framework, config = enhanced_crawler_config.doc_site_config.get_config_for_url('https://docs.example.com/')
            logger.info(f"Framework detection test passed: {framework}")
            
            # Test content analysis
            test_content = "This is a test content for validation."
            metrics = content_quality_analyzer.content_analyzer.analyze_content(test_content)
            logger.info(f"Content analysis test passed: quality score {metrics.quality_score:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def rollback(self) -> bool:
        """Rollback deployment to previous state."""
        
        if not self.backup_dir.exists():
            logger.error("No backup found for rollback")
            return False
        
        try:
            logger.info("Rolling back deployment...")
            
            # Restore backed up files
            for backup_file in self.backup_dir.iterdir():
                target_file = self.target_dir / backup_file.name
                shutil.copy2(backup_file, target_file)
                logger.info(f"Restored {backup_file.name}")
            
            # Remove deployed component files
            for component_file in self.component_files:
                component_path = self.target_dir / component_file
                if component_path.exists():
                    component_path.unlink()
                    logger.info(f"Removed {component_file}")
            
            logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


def main():
    """Main deployment function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy enhanced crawler system')
    parser.add_argument('target_dir', help='Target directory containing your RAG system')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup of existing files')
    parser.add_argument('--rollback', action='store_true', help='Rollback previous deployment')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create deployer
    deployer = EnhancedCrawlerDeployer(
        target_directory=args.target_dir,
        backup_existing=not args.no_backup
    )
    
    # Perform deployment or rollback
    if args.rollback:
        success = deployer.rollback()
    else:
        success = deployer.deploy()
    
    if success:
        logger.info("Operation completed successfully!")
        return 0
    else:
        logger.error("Operation failed!")
        return 1


if __name__ == '__main__':
    exit(main())
```

### Step 13: Documentation and User Guide

Create comprehensive documentation for the enhanced crawler system.

**13.1 Create User Guide**

Create a new file `ENHANCED_CRAWLER_GUIDE.md`:

```markdown
# Enhanced RAG Crawler User Guide

## Overview

The Enhanced RAG Crawler is a sophisticated content extraction system designed specifically for documentation sites. It addresses the common problem of poor content quality when crawling documentation sites with complex navigation structures.

## Key Features

### 🎯 Intelligent Framework Detection
- Automatically detects documentation frameworks (Material Design, ReadMe.io, GitBook, etc.)
- Applies framework-specific extraction configurations
- Supports custom configuration for unknown frameworks

### 🧠 Machine Learning Content Classification
- Uses ML techniques to identify substantive content vs navigation
- Automatically filters out boilerplate and navigation content
- Provides confidence scores for extracted content

### 📊 Quality Monitoring and Validation
- Comprehensive content quality analysis
- Real-time quality metrics and scoring
- Automatic quality validation with detailed reports

### 🔄 Adaptive Learning
- Learns from extraction results over time
- Automatically optimizes configurations based on performance
- Maintains extraction history for continuous improvement

### 📈 Performance Monitoring
- Detailed performance reports and analytics
- Framework-specific performance tracking
- Actionable recommendations for improvement

## Quick Start

### 1. Basic Enhanced Extraction

```python
# Use enhanced single page crawler
result = await crawl_single_page_enhanced_tool(ctx, "https://docs.example.com/")
```

### 2. ML-Enhanced Extraction

```python
# Use ML-based content classification
result = await crawl_single_page_ml_enhanced_tool(ctx, "https://docs.example.com/")
```

### 3. Site Analysis

```python
# Analyze a site's framework and configuration
analysis = await analyze_site_framework_tool(ctx, "https://docs.example.com/")
```

## Configuration

### Framework-Specific Settings

The system automatically detects and applies optimal settings for different documentation frameworks:

- **Material Design** (MkDocs, n8n Docs): Targets `.md-content`, excludes `.md-sidebar`
- **ReadMe.io** (VirusTotal Docs): Targets `.rm-Content`, excludes `.rm-Sidebar`
- **GitBook**: Targets main content areas, excludes navigation
- **Docusaurus**: Targets `.docMainContainer`, excludes sidebars
- **Sphinx** (Python Docs): Targets `.document`, excludes `.sphinxsidebar`

### Custom Configuration

You can override default settings:

```python
custom_overrides = {
    'word_count_threshold': 25,
    'exclude_external_links': True,
    'process_iframes': False
}

config, framework = smart_config_factory.create_adaptive_config(
    url, 
    custom_overrides=custom_overrides
)
```

## Quality Metrics

### Content Quality Scores

- **Excellent (0.85+)**: High-quality, substantive content with minimal navigation
- **Good (0.70-0.84)**: Good content quality with some navigation elements
- **Acceptable (0.50-0.69)**: Usable content but may need improvement
- **Poor (<0.50)**: Low quality, mostly navigation or boilerplate

### Key Metrics

- **Content Ratio**: Percentage of substantive content vs navigation
- **Link Density**: Ratio of links to total words
- **Technical Relevance**: Presence of technical documentation terms
- **Readability Score**: Content structure and complexity assessment

## Troubleshooting

### Common Issues

#### Low Content Quality
**Symptoms**: Quality scores below 0.5, high navigation ratio
**Solutions**:
- Enable ML classification: `use_ml_classification=True`
- Use CSS selector approach for specific targeting
- Add site-specific exclusion rules

#### Framework Detection Issues
**Symptoms**: Generic framework detected for known documentation sites
**Solutions**:
- Provide HTML content for analysis
- Add custom domain patterns
- Use manual framework specification

#### Performance Issues
**Symptoms**: Slow extraction, high resource usage
**Solutions**:
- Reduce concurrent sessions
- Enable caching
- Use batch processing for multiple URLs

### Error Codes

- **Framework Detection Failed**: Unknown site structure, using generic configuration
- **Content Validation Failed**: Extracted content doesn't meet quality thresholds
- **ML Classification Error**: Machine learning processing failed, falling back to standard extraction

## Best Practices

### 1. Site-Specific Optimization

For frequently crawled sites, analyze the framework and create custom configurations:

```python
# Analyze site structure
analysis = await analyze_site_framework_tool(ctx, url)

# Review recommendations and apply custom settings
custom_config = create_custom_config_based_on_analysis(analysis)
```

### 2. Quality Monitoring

Regularly review performance reports to identify improvement opportunities:

```python
# Generate monthly performance report
report = await generate_performance_report_tool(ctx, days=30)
```

### 3. Adaptive Learning

Enable adaptive learning for continuous improvement:

```python
# Enable learning from extraction results
config, framework = smart_config_factory.create_adaptive_config(
    url, 
    enable_learning=True
)
```

### 4. Content Validation

Always validate content quality before storage:

```python
# Validate extracted content
validation_result = content_validator.validate_content(content, url)

if not validation_result.is_valid:
    # Handle low-quality content
    apply_alternative_extraction_method()
```

## API Reference

### Enhanced Tools

#### `crawl_single_page_enhanced_tool`
Enhanced single page crawler with intelligent framework detection.

**Parameters**:
- `url` (str): URL to crawl
- `use_enhanced_config` (bool): Whether to use enhanced configuration (default: True)

**Returns**: JSON with crawl results and quality metrics

#### `crawl_single_page_ml_enhanced_tool`
ML-enhanced crawler with intelligent content classification.

**Parameters**:
- `url` (str): URL to crawl
- `use_ml_classification` (bool): Whether to use ML classification (default: True)

**Returns**: JSON with crawl results and ML classification summary

#### `analyze_site_framework_tool`
Analyze a site's framework and optimal configuration.

**Parameters**:
- `url` (str): URL to analyze

**Returns**: JSON with framework analysis and recommendations

#### `generate_performance_report_tool`
Generate performance report for the enhanced crawler.

**Parameters**:
- `days` (int): Number of days to include in report (default: 30)

**Returns**: JSON with performance report and metrics

### Configuration Classes

#### `DocumentationFramework`
Enumeration of supported documentation frameworks.

#### `ExtractionConfig`
Configuration for content extraction from a specific framework.

#### `ContentMetrics`
Comprehensive content quality metrics.

#### `ValidationResult`
Result of content validation with issues and recommendations.

## Advanced Usage

### Custom Framework Support

Add support for new documentation frameworks:

```python
# Define custom framework
class CustomFramework(DocumentationFramework):
    CUSTOM_DOCS = "custom_docs"

# Create custom configuration
custom_config = ExtractionConfig(
    target_elements=["main.custom-content"],
    excluded_tags=["nav", "aside"],
    excluded_selectors=[".custom-sidebar"],
    word_count_threshold=20
)

# Register with configuration system
doc_site_config.framework_configs[CustomFramework.CUSTOM_DOCS] = custom_config
```

### Batch Processing Optimization

Optimize batch processing for large-scale crawling:

```python
# Group URLs by framework for optimized batch processing
url_groups = {}
for url in urls:
    framework, _ = doc_site_config.get_config_for_url(url)
    if framework not in url_groups:
        url_groups[framework] = []
    url_groups[framework].append(url)

# Process each group with framework-specific configuration
for framework, framework_urls in url_groups.items():
    config = smart_config_factory.create_config(framework_urls[0])
    results = await crawler.arun_many(urls=framework_urls, config=config)
```

### Performance Tuning

Optimize performance for your specific use case:

```python
# Memory-adaptive dispatcher for large batches
dispatcher = MemoryAdaptiveDispatcher(
    memory_threshold_percent=70.0,
    check_interval=1.0,
    max_session_permit=max_concurrent
)

# Optimized configuration for speed vs quality
speed_optimized_config = {
    'word_count_threshold': 10,  # Lower threshold for speed
    'exclude_external_links': True,  # Reduce processing overhead
    'process_iframes': False,  # Skip iframe processing
}
```

## Support and Contributing

### Getting Help

1. Check the troubleshooting section above
2. Review performance reports for insights
3. Analyze site framework for configuration recommendations
4. Check logs for detailed error information

### Contributing

To contribute improvements:

1. Run the test suite: `python test_enhanced_crawler.py`
2. Add tests for new functionality
3. Update documentation
4. Submit pull request with detailed description

### Reporting Issues

When reporting issues, include:

- URL that's causing problems
- Framework detection results
- Quality metrics and validation results
- Error logs and stack traces
- Performance report if relevant

## Changelog

### Version 1.0.0
- Initial release with framework detection
- Basic quality analysis and validation
- ML-based content classification
- Adaptive configuration management
- Performance monitoring and reporting

---

For more detailed technical information, see the implementation documentation and API reference.
```

This comprehensive implementation plan provides a complete solution for enhancing your RAG crawler to handle documentation sites effectively. The system includes intelligent framework detection, ML-based content classification, quality validation, adaptive learning, and comprehensive monitoring capabilities.

The implementation is designed to be:
- **Backward compatible** with your existing system
- **Incrementally deployable** through phases
- **Thoroughly tested** with comprehensive test suites
- **Well documented** with clear user guides
- **Performance optimized** for production use

Each phase builds upon the previous one, allowing you to implement and validate improvements incrementally while maintaining system stability.

