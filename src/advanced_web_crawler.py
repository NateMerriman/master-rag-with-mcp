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

from typing import Dict, List, Optional, Any, Tuple
import asyncio
import time
import logging
import re
import json
from dataclasses import dataclass
from urllib.parse import urlparse

# Import specific exception types for granular error handling
try:
    from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError
    from playwright._impl._errors import Error as PlaywrightError
except ImportError:
    PlaywrightTimeoutError = TimeoutError
    PlaywrightError = Exception

try:
    import aiohttp
    from aiohttp import ClientError, ClientTimeout, ClientConnectionError
except ImportError:
    ClientError = ConnectionError
    ClientTimeout = TimeoutError  
    ClientConnectionError = ConnectionError

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
class UnifiedQualityResult:
    """Unified result from quality validation system with clear prioritization."""
    
    quality_score: float = 0.0
    quality_passed: bool = False
    validation_method: str = "none"  # "enhanced", "legacy", "disabled", "failed"
    issues: List[str] = None
    
    # Enhanced quality metrics if available
    enhanced_metrics: Optional['ContentQualityMetrics'] = None
    
    # Legacy validation result if available  
    legacy_validation: Optional['QualityValidationResult'] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


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
        
        # ENHANCED: 4-tier fallback selector hierarchy for robust content extraction
        self.fallback_css_selectors = [
            # TIER 1: High-precision framework-specific selectors (if framework detection fails)
            "main.md-main .md-content, .rm-Article .markdown-body, .gitbook-content .markdown-section, main.docMainContainer .markdown",
            
            # TIER 2: Standard semantic HTML5 containers  
            "main article, article[role='main'], main[role='main'], [role='main'] article",
            
            # TIER 3: Common generic class names (ordered by specificity)
            ".content, .main-content, .page-content, .docs-content, .documentation, .article-content",
            
            # TIER 4: Last resort with aggressive exclusion of navigation
            "body > main, body > article, body > .content, body > div:not(.sidebar):not(.navigation):not(.nav):not(.menu):not(.header):not(.footer)"
        ]
        
        # Enhanced exclusion selectors for each tier
        self.tier_exclusions = {
            1: [  # Minimal exclusions for high-precision selectors
                ".sidebar", ".navigation", ".nav", ".menu", ".breadcrumb"
            ],
            2: [  # Moderate exclusions for semantic selectors
                ".sidebar", ".navigation", ".nav", ".menu", ".breadcrumb", 
                ".toc", ".table-of-contents", ".pagination", "header", "footer"
            ],
            3: [  # More aggressive exclusions for generic selectors
                ".sidebar", ".navigation", ".nav", ".menu", ".breadcrumb",
                ".toc", ".table-of-contents", ".pagination", "header", "footer",
                ".related", ".recommended", ".social", ".share", ".comments"
            ],
            4: [  # Very aggressive exclusions for last resort
                ".sidebar", ".navigation", ".nav", ".menu", ".breadcrumb",
                ".toc", ".table-of-contents", ".pagination", "header", "footer", 
                ".related", ".recommended", ".social", ".share", ".comments",
                "aside", ".widget", ".advertisement", ".ad", ".banner", ".promo"
            ]
        }
        
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
        Create a tier-based fallback configuration for when primary extraction fails.
        
        Uses a 4-tier fallback hierarchy with progressively more aggressive exclusions:
        - Tier 1: High-precision framework selectors with minimal exclusions
        - Tier 2: Semantic HTML5 containers with moderate exclusions  
        - Tier 3: Generic class names with aggressive exclusions
        - Tier 4: Last resort with very aggressive exclusions
        
        Args:
            attempt_number: The attempt number (1-based)
            
        Returns:
            Tier-appropriate CrawlerRunConfig
        """
        # Determine tier and selector
        if attempt_number <= len(self.fallback_css_selectors):
            css_selector = self.fallback_css_selectors[attempt_number - 1]
            tier = attempt_number
        else:
            # Emergency fallback beyond tier 4
            css_selector = "body"
            tier = 4
            
        # Get tier-appropriate exclusions
        tier_exclusions = self.tier_exclusions.get(tier, self.tier_exclusions[4])
        combined_exclusions = tier_exclusions + self.excluded_selectors
        
        # Adjust word count threshold based on tier (lower thresholds for more desperate attempts)
        word_threshold = max(5, 20 - (tier * 3))  # 17, 14, 11, 8, then minimum 5
        
        # Create tier-specific excluded tags
        if tier <= 2:
            excluded_tags = ["script", "style", "noscript"]  # Minimal for precise selectors
        elif tier == 3:
            excluded_tags = ["nav", "script", "style", "noscript"]  # Add nav for generic
        else:
            excluded_tags = ["nav", "header", "footer", "aside", "script", "style", "noscript"]  # Aggressive
        
        logger.info(f"Creating Tier {tier} fallback config with {len(combined_exclusions)} exclusions, word threshold: {word_threshold}")
        
        return CrawlerRunConfig(
            # Use NoExtractionStrategy for consistent extraction
            extraction_strategy=NoExtractionStrategy(),
            
            # Tier-optimized markdown generation
            markdown_generator=DefaultMarkdownGenerator(
                options={
                    'ignore_links': tier >= 3,  # Preserve links in precise tiers, remove in desperate ones
                    'ignore_images': True,
                    'protect_links': tier <= 2,  # Protect links only in precise tiers
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
            excluded_tags=excluded_tags,
            excluded_selector=", ".join(combined_exclusions),
            word_count_threshold=word_threshold,
            exclude_external_links=tier >= 3,  # More aggressive link filtering in desperate tiers
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
                    
                    # ENHANCED: Real-time selector validation (Task 20.4)
                    tier = 0 if attempt == 1 else attempt - 1  # Primary = tier 0, fallbacks = tier 1-4
                    is_valid_content, rt_quality_score, rt_suggestions = self._perform_real_time_selector_validation(
                        cleaned_markdown, url, tier
                    )
                    
                    # Log real-time validation results
                    logger.info(f"Real-time validation (tier {tier}): valid={is_valid_content}, score={rt_quality_score:.3f}")
                    for suggestion in rt_suggestions[:3]:  # Log first 3 suggestions
                        logger.info(f"  Suggestion: {suggestion}")
                    
                    # Enhanced quality validation if available
                    enhanced_quality_metrics = None
                    if self.enable_enhanced_quality and self.enhanced_quality_analyzer and ENHANCED_QUALITY_AVAILABLE:
                        enhanced_quality_metrics = calculate_content_quality(cleaned_markdown)
                        
                        # Log quality metrics
                        log_quality_metrics(enhanced_quality_metrics, url, framework_detected)
                        
                        # Combine real-time validation with enhanced quality metrics
                        should_retry_enhanced = should_retry_extraction(enhanced_quality_metrics)
                        combined_should_retry = (not is_valid_content) or should_retry_enhanced
                        
                        logger.info(f"Combined validation: real-time={is_valid_content}, enhanced={not should_retry_enhanced}, overall={'ACCEPT' if not combined_should_retry else 'RETRY'}")
                        
                        # Check if quality is acceptable (both real-time and enhanced must pass)
                        if not combined_should_retry or attempt >= self.max_fallback_attempts + 1:
                            # Quality is acceptable or we've exhausted attempts
                            break
                        else:
                            logger.info(f"Quality validation failed for {url}, trying tier {attempt} fallback")
                            # Store this result in case fallbacks also fail
                            if best_result is None:
                                best_result = (result, cleaned_markdown, enhanced_quality_metrics, rt_quality_score)
                            continue
                    else:
                        # Use only real-time validation if enhanced quality not available
                        if is_valid_content or attempt >= self.max_fallback_attempts + 1:
                            # Real-time validation passed or we've exhausted attempts
                            break
                        else:
                            logger.info(f"Real-time validation failed for {url}, trying tier {attempt} fallback")
                            # Store this result in case fallbacks also fail
                            if best_result is None:
                                best_result = (result, cleaned_markdown, None, rt_quality_score)
                            continue
                        
                except PlaywrightTimeoutError as e:
                    self._log_structured("warning", "Playwright timeout during extraction", 
                                       url=url, attempt=attempt, exception=e,
                                       error_category="timeout", should_retry=True)
                    continue
                except PlaywrightError as e:
                    self._log_structured("error", "Playwright error during extraction", 
                                       url=url, attempt=attempt, exception=e,
                                       error_category="browser", should_retry=True)
                    continue
                except ClientConnectionError as e:
                    self._log_structured("warning", "Network connection error during extraction", 
                                       url=url, attempt=attempt, exception=e,
                                       error_category="network", should_retry=True)
                    continue
                except ClientTimeout as e:
                    self._log_structured("warning", "Network timeout during extraction", 
                                       url=url, attempt=attempt, exception=e,
                                       error_category="timeout", should_retry=True)
                    continue
                except ClientError as e:
                    self._log_structured("warning", "HTTP client error during extraction", 
                                       url=url, attempt=attempt, exception=e,
                                       error_category="http", should_retry=True)
                    continue
                except ValueError as e:
                    self._log_structured("error", "Data validation error during extraction", 
                                       url=url, attempt=attempt, exception=e,
                                       error_category="validation", should_retry=True)
                    continue
                except KeyError as e:
                    self._log_structured("error", "Missing required data during extraction", 
                                       url=url, attempt=attempt, exception=e,
                                       error_category="data", should_retry=True)
                    continue
                except Exception as e:
                    # Fallback for unexpected errors
                    self._log_structured("error", "Unexpected error during extraction", 
                                       url=url, attempt=attempt, exception=e,
                                       error_category="unexpected", should_retry=True)
                    continue
            
            extraction_time_ms = (time.time() - start_time) * 1000
            
            if result and result.success and cleaned_markdown:
                # Extract metadata for pipeline processing
                word_count = len(cleaned_markdown.split())
                title = self._extract_title_from_markdown(cleaned_markdown)
                
                # Calculate quality indicators
                content_ratio = self._calculate_content_ratio(cleaned_markdown)
                has_dynamic = self._detect_dynamic_content_indicators(cleaned_markdown)
                
                # Unified quality validation with clear prioritization
                unified_quality = self._perform_unified_quality_validation(cleaned_markdown, url, enhanced_quality_metrics)
                
                # Extract results from unified validation
                quality_score = unified_quality.quality_score
                quality_passed = unified_quality.quality_passed
                quality_validation = unified_quality.legacy_validation  # For backward compatibility
                
                # Log final quality decision
                logger.info(f"Quality validation complete: method={unified_quality.validation_method}, score={quality_score:.3f}, passed={quality_passed}")
                if unified_quality.issues:
                    logger.warning(f"Quality issues found: {', '.join(unified_quality.issues)}")
                
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
                    # Handle both old format (3 items) and new format (4 items) for compatibility
                    if len(best_result) == 4:
                        result, cleaned_markdown, enhanced_quality_metrics, rt_quality_score = best_result
                    else:
                        result, cleaned_markdown, enhanced_quality_metrics = best_result
                        rt_quality_score = 0.0
                        
                    word_count = len(cleaned_markdown.split())
                    title = self._extract_title_from_markdown(cleaned_markdown)
                    content_ratio = self._calculate_content_ratio(cleaned_markdown)
                    has_dynamic = self._detect_dynamic_content_indicators(cleaned_markdown)
                    
                    # Use real-time quality score if available, otherwise fall back to enhanced metrics
                    final_quality_score = rt_quality_score if rt_quality_score > 0 else (
                        enhanced_quality_metrics.overall_quality_score if enhanced_quality_metrics else 0.0
                    )
                    
                    logger.warning(f"Using best available result for {url} with low quality (score: {final_quality_score:.3f})")
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
                        quality_score=final_quality_score
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
                
        except PlaywrightTimeoutError as e:
            extraction_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Browser timeout: {str(e)}"
            self._log_structured("error", "Top-level browser timeout", 
                               url=url, exception=e, error_category="timeout",
                               extraction_time_ms=extraction_time_ms, should_retry=False)
            
            return AdvancedCrawlResult(
                url=url, markdown="", success=False, error_message=error_msg,
                extraction_time_ms=extraction_time_ms
            )
        except PlaywrightError as e:
            extraction_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Browser error: {str(e)}"
            self._log_structured("error", "Top-level browser error", 
                               url=url, exception=e, error_category="browser",
                               extraction_time_ms=extraction_time_ms, should_retry=False)
            
            return AdvancedCrawlResult(
                url=url, markdown="", success=False, error_message=error_msg,
                extraction_time_ms=extraction_time_ms
            )
        except (ClientConnectionError, ClientTimeout, ClientError) as e:
            extraction_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Network error: {str(e)}"
            self._log_structured("error", "Top-level network error", 
                               url=url, exception=e, error_category="network",
                               extraction_time_ms=extraction_time_ms, should_retry=True)
            
            return AdvancedCrawlResult(
                url=url, markdown="", success=False, error_message=error_msg,
                extraction_time_ms=extraction_time_ms
            )
        except (ValueError, KeyError, AttributeError) as e:
            extraction_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Data processing error: {str(e)}"
            self._log_structured("error", "Top-level data processing error", 
                               url=url, exception=e, error_category="data",
                               extraction_time_ms=extraction_time_ms, should_retry=False)
            
            return AdvancedCrawlResult(
                url=url, markdown="", success=False, error_message=error_msg,
                extraction_time_ms=extraction_time_ms
            )
        except Exception as e:
            extraction_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Unexpected crawling error: {str(e)}"
            self._log_structured("error", "Top-level unexpected error", 
                               url=url, exception=e, error_category="unexpected",
                               extraction_time_ms=extraction_time_ms, should_retry=False)
            
            return AdvancedCrawlResult(
                url=url, markdown="", success=False, error_message=error_msg,
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
    
    def _log_structured(self, level: str, message: str, url: str = None, attempt: int = None, exception: Exception = None, **extra_context):
        """
        Create structured JSON log entries with consistent context.
        
        Args:
            level: Log level (info, warning, error, debug)
            message: Human readable message
            url: URL being processed (if applicable)
            attempt: Attempt number (if applicable) 
            exception: Exception object (if applicable)
            **extra_context: Additional context fields
        """
        # Create structured log entry
        log_data = {
            "message": message,
            "timestamp": time.time(),
        }
        
        # Add context if provided
        if url:
            log_data["url"] = url
            log_data["domain"] = urlparse(url).netloc
        if attempt is not None:
            log_data["attempt"] = attempt
        if exception:
            log_data["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "module": getattr(exception, "__module__", "unknown")
            }
        
        # Add any additional context
        log_data.update(extra_context)
        
        # Log with appropriate level
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(json.dumps(log_data, default=str))
    
    def _perform_unified_quality_validation(self, markdown: str, url: str, enhanced_quality_metrics: Optional['ContentQualityMetrics'] = None) -> UnifiedQualityResult:
        """
        Perform unified quality validation with clear prioritization.
        
        Priority order:
        1. Enhanced quality system (if available and metrics provided)
        2. Legacy quality validation (if enabled and available) 
        3. Disabled validation (pass-through)
        4. Failed validation (fail-closed)
        
        Args:
            markdown: The content to validate
            url: The URL being validated (for logging)
            enhanced_quality_metrics: Pre-calculated enhanced quality metrics if available
            
        Returns:
            UnifiedQualityResult with clear validation outcome and method used
        """
        logger.debug(f"Starting unified quality validation for {url}")
        
        # Priority 1: Enhanced quality system
        if enhanced_quality_metrics and ENHANCED_QUALITY_AVAILABLE:
            logger.info(f"Using enhanced quality validation for {url}")
            quality_score = enhanced_quality_metrics.overall_quality_score
            quality_passed = not enhanced_quality_metrics.should_retry_with_fallback
            
            # Log enhanced quality decision
            logger.info(f"Enhanced quality: score={quality_score:.3f}, passed={quality_passed}")
            
            return UnifiedQualityResult(
                quality_score=quality_score,
                quality_passed=quality_passed,
                validation_method="enhanced",
                enhanced_metrics=enhanced_quality_metrics,
                issues=[] if quality_passed else ["Enhanced quality metrics suggest retry needed"]
            )
        
        # Priority 2: Legacy quality validation
        if self.enable_quality_validation and self.quality_validator and QUALITY_VALIDATION_AVAILABLE:
            logger.info(f"Using legacy quality validation for {url}")
            legacy_validation = self.quality_validator.validate_content(markdown, url)
            
            # Log legacy quality decision
            logger.info(f"Legacy quality: score={legacy_validation.score:.3f}, passed={legacy_validation.passed}")
            if not legacy_validation.passed:
                logger.warning(f"Legacy quality validation failed: {', '.join(legacy_validation.issues)}")
            
            return UnifiedQualityResult(
                quality_score=legacy_validation.score,
                quality_passed=legacy_validation.passed,
                validation_method="legacy",
                legacy_validation=legacy_validation,
                issues=legacy_validation.issues
            )
        
        # Priority 3: Validation explicitly disabled
        if not self.enable_quality_validation:
            logger.info(f"Quality validation disabled for {url} - accepting content")
            return UnifiedQualityResult(
                quality_score=1.0,
                quality_passed=True,
                validation_method="disabled",
                issues=[]
            )
        
        # Priority 4: Validation enabled but failed/unavailable (fail-closed)
        logger.warning(f"Quality validation enabled but unavailable for {url} - failing closed")
        return UnifiedQualityResult(
            quality_score=0.0,
            quality_passed=False,
            validation_method="failed",
            issues=["Quality validation system enabled but not available"]
        )
    
    def _perform_real_time_selector_validation(self, markdown: str, url: str, tier: int = 0) -> Tuple[bool, float, List[str]]:
        """
        Perform real-time validation to determine if the selected content is navigation vs. main content.
        
        This function implements the specific validation criteria from Task 20.4:
        - Link density validation (flag if link/word ratio > 0.4)
        - Average text length validation for list items
        - Navigation keyword detection
        - Text-to-tag ratio analysis
        
        Args:
            markdown: Extracted markdown content to validate
            url: URL being crawled (for logging)
            tier: Fallback tier being used (0 = primary, 1-4 = fallback tiers)
            
        Returns:
            Tuple of (is_valid_content, quality_score, improvement_suggestions)
        """
        
        if not markdown or len(markdown.strip()) < 10:
            return False, 0.0, ["Content is too short or empty"]
        
        suggestions = []
        quality_score = 1.0  # Start with perfect score, apply penalties
        
        # 1. LINK DENSITY VALIDATION (Task 20.4 requirement)
        words = len(markdown.split())
        links = re.findall(r'\[([^\]]*)\]\([^)]*\)', markdown)
        link_density = len(links) / words if words > 0 else 0
        
        if link_density > 0.4:  # Critical threshold from task
            quality_score *= 0.1  # 90% penalty - almost certainly navigation
            suggestions.append(f"Critical: Link density {link_density:.3f} > 0.4 indicates navigation content")
        elif link_density > 0.25:
            quality_score *= 0.3  # 70% penalty
            suggestions.append(f"High link density {link_density:.3f} suggests navigation elements")
        elif link_density > 0.15:
            quality_score *= 0.6  # 40% penalty  
            suggestions.append(f"Moderate link density {link_density:.3f} may include navigation")
        
        # 2. NAVIGATION KEYWORD DETECTION (Task 20.4 requirement)
        nav_keywords = ['next', 'previous', 'prev', 'home', 'back', 'menu', 'navigation', 'nav', 
                       'breadcrumb', 'sidebar', 'toc', 'table of contents', 'edit', 'edit page']
        
        markdown_lower = markdown.lower()
        nav_keyword_count = sum(markdown_lower.count(keyword) for keyword in nav_keywords)
        nav_keyword_density = nav_keyword_count / words if words > 0 else 0
        
        if nav_keyword_density > 0.05:  # >5% navigation keywords
            quality_score *= 0.2  # 80% penalty
            suggestions.append(f"High navigation keyword density {nav_keyword_density:.3f} indicates navigation content")
        elif nav_keyword_density > 0.02:  # >2% navigation keywords
            quality_score *= 0.5  # 50% penalty
            suggestions.append(f"Moderate navigation keywords detected {nav_keyword_density:.3f}")
        
        # 3. AVERAGE TEXT LENGTH VALIDATION FOR LIST ITEMS (Task 20.4 requirement)
        list_items = re.findall(r'^\s*[-*+]\s+(.+)$', markdown, re.MULTILINE)
        if list_items:
            avg_list_item_length = sum(len(item.split()) for item in list_items) / len(list_items)
            
            if avg_list_item_length < 3:  # Very short list items likely navigation
                quality_score *= 0.15  # 85% penalty
                suggestions.append(f"List items too short (avg {avg_list_item_length:.1f} words) - likely navigation")
            elif avg_list_item_length < 5:  # Short list items suspicious
                quality_score *= 0.4  # 60% penalty
                suggestions.append(f"Short list items (avg {avg_list_item_length:.1f} words) may be navigation")
        
        # 4. TEXT-TO-TAG RATIO ANALYSIS (Task 20.4 requirement)
        # Count markdown formatting tags vs plain text
        markdown_tags = len(re.findall(r'[#*`\[\]()_~]', markdown))
        plain_chars = len(re.sub(r'[#*`\[\]()_~]', '', markdown))
        tag_ratio = markdown_tags / (plain_chars + markdown_tags) if (plain_chars + markdown_tags) > 0 else 0
        
        if tag_ratio > 0.3:  # >30% formatting suggests navigation/links
            quality_score *= 0.3  # 70% penalty
            suggestions.append(f"High tag-to-text ratio {tag_ratio:.3f} indicates link-heavy navigation")
        elif tag_ratio > 0.2:  # >20% formatting
            quality_score *= 0.6  # 40% penalty
            suggestions.append(f"Moderate tag-to-text ratio {tag_ratio:.3f}")
        
        # 5. REPETITIVE PATTERN DETECTION
        lines = [line.strip() for line in markdown.split('\n') if line.strip()]
        if len(lines) > 5:
            # Check for repetitive short lines (breadcrumbs, navigation)
            short_lines = [line for line in lines if len(line.split()) <= 4]
            short_line_ratio = len(short_lines) / len(lines)
            
            if short_line_ratio > 0.7:  # >70% short lines
                quality_score *= 0.15  # 85% penalty
                suggestions.append(f"High ratio of short lines {short_line_ratio:.3f} indicates navigation")
            elif short_line_ratio > 0.5:  # >50% short lines
                quality_score *= 0.4  # 60% penalty
                suggestions.append(f"Many short lines {short_line_ratio:.3f} may indicate navigation")
        
        # 6. MINIMUM WORD COUNT WITH TIER-BASED THRESHOLDS
        min_words_by_tier = {0: 50, 1: 40, 2: 30, 3: 20, 4: 15}  # Lower thresholds for fallback tiers
        min_words = min_words_by_tier.get(tier, 15)
        
        if words < min_words:
            quality_score *= 0.2  # 80% penalty for insufficient content
            suggestions.append(f"Word count {words} below tier {tier} minimum of {min_words}")
        
        # Determine if content should be accepted
        # Use tier-based quality thresholds (more lenient for fallback tiers)
        tier_thresholds = {0: 0.6, 1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2}
        threshold = tier_thresholds.get(tier, 0.3)
        
        is_valid = quality_score >= threshold
        
        logger.info(f"Real-time validation for {url} (tier {tier}): quality={quality_score:.3f}, threshold={threshold}, valid={is_valid}")
        if not is_valid:
            logger.warning(f"Validation failed - link_density={link_density:.3f}, nav_keywords={nav_keyword_count}, words={words}")
        
        return is_valid, quality_score, suggestions

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
        
    except ClientConnectionError as e:
        logger.error(f"Network connection error downloading sitemap {sitemap_url}: {str(e)}")
    except ClientTimeout as e:
        logger.error(f"Timeout downloading sitemap {sitemap_url}: {str(e)}")
    except ClientError as e:
        logger.error(f"HTTP error downloading sitemap {sitemap_url}: {str(e)}")
    except ET.ParseError as e:
        logger.error(f"XML parsing error for sitemap {sitemap_url}: {str(e)}")
    except ValueError as e:
        logger.error(f"Data validation error for sitemap {sitemap_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error crawling sitemap {sitemap_url}: {str(e)}")
    
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
        
    except ClientConnectionError as e:
        logger.error(f"Network connection error downloading text file {text_file_url}: {str(e)}")
    except ClientTimeout as e:
        logger.error(f"Timeout downloading text file {text_file_url}: {str(e)}")
    except ClientError as e:
        logger.error(f"HTTP error downloading text file {text_file_url}: {str(e)}")
    except UnicodeDecodeError as e:
        logger.error(f"Text encoding error for file {text_file_url}: {str(e)}")
    except ValueError as e:
        logger.error(f"Data validation error for text file {text_file_url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error crawling text file {text_file_url}: {str(e)}")
    
    return results