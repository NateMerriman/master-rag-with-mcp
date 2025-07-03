#!/usr/bin/env python3
"""
Enhanced crawler configuration for documentation sites.

This module provides framework-specific configurations for optimal content extraction
from various documentation platforms, addressing the navigation overload problem
identified in documentation sites.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import re
from urllib.parse import urlparse
import asyncio
from crawl4ai import CrawlerRunConfig
import logging

logger = logging.getLogger(__name__)


class DocumentationFramework(Enum):
    """Supported documentation frameworks."""
    MATERIAL_DESIGN = "material_design"  # n8n, MkDocs sites
    README_IO = "readme_io"              # VirusTotal, many API docs
    GITBOOK = "gitbook"                  # GitBook hosted docs
    DOCUSAURUS = "docusaurus"            # Facebook's Docusaurus
    SPHINX = "sphinx"                    # Python Sphinx docs
    VUEPRESS = "vuepress"                # VuePress sites
    JEKYLL = "jekyll"                    # Jekyll-based docs
    GENERIC = "generic"                  # Fallback for unknown frameworks


@dataclass
class FrameworkConfig:
    """Configuration for content extraction from a specific framework."""
    
    # Target elements for main content (CSS selectors)
    target_elements: List[str] = field(default_factory=list)
    
    # CSS selectors to exclude from extraction
    excluded_selectors: List[str] = field(default_factory=list)
    
    # HTML tags to exclude
    excluded_tags: List[str] = field(default_factory=list)
    
    # Minimum word count threshold for content blocks
    word_count_threshold: int = 15
    
    # Framework-specific settings  
    exclude_external_links: bool = True
    exclude_social_media_links: bool = True
    process_iframes: bool = False
    
    # Quality validation thresholds
    min_content_ratio: float = 0.6  # Minimum content-to-navigation ratio
    max_link_density: float = 0.3   # Maximum links per word
    
    # Framework identification patterns
    detection_patterns: Dict[str, List[str]] = field(default_factory=dict)


class DocumentationSiteConfigManager:
    """Configuration manager for documentation site extraction."""
    
    def __init__(self):
        self._framework_configs = self._initialize_framework_configs()
        self._domain_patterns = self._initialize_domain_patterns()
        self._detection_cache: Dict[str, DocumentationFramework] = {}
    
    def _initialize_framework_configs(self) -> Dict[DocumentationFramework, FrameworkConfig]:
        """Initialize framework-specific extraction configurations."""
        
        configs = {}
        
        # Material Design (used by n8n, MkDocs sites)
        configs[DocumentationFramework.MATERIAL_DESIGN] = FrameworkConfig(
            target_elements=[
                "article.md-content__inner"  # This was the most effective selector with 554 words
            ],
            excluded_selectors=[
                # Minimal exclusions to test if content extraction works
                ".md-sidebar",
                ".md-nav",
                "nav.md-path"  # Just breadcrumbs for now
            ],
            excluded_tags=["nav", "footer", "aside"],  # Removed "header" to preserve content headers
            word_count_threshold=15,
            min_content_ratio=0.7,
            max_link_density=0.25,
            detection_patterns={
                "css_classes": ["md-main", "md-content", "md-sidebar", "md-nav"],
                "meta_tags": ["mkdocs", "material"],
                "generator": ["mkdocs"]
            }
        )
        
        # ReadMe.io (used by VirusTotal, many API documentation sites)
        configs[DocumentationFramework.README_IO] = FrameworkConfig(
            target_elements=[
                "main.rm-Guides",
                "main[class*='rm-']",
                "article.rm-Content",
                ".markdown-body",
                ".rm-Article"
            ],
            excluded_selectors=[
                ".rm-Sidebar",
                ".hub-sidebar",
                ".hub-reference-sidebar", 
                ".content-toc",
                ".rm-Nav",
                "nav[class*='rm-']",
                "nav[class*='hub-']",
                ".md-sidebar",
                ".md-nav",
                ".md-header",
                ".md-footer",
                ".md-tabs",
                ".md-search",
                ".pagination-nav",
                "[role='navigation']",
                "[role='banner']",
                "[role='contentinfo']",
                "#__docusaurus",
                ".DocSearch-Button",
                ".navbar",
                ".menu",
                ".sidebar",
                ".table-of-contents",
                ".theme-doc-toc",
                ".theme-edit-this-page",
                ".pagination-nav",
                ".docItemFooter_node_modules-",
                ".footer",
                ".cookie-consent",
                ".feedback-section",
                ".related-content",
                ".popular-integrations",
                ".trending-combinations",
                ".top-integration-categories",
                ".trending-templates",
                ".top-guides",
                ".md-copyright",
                ".md-source",
                ".md-nav",
                ".md-tabs",
                ".md-search",
                ".md-footer",
                ".md-header",
                ".md-sidebar",
                ".md-content__button",
                ".md-content__inner",
                ".md-content__heading",
                ".md-content__source",
                ".md-content__toc",
                ".md-content__footnote",
                ".md-content__admonition",
                ".md-content__code",
                ".md-content__table",
                ".md-content__figure",
                ".md-content__image",
                ".md-content__video",
                ".md-content__iframe",
                ".md-content__button",
                ".md-content__link",
                ".md-content__list",
                ".md-content__list-item",
                ".md-content__paragraph",
                ".md-content__blockquote",
                ".md-content__hr",
                ".md-content__details",
                ".md-content__summary",
                ".md-content__kbd",
                ".md-content__mark",
                ".md-content__del",
                ".md-content__ins",
                ".md-content__sup",
                ".md-content__sub",
                ".md-content__small",
                ".md-content__strong",
                ".md-content__em",
                ".md-content__code-inline",
                ".md-content__code-block",
                ".md-content__table-wrapper",
                ".md-content__figure-wrapper",
                ".md-content__image-wrapper",
                ".md-content__video-wrapper",
                ".md-content__iframe-wrapper",
                ".md-content__button-wrapper",
                ".md-content__link-wrapper",
                ".md-content__list-wrapper",
                ".md-content__list-item-wrapper",
                ".md-content__paragraph-wrapper",
                ".md-content__blockquote-wrapper",
                ".md-content__hr-wrapper",
                ".md-content__details-wrapper",
                ".md-content__summary-wrapper",
                ".md-content__kbd-wrapper",
                ".md-content__mark-wrapper",
                ".md-content__del-wrapper",
                ".md-content__ins-wrapper",
                ".md-content__sup-wrapper",
                ".md-content__sub-wrapper",
                ".md-content__small-wrapper",
                ".md-content__strong-wrapper",
                ".md-content__em-wrapper",
                ".md-content__code-inline-wrapper",
                ".md-content__code-block-wrapper",
                ".md-content__table-wrapper",
                ".md-content__figure-wrapper",
                ".md-content__image-wrapper",
                ".md-content__video-wrapper",
                ".md-content__iframe-wrapper",
                ".md-content__button-wrapper",
                ".md-content__link-wrapper",
                ".md-content__list-wrapper",
                ".md-content__list-item-wrapper",
                ".md-content__paragraph-wrapper",
                ".md-content__blockquote-wrapper",
                ".md-content__hr-wrapper",
                ".md-content__details-wrapper",
                ".md-content__summary-wrapper",
                ".md-content__kbd-wrapper",
                ".md-content__mark-wrapper",
                ".md-content__del-wrapper",
                ".md-content__ins-wrapper",
                ".md-content__sup-wrapper",
                ".md-content__sub-wrapper",
                ".md-content__small-wrapper",
                ".md-content__strong-wrapper",
                ".md-content__em-wrapper",
                ".md-content__code-inline-wrapper",
                ".md-content__code-block-wrapper"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            word_count_threshold=20,  # ReadMe tends to have more concise navigation
            min_content_ratio=0.65,
            max_link_density=0.3,
            detection_patterns={
                "css_classes": ["rm-Guides", "rm-Sidebar", "rm-Content", "hub-sidebar"],
                "meta_tags": ["readme", "readme.io"],
                "domain_patterns": ["readme.io", "docs.readme.io"]
            }
        )
        
        # GitBook
        configs[DocumentationFramework.GITBOOK] = FrameworkConfig(
            target_elements=[
                ".gitbook-content",
                ".page-inner",
                ".book-body",
                ".markdown-section"
            ],
            excluded_selectors=[
                ".book-summary",
                ".book-header",
                ".navigation",
                ".gitbook-link",
                ".book-navigation"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            word_count_threshold=15,
            min_content_ratio=0.65,
            detection_patterns={
                "css_classes": ["gitbook-content", "book-summary", "book-body"],
                "meta_tags": ["gitbook"],
                "generator": ["gitbook"]
            }
        )
        
        # Docusaurus (Facebook's documentation framework)
        configs[DocumentationFramework.DOCUSAURUS] = FrameworkConfig(
            target_elements=[
                "main.docMainContainer",
                ".docItemContainer",
                ".markdown",
                "article"
            ],
            excluded_selectors=[
                ".navbar",
                ".sidebar",
                ".menu",
                ".table-of-contents",
                ".pagination-nav",
                ".docusaurus-highlight-code-line"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            word_count_threshold=15,
            min_content_ratio=0.7,
            detection_patterns={
                "css_classes": ["docMainContainer", "docItemContainer"],
                "meta_tags": ["docusaurus"],
                "generator": ["docusaurus"]
            }
        )
        
        # Sphinx (Python documentation)
        configs[DocumentationFramework.SPHINX] = FrameworkConfig(
            target_elements=[
                ".document",
                ".body",
                ".section",
                "#sphinx-content"
            ],
            excluded_selectors=[
                ".sphinxsidebar",
                ".related",
                ".footer",
                ".navigation",
                ".breadcrumbs"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            word_count_threshold=15,
            min_content_ratio=0.7,
            detection_patterns={
                "css_classes": ["sphinxsidebar", "sphinx-content"],
                "meta_tags": ["sphinx"],
                "generator": ["sphinx"]
            }
        )
        
        # VuePress
        configs[DocumentationFramework.VUEPRESS] = FrameworkConfig(
            target_elements=[
                ".page",
                ".content",
                ".theme-default-content"
            ],
            excluded_selectors=[
                ".sidebar",
                ".navbar",
                ".page-nav",
                ".sidebar-links"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            word_count_threshold=15,
            min_content_ratio=0.7,
            detection_patterns={
                "css_classes": ["theme-default-content", "sidebar-links"],
                "meta_tags": ["vuepress"],
                "generator": ["vuepress"]
            }
        )
        
        # Jekyll (GitHub Pages, etc.)
        configs[DocumentationFramework.JEKYLL] = FrameworkConfig(
            target_elements=[
                ".post-content",
                ".content",
                "main",
                "article"
            ],
            excluded_selectors=[
                ".sidebar",
                ".navigation",
                ".site-nav",
                ".post-nav"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            word_count_threshold=15,
            min_content_ratio=0.65,
            detection_patterns={
                "css_classes": ["post-content", "site-nav"],
                "meta_tags": ["jekyll"],
                "generator": ["jekyll"]
            }
        )
        
        # Generic fallback configuration
        configs[DocumentationFramework.GENERIC] = FrameworkConfig(
            target_elements=[
                "main",
                "article", 
                ".content",
                ".main-content",
                ".documentation",
                ".docs-content",
                ".page-content",
                "[role='main']"
            ],
            excluded_selectors=[
                ".sidebar",
                ".navigation", 
                ".nav-sidebar",
                ".breadcrumb",
                ".toc",
                ".table-of-contents"
            ],
            excluded_tags=["nav", "header", "footer", "aside"],
            word_count_threshold=15,
            min_content_ratio=0.5,  # Lower threshold for unknown frameworks
            max_link_density=0.4,
            detection_patterns={}
        )
        
        return configs
    
    def _initialize_domain_patterns(self) -> Dict[DocumentationFramework, List[str]]:
        """Initialize domain-based framework detection patterns."""
        return {
            DocumentationFramework.MATERIAL_DESIGN: [
                r".*\.n8n\.io$",  # n8n uses Material Design
                r".*docs\.n8n\.io$",
                r".*\.mkdocs\..*$"  # MkDocs sites typically use Material Design
            ],
            DocumentationFramework.README_IO: [
                r".*\.readme\.io$",
                r".*\.readme\.com$",
                # Removed the overly broad r".*docs\..*\.io$" pattern
            ],
            DocumentationFramework.GITBOOK: [
                r".*\.gitbook\.io$",
                r".*\.gitbook\.com$"
            ],
            # Other frameworks detected via HTML analysis
        }
    
    def detect_documentation_framework(self, url: str) -> DocumentationFramework:
        """
        Detect the documentation framework for a given URL and HTML content.
        
        Args:
            url: The URL being crawled
            html_content: Optional HTML content for analysis
            
        Returns:
            Detected DocumentationFramework enum value
        """
        # Check cache first
        domain = urlparse(url).netloc
        if domain in self._detection_cache:
            return self._detection_cache[domain]
        
        detected_framework = DocumentationFramework.GENERIC
        
        # 1. Domain-based detection
        for framework, patterns in self._domain_patterns.items():
            for pattern in patterns:
                if re.match(pattern, domain, re.IGNORECASE):
                    detected_framework = framework
                    break
            if detected_framework != DocumentationFramework.GENERIC:
                break
        
        # Cache the result
        self._detection_cache[domain] = detected_framework
        
        logger.info(f"Detected framework {detected_framework.value} for domain {domain}")
        return detected_framework
    
    def _analyze_html_for_framework(self, html_content: str) -> DocumentationFramework:
        """
        Analyze HTML content to detect documentation framework.
        
        Args:
            html_content: The HTML content to analyze
            
        Returns:
            Detected DocumentationFramework enum value
        """
        html_lower = html_content.lower()
        
        # Check each framework's detection patterns
        for framework, config in self._framework_configs.items():
            if framework == DocumentationFramework.GENERIC:
                continue
                
            score = 0
            patterns = config.detection_patterns
            
            # Check CSS classes
            for css_class in patterns.get("css_classes", []):
                if css_class.lower() in html_lower:
                    score += 2
            
            # Check meta tags
            for meta_tag in patterns.get("meta_tags", []):
                if f'content="{meta_tag}"' in html_lower or f"name=\"{meta_tag}\"" in html_lower:
                    score += 3
            
            # Check generator meta tags
            for generator in patterns.get("generator", []):
                if f'name="generator"' in html_lower and generator.lower() in html_lower:
                    score += 5
            
            # Framework-specific strong indicators
            if framework == DocumentationFramework.MATERIAL_DESIGN:
                if "md-main" in html_lower and "md-sidebar" in html_lower:
                    score += 10
            elif framework == DocumentationFramework.README_IO:
                if "rm-guides" in html_lower or "hub-sidebar" in html_lower:
                    score += 10
            
            # If we have a strong match, return it
            if score >= 8:
                return framework
        
        return DocumentationFramework.GENERIC
    
    def get_framework_config(self, framework: DocumentationFramework) -> FrameworkConfig:
        """Get configuration for a specific framework."""
        return self._framework_configs.get(framework, self._framework_configs[DocumentationFramework.GENERIC])
    
    def create_crawl4ai_config(self, framework: DocumentationFramework, custom_overrides: Optional[Dict] = None) -> CrawlerRunConfig:
        """
        Create an optimized Crawl4AI configuration for the detected framework.
        
        Args:
            framework: The detected documentation framework
            custom_overrides: Optional dictionary of custom configuration overrides
            
        Returns:
            Optimized CrawlerRunConfig for the framework
        """
        config = self.get_framework_config(framework)
        
        # Build the Crawl4AI configuration
        crawl_config = CrawlerRunConfig(
            # Target main content areas
            target_elements=config.target_elements,
            
            # Exclude navigation and other noise
            excluded_tags=config.excluded_tags,
            
            # Content filtering
            word_count_threshold=config.word_count_threshold,
            exclude_external_links=config.exclude_external_links,
            exclude_social_media_links=config.exclude_social_media_links,
            process_iframes=config.process_iframes,
            
            # Additional excluded selectors need to be handled via CSS selector
            # We'll construct a combined CSS selector for exclusions
        )
        
        # Apply custom overrides if provided
        if custom_overrides:
            for key, value in custom_overrides.items():
                if hasattr(crawl_config, key):
                    setattr(crawl_config, key, value)
        
        return crawl_config
    
    def get_quality_thresholds(self, framework: DocumentationFramework) -> Tuple[float, float]:
        """
        Get quality thresholds for the framework.
        
        Returns:
            Tuple of (min_content_ratio, max_link_density)
        """
        config = self.get_framework_config(framework)
        return (config.min_content_ratio, config.max_link_density)
    
    def clear_cache(self):
        """Clear the framework detection cache."""
        self._detection_cache.clear()


# Global instance for easy access
config_manager = DocumentationSiteConfigManager()


def detect_framework(url: str) -> DocumentationFramework:
    """Convenience function for framework detection."""
    return config_manager.detect_documentation_framework(url)


def get_optimized_config(framework: DocumentationFramework, custom_overrides: Optional[Dict] = None) -> CrawlerRunConfig:
    """Convenience function for getting optimized crawl configuration."""
    return config_manager.create_crawl4ai_config(framework, custom_overrides)


def get_quality_thresholds(framework: DocumentationFramework) -> Tuple[float, float]:
    """Convenience function for getting quality thresholds."""
    return config_manager.get_quality_thresholds(framework)