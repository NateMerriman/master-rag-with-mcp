#!/usr/bin/env python3
"""
Simple Documentation Site Configuration for Framework Detection and CSS Targeting.

This module provides the exact API requested by Task 16 for DocumentationSiteConfig,
acting as a simple wrapper around the more advanced enhanced_crawler_config system.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
from pydantic import BaseModel, validator
from urllib.parse import urlparse

# Import the existing enhanced system
from enhanced_crawler_config import (
    DocumentationFramework,
    config_manager as enhanced_config_manager,
    detect_framework as enhanced_detect_framework
)


@dataclass
class DocumentationSiteConfig:
    """
    Simple configuration for documentation site crawling.
    
    This structure holds site-specific configurations as requested by Task 16,
    including the domain name and a list of CSS selectors for content extraction.
    """
    
    domain: str
    content_selectors: List[str]
    
    def __post_init__(self):
        """Validate the configuration after initialization."""
        if not isinstance(self.domain, str):
            raise TypeError("domain must be a string")
        if not isinstance(self.content_selectors, list):
            raise TypeError("content_selectors must be a list")
        if not all(isinstance(selector, str) for selector in self.content_selectors):
            raise TypeError("all content_selectors must be strings")


class DocumentationSiteConfigPydantic(BaseModel):
    """
    Pydantic version of DocumentationSiteConfig for enhanced validation.
    
    This provides automatic validation and serialization capabilities
    as an alternative to the dataclass version.
    """
    
    domain: str
    content_selectors: List[str]
    
    @validator('domain')
    def validate_domain(cls, v):
        """Validate domain format."""
        if not v or not isinstance(v, str):
            raise ValueError("domain must be a non-empty string")
        return v.lower().strip()
    
    @validator('content_selectors')
    def validate_content_selectors(cls, v):
        """Validate content selectors list."""
        if not v or not isinstance(v, list):
            raise ValueError("content_selectors must be a non-empty list")
        if not all(isinstance(selector, str) for selector in v):
            raise ValueError("all content_selectors must be strings")
        return [selector.strip() for selector in v if selector.strip()]


class ConfigurationRegistry:
    """
    Central registry for storing and retrieving DocumentationSiteConfig objects.
    
    This acts as a simple wrapper around the enhanced configuration system,
    providing the exact API requested by Task 16.
    """
    
    def __init__(self):
        """Initialize the configuration registry."""
        self._configs: Dict[str, DocumentationSiteConfig] = {}
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default configurations from the enhanced system."""
        # Create simplified configs for the main frameworks
        
        # n8n.io configuration (Material Design framework)
        n8n_config = DocumentationSiteConfig(
            domain="n8n.io",
            content_selectors=["main.md-main .md-content", "article.md-content__inner", "main.md-main"]
        )
        self._configs["n8n.io"] = n8n_config
        self._configs["docs.n8n.io"] = n8n_config
        
        # ReadMe.io configurations
        readme_config = DocumentationSiteConfig(
            domain="readme.io",
            content_selectors=["main.rm-Guides", "article.rm-Content", ".rm-Article"]
        )
        self._configs["readme.io"] = readme_config
        
        # GitBook configurations
        gitbook_config = DocumentationSiteConfig(
            domain="gitbook.io",
            content_selectors=[".gitbook-content", ".page-inner", ".book-body"]
        )
        self._configs["gitbook.io"] = gitbook_config
        self._configs["gitbook.com"] = gitbook_config
    
    def add_config(self, config: DocumentationSiteConfig) -> None:
        """
        Add a configuration to the registry.
        
        Args:
            config: DocumentationSiteConfig instance to add
        """
        if not isinstance(config, DocumentationSiteConfig):
            raise TypeError("config must be a DocumentationSiteConfig instance")
        
        self._configs[config.domain] = config
    
    def get_config_by_domain(self, domain: str) -> Optional[DocumentationSiteConfig]:
        """
        Retrieve configuration for a given domain.
        
        Args:
            domain: Domain name to look up
            
        Returns:
            DocumentationSiteConfig instance if found, None otherwise
        """
        if not isinstance(domain, str):
            return None
        
        # Normalize domain
        normalized_domain = domain.lower().strip()
        
        # Direct lookup first
        if normalized_domain in self._configs:
            return self._configs[normalized_domain]
        
        # Try partial matching for subdomains
        for registered_domain, config in self._configs.items():
            if normalized_domain.endswith(registered_domain):
                return config
        
        return None
    
    def list_domains(self) -> List[str]:
        """Get list of all registered domains."""
        return list(self._configs.keys())
    
    def remove_config(self, domain: str) -> bool:
        """
        Remove a configuration from the registry.
        
        Args:
            domain: Domain to remove
            
        Returns:
            True if removed, False if not found
        """
        if domain in self._configs:
            del self._configs[domain]
            return True
        return False
    
    def clear_configs(self) -> None:
        """Clear all configurations from the registry."""
        self._configs.clear()


# Global registry instance for easy access
config_registry = ConfigurationRegistry()


def get_config_by_domain(domain: str) -> Optional[DocumentationSiteConfig]:
    """
    Convenience function for looking up configuration by domain.
    
    This is the main function requested by Task 16 subtask 2.
    
    Args:
        domain: Domain name to look up
        
    Returns:
        DocumentationSiteConfig instance if found, None otherwise
    """
    return config_registry.get_config_by_domain(domain)


def register_site_config(domain: str, content_selectors: List[str]) -> DocumentationSiteConfig:
    """
    Convenience function to create and register a site configuration.
    
    Args:
        domain: Domain name
        content_selectors: List of CSS selectors for content extraction
        
    Returns:
        Created DocumentationSiteConfig instance
    """
    config = DocumentationSiteConfig(domain=domain, content_selectors=content_selectors)
    config_registry.add_config(config)
    return config


def extract_domain_from_url(url: str) -> str:
    """
    Extract domain from URL for configuration lookup.
    
    Args:
        url: Full URL to extract domain from
        
    Returns:
        Domain name (netloc)
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""