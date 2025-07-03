#!/usr/bin/env python3
"""
Unit tests for DocumentationSiteConfig data structure and registry.

Tests the functionality requested by Task 16 subtasks 1 and 2.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from documentation_site_config import (
    DocumentationSiteConfig,
    DocumentationSiteConfigPydantic,
    ConfigurationRegistry,
    config_registry,
    get_config_by_domain,
    register_site_config,
    extract_domain_from_url
)


class TestDocumentationSiteConfig:
    """Test the DocumentationSiteConfig data structure (Task 16.1)."""
    
    def test_valid_config_creation(self):
        """Test successful creation of DocumentationSiteConfig with valid data."""
        config = DocumentationSiteConfig(
            domain="example.com",
            content_selectors=["main", "article", ".content"]
        )
        
        assert config.domain == "example.com"
        assert config.content_selectors == ["main", "article", ".content"]
        assert isinstance(config.content_selectors, list)
    
    def test_config_with_single_selector(self):
        """Test config creation with single selector."""
        config = DocumentationSiteConfig(
            domain="test.io",
            content_selectors=["main.content"]
        )
        
        assert config.domain == "test.io"
        assert config.content_selectors == ["main.content"]
        assert len(config.content_selectors) == 1
    
    def test_config_validation_invalid_domain_type(self):
        """Test validation error for incorrect domain type."""
        with pytest.raises(TypeError, match="domain must be a string"):
            DocumentationSiteConfig(
                domain=123,  # Invalid: should be string
                content_selectors=["main"]
            )
    
    def test_config_validation_invalid_selectors_type(self):
        """Test validation error for incorrect content_selectors type."""
        with pytest.raises(TypeError, match="content_selectors must be a list"):
            DocumentationSiteConfig(
                domain="example.com",
                content_selectors="main"  # Invalid: should be list
            )
    
    def test_config_validation_invalid_selector_items(self):
        """Test validation error for non-string items in content_selectors."""
        with pytest.raises(TypeError, match="all content_selectors must be strings"):
            DocumentationSiteConfig(
                domain="example.com",
                content_selectors=["main", 123, "article"]  # Invalid: contains non-string
            )
    
    def test_config_with_empty_selectors_list(self):
        """Test config creation with empty selectors list."""
        config = DocumentationSiteConfig(
            domain="example.com",
            content_selectors=[]
        )
        
        assert config.domain == "example.com"
        assert config.content_selectors == []
    
    def test_config_immutability_after_creation(self):
        """Test that config can be modified after creation (dataclass behavior)."""
        config = DocumentationSiteConfig(
            domain="example.com",
            content_selectors=["main"]
        )
        
        # Dataclasses are mutable by default
        config.domain = "modified.com"
        config.content_selectors.append("article")
        
        assert config.domain == "modified.com"
        assert "article" in config.content_selectors


class TestDocumentationSiteConfigPydantic:
    """Test the Pydantic version of DocumentationSiteConfig."""
    
    def test_pydantic_valid_config(self):
        """Test Pydantic config with valid data."""
        config = DocumentationSiteConfigPydantic(
            domain="Example.Com",  # Test case normalization
            content_selectors=["  main  ", "article", "  .content  "]  # Test trimming
        )
        
        assert config.domain == "example.com"  # Should be normalized
        assert config.content_selectors == ["main", "article", ".content"]  # Should be trimmed
    
    def test_pydantic_validation_empty_domain(self):
        """Test Pydantic validation for empty domain."""
        with pytest.raises(ValueError, match="domain must be a non-empty string"):
            DocumentationSiteConfigPydantic(
                domain="",
                content_selectors=["main"]
            )
    
    def test_pydantic_validation_empty_selectors(self):
        """Test Pydantic validation for empty selectors."""
        with pytest.raises(ValueError, match="content_selectors must be a non-empty list"):
            DocumentationSiteConfigPydantic(
                domain="example.com",
                content_selectors=[]
            )
    
    def test_pydantic_validation_invalid_selector_types(self):
        """Test Pydantic validation for invalid selector types."""
        with pytest.raises(ValueError, match="all content_selectors must be strings"):
            DocumentationSiteConfigPydantic(
                domain="example.com",
                content_selectors=["main", 123]
            )


class TestConfigurationRegistry:
    """Test the ConfigurationRegistry class (Task 16.2)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ConfigurationRegistry()
    
    def test_registry_initialization(self):
        """Test that registry initializes with default configurations."""
        # Should have n8n.io config by default
        config = self.registry.get_config_by_domain("n8n.io")
        assert config is not None
        assert config.domain == "n8n.io"
        assert "main.md-main" in config.content_selectors
        assert "article.md-content__inner" in config.content_selectors
    
    def test_add_and_retrieve_config(self):
        """Test adding and retrieving configuration."""
        test_config = DocumentationSiteConfig(
            domain="test.example.com",
            content_selectors=["main.test", ".test-content"]
        )
        
        self.registry.add_config(test_config)
        retrieved = self.registry.get_config_by_domain("test.example.com")
        
        assert retrieved is not None
        assert retrieved.domain == "test.example.com"
        assert retrieved.content_selectors == ["main.test", ".test-content"]
    
    def test_get_config_by_domain_not_found(self):
        """Test lookup of non-existent domain returns None."""
        result = self.registry.get_config_by_domain("nonexistent.com")
        assert result is None
    
    def test_get_config_by_domain_invalid_input(self):
        """Test lookup with invalid input returns None gracefully."""
        assert self.registry.get_config_by_domain(None) is None
        assert self.registry.get_config_by_domain(123) is None
        assert self.registry.get_config_by_domain("") is None
    
    def test_subdomain_matching(self):
        """Test that subdomain lookup works correctly."""
        # Should find config for docs.n8n.io using n8n.io config
        config = self.registry.get_config_by_domain("docs.n8n.io")
        assert config is not None
        assert config.domain == "n8n.io"  # Should match the base domain config
    
    def test_list_domains(self):
        """Test listing all registered domains."""
        domains = self.registry.list_domains()
        assert isinstance(domains, list)
        assert "n8n.io" in domains
        assert "readme.io" in domains
        assert "gitbook.io" in domains
    
    def test_remove_config(self):
        """Test removing configuration."""
        # Add a test config
        test_config = DocumentationSiteConfig(
            domain="remove-test.com",
            content_selectors=["main"]
        )
        self.registry.add_config(test_config)
        
        # Verify it exists
        assert self.registry.get_config_by_domain("remove-test.com") is not None
        
        # Remove it
        removed = self.registry.remove_config("remove-test.com")
        assert removed is True
        
        # Verify it's gone
        assert self.registry.get_config_by_domain("remove-test.com") is None
        
        # Try removing non-existent
        removed = self.registry.remove_config("non-existent.com")
        assert removed is False
    
    def test_clear_configs(self):
        """Test clearing all configurations."""
        # Add test config
        test_config = DocumentationSiteConfig(
            domain="clear-test.com",
            content_selectors=["main"]
        )
        self.registry.add_config(test_config)
        
        # Clear all
        self.registry.clear_configs()
        
        # Verify all gone
        assert self.registry.get_config_by_domain("n8n.io") is None
        assert self.registry.get_config_by_domain("clear-test.com") is None
        assert len(self.registry.list_domains()) == 0
    
    def test_add_config_invalid_type(self):
        """Test adding invalid config type raises error."""
        with pytest.raises(TypeError, match="config must be a DocumentationSiteConfig instance"):
            self.registry.add_config("not a config")


class TestGlobalRegistry:
    """Test the global registry instance and convenience functions."""
    
    def test_global_get_config_by_domain_function(self):
        """Test the global get_config_by_domain function."""
        # Should work with global registry
        config = get_config_by_domain("n8n.io")
        assert config is not None
        assert config.domain == "n8n.io"
        
        # Should return None for non-existent
        config = get_config_by_domain("non-existent.test")
        assert config is None
    
    def test_register_site_config_function(self):
        """Test the register_site_config convenience function."""
        config = register_site_config(
            domain="convenience-test.com",
            content_selectors=["main.convenience", ".convenience-content"]
        )
        
        assert isinstance(config, DocumentationSiteConfig)
        assert config.domain == "convenience-test.com"
        assert config.content_selectors == ["main.convenience", ".convenience-content"]
        
        # Should be retrievable via global function
        retrieved = get_config_by_domain("convenience-test.com")
        assert retrieved is not None
        assert retrieved.domain == "convenience-test.com"
    
    def test_extract_domain_from_url(self):
        """Test URL domain extraction utility."""
        assert extract_domain_from_url("https://docs.n8n.io/workflows/") == "docs.n8n.io"
        assert extract_domain_from_url("http://example.com/path") == "example.com"
        assert extract_domain_from_url("https://api.readme.io/docs") == "api.readme.io"
        assert extract_domain_from_url("ftp://files.example.org/file.txt") == "files.example.org"
        
        # Test edge cases
        assert extract_domain_from_url("invalid-url") == ""
        assert extract_domain_from_url("") == ""
        
        # Test case normalization
        assert extract_domain_from_url("https://EXAMPLE.COM/path") == "example.com"


class TestTaskSpecificRequirements:
    """Test specific requirements from Task 16 subtasks."""
    
    def test_n8n_io_configuration_present(self):
        """Test that n8n.io configuration is present with correct selectors (Task 16.3)."""
        config = get_config_by_domain("n8n.io")
        
        assert config is not None, "n8n.io configuration should be present"
        assert config.domain == "n8n.io"
        
        # Task 16.3 specifically requires these selectors
        expected_selectors = ["main.md-main", "article.md-content__inner"]
        for selector in expected_selectors:
            assert selector in config.content_selectors, f"Selector '{selector}' should be present"
    
    def test_docs_n8n_io_subdomain_lookup(self):
        """Test that docs.n8n.io also resolves to the n8n.io configuration."""
        config = get_config_by_domain("docs.n8n.io")
        
        assert config is not None
        assert config.domain == "n8n.io"  # Should resolve to base domain
        assert "main.md-main" in config.content_selectors
        assert "article.md-content__inner" in config.content_selectors
    
    def test_registry_lookup_function_signature(self):
        """Test that get_config_by_domain function has correct signature as per Task 16.2."""
        # The function should accept domain string and return Optional[DocumentationSiteConfig]
        result = get_config_by_domain("n8n.io")
        assert isinstance(result, DocumentationSiteConfig)
        
        result = get_config_by_domain("non-existent.domain")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])