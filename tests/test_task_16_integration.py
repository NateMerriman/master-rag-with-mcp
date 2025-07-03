#!/usr/bin/env python3
"""
Unit tests for Task 16 integration into AdvancedWebCrawler.

Tests the integration functionality requested by Task 16 subtasks 4 and 5.
"""

import pytest
import sys
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from documentation_site_config import (
    DocumentationSiteConfig,
    get_config_by_domain,
    extract_domain_from_url
)
from advanced_web_crawler import AdvancedWebCrawler


class TestTask16Integration:
    """Test Task 16 integration into AdvancedWebCrawler (subtasks 16.4 and 16.5)."""
    
    def test_extract_domain_from_url(self):
        """Test domain extraction utility works correctly."""
        # Test cases from Task 16.4
        assert extract_domain_from_url("https://docs.n8n.io/workflows/") == "docs.n8n.io"
        assert extract_domain_from_url("https://n8n.io/glossary/") == "n8n.io"
        assert extract_domain_from_url("https://example.com/path") == "example.com"
    
    def test_n8n_configuration_lookup(self):
        """Test that n8n.io configurations can be looked up correctly."""
        # Test direct n8n.io lookup
        config = get_config_by_domain("n8n.io")
        assert config is not None
        assert config.domain == "n8n.io"
        assert "main.md-main" in config.content_selectors
        assert "article.md-content__inner" in config.content_selectors
        
        # Test subdomain lookup (docs.n8n.io should match n8n.io config)
        config = get_config_by_domain("docs.n8n.io")
        assert config is not None
        assert config.domain == "n8n.io"
    
    def test_unconfigured_site_lookup(self):
        """Test lookup of unconfigured site returns None."""
        config = get_config_by_domain("unconfigured-site.com")
        assert config is None
    
    @patch('advanced_web_crawler.get_config_by_domain')
    @patch('advanced_web_crawler.extract_domain_from_url')
    def test_crawler_uses_simple_config_when_available(self, mock_extract_domain, mock_get_config):
        """Test that AdvancedWebCrawler uses simple config when available (Task 16.4)."""
        # Setup mocks
        mock_extract_domain.return_value = "docs.n8n.io"
        mock_config = DocumentationSiteConfig(
            domain="n8n.io",
            content_selectors=["main.md-main", "article.md-content__inner"]
        )
        mock_get_config.return_value = mock_config
        
        # Create crawler and test config creation
        crawler = AdvancedWebCrawler()
        config = crawler._create_optimized_run_config("https://docs.n8n.io/nodes/")
        
        # Verify domain extraction was called
        mock_extract_domain.assert_called_once_with("https://docs.n8n.io/nodes/")
        
        # Verify config lookup was called
        mock_get_config.assert_called_once_with("docs.n8n.io")
        
        # Verify the CSS selectors are set correctly (Task 16.5)
        expected_selector = "main.md-main, article.md-content__inner"
        assert config.css_selector == expected_selector
    
    @patch('advanced_web_crawler.get_config_by_domain')
    @patch('advanced_web_crawler.extract_domain_from_url')
    def test_crawler_fallback_when_no_simple_config(self, mock_extract_domain, mock_get_config):
        """Test that crawler falls back to enhanced config when no simple config found."""
        # Setup mocks
        mock_extract_domain.return_value = "unknown-site.com"
        mock_get_config.return_value = None  # No config found
        
        # Create crawler and test config creation
        crawler = AdvancedWebCrawler()
        config = crawler._create_optimized_run_config("https://unknown-site.com/docs/")
        
        # Verify domain lookup was attempted
        mock_extract_domain.assert_called_once_with("https://unknown-site.com/docs/")
        mock_get_config.assert_called_once_with("unknown-site.com")
        
        # Should still create a valid config (using enhanced system or fallback)
        assert config is not None
        assert hasattr(config, 'css_selector')
    
    @patch('advanced_web_crawler.get_config_by_domain')
    @patch('advanced_web_crawler.extract_domain_from_url')  
    def test_crawler_handles_empty_selectors_list(self, mock_extract_domain, mock_get_config):
        """Test crawler handles config with empty selectors list."""
        # Setup mocks
        mock_extract_domain.return_value = "test.com"
        mock_config = DocumentationSiteConfig(
            domain="test.com",
            content_selectors=[]  # Empty list
        )
        mock_get_config.return_value = mock_config
        
        # Create crawler and test config creation
        crawler = AdvancedWebCrawler()
        config = crawler._create_optimized_run_config("https://test.com/docs/")
        
        # Should fall back to enhanced config when selectors list is empty
        assert config is not None
        assert hasattr(config, 'css_selector')
    
    def test_css_selector_string_formatting(self):
        """Test that CSS selectors are properly formatted as comma-separated string."""
        # Test multiple selectors
        selectors = ["main.md-main", "article.md-content__inner", ".content"]
        expected = "main.md-main, article.md-content__inner, .content"
        actual = ", ".join(selectors)
        assert actual == expected
        
        # Test single selector
        selectors = ["main.content"]
        expected = "main.content"
        actual = ", ".join(selectors)
        assert actual == expected
    
    @patch('advanced_web_crawler.SIMPLE_CONFIG_AVAILABLE', False)
    def test_crawler_works_when_simple_config_unavailable(self):
        """Test that crawler still works when simple config system is unavailable."""
        # Create crawler when simple config is not available
        crawler = AdvancedWebCrawler()
        config = crawler._create_optimized_run_config("https://docs.n8n.io/workflows/")
        
        # Should still create a valid config
        assert config is not None
        assert hasattr(config, 'css_selector')
    
    def test_n8n_specific_selectors_from_task_16_3(self):
        """Test that n8n.io has the exact selectors specified in Task 16.3."""
        config = get_config_by_domain("n8n.io")
        
        assert config is not None
        # Task 16.3 specifically requires these selectors
        required_selectors = ["main.md-main", "article.md-content__inner"]
        
        for selector in required_selectors:
            assert selector in config.content_selectors, f"Required selector '{selector}' not found"
    
    @patch('advanced_web_crawler.logger')
    @patch('advanced_web_crawler.get_config_by_domain')
    @patch('advanced_web_crawler.extract_domain_from_url')
    def test_crawler_logs_config_usage(self, mock_extract_domain, mock_get_config, mock_logger):
        """Test that crawler logs when using Task 16 simple config."""
        # Setup mocks
        mock_extract_domain.return_value = "n8n.io"
        mock_config = DocumentationSiteConfig(
            domain="n8n.io",
            content_selectors=["main.md-main", "article.md-content__inner"]
        )
        mock_get_config.return_value = mock_config
        
        # Create crawler and test config creation
        crawler = AdvancedWebCrawler()
        config = crawler._create_optimized_run_config("https://n8n.io/glossary/")
        
        # Verify logging was called
        mock_logger.info.assert_called()
        
        # Check that the log message mentions Task 16 simple config
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        task_16_logged = any("Task 16 simple config" in msg for msg in log_calls)
        assert task_16_logged, "Should log usage of Task 16 simple config"


class TestTask16ErrorHandling:
    """Test error handling in Task 16 integration."""
    
    @patch('advanced_web_crawler.extract_domain_from_url')
    def test_crawler_handles_domain_extraction_error(self, mock_extract_domain):
        """Test crawler handles errors in domain extraction gracefully."""
        # Mock domain extraction to raise exception
        mock_extract_domain.side_effect = Exception("URL parsing error")
        
        # Create crawler and test config creation
        crawler = AdvancedWebCrawler()
        config = crawler._create_optimized_run_config("invalid-url")
        
        # Should still create a valid config despite error
        assert config is not None
        assert hasattr(config, 'css_selector')
    
    @patch('advanced_web_crawler.get_config_by_domain')
    @patch('advanced_web_crawler.extract_domain_from_url')
    def test_crawler_handles_config_lookup_error(self, mock_extract_domain, mock_get_config):
        """Test crawler handles errors in config lookup gracefully."""
        # Setup mocks
        mock_extract_domain.return_value = "test.com"
        mock_get_config.side_effect = Exception("Config lookup error")
        
        # Create crawler and test config creation
        crawler = AdvancedWebCrawler()
        config = crawler._create_optimized_run_config("https://test.com/docs/")
        
        # Should still create a valid config despite error
        assert config is not None
        assert hasattr(config, 'css_selector')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])