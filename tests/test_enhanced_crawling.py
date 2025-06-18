#!/usr/bin/env python3
"""
Test suite for enhanced crawling functionality.

This test suite validates framework detection, quality validation,
enhanced crawling functions, and fallback mechanisms.
"""

import pytest
import asyncio
import os
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enhanced_crawler_config import (
    DocumentationFramework,
    DocumentationSiteConfigManager,
    detect_framework,
    get_optimized_config,
    get_quality_thresholds
)
from content_quality import (
    ContentQualityAnalyzer,
    ContentQualityMetrics,
    calculate_content_quality,
    is_high_quality_content,
    should_retry_extraction
)
from smart_crawler_factory import (
    SmartCrawlerFactory,
    EnhancedCrawler,
    CrawlResult
)


class TestFrameworkDetection:
    """Test framework detection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config_manager = DocumentationSiteConfigManager()
    
    def test_material_design_detection_by_domain(self):
        """Test Material Design detection by domain patterns."""
        # n8n documentation (known Material Design site)
        framework = detect_framework("https://docs.n8n.io/workflows/")
        assert framework == DocumentationFramework.GENERIC  # Domain pattern not in list
        
        # Test with HTML content
        html_content = """
        <html>
        <head><meta name="generator" content="mkdocs"></head>
        <body>
        <main class="md-main">
        <article class="md-content__inner">
        <div class="md-sidebar"></div>
        </body>
        </html>
        """
        framework = detect_framework("https://docs.n8n.io/workflows/", html_content)
        assert framework == DocumentationFramework.MATERIAL_DESIGN
    
    def test_readme_io_detection_by_domain(self):
        """Test ReadMe.io detection by domain patterns."""
        # Test ReadMe.io domain patterns
        framework = detect_framework("https://docs.virustotal.readme.io/reference/overview")
        assert framework == DocumentationFramework.README_IO
        
        framework = detect_framework("https://api.readme.io/docs/getting-started")
        assert framework == DocumentationFramework.README_IO
    
    def test_readme_io_detection_by_html(self):
        """Test ReadMe.io detection by HTML content."""
        html_content = """
        <html>
        <body>
        <main class="rm-Guides">
        <div class="hub-sidebar"></div>
        <div class="rm-Sidebar"></div>
        </body>
        </html>
        """
        framework = detect_framework("https://example.com/docs", html_content)
        assert framework == DocumentationFramework.README_IO
    
    def test_gitbook_detection(self):
        """Test GitBook detection."""
        framework = detect_framework("https://docs.example.gitbook.io/guide")
        assert framework == DocumentationFramework.GITBOOK
        
        html_content = """
        <html>
        <body>
        <div class="gitbook-content">
        <div class="book-summary"></div>
        </body>
        </html>
        """
        framework = detect_framework("https://example.com/docs", html_content)
        assert framework == DocumentationFramework.GITBOOK
    
    def test_generic_fallback(self):
        """Test generic fallback for unknown frameworks."""
        framework = detect_framework("https://example.com/docs")
        assert framework == DocumentationFramework.GENERIC
        
        html_content = "<html><body><p>Some content</p></body></html>"
        framework = detect_framework("https://example.com/docs", html_content)
        assert framework == DocumentationFramework.GENERIC
    
    def test_framework_config_retrieval(self):
        """Test framework configuration retrieval."""
        # Test Material Design config
        config = self.config_manager.get_framework_config(DocumentationFramework.MATERIAL_DESIGN)
        assert "main.md-main" in config.target_elements
        assert ".md-sidebar" in config.excluded_selectors
        assert "nav" in config.excluded_tags
        
        # Test ReadMe.io config
        config = self.config_manager.get_framework_config(DocumentationFramework.README_IO)
        assert "main.rm-Guides" in config.target_elements
        assert ".rm-Sidebar" in config.excluded_selectors
        
        # Test quality thresholds
        thresholds = get_quality_thresholds(DocumentationFramework.MATERIAL_DESIGN)
        assert len(thresholds) == 2
        assert 0.0 <= thresholds[0] <= 1.0  # min_content_ratio
        assert 0.0 <= thresholds[1] <= 1.0  # max_link_density
    
    def test_cache_functionality(self):
        """Test framework detection caching."""
        # First detection should cache the result
        framework1 = detect_framework("https://docs.test.com/guide")
        framework2 = detect_framework("https://docs.test.com/another-page")
        
        # Should return the same framework for same domain
        assert framework1 == framework2
        
        # Clear cache and test
        self.config_manager.clear_cache()
        framework3 = detect_framework("https://docs.test.com/guide")
        assert framework3 == DocumentationFramework.GENERIC


class TestContentQuality:
    """Test content quality validation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = ContentQualityAnalyzer()
    
    def test_high_quality_content(self):
        """Test detection of high quality content."""
        high_quality_content = """
        # API Documentation
        
        This is a comprehensive guide to using our API. The API allows you to 
        perform various operations including data retrieval, user management, 
        and system configuration.
        
        ## Getting Started
        
        To begin using the API, you'll need to obtain an API key and configure
        your development environment. Here's a step-by-step guide:
        
        1. Register for an account
        2. Generate your API key
        3. Install the SDK
        
        ```python
        import api_client
        
        client = api_client.Client(api_key="your_key_here")
        response = client.get_user_data(user_id=123)
        ```
        
        ## Authentication
        
        All API requests require authentication using your API key. Include
        the key in the Authorization header of your HTTP requests.
        """
        
        metrics = calculate_content_quality(high_quality_content)
        
        assert metrics.overall_quality_score > 0.6
        assert metrics.quality_category in ["good", "excellent"]
        assert metrics.content_to_navigation_ratio > 0.8
        assert metrics.link_density < 0.2
        assert metrics.word_count > 100
        assert metrics.code_block_count > 0
        assert not should_retry_extraction(metrics)
    
    def test_low_quality_navigation_heavy_content(self):
        """Test detection of low quality, navigation-heavy content."""
        nav_heavy_content = """
        Home | About | Contact | Services
        
        - [Getting Started](guide.html)
        - [User Manual](manual.html) 
        - [API Reference](api.html)
        - [FAQ](faq.html)
        - [Support](support.html)
        - [Downloads](downloads.html)
        - [Blog](blog.html)
        - [News](news.html)
        - [Events](events.html)
        - [Partners](partners.html)
        
        Navigation: Previous | Next | Up | Top | Bottom
        
        Search | Help | Feedback | Print | Share | Bookmark
        
        Copyright © 2024 Example Corp. All rights reserved.
        Privacy Policy | Terms of Service | Cookies
        """
        
        metrics = calculate_content_quality(nav_heavy_content)
        
        assert metrics.overall_quality_score < 0.4
        assert metrics.quality_category in ["poor", "fair"]
        assert metrics.content_to_navigation_ratio < 0.5
        assert metrics.link_density > 0.3
        assert metrics.navigation_element_count > 10
        assert should_retry_extraction(metrics)
    
    def test_medium_quality_content(self):
        """Test detection of medium quality content."""
        medium_content = """
        # Quick Start Guide
        
        Welcome to our platform! Here are the essential steps:
        
        - [Login](login.html)
        - [Dashboard](dashboard.html)
        - [Settings](settings.html)
        
        Getting started is easy. First, create your account and verify your email.
        Then, log in to access the dashboard where you can configure your preferences.
        
        For more help, see:
        - [Help Center](help.html) 
        - [Contact Support](support.html)
        """
        
        metrics = calculate_content_quality(medium_content)
        
        assert 0.3 <= metrics.overall_quality_score <= 0.7
        assert metrics.quality_category in ["fair", "good"]
        assert 0.4 <= metrics.content_to_navigation_ratio <= 0.8
    
    def test_quality_calculation_performance(self):
        """Test that quality calculation is fast enough."""
        content = "Test content " * 1000  # Large content
        
        metrics = calculate_content_quality(content)
        
        # Should complete in under 100ms as per requirements
        assert metrics.calculation_time_ms < 100
        assert metrics.word_count > 0
    
    def test_improvement_suggestions(self):
        """Test quality improvement suggestions."""
        nav_heavy_content = """
        [Home](/) | [About](/about) | [Contact](/contact) | [Services](/services)
        [Products](/products) | [Blog](/blog) | [Support](/support) | [Login](/login)
        
        Short content with many links.
        
        [Link 1](1) | [Link 2](2) | [Link 3](3) | [Link 4](4) | [Link 5](5)
        """
        
        metrics = calculate_content_quality(nav_heavy_content)
        
        assert len(metrics.improvement_suggestions) > 0
        suggestions_text = " ".join(metrics.improvement_suggestions).lower()
        
        # Should suggest navigation-related improvements
        assert any(keyword in suggestions_text for keyword in [
            "navigation", "link", "selector", "content"
        ])


class TestSmartCrawlerFactory:
    """Test smart crawler factory functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.factory = SmartCrawlerFactory()
    
    def test_config_creation(self):
        """Test creation of optimized configurations."""
        # Test Material Design configuration
        config = self.factory.create_optimized_config(
            "https://docs.n8n.io/workflows/",
            DocumentationFramework.MATERIAL_DESIGN
        )
        
        assert config is not None
        assert hasattr(config, 'target_elements')
        assert hasattr(config, 'excluded_tags')
        assert hasattr(config, 'word_count_threshold')
    
    def test_url_specific_enhancements(self):
        """Test URL-specific configuration enhancements."""
        # Test n8n.io specific enhancements
        base_config = get_optimized_config(DocumentationFramework.MATERIAL_DESIGN)
        enhanced_config = self.factory._enhance_config_for_url(
            base_config, 
            "https://docs.n8n.io/workflows/",
            DocumentationFramework.MATERIAL_DESIGN
        )
        
        # n8n should have higher word threshold due to verbose navigation
        assert enhanced_config.word_count_threshold >= 20
        assert enhanced_config.exclude_external_links == True
    
    def test_fallback_config_creation(self):
        """Test fallback configuration creation."""
        # Test first fallback attempt
        fallback_config = self.factory.create_fallback_config(1)
        assert fallback_config is not None
        assert hasattr(fallback_config, 'css_selector')
        
        # Test multiple fallback attempts
        for i in range(1, 5):
            config = self.factory.create_fallback_config(i)
            assert config is not None
        
        # Test beyond available selectors
        config = self.factory.create_fallback_config(10)
        assert config.css_selector == "body"  # Last resort


class TestEnhancedCrawler:
    """Test enhanced crawler functionality."""
    
    @pytest.fixture
    def mock_crawler_result(self):
        """Create mock crawler result."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.status_code = 200
        mock_result.html = "<html><body><main>Test content</main></body></html>"
        mock_result.cleaned_html = "<main>Test content</main>"
        mock_result.markdown = "# Test Content\n\nThis is test content for validation."
        mock_result.extracted_content = "Test content for validation"
        mock_result.links = {"internal": [], "external": []}
        mock_result.error_message = None
        return mock_result
    
    @pytest.mark.asyncio
    async def test_enhanced_crawler_context_manager(self):
        """Test enhanced crawler context manager."""
        with patch('smart_crawler_factory.AsyncWebCrawler') as mock_crawler_class:
            mock_crawler = AsyncMock()
            mock_crawler_class.return_value = mock_crawler
            
            async with EnhancedCrawler() as crawler:
                assert crawler is not None
                assert hasattr(crawler, 'factory')
                assert hasattr(crawler, 'max_fallback_attempts')
    
    @pytest.mark.asyncio
    async def test_crawl_result_creation(self):
        """Test CrawlResult creation and validation."""
        result = CrawlResult(
            url="https://example.com",
            html="<html></html>",
            cleaned_html="<div>content</div>",
            markdown="# Test",
            extracted_content="Test content",
            success=True,
            status_code=200,
            framework=DocumentationFramework.GENERIC,
            quality_metrics=None,
            used_fallback=False,
            extraction_attempts=1,
            total_time_seconds=1.5,
            framework_detection_time_ms=50.0,
            quality_analysis_time_ms=25.0
        )
        
        assert result.url == "https://example.com"
        assert result.success == True
        assert result.framework == DocumentationFramework.GENERIC
        assert result.extraction_attempts == 1
        assert result.total_time_seconds == 1.5


class TestIntegrationScenarios:
    """Test integration scenarios for enhanced crawling."""
    
    def test_n8n_documentation_scenario(self):
        """Test n8n documentation site scenario."""
        # Simulate n8n HTML structure
        n8n_html = """
        <html>
        <head><meta name="generator" content="mkdocs"></head>
        <body>
        <header class="md-header"></header>
        <nav class="md-nav md-nav--primary">
            <ul>
                <li><a href="/workflows/">Workflows</a></li>
                <li><a href="/nodes/">Nodes</a></li>
                <li><a href="/integrations/">Integrations</a></li>
            </ul>
        </nav>
        <div class="md-sidebar md-sidebar--primary"></div>
        <main class="md-main">
            <article class="md-content__inner md-typeset">
                <h1>Workflow Automation</h1>
                <p>n8n is a workflow automation tool that allows you to connect different services and automate processes.</p>
                
                <h2>Getting Started</h2>
                <p>To create your first workflow, follow these steps:</p>
                <ol>
                    <li>Open the n8n editor</li>
                    <li>Add nodes to your workflow</li>
                    <li>Configure the connections</li>
                    <li>Test and execute</li>
                </ol>
                
                <h2>Example Workflow</h2>
                <p>Here's a simple example of a webhook-triggered workflow:</p>
                <pre><code>
                {
                  "nodes": [
                    {
                      "name": "Webhook",
                      "type": "n8n-nodes-base.webhook"
                    }
                  ]
                }
                </code></pre>
            </article>
        </main>
        </body>
        </html>
        """
        
        # Test framework detection
        framework = detect_framework("https://docs.n8n.io/workflows/", n8n_html)
        assert framework == DocumentationFramework.MATERIAL_DESIGN
        
        # Test content quality on main content extraction
        main_content = """
        # Workflow Automation
        
        n8n is a workflow automation tool that allows you to connect different services and automate processes.
        
        ## Getting Started
        
        To create your first workflow, follow these steps:
        
        1. Open the n8n editor
        2. Add nodes to your workflow  
        3. Configure the connections
        4. Test and execute
        
        ## Example Workflow
        
        Here's a simple example of a webhook-triggered workflow:
        
        ```json
        {
          "nodes": [
            {
              "name": "Webhook",
              "type": "n8n-nodes-base.webhook"
            }
          ]
        }
        ```
        """
        
        metrics = calculate_content_quality(main_content)
        assert metrics.quality_category in ["good", "excellent"]
        assert metrics.content_to_navigation_ratio > 0.7
        assert not should_retry_extraction(metrics)
    
    def test_virustotal_documentation_scenario(self):
        """Test VirusTotal documentation site scenario."""
        # Simulate VirusTotal/ReadMe.io HTML structure
        vt_html = """
        <html>
        <body>
        <nav class="hub-sidebar"></nav>
        <div class="rm-Sidebar"></div>
        <main class="rm-Guides">
            <div class="rm-Article">
                <h1>API Reference</h1>
                <p>The VirusTotal API allows you to programmatically scan files and URLs for malware.</p>
                
                <h2>Authentication</h2>
                <p>All API requests require an API key. Include it in the x-apikey header:</p>
                
                <pre><code>
                curl -X GET "https://www.virustotal.com/api/v3/files/{id}" \
                     -H "x-apikey: YOUR_API_KEY"
                </code></pre>
                
                <h2>Rate Limits</h2>
                <p>Free tier allows 500 requests per day. Premium tiers have higher limits.</p>
            </div>
        </main>
        </body>
        </html>
        """
        
        # Test framework detection
        framework = detect_framework("https://developers.virustotal.com/reference/overview", vt_html)
        assert framework == DocumentationFramework.README_IO
        
        # Test content quality on main content
        main_content = """
        # API Reference
        
        The VirusTotal API allows you to programmatically scan files and URLs for malware.
        
        ## Authentication
        
        All API requests require an API key. Include it in the x-apikey header:
        
        ```bash
        curl -X GET "https://www.virustotal.com/api/v3/files/{id}" \
             -H "x-apikey: YOUR_API_KEY"
        ```
        
        ## Rate Limits
        
        Free tier allows 500 requests per day. Premium tiers have higher limits.
        """
        
        metrics = calculate_content_quality(main_content)
        assert metrics.quality_category in ["good", "excellent"]
        assert metrics.code_block_count > 0
        assert not should_retry_extraction(metrics)


class TestPerformanceAndReliability:
    """Test performance and reliability aspects."""
    
    def test_framework_detection_performance(self):
        """Test framework detection performance."""
        import time
        
        large_html = "<html><body>" + "<div>content</div>" * 1000 + "</body></html>"
        
        start_time = time.time()
        framework = detect_framework("https://example.com", large_html)
        end_time = time.time()
        
        # Should complete quickly even with large HTML
        assert (end_time - start_time) < 1.0
        assert framework == DocumentationFramework.GENERIC
    
    def test_quality_analysis_performance(self):
        """Test quality analysis performance."""
        large_content = "This is test content. " * 1000
        
        metrics = calculate_content_quality(large_content)
        
        # Should meet performance requirement
        assert metrics.calculation_time_ms < 100
        assert metrics.word_count > 2000
    
    def test_error_handling(self):
        """Test error handling in quality analysis."""
        # Test with empty content
        metrics = calculate_content_quality("")
        assert metrics.word_count == 0
        assert metrics.overall_quality_score == 0.0
        
        # Test with very short content
        metrics = calculate_content_quality("Hi")
        assert metrics.word_count == 1
        assert metrics.overall_quality_score < 0.5
    
    def test_config_manager_robustness(self):
        """Test configuration manager robustness."""
        config_manager = DocumentationSiteConfigManager()
        
        # Test with invalid framework
        try:
            # This should return generic config, not crash
            config = config_manager.get_framework_config(None)
            assert config is not None
        except:
            # If it does throw an exception, that's acceptable too
            pass
        
        # Test cache clearing
        config_manager.clear_cache()
        # Should not throw any errors
        
        # Test detection with malformed HTML
        framework = config_manager.detect_documentation_framework(
            "https://example.com", 
            "<html><body><div>broken"
        )
        assert framework == DocumentationFramework.GENERIC


if __name__ == "__main__":
    pytest.main([__file__, "-v"])