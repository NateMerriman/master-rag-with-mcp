#!/usr/bin/env python3
"""
Debug script to examine what CSS selectors and elements are present on the n8n page.
"""

import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import NoExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

async def debug_page_structure():
    """Debug what elements are actually on the page."""
    
    print("=== DEBUGGING PAGE STRUCTURE ===\n")
    
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        browser_type="playwright"
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        
        # First, get the page with minimal exclusions to see all content
        minimal_config = CrawlerRunConfig(
            extraction_strategy=NoExtractionStrategy(),
            markdown_generator=DefaultMarkdownGenerator(),
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=1,
            excluded_selector="",  # No exclusions to see everything
        )
        
        print("1. EXTRACTING WITH NO EXCLUSIONS...")
        result = await crawler.arun(
            url='https://docs.n8n.io/try-it-out/quickstart/',
            config=minimal_config
        )
        
        print(f"Success: {result.success}")
        print(f"Total length: {len(result.markdown)} characters")
        print(f"Word count: {len(result.markdown.split())}")
        
        print("\n--- FULL CONTENT (first 3000 chars) ---")
        print(result.markdown[:3000])
        
        # Check for specific content patterns
        print(f"\n--- CONTENT ANALYSIS ---")
        print(f"Contains 'The very quick quickstart': {'The very quick quickstart' in result.markdown}")
        print(f"Contains 'workflow': {'workflow' in result.markdown}")
        print(f"Contains 'template': {'template' in result.markdown}")
        print(f"Contains 'quickstart': {'quickstart' in result.markdown}")
        
        # Now test with current targeting
        print("\n2. TESTING CURRENT TARGET SELECTORS...")
        target_selectors = [
            "main.md-main",
            "article.md-content__inner", 
            ".md-content",
            "main[class*='md-']",
            "article[class*='md-']"
        ]
        
        for selector in target_selectors:
            targeted_config = CrawlerRunConfig(
                extraction_strategy=NoExtractionStrategy(),
                markdown_generator=DefaultMarkdownGenerator(),
                cache_mode=CacheMode.BYPASS,
                word_count_threshold=1,
                css_selector=selector,
                excluded_selector=""
            )
            
            result = await crawler.arun(
                url='https://docs.n8n.io/try-it-out/quickstart/',
                config=targeted_config
            )
            
            print(f"\nSelector '{selector}':")
            print(f"  Success: {result.success}")
            print(f"  Word count: {len(result.markdown.split()) if result.markdown else 0}")
            if result.markdown and len(result.markdown) > 100:
                print(f"  Preview: {result.markdown[:200]}...")

if __name__ == "__main__":
    asyncio.run(debug_page_structure())