#!/usr/bin/env python3
"""
Direct test of the exact configuration the AdvancedWebCrawler is using.
"""

import asyncio
from src.advanced_web_crawler import AdvancedWebCrawler

async def debug_exact_config():
    """Test the exact configuration being used by AdvancedWebCrawler."""
    
    print("=== DEBUGGING EXACT ADVANCEDWEBCRAWLER CONFIG ===\n")
    
    async with AdvancedWebCrawler() as crawler:
        # Get the exact configuration that would be used
        config = crawler._create_optimized_run_config('https://docs.n8n.io/try-it-out/quickstart/')
        
        print("Configuration details:")
        print(f"CSS Selector: {config.css_selector}")
        print(f"Excluded Selector: {config.excluded_selector}")
        print(f"Word threshold: {config.word_count_threshold}")
        
        # Test with this exact config
        result = await crawler.crawler.arun(
            url='https://docs.n8n.io/try-it-out/quickstart/',
            config=config
        )
        
        print(f"\nRaw result:")
        print(f"Success: {result.success}")
        print(f"Length: {len(result.markdown) if result.markdown else 0}")
        print(f"Word count: {len(result.markdown.split()) if result.markdown else 0}")
        
        if result.markdown:
            print(f"\nFirst 1000 chars:")
            print(result.markdown[:1000])
        else:
            print(f"\nError: {result.error_message}")
            
        # Now test with absolutely minimal config
        print(f"\n--- TESTING WITH MINIMAL CONFIG ---")
        from crawl4ai import CrawlerRunConfig, CacheMode
        from crawl4ai.extraction_strategy import NoExtractionStrategy
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
        
        minimal_config = CrawlerRunConfig(
            extraction_strategy=NoExtractionStrategy(),
            markdown_generator=DefaultMarkdownGenerator(),
            css_selector="article.md-content__inner",
            excluded_selector="",  # No exclusions
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=1
        )
        
        result_minimal = await crawler.crawler.arun(
            url='https://docs.n8n.io/try-it-out/quickstart/',
            config=minimal_config
        )
        
        print(f"Minimal config result:")
        print(f"Success: {result_minimal.success}")
        print(f"Length: {len(result_minimal.markdown) if result_minimal.markdown else 0}")
        print(f"Word count: {len(result_minimal.markdown.split()) if result_minimal.markdown else 0}")
        
        if result_minimal.markdown:
            print(f"Preview: {result_minimal.markdown[:500]}")

if __name__ == "__main__":
    asyncio.run(debug_exact_config())