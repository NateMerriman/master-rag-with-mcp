#!/usr/bin/env python3
"""
Debug script to test the _post_process_markdown function and see what's happening.
"""

import asyncio
import re
from src.advanced_web_crawler import AdvancedWebCrawler

async def debug_post_processing():
    """Debug the post-processing step to see what's being removed vs preserved."""
    
    print("=== DEBUGGING POST-PROCESSING ===\n")
    
    async with AdvancedWebCrawler() as crawler:
        # Get the raw result before post-processing
        result = await crawler.crawler.arun(
            url='https://docs.n8n.io/try-it-out/quickstart/',
            config=crawler._create_optimized_run_config('https://docs.n8n.io/try-it-out/quickstart/')
        )
        
        print(f"Raw extraction success: {result.success}")
        print(f"Raw markdown length: {len(result.markdown)} characters")
        print(f"Raw word count: {len(result.markdown.split())}")
        
        print("\n--- RAW MARKDOWN (first 2000 chars) ---")
        print(result.markdown[:2000])
        
        # Apply our post-processing
        cleaned = crawler._post_process_markdown(result.markdown)
        
        print(f"\nCleaned markdown length: {len(cleaned)} characters")  
        print(f"Cleaned word count: {len(cleaned.split())}")
        
        print("\n--- CLEANED MARKDOWN (first 2000 chars) ---")
        print(cleaned[:2000])
        
        # Check for specific patterns
        print(f"\n--- PATTERN ANALYSIS ---")
        print(f"Raw contains 'workflow': {'workflow' in result.markdown}")
        print(f"Cleaned contains 'workflow': {'workflow' in cleaned}")
        print(f"Raw contains 'template': {'template' in result.markdown}")
        print(f"Cleaned contains 'template': {'template' in cleaned}")
        
        # Count HTML tags in both
        html_tags_raw = len(re.findall(r'<[^>]+>', result.markdown))
        html_tags_cleaned = len(re.findall(r'<[^>]+>', cleaned))
        print(f"HTML tags raw: {html_tags_raw}")
        print(f"HTML tags cleaned: {html_tags_cleaned}")

if __name__ == "__main__":
    asyncio.run(debug_post_processing())