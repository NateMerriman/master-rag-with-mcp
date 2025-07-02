#!/usr/bin/env python3
"""
Test script for AdvancedWebCrawler with Playwright and Quality Validation.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from advanced_web_crawler import AdvancedWebCrawler, crawl_single_page_advanced


async def test_playwright_rendering():
    """Test that Playwright can handle JavaScript-heavy sites."""
    
    # Test URL that requires JavaScript rendering (promptingguide.ai mentioned in task)
    test_url = "https://www.promptingguide.ai/"
    
    print(f"Testing AdvancedWebCrawler with: {test_url}")
    print("This should successfully extract content from a JavaScript-heavy site...")
    
    try:
        result = await crawl_single_page_advanced(test_url, enable_quality_validation=True)
        
        if result.success:
            print(f"✅ Success! Extracted {result.word_count} words")
            print(f"   Title: {result.title}")
            print(f"   Framework: {result.framework_detected}")
            print(f"   Extraction time: {result.extraction_time_ms:.1f}ms")
            print(f"   Content ratio: {result.content_to_navigation_ratio:.2f}")
            
            # Show quality validation results
            if result.quality_validation:
                print(f"   Quality Score: {result.quality_score:.3f}")
                print(f"   Quality Category: {result.quality_validation.category}")
                print(f"   Quality Passed: {result.quality_passed}")
                
                if result.quality_validation.issues:
                    print(f"   Issues: {', '.join(result.quality_validation.issues)}")
                
                if result.quality_validation.warnings:
                    print(f"   Warnings: {', '.join(result.quality_validation.warnings[:2])}")
            
            # Show first 200 characters of markdown
            preview = result.markdown[:200].replace('\n', ' ')
            print(f"   Preview: {preview}...")
            
            return True
        else:
            print(f"❌ Failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


async def test_quality_validation():
    """Test quality validation with known good and bad examples."""
    
    print("\n" + "=" * 50)
    print("🧪 Testing Quality Validation")
    print("=" * 50)
    
    # Test with a well-structured documentation site
    test_urls = [
        "https://docs.python.org/3/",  # Should be high quality
        "https://www.promptingguide.ai/",  # May have JavaScript challenges
    ]
    
    async with AdvancedWebCrawler(enable_quality_validation=True) as crawler:
        
        for i, url in enumerate(test_urls, 1):
            print(f"\nTest {i}: {url}")
            print("-" * 30)
            
            try:
                result = await crawler.crawl_single_page(url)
                
                if result.success:
                    print(f"Extraction: ✅ {result.word_count} words")
                    
                    if result.quality_validation:
                        qv = result.quality_validation
                        print(f"Quality Score: {qv.score:.3f} ({qv.category})")
                        print(f"HTML Artifacts: {qv.html_artifacts_found}")
                        print(f"Script Contamination: {qv.script_contamination}")
                        print(f"Content Ratio: {qv.content_to_navigation_ratio:.3f}")
                        
                        if qv.issues:
                            print(f"Issues: {qv.issues[0]}")  # Show first issue
                        
                        if qv.recommendations:
                            print(f"Recommendations: {qv.recommendations[0]}")  # Show first rec
                    else:
                        print("Quality validation not available")
                else:
                    print(f"Extraction: ❌ {result.error_message}")
                    
            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("Testing AdvancedWebCrawler with Playwright and Quality Validation...")
    
    # Test 1: Basic Playwright functionality
    success = asyncio.run(test_playwright_rendering())
    
    if success:
        print("\n🎉 Basic Playwright test passed!")
        
        # Test 2: Quality validation functionality
        asyncio.run(test_quality_validation())
        
        print("\n✅ All tests completed!")
    else:
        print("\n⚠️  Basic test failed - may need Docker environment")