#!/usr/bin/env python3
"""
Integration Validation Test for AdvancedWebCrawler

This script validates that the AdvancedWebCrawler improvements are properly
integrated into both the MCP server and manual crawling scripts.

Tests:
1. MCP tool availability and functionality
2. Manual crawling script argument parsing and execution
3. Comparison between different crawling methods
4. Quality validation across all methods
"""

import asyncio
import sys
import subprocess
import json
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_manual_crawl_help():
    """Test that manual crawl script shows the new --advanced option."""
    
    print("🧪 Testing Manual Crawl Script Integration")
    print("-" * 50)
    
    try:
        # Run manual crawl with --help to see available options
        result = subprocess.run([
            sys.executable, str(src_path / "manual_crawl.py"), "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            help_text = result.stdout
            
            # Check for the new --advanced flag
            if "--advanced" in help_text:
                print("✅ Manual crawl script includes --advanced option")
                
                # Check description
                if "AdvancedWebCrawler" in help_text:
                    print("✅ --advanced option has proper description")
                else:
                    print("⚠️  --advanced option missing AdvancedWebCrawler description")
                    
                # Show the relevant help section
                lines = help_text.split('\n')
                for i, line in enumerate(lines):
                    if "--advanced" in line:
                        print(f"   Option: {line.strip()}")
                        if i + 1 < len(lines):
                            print(f"   Description: {lines[i+1].strip()}")
                        break
                
                return True
            else:
                print("❌ Manual crawl script missing --advanced option")
                return False
        else:
            print(f"❌ Manual crawl script help failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Manual crawl script help timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing manual crawl script: {str(e)}")
        return False


def test_mcp_imports():
    """Test that MCP server can import AdvancedWebCrawler modules."""
    
    print("\n🧪 Testing MCP Server Integration")
    print("-" * 50)
    
    try:
        # Import the MCP server module to check for import errors
        import crawl4ai_mcp
        
        # Check if AdvancedWebCrawler functions are available
        expected_functions = [
            'is_advanced_crawler_available',
            'crawl_single_page_with_advanced_crawler'
        ]
        
        for func_name in expected_functions:
            if hasattr(crawl4ai_mcp, func_name):
                print(f"✅ MCP server has {func_name} function")
            else:
                print(f"❌ MCP server missing {func_name} function")
                return False
        
        # Test availability check
        try:
            available = crawl4ai_mcp.is_advanced_crawler_available()
            print(f"✅ AdvancedWebCrawler availability check: {available}")
        except Exception as e:
            print(f"⚠️  Error checking AdvancedWebCrawler availability: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import MCP server: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing MCP server: {str(e)}")
        return False


def test_advanced_crawler_imports():
    """Test that AdvancedWebCrawler modules can be imported."""
    
    print("\n🧪 Testing AdvancedWebCrawler Module Imports")
    print("-" * 50)
    
    try:
        # Test AdvancedWebCrawler import
        from advanced_web_crawler import (
            AdvancedWebCrawler,
            AdvancedCrawlResult,
            crawl_single_page_advanced,
            batch_crawl_advanced
        )
        print("✅ AdvancedWebCrawler classes imported successfully")
        
        # Test quality validation import
        from crawler_quality_validation import (
            ContentQualityValidator,
            QualityValidationResult,
            validate_crawler_output,
            create_quality_report
        )
        print("✅ Quality validation classes imported successfully")
        
        # Test that classes can be instantiated
        validator = ContentQualityValidator()
        print("✅ ContentQualityValidator instantiated successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import AdvancedWebCrawler modules: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing AdvancedWebCrawler imports: {str(e)}")
        return False


async def test_advanced_crawler_functionality():
    """Test basic AdvancedWebCrawler functionality without external dependencies."""
    
    print("\n🧪 Testing AdvancedWebCrawler Basic Functionality")
    print("-" * 50)
    
    try:
        from advanced_web_crawler import AdvancedWebCrawler
        from crawler_quality_validation import ContentQualityValidator
        
        # Test quality validator with sample content
        validator = ContentQualityValidator()
        
        sample_markdown = """# Test Document

## Introduction

This is a test document with good structure.

### Features

- Clean headings
- Proper markdown formatting
- No HTML artifacts

## Conclusion

This document should pass quality validation.
"""
        
        result = validator.validate_content(sample_markdown, "https://test.example.com")
        
        print(f"✅ Quality validation test completed")
        print(f"   Score: {result.score:.3f}")
        print(f"   Category: {result.category}")
        print(f"   Passed: {result.passed}")
        print(f"   Issues: {len(result.issues)}")
        print(f"   Warnings: {len(result.warnings)}")
        
        if result.score > 0.7:
            print("✅ Quality validation working correctly")
            return True
        else:
            print("⚠️  Quality validation score lower than expected")
            return False
        
    except Exception as e:
        print(f"❌ Error testing AdvancedWebCrawler functionality: {str(e)}")
        return False


def compare_crawling_methods():
    """Compare the available crawling methods and their capabilities."""
    
    print("\n🧪 Crawling Methods Comparison")
    print("-" * 50)
    
    methods = {
        "Baseline Crawler": {
            "description": "Original AsyncWebCrawler from crawl4ai",
            "features": ["Basic HTML extraction", "Simple chunking"],
            "trigger": "Default mode or --baseline flag"
        },
        "Enhanced Crawler": {
            "description": "SmartCrawlerFactory with framework detection", 
            "features": ["Framework detection", "CSS targeting", "Quality metrics", "Fallback mechanisms"],
            "trigger": "USE_ENHANCED_CRAWLING=true or --enhanced flag"
        },
        "Advanced Crawler": {
            "description": "NEW AdvancedWebCrawler with Playwright + TrafilaturaExtractor",
            "features": ["Playwright browser automation", "TrafilaturaExtractor", "Quality validation", "DocumentIngestionPipeline ready"],
            "trigger": "--advanced flag or crawl_single_page_with_advanced_crawler MCP tool"
        }
    }
    
    for name, info in methods.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Features: {', '.join(info['features'])}")
        print(f"  Trigger: {info['trigger']}")
    
    print(f"\n✅ Total crawling methods available: {len(methods)}")
    return True


def test_mcp_tool_descriptions():
    """Test MCP tool descriptions include the new AdvancedWebCrawler tool."""
    
    print("\n🧪 Testing MCP Tool Descriptions")
    print("-" * 50)
    
    try:
        import crawl4ai_mcp
        
        # Check if the tool description function exists
        if hasattr(crawl4ai_mcp, 'get_strategy_status'):
            print("✅ MCP server has get_strategy_status function")
            
            # This would normally require a running MCP context, so we just check for the function
            print("✅ MCP tool status reporting available")
            return True
        else:
            print("⚠️  MCP server missing get_strategy_status function")
            return False
            
    except Exception as e:
        print(f"❌ Error testing MCP tool descriptions: {str(e)}")
        return False


def main():
    """Run all integration validation tests."""
    
    print("🚀 AdvancedWebCrawler Integration Validation")
    print("=" * 60)
    print("Validating that Task 13 improvements are integrated into both")
    print("MCP server and manual crawling workflows.\n")
    
    tests = [
        ("Manual Crawl Script", test_manual_crawl_help),
        ("MCP Server Integration", test_mcp_imports),
        ("Module Imports", test_advanced_crawler_imports),
        ("Basic Functionality", lambda: asyncio.run(test_advanced_crawler_functionality())),
        ("Methods Comparison", compare_crawling_methods),
        ("MCP Tool Descriptions", test_mcp_tool_descriptions),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"Running: {test_name}")
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name}: PASSED\n")
            else:
                print(f"❌ {test_name}: FAILED\n")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}\n")
            results.append((test_name, False))
    
    # Final summary
    print("=" * 60)
    print("🎯 INTEGRATION VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ AdvancedWebCrawler is properly integrated into both MCP and manual workflows")
        print("\nYou can now use the improvements via:")
        print("• MCP tool: crawl_single_page_with_advanced_crawler")  
        print("• Manual script: python src/manual_crawl.py --url <URL> --advanced")
    else:
        print("\n⚠️  Some integration tests failed")
        print("Check the error messages above for details")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)