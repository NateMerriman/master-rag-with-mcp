#!/usr/bin/env python3
"""
Test script for the Content Quality Validation Suite.

This script tests the quality validation functionality with various
markdown samples to verify the validation logic works correctly.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from crawler_quality_validation import (
    ContentQualityValidator, 
    validate_crawler_output,
    create_quality_report,
    GoldenSetManager
)


def test_quality_validation():
    """Test quality validation with various markdown samples."""
    
    validator = ContentQualityValidator()
    
    # Test cases with different quality levels
    test_cases = [
        {
            "name": "Excellent Quality",
            "markdown": """# Complete Guide to API Documentation

## Introduction

This guide provides comprehensive information about our REST API endpoints.

### Authentication

All API requests require authentication using Bearer tokens:

```bash
curl -H "Authorization: Bearer TOKEN" https://api.example.com/v1/users
```

### Rate Limits

API requests are limited to 1000 requests per hour per user.

## Available Endpoints

### Users

- `GET /v1/users` - List all users
- `POST /v1/users` - Create a new user

For more information, see the [official documentation](https://docs.example.com).
""",
            "expected_category": "excellent"
        },
        
        {
            "name": "Good Quality with Minor Issues", 
            "markdown": """# User Management

## Overview
This section covers user management features.

Navigation: Home | Users | Settings | Help

### Creating Users
To create a new user, follow these steps:

1. Navigate to the Users page
2. Click "Add User"
3. Fill in the required fields

### Deleting Users
<div class="warning">Be careful when deleting users</div>

Contact support for more information.
""",
            "expected_category": "good"
        },
        
        {
            "name": "Poor Quality with HTML Artifacts",
            "markdown": """<nav class="sidebar">
<ul>
<li><a href="/home">Home</a></li>
<li><a href="/users">Users</a></li>
</ul>
</nav>

<script>
function toggleMenu() {
    document.getElementById('menu').style.display = 'block';
}
</script>

# Users

<footer>&copy; 2024 Company</footer>

Some content here.

<div onclick="alert('clicked')">Click me</div>
""",
            "expected_category": "poor"
        },
        
        {
            "name": "Empty Content",
            "markdown": "",
            "expected_category": "poor"
        },
        
        {
            "name": "Navigation Heavy",
            "markdown": """Menu Navigation Sidebar Toggle Home Back Next Previous
Table of Contents Skip to content Breadcrumb Menu

# Page Title

Navigation Menu Sidebar Toggle Home Users Settings Help
Some actual content here but very little.
Menu Navigation Home Back Forward Toggle Sidebar
""",
            "expected_category": "fair"
        }
    ]
    
    print("🧪 Testing Content Quality Validation Suite")
    print("=" * 60)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 40)
        
        # Run validation
        result = validator.validate_content(
            test_case['markdown'], 
            url=f"https://test{i}.example.com"
        )
        
        results.append(result)
        
        # Display results
        print(f"Score: {result.score:.3f}")
        print(f"Category: {result.category}")
        print(f"Passed: {result.passed}")
        print(f"Word Count: {result.word_count}")
        print(f"HTML Artifacts: {result.html_artifacts_found}")
        print(f"Script Contamination: {result.script_contamination}")
        print(f"Content Ratio: {result.content_to_navigation_ratio:.3f}")
        print(f"Link Density: {result.link_density:.3f}")
        
        if result.issues:
            print(f"Issues: {', '.join(result.issues)}")
        
        if result.warnings:
            print(f"Warnings: {', '.join(result.warnings)}")
        
        # Check if category matches expectation
        expected = test_case['expected_category']
        if result.category == expected:
            print(f"✅ Category matches expectation: {expected}")
        else:
            print(f"⚠️  Category mismatch: got {result.category}, expected {expected}")
    
    # Generate quality report
    print("\n" + "=" * 60)
    print("📊 QUALITY VALIDATION REPORT")
    print("=" * 60)
    
    report = create_quality_report(results)
    print(report)
    
    # Test golden set functionality
    print("\n" + "=" * 60)
    print("🏆 GOLDEN SET TESTING")
    print("=" * 60)
    
    golden_manager = GoldenSetManager(Path("test_golden_set"))
    
    # Add the excellent quality example to golden set
    excellent_markdown = test_cases[0]['markdown']
    success = golden_manager.create_golden_example(
        url="https://example.com/api-docs",
        markdown=excellent_markdown,
        description="Example of well-structured API documentation"
    )
    
    if success:
        print("✅ Successfully added example to golden set")
        
        # Test comparison
        similarities = golden_manager.compare_against_golden_set(test_cases[1]['markdown'])
        print(f"Similarity scores: {similarities}")
    else:
        print("❌ Failed to add example to golden set")
    
    print("\n🎉 Quality validation testing complete!")
    
    return results


if __name__ == "__main__":
    test_quality_validation()