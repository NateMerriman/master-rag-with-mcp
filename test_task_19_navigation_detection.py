#!/usr/bin/env python3
"""
Test Suite for Task 19: Enhanced Navigation Detection and Quality Validation

This test validates the critical fixes to navigation detection patterns in the
content quality analyzer, ensuring that navigation content like bulleted
integration lists receive low quality scores instead of perfect 1.000 scores.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from content_quality import ContentQualityAnalyzer, calculate_content_quality


def test_n8n_integration_list_detection():
    """Test that n8n-style integration lists are properly detected as navigation."""
    
    # Problematic content from n8n.io that was getting perfect 1.000 scores
    n8n_integration_content = """
# Credentials

This section contains information about various credentials and integrations:

* [S3 credentials](https://docs.n8n.io/integrations/builtin/credentials/s3/)
* [Salesforce credentials](https://docs.n8n.io/integrations/builtin/credentials/salesforce/)
* [Salesmate credentials](https://docs.n8n.io/integrations/builtin/credentials/salesmate/)
* [SearXNG credentials](https://docs.n8n.io/integrations/builtin/credentials/searxng/)
* [SeaTable credentials](https://docs.n8n.io/integrations/builtin/credentials/seatable/)
* [SecurityScorecard credentials](https://docs.n8n.io/integrations/builtin/credentials/securityscorecard/)
* [Segment credentials](https://docs.n8n.io/integrations/builtin/credentials/segment/)
* [Sekoia credentials](https://docs.n8n.io/integrations/builtin/credentials/sekoia/)
* [Send Email](https://docs.n8n.io/integrations/builtin/credentials/sendemail/)
* [SendGrid credentials](https://docs.n8n.io/integrations/builtin/credentials/sendgrid/)
* [Sendy credentials](https://docs.n8n.io/integrations/builtin/credentials/sendy/)
* [Sentry.io credentials](https://docs.n8n.io/integrations/builtin/credentials/sentryio/)
* [Serp credentials](https://docs.n8n.io/integrations/builtin/credentials/serp/)
* [ServiceNow credentials](https://docs.n8n.io/integrations/builtin/credentials/servicenow/)
* [Shopify credentials](https://docs.n8n.io/integrations/builtin/credentials/shopify/)
* [Slack credentials](https://docs.n8n.io/integrations/builtin/credentials/slack/)
* [Snowflake credentials](https://docs.n8n.io/integrations/builtin/credentials/snowflake/)
* [Spotify credentials](https://docs.n8n.io/integrations/builtin/credentials/spotify/)
* [SSH credentials](https://docs.n8n.io/integrations/builtin/credentials/ssh/)
* [Stripe credentials](https://docs.n8n.io/integrations/builtin/credentials/stripe/)
"""
    
    print("🧪 Testing Enhanced Navigation Detection (Task 19)")
    print("=" * 60)
    
    # Analyze the problematic content
    metrics = calculate_content_quality(n8n_integration_content)
    
    print(f"Content Quality Analysis Results:")
    print(f"  Overall Quality Score: {metrics.overall_quality_score:.3f}")
    print(f"  Quality Category: {metrics.quality_category}")
    print(f"  Content-to-Navigation Ratio: {metrics.content_to_navigation_ratio:.3f}")
    print(f"  Link Density: {metrics.link_density:.3f}")
    print(f"  Navigation Elements Detected: {metrics.navigation_element_count}")
    print(f"  Word Count: {metrics.word_count}")
    print(f"  Unique Links: {metrics.unique_link_count}")
    
    # CRITICAL VALIDATION: This content should now receive a LOW quality score
    assert metrics.overall_quality_score < 0.3, f"FAILED: Integration list still receiving high score: {metrics.overall_quality_score:.3f}"
    assert metrics.quality_category in ["poor", "fair"], f"FAILED: Integration list categorized as {metrics.quality_category}"
    assert metrics.link_density > 0.15, f"FAILED: Link density not detected: {metrics.link_density:.3f}"
    assert metrics.navigation_element_count > 10, f"FAILED: Navigation elements not detected: {metrics.navigation_element_count}"
    
    print("\n✅ SUCCESS: Integration list correctly identified as low-quality navigation content!")
    print(f"   Quality Score: {metrics.overall_quality_score:.3f} (target: <0.3)")
    print(f"   Category: {metrics.quality_category} (target: poor/fair)")
    
    return metrics


def test_legitimate_content_still_scores_high():
    """Test that legitimate documentation content still receives high scores."""
    
    legitimate_content = """
# Getting Started with n8n

## Introduction

n8n is a powerful workflow automation tool that allows you to connect different services and APIs. 
This guide will walk you through the basic concepts and help you create your first workflow.

## Core Concepts

### Workflows
A workflow in n8n is a series of connected nodes that process data. Each workflow starts with a 
trigger node and can include multiple processing nodes to transform, filter, or route data.

### Nodes
Nodes are the building blocks of workflows. There are several types:
- **Trigger nodes**: Start the workflow when certain conditions are met
- **Regular nodes**: Process, transform, or send data  
- **Core nodes**: Provide essential functionality like conditional logic

### Connections
Nodes are connected with lines that define the flow of data. Data flows from left to right,
and you can create complex branching logic by connecting multiple nodes.

## Creating Your First Workflow

1. **Add a trigger node**: Start by adding a trigger like "Manual Trigger" or "Webhook"
2. **Add processing nodes**: Connect nodes that will process your data
3. **Configure each node**: Set up the specific parameters for each node
4. **Test your workflow**: Use the test functionality to verify everything works
5. **Activate your workflow**: Enable it to run automatically

## Best Practices

- Keep workflows simple and focused on specific tasks
- Use descriptive names for your workflows and nodes
- Add notes to complex workflows to help with maintenance
- Test thoroughly before activating production workflows

This documentation provides a solid foundation for understanding n8n's core functionality.
"""
    
    print("\n🧪 Testing Legitimate Content Quality")
    print("=" * 60)
    
    metrics = calculate_content_quality(legitimate_content)
    
    print(f"Legitimate Content Analysis Results:")
    print(f"  Overall Quality Score: {metrics.overall_quality_score:.3f}")
    print(f"  Quality Category: {metrics.quality_category}")
    print(f"  Content-to-Navigation Ratio: {metrics.content_to_navigation_ratio:.3f}")
    print(f"  Link Density: {metrics.link_density:.3f}")
    print(f"  Navigation Elements: {metrics.navigation_element_count}")
    print(f"  Word Count: {metrics.word_count}")
    
    # Legitimate content should still score well
    assert metrics.overall_quality_score > 0.7, f"FAILED: Legitimate content scored too low: {metrics.overall_quality_score:.3f}"
    assert metrics.quality_category in ["excellent", "good"], f"FAILED: Legitimate content categorized as {metrics.quality_category}"
    assert metrics.link_density < 0.1, f"FAILED: Legitimate content has high link density: {metrics.link_density:.3f}"
    
    print(f"\n✅ SUCCESS: Legitimate content correctly scored high!")
    print(f"   Quality Score: {metrics.overall_quality_score:.3f} (target: >0.7)")
    print(f"   Category: {metrics.quality_category} (target: excellent/good)")
    
    return metrics


def test_mixed_content_detection():
    """Test content that mixes legitimate text with some navigation elements."""
    
    mixed_content = """
# API Authentication Guide

n8n provides several methods for API authentication. This guide covers the most common approaches.

## Basic Authentication

Basic authentication uses a username and password combination. To set this up:

1. Go to your credentials page
2. Create a new Basic Auth credential  
3. Enter your username and password
4. Save the credential

## OAuth 2.0

OAuth 2.0 is more secure for production use. The setup process varies by service.

### Common OAuth Services

* [Google OAuth](https://docs.n8n.io/integrations/builtin/credentials/google/)
* [GitHub OAuth](https://docs.n8n.io/integrations/builtin/credentials/github/) 
* [Microsoft OAuth](https://docs.n8n.io/integrations/builtin/credentials/microsoft/)

## API Keys

Many services use API keys for authentication. Store these securely in n8n credentials.

The authentication method you choose depends on your security requirements and the APIs you're integrating with.
"""
    
    print("\n🧪 Testing Mixed Content Detection")
    print("=" * 60)
    
    metrics = calculate_content_quality(mixed_content)
    
    print(f"Mixed Content Analysis Results:")
    print(f"  Overall Quality Score: {metrics.overall_quality_score:.3f}")
    print(f"  Quality Category: {metrics.quality_category}")
    print(f"  Content-to-Navigation Ratio: {metrics.content_to_navigation_ratio:.3f}")
    print(f"  Link Density: {metrics.link_density:.3f}")
    print(f"  Navigation Elements: {metrics.navigation_element_count}")
    print(f"  Word Count: {metrics.word_count}")
    
    # Mixed content should receive moderate to good score (it has substantial legitimate content)
    assert 0.4 < metrics.overall_quality_score < 1.0, f"Mixed content score out of expected range: {metrics.overall_quality_score:.3f}"
    
    print(f"\n✅ SUCCESS: Mixed content appropriately scored!")
    print(f"   Quality Score: {metrics.overall_quality_score:.3f} (target: 0.4-1.0)")
    print(f"   Category: {metrics.quality_category}")
    
    return metrics


def run_comprehensive_test():
    """Run all navigation detection tests."""
    
    print("🚀 TASK 19 COMPREHENSIVE NAVIGATION DETECTION TEST")
    print("=" * 70)
    print("Testing enhanced navigation detection and non-linear quality scoring")
    print("to ensure navigation content receives appropriate low quality scores.\n")
    
    try:
        # Test 1: Navigation content should score low
        nav_metrics = test_n8n_integration_list_detection()
        
        # Test 2: Legitimate content should score high  
        content_metrics = test_legitimate_content_still_scores_high()
        
        # Test 3: Mixed content should score moderately
        mixed_metrics = test_mixed_content_detection()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("TASK 19 VALIDATION SUMMARY:")
        print(f"  Navigation Content Score: {nav_metrics.overall_quality_score:.3f} (✅ Low)")
        print(f"  Legitimate Content Score: {content_metrics.overall_quality_score:.3f} (✅ High)")
        print(f"  Mixed Content Score: {mixed_metrics.overall_quality_score:.3f} (✅ Moderate)")
        print("\nThe enhanced navigation detection system is working correctly!")
        print("Navigation content like n8n.io integration lists now receive")
        print("appropriately low quality scores instead of perfect 1.000 scores.")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\nThe navigation detection system still has issues.")
        print("Check the implementation in content_quality.py")
        return False
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)