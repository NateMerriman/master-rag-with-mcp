#!/usr/bin/env python3
"""
Test script for developing new structure-aware regex patterns for markdown post-processing.
This script helps validate regex patterns against HTML snippets to ensure proper content preservation.
"""

import re
from typing import List, Tuple

# Test HTML snippets representing navigation elements (should be removed)
NAVIGATION_HTML_SNIPPETS = [
    # Navigation block
    '''<nav class="md-nav md-nav--primary">
  <label class="md-nav__title">
    <a href="." title="n8n Docs">n8n Docs</a>
  </label>
  <ul class="md-nav__list">
    <li class="md-nav__item"><a href="/quickstart/">Quickstart</a></li>
  </ul>
</nav>''',
    
    # Breadcrumb navigation
    '''<nav class="md-path">
  <ol class="md-path__list">
    <li class="md-path__item">
      <a href="../.." class="md-path__link">Using n8n</a>
    </li>
  </ol>
</nav>''',
    
    # Footer block
    '''<footer class="md-footer">
  <div class="md-footer-meta">
    <nav class="md-footer-nav">
      <a href="/privacy/" class="md-footer-nav__link">Privacy</a>
    </nav>
  </div>
</footer>''',
    
    # Sidebar navigation
    '''<div class="md-sidebar md-sidebar--primary">
  <div class="md-sidebar__scrollwrap">
    <nav class="md-nav">
      <a href="/integrations/" class="md-nav__link">Integrations</a>
    </nav>
  </div>
</div>''',
]

# Test HTML snippets representing content elements (should be preserved) 
CONTENT_HTML_SNIPPETS = [
    # Paragraph with inline links (glossary-style)
    '''<p>This quickstart introduces two key features: <a href="/glossary/#template-n8n">workflow templates</a> and <a href="/glossary/#expression-n8n">expressions</a>.</p>''',
    
    # List item with content link
    '''<li>Load a <a href="/glossary/#workflow-n8n">workflow</a> from the workflow templates library</li>''',
    
    # Content with integration links
    '''<p>Gets example data from the <a href="/integrations/builtin/app-nodes/n8n-nodes-base.n8ntrainingcustomerdatastore/">Customer Datastore</a> node.</p>''',
    
    # Definition-style content
    '''<dt>API Key</dt>
<dd>A <a href="/security/credentials/">credential</a> used to authenticate with external services.</dd>''',
]

def test_regex_patterns() -> None:
    """Test new regex patterns against HTML snippets."""
    
    print("=== DEVELOPING STRUCTURE-AWARE REGEX PATTERNS ===\n")
    
    # New regex patterns targeting entire HTML blocks (optimized and finalized)
    new_patterns = [
        # Remove entire nav blocks (covers most navigation)
        (r'<nav\b[^>]*>.*?</nav>', 'Navigation blocks'),
        
        # Remove entire footer blocks  
        (r'<footer\b[^>]*>.*?</footer>', 'Footer blocks'),
        
        # Remove sidebar and menu blocks
        (r'<div\b[^>]*class="[^"]*(?:sidebar|menu|navigation)[^"]*"[^>]*>.*?</div>', 'Sidebar/menu blocks'),
        
        # Remove header blocks that contain only navigation
        (r'<header\b[^>]*>.*?</header>', 'Header blocks'),
        
        # Remove breadcrumb navigation specifically (redundant but explicit)
        (r'<[^>]*class="[^"]*(?:breadcrumb|path|crumb)[^"]*"[^>]*>.*?</[^>]*>', 'Breadcrumb elements'),
        
        # Remove edit buttons and action buttons
        (r'<a\b[^>]*class="[^"]*(?:button|btn|edit|action)[^"]*"[^>]*>.*?</a>', 'Action buttons'),
        
        # Remove standalone header anchor links (#) 
        (r'<a\b[^>]*class="[^"]*headerlink[^"]*"[^>]*>\s*#\s*</a>', 'Header anchor links'),
        
        # Remove table of contents blocks
        (r'<[^>]*class="[^"]*(?:toc|table-of-contents)[^"]*"[^>]*>.*?</[^>]*>', 'Table of contents'),
    ]
    
    print("Testing patterns against NAVIGATION snippets (should match):")
    print("-" * 60)
    
    for i, snippet in enumerate(NAVIGATION_HTML_SNIPPETS, 1):
        print(f"\nNavigation snippet {i}:")
        print(snippet[:100] + "..." if len(snippet) > 100 else snippet)
        
        matches_found = 0
        for pattern, description in new_patterns:
            if re.search(pattern, snippet, re.DOTALL | re.IGNORECASE):
                print(f"  ✓ MATCHED by: {description}")
                matches_found += 1
        
        if matches_found == 0:
            print(f"  ❌ NO MATCHES FOUND - this navigation element would NOT be removed!")
    
    print("\n" + "=" * 60)
    print("Testing patterns against CONTENT snippets (should NOT match):")
    print("-" * 60)
    
    for i, snippet in enumerate(CONTENT_HTML_SNIPPETS, 1):
        print(f"\nContent snippet {i}:")
        print(snippet)
        
        matches_found = 0
        for pattern, description in new_patterns:
            if re.search(pattern, snippet, re.DOTALL | re.IGNORECASE):
                print(f"  ❌ INCORRECTLY MATCHED by: {description}")
                matches_found += 1
        
        if matches_found == 0:
            print(f"  ✓ CORRECTLY PRESERVED - no unwanted pattern matches")


def get_final_regex_patterns() -> List[str]:
    """Return the final validated regex patterns for use in _post_process_markdown."""
    return [
        # Remove entire nav blocks (covers most navigation)
        r'<nav\b[^>]*>.*?</nav>',
        
        # Remove entire footer blocks  
        r'<footer\b[^>]*>.*?</footer>',
        
        # Remove sidebar and menu blocks
        r'<div\b[^>]*class="[^"]*(?:sidebar|menu|navigation)[^"]*"[^>]*>.*?</div>',
        
        # Remove header blocks that contain only navigation
        r'<header\b[^>]*>.*?</header>',
        
        # Remove breadcrumb navigation specifically
        r'<[^>]*class="[^"]*(?:breadcrumb|path|crumb)[^"]*"[^>]*>.*?</[^>]*>',
        
        # Remove edit buttons and action buttons
        r'<a\b[^>]*class="[^"]*(?:button|btn|edit|action)[^"]*"[^>]*>.*?</a>',
        
        # Remove standalone header anchor links (#) 
        r'<a\b[^>]*class="[^"]*headerlink[^"]*"[^>]*>\s*#\s*</a>',
        
        # Remove table of contents blocks
        r'<[^>]*class="[^"]*(?:toc|table-of-contents)[^"]*"[^>]*>.*?</[^>]*>',
    ]


def apply_patterns_to_markdown(markdown: str) -> str:
    """Apply the final regex patterns to markdown content."""
    patterns = get_final_regex_patterns()
    
    cleaned_markdown = markdown
    for pattern in patterns:
        cleaned_markdown = re.sub(pattern, '', cleaned_markdown, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any double newlines created by removing blocks
    cleaned_markdown = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_markdown)
    
    return cleaned_markdown.strip()


if __name__ == "__main__":
    test_regex_patterns()
    
    print("\n" + "=" * 60)
    print("FINAL REGEX PATTERNS FOR IMPLEMENTATION:")
    print("-" * 60)
    patterns = get_final_regex_patterns()
    for i, pattern in enumerate(patterns, 1):
        print(f"{i}. {pattern}")
    
    print(f"\nTotal patterns: {len(patterns)}")
    print("These patterns target HTML blocks and preserve inline content links.")