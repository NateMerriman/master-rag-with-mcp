# Site Structure Analysis

## n8n Documentation Site Analysis

### DOM Structure Findings:
- **Main Content Container**: `<main class="md-main">`
- **Article Content**: `<article class="md-content__inner md-typeset">`
- **Content Area**: `.md-content`
- **Sidebar Navigation**: `.md-sidebar.md-sidebar--primary`
- **Navigation Count**: 117 nav elements (massive navigation structure!)

### Key Issues Identified:
1. **Excessive Navigation**: 117 navigation elements on a single page
2. **Material Design Framework**: Uses Material Design (md-*) classes
3. **Complex Sidebar**: Primary sidebar with extensive nested navigation

### Optimal Crawl4AI Configuration for n8n:
```python
config = CrawlerRunConfig(
    # Target the main content areas
    target_elements=["main.md-main", "article.md-content__inner", ".md-content"],
    
    # Exclude navigation elements
    excluded_tags=["nav", "header", "footer", "aside"],
    
    # Filter out short navigation text
    word_count_threshold=15,
    
    # Remove external noise
    exclude_external_links=True,
    exclude_social_media_links=True
)
```

### CSS Selectors for n8n Docs:
- **Target**: `main.md-main, article.md-content__inner, .md-content`
- **Exclude**: `.md-sidebar, nav, header, footer`


## VirusTotal Documentation Site Analysis

### DOM Structure Findings:
- **Main Content Container**: `<main class="rm-Guides">`
- **Article Content**: 1 article element
- **Content Area**: `.Sidebar1t2G1ZJq-vU1.rm-Sidebar.hub-sidebar-content`
- **Table of Contents**: `.content-toc.grid-25`
- **Navigation Count**: 4 nav elements (much cleaner than n8n)
- **Sidebar**: Complex class structure with theme and layout classes

### Key Issues Identified:
1. **Complex CSS Classes**: Uses generated/hashed class names
2. **ReadMe.io Framework**: Appears to use ReadMe.io documentation platform
3. **Sidebar Content**: Extensive sidebar navigation with nested categories
4. **Table of Contents**: Separate TOC element that could be filtered

### Optimal Crawl4AI Configuration for VirusTotal:
```python
config = CrawlerRunConfig(
    # Target the main content areas
    target_elements=["main.rm-Guides", "article", ".rm-Content"],
    
    # Exclude navigation elements
    excluded_tags=["nav", "header", "footer", "aside"],
    
    # Filter out short navigation text and TOC
    word_count_threshold=20,
    
    # Remove external noise
    exclude_external_links=True,
    exclude_social_media_links=True
)
```

### CSS Selectors for VirusTotal Docs:
- **Target**: `main.rm-Guides, article, .rm-Content`
- **Exclude**: `.rm-Sidebar, .content-toc, nav, header, footer`

## Comparison: n8n vs VirusTotal

| Aspect | n8n Docs | VirusTotal Docs |
|--------|----------|-----------------|
| Framework | Material Design | ReadMe.io |
| Navigation Elements | 117 nav elements | 4 nav elements |
| Main Content | `main.md-main` | `main.rm-Guides` |
| Sidebar | `.md-sidebar` | `.rm-Sidebar` |
| Complexity | Very High | Moderate |
| CSS Classes | Semantic (md-*) | Generated/Hashed |

