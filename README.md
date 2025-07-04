<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Supabase](https://supabase.com/) for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities.

With this MCP server, you can <b>scrape anything</b> and then <b>use that knowledge anywhere</b> for RAG.

The primary goal is to bring this MCP server into [Archon](https://github.com/coleam00/Archon) as I evolve it to be more of a knowledge engine for AI coding assistants to build AI agents. This first version of the Crawl4AI/RAG MCP server will be improved upon greatly soon, especially making it more configurable so you can use different embedding models and run everything locally with Ollama.

## Overview

This MCP server provides tools that enable AI agents to crawl websites, store content in a vector database (Supabase), and perform RAG over the crawled content. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/) I provided on my channel previously.

## Vision

The Crawl4AI RAG MCP server is just the beginning. Here's where we're headed:

1. **Integration with Archon**: Building this system directly into [Archon](https://github.com/coleam00/Archon) to create a comprehensive knowledge engine for AI coding assistants to build better AI agents.

2. **Multiple Embedding Models**: Expanding beyond OpenAI to support a variety of embedding models, including the ability to run everything locally with Ollama for complete control and privacy.

3. **Advanced RAG Strategies**: Implementing sophisticated retrieval techniques like contextual retrieval, late chunking, and others to move beyond basic "naive lookups" and significantly enhance the power and precision of the RAG system, especially as it integrates with Archon.

4. **Enhanced Chunking Strategy**: Implementing a Context 7-inspired chunking approach that focuses on examples and creates distinct, semantically meaningful sections for each chunk, improving retrieval precision.

5. **Performance Optimization**: Increasing crawling and indexing speed to make it more realistic to "quickly" index new documentation to then leverage it within the same prompt in an AI coding assistant.

## Features

### Core Capabilities
- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Content Chunking**: Intelligently splits content by headers and size for better processing
- **Hybrid Search**: Combines semantic vector search with full-text search using Reciprocal Rank Fusion (RRF)
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process

### Advanced RAG Strategies (Optional)
- **Cross-Encoder Reranking**: Improves search result quality by reordering results based on query-document relevance
- **Contextual Embeddings**: Enhanced semantic understanding through document-level context generation
- **Agentic RAG**: Specialized code extraction and search capabilities for AI coding assistants
- **Performance Monitoring**: Real-time performance tracking and regression testing framework

## Tools

The server provides a dynamic set of tools that adapt based on your enabled RAG strategies:

### Core Tools (Always Available)
1. **`crawl_single_page`**: Quickly crawl a single web page and store its content in the vector database
2. **`smart_crawl_url`**: Intelligently crawl a full website based on the type of URL provided (sitemap, llms-full.txt, or a regular webpage that needs to be crawled recursively)
3. **`get_available_sources`**: Get a list of all available sources (domains) in the database
4. **`perform_rag_query`**: Search for relevant content using semantic search with optional source filtering

### Strategy-Specific Tools (Conditional)
- **`perform_rag_query_with_reranking`**: Enhanced search with cross-encoder reranking (requires `USE_RERANKING=true`)
- **`search_code_examples`**: Specialized code search with language and complexity filtering (requires `USE_AGENTIC_RAG=true`)
- **`perform_contextual_rag_query`**: Context-enhanced search queries (requires `USE_CONTEXTUAL_EMBEDDINGS=true`)

### Enhanced Crawling Tools (Conditional)
- **`crawl_single_page_enhanced`**: Enhanced single page crawling with framework detection and quality validation (requires `USE_ENHANCED_CRAWLING=true`)
- **`smart_crawl_url_enhanced`**: Enhanced batch crawling with quality metrics and framework optimization (requires `USE_ENHANCED_CRAWLING=true`)
- **`analyze_site_framework`**: Diagnostic tool for framework detection and optimal extraction configuration (requires `USE_ENHANCED_CRAWLING=true`)

> **Note**: Strategy-specific tools automatically appear when their corresponding strategies are enabled via environment variables. This ensures optimal resource usage and a clean tool interface.

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
- [Supabase](https://supabase.com/) (database for RAG)
- [OpenAI API key](https://platform.openai.com/api-keys) (for generating embeddings)

## Installation

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Build the Docker image:
   ```bash
   docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
   ```

3. Create a `.env` file based on the configuration section below

### Using uv directly (no Docker)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   .venv\Scripts\activate
   # on Mac/Linux: source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -e .
   crawl4ai-setup
   ```

5. Create a `.env` file based on the configuration section below

## Database Setup

Before running the server, you need to set up the database with the pgvector extension and all required tables:

1. Go to the SQL Editor in your Supabase dashboard (create a new project first if necessary)

2. Create a new query and paste the contents of `crawled_pages.sql`

3. Run the query to create the core tables and functions

4. **For advanced features**: If you plan to use agentic RAG code extraction, also run the migration scripts in `src/database/migrations/` to set up the enhanced schema. Run scripts in numerical order for proper dependencies:
   - `001_create_sources_table.sql` - Creates centralized source management.
   - `002_create_code_examples_table.sql` - Creates the specialized table for storing code examples with the latest hybrid-search schema.
   - `003_populate_source_ids.sql` - A utility script to backfill `source_id` in the `crawled_pages` table if you have existing data from before the multi-source system was added.

> **Note**: The core `crawled_pages` table is sufficient for basic functionality. Advanced tables are only needed when using specific RAG strategies.

## Configuration

Create a `.env` file in the project root with the following variables:

### Basic Configuration

```bash
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

### Advanced RAG Strategy Configuration

All advanced strategies are **disabled by default** for backward compatibility. Enable only the strategies you need:

```bash
# RAG Strategy Toggles (all default to false)
USE_CONTEXTUAL_EMBEDDINGS=false     # Enhanced semantic understanding
USE_RERANKING=false                 # Cross-encoder result reranking
USE_AGENTIC_RAG=false              # Specialized code search capabilities
USE_ENHANCED_CRAWLING=false        # Enhanced documentation site crawling
USE_HYBRID_SEARCH_ENHANCED=false   # Advanced hybrid search algorithms

# Model Configuration for Advanced Strategies
CONTEXTUAL_MODEL=gpt-3.5-turbo                           # For contextual embeddings
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2    # For cross-encoder reranking

# Legacy Model Choice (existing feature)
MODEL_CHOICE=gpt-4o-mini           # Optional: enables basic contextual embeddings
```

### Configuration Examples

#### Basic Setup (Default)
```bash
# Only core hybrid search functionality
OPENAI_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_SERVICE_KEY=your_key
```

#### Enhanced Search Quality
```bash
# Core + reranking for better result ordering
OPENAI_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_SERVICE_KEY=your_key
USE_RERANKING=true
```

#### Documentation Site Optimization Setup
```bash
# Core + enhanced crawling for documentation sites
OPENAI_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_SERVICE_KEY=your_key
USE_ENHANCED_CRAWLING=true
USE_RERANKING=true
```

#### AI Coding Assistant Setup
```bash
# Core + contextual embeddings + code search + reranking + enhanced crawling
OPENAI_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_SERVICE_KEY=your_key
USE_CONTEXTUAL_EMBEDDINGS=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_ENHANCED_CRAWLING=true
CONTEXTUAL_MODEL=gpt-4o-mini
```

#### Maximum Performance Setup
```bash
# All strategies enabled for best search quality
OPENAI_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_SERVICE_KEY=your_key
USE_CONTEXTUAL_EMBEDDINGS=true
USE_RERANKING=true
USE_AGENTIC_RAG=true
USE_ENHANCED_CRAWLING=true
USE_HYBRID_SEARCH_ENHANCED=true
CONTEXTUAL_MODEL=gpt-4o-mini
```

## RAG Strategy Guide

### Core Strategy: Hybrid Search
**Always enabled** - This is the foundation of the system.

**What it does**: Combines semantic vector search with PostgreSQL full-text search using Reciprocal Rank Fusion (RRF) to merge results.

**Benefits**:
- Best of both worlds: semantic understanding + keyword precision
- Handles both conceptual queries and specific term searches
- Production-proven with consistent performance

**Performance**: ~790ms average response time (baseline)

### Cross-Encoder Reranking (`USE_RERANKING=true`)
**Purpose**: Improves search result quality by reordering the top results from hybrid search.

**What it does**: Uses a cross-encoder model (default: `ms-marco-MiniLM-L-6-v2`) to score query-document pairs and reorder results by relevance. Works as a post-processing step that preserves all benefits of the hybrid search while adding quality improvements.

**Integration**: 
- **Tool**: `perform_rag_query_with_reranking` becomes available when enabled
- **Pipeline**: Hybrid Search (RRF) → Cross-Encoder Reranking → Final Results
- **Preservation**: Original RRF scores and rankings are preserved in metadata
- **Fallback**: Gracefully degrades to hybrid search if reranking fails

**How the Two-Stage Pipeline Works**:

*Stage 1: Hybrid Search (Foundation)*
1. **Semantic search** retrieves content via vector embeddings (OpenAI text-embedding-3-small)
2. **Full-text search** finds keyword matches using PostgreSQL FTS with tsvector/GIN indexing  
3. **RRF (Reciprocal Rank Fusion)** intelligently combines both rankings into unified scores
4. Returns initial result set (default: 20 documents) with comprehensive scoring metadata

*Stage 2: Cross-Encoder Reranking (Enhancement)*
1. Takes the hybrid search results as input (preserving all original scores)
2. **Query-document pair scoring** using transformer-based cross-encoder model
3. **Reorders results** based on nuanced semantic relevance understanding
4. Returns top results (default: 5) with both original and reranking scores

**Key Architectural Benefits**:
- **Additive Enhancement**: Reranking builds upon hybrid search rather than replacing it
- **Best of Both Worlds**: Fast initial filtering (hybrid) + precise final ranking (cross-encoder)
- **Score Preservation**: All original RRF, semantic, and full-text scores remain available
- **Performance Optimization**: Cross-encoder only processes the most promising candidates

**Benefits**:
- **Quality**: Significantly improves result relevance for complex queries
- **Compatibility**: Preserves all hybrid search benefits (RRF, semantic + full-text)
- **Flexibility**: Works with source filtering and existing search parameters
- **Natural Language**: Particularly effective for conversational and question-based queries

**Trade-offs**:
- **Performance**: Adds ~150-500ms processing time for reranking
- **Dependencies**: Requires sentence-transformers library (~500MB download)
- **Resources**: Uses additional CPU for local cross-encoder inference
- **Memory**: Loads model into memory (~100-200MB RAM)

**Performance Monitoring**: Built-in timing metrics and overhead tracking included.

**Best for**: Users who prioritize search quality over speed, complex query scenarios, Q&A applications

### Contextual Embeddings (`USE_CONTEXTUAL_EMBEDDINGS=true`)
**Purpose**: Enhanced semantic understanding through document-level context.

**What it does**: Generates contextual summaries for content chunks to improve embedding quality.

**Benefits**:
- Better semantic retrieval for complex documents
- Improved understanding of context and relationships
- More accurate embeddings for specialized content

**Trade-offs**:
- Requires additional LLM API calls during indexing
- Increased indexing time and costs
- Higher storage requirements for context data

**Best for**: Complex documentation, technical content, research papers

### Agentic RAG (`USE_AGENTIC_RAG=true`)
**Purpose**: Specialized capabilities for AI coding assistants with automatic code extraction.

**What it does**: Automatically detects, extracts, and indexes code examples with dual embeddings (code content + natural language summaries) for enhanced code search capabilities.

**Features**:
- **Automatic Code Detection**: Supports 18+ programming languages (Python, JavaScript, SQL, Java, C++, etc.)
- **Smart Code Processing**: Language detection, complexity scoring (1-10 scale), and contextual summarization
- **Dual Embeddings**: Separate embeddings for code content and natural language descriptions
- **Hybrid Code Search**: Combines semantic search with language/complexity filtering
- **Code-to-Code Search**: Find similar code patterns and implementations

**Benefits**:
- Specialized code search and extraction with high accuracy
- Better retrieval for programming-related queries
- Supports both natural language to code and code-to-code search
- Enables powerful AI coding assistant scenarios

**Trade-offs**:
- Additional database tables and processing complexity
- Code extraction overhead during crawling (~10-20% additional time)
- Most beneficial only for content with code examples
- Requires source_id relationships (automatic setup)

**Best for**: AI coding assistants, technical documentation with code examples, programming tutorials

### Enhanced Hybrid Search (`USE_HYBRID_SEARCH_ENHANCED=true`)
**Purpose**: More sophisticated hybrid search algorithms.

**What it does**: Advanced RRF implementation with better weight balancing and query expansion.

**Benefits**:
- Improved search quality over basic hybrid search
- Better handling of different query types
- Optimized ranking algorithms

**Trade-offs**:
- Slightly increased query processing time
- More complex ranking logic
- May require tuning for specific content types

**Best for**: Large knowledge bases with diverse content types

### Enhanced Crawling (`USE_ENHANCED_CRAWLING=true`)
**Purpose**: Intelligent documentation site extraction with framework detection and quality validation.

**What it does**: Automatically detects documentation frameworks (Material Design, ReadMe.io, GitBook, Docusaurus, Sphinx, etc.) and applies optimized CSS selectors to extract main content while filtering out navigation noise.

**Key Features**:
- **Framework Detection**: Identifies documentation platforms using domain patterns and HTML analysis
- **Smart CSS Targeting**: Uses framework-specific selectors to target main content areas
- **Navigation Filtering**: Excludes sidebars, headers, footers, and table of contents
- **Quality Validation**: Measures content-to-navigation ratio, link density, and semantic coherence
- **Automatic Fallback**: Falls back to alternative extraction strategies if quality is poor
- **Performance Monitoring**: Tracks extraction metrics and quality improvements

**Supported Frameworks**:
- **Material Design** (n8n, MkDocs sites): Targets `main.md-main`, excludes `.md-sidebar`
- **ReadMe.io** (VirusTotal, API docs): Targets `main.rm-Guides`, excludes `.rm-Sidebar`
- **GitBook**: Targets `.gitbook-content`, excludes `.book-summary`
- **Docusaurus**: Targets `.docMainContainer`, excludes `.sidebar`
- **Sphinx** (Python docs): Targets `.document`, excludes `.sphinxsidebar`
- **VuePress**: Targets `.theme-default-content`, excludes `.sidebar`
- **Jekyll** (GitHub Pages): Targets `.post-content`, excludes `.site-nav`
- **Generic**: Fallback configuration for unknown frameworks

**Tools Available**:
- **`crawl_single_page_enhanced`**: Enhanced single page crawling with quality metrics
- **`smart_crawl_url_enhanced`**: Enhanced batch crawling with framework optimization
- **`analyze_site_framework`**: Diagnostic tool for framework detection and configuration

**Benefits**:
- **Dramatic Quality Improvement**: Reduces navigation noise from 70-80% to 20-30%
- **Better Content Ratio**: Improves content-to-navigation ratio from ~30:70 to ~80:20  
- **Optimized for Documentation**: Specifically designed for sites like n8n docs, VirusTotal API docs, GitHub Pages
- **Automatic Configuration**: No manual CSS selector configuration needed
- **Quality Assurance**: Built-in validation ensures extraction quality

**Trade-offs**:
- **Minimal Overhead**: Framework detection adds ~50ms per domain (cached)
- **Quality Analysis**: Adds ~25-50ms per page for content validation
- **Processing Time**: Maintains <2s per page extraction time requirement
- **Storage**: Enhanced metadata adds ~200-300 bytes per chunk

**Performance Impact**:
- Framework detection: ~50ms per domain (cached after first detection)
- Quality validation: ~25-50ms per page
- Total overhead: <100ms per page
- Quality improvement: 60-80% better content-to-navigation ratio

**Use Cases**:
- **Documentation Sites**: n8n, VirusTotal, Kubernetes, React, Vue.js docs
- **API Documentation**: ReadMe.io-based sites, Swagger/OpenAPI docs
- **Technical Guides**: GitHub Pages, GitBook-hosted documentation
- **Knowledge Bases**: Company wikis, internal documentation portals

**Example Quality Improvement**:
```
Before Enhanced Crawling:
- Content: 30% (actual documentation)
- Navigation: 70% (sidebars, menus, links)
- Quality Score: 0.3 (poor)

After Enhanced Crawling:
- Content: 80% (actual documentation) 
- Navigation: 20% (contextual links only)
- Quality Score: 0.8 (excellent)
```

**Configuration Options**:
```bash
USE_ENHANCED_CRAWLING=true

# Quality thresholds (optional tuning)
ENHANCED_CRAWLING_MIN_CONTENT_RATIO=0.6
ENHANCED_CRAWLING_MAX_LINK_DENSITY=0.3
ENHANCED_CRAWLING_MIN_QUALITY_SCORE=0.5

# Performance settings (optional tuning)
ENHANCED_CRAWLING_MAX_EXTRACTION_TIME=5.0
ENHANCED_CRAWLING_MAX_FALLBACK_ATTEMPTS=3
```

**Best for**: Documentation sites, API reference pages, technical guides, knowledge bases with extensive navigation

## Performance Monitoring

The system includes comprehensive performance monitoring capabilities:

### Commands
```bash
# Capture baseline metrics
python src/performance_baseline.py

# Run performance regression tests
python tests/test_performance_regression.py

# Quick performance validation
python tests/test_performance_regression.py --quick
```

### Current Baseline
- **Response Time**: 790.81ms average for hybrid search
- **Database Size**: 9,149 documents across 8 sources
- **Search Quality**: All test queries return 10 relevant results

### Performance Tuning Recommendations

#### For Speed-Optimized Setup
```bash
# Minimum latency configuration
USE_RERANKING=false              # Skip reranking overhead
USE_CONTEXTUAL_EMBEDDINGS=false  # Skip context generation
USE_AGENTIC_RAG=false           # Skip code extraction
USE_HYBRID_SEARCH_ENHANCED=false # Use basic hybrid search
```
**Expected performance**: ~790ms (baseline)

#### For Quality-Optimized Setup
```bash
# Maximum search quality configuration
USE_RERANKING=true
USE_CONTEXTUAL_EMBEDDINGS=true
USE_AGENTIC_RAG=true
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2  # Larger model
CONTEXTUAL_MODEL=gpt-4o-mini
```
**Expected performance**: ~1,200-1,500ms (+reranking overhead)

#### For Balanced Performance
```bash
# Good balance of speed and quality
USE_RERANKING=true               # Significant quality improvement
USE_CONTEXTUAL_EMBEDDINGS=false  # Skip costly context generation
USE_AGENTIC_RAG=false           # Skip unless needed for code
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  # Faster model
```
**Expected performance**: ~1,000-1,200ms

#### Strategy-Specific Tuning

**Reranking Performance**:
- `ms-marco-TinyBERT-L-2-v2`: Fastest, good quality (~100-200ms overhead)
- `ms-marco-MiniLM-L-6-v2`: Balanced, better quality (~200-300ms overhead)
- `ms-marco-MiniLM-L-12-v2`: Slowest, best quality (~400-500ms overhead)

**Contextual Embeddings Performance**:
- `gpt-3.5-turbo`: Faster, lower cost for context generation
- `gpt-4o-mini`: Better context quality, moderate speed
- `gpt-4`: Best quality, highest cost and latency

**Database Optimization**:
```sql
-- Ensure proper indexes exist
-- For crawled_pages table
CREATE INDEX IF NOT EXISTS idx_crawled_pages_embedding_hnsw 
ON crawled_pages USING hnsw (embedding vector_ip_ops);

-- For full-text search
CREATE INDEX IF NOT EXISTS idx_crawled_pages_fts 
ON crawled_pages USING GIN (fts);

-- For source filtering
CREATE INDEX IF NOT EXISTS idx_crawled_pages_source_url 
ON crawled_pages (source_url);
```

#### Memory Optimization

**For resource-constrained environments**:
```bash
# Minimize memory usage
USE_RERANKING=false              # Saves ~500MB for model
USE_CONTEXTUAL_EMBEDDINGS=false  # Reduces API calls
USE_AGENTIC_RAG=false           # Fewer database tables

# If reranking needed, use smallest model
RERANKING_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2
```

**Monitor memory usage**:
```bash
# Check model memory usage
python -c "
from src.reranking import get_global_reranker
reranker = get_global_reranker()
print(f'Reranker loaded: {reranker is not None}')
"
```

#### Monitoring Performance Changes

**Before enabling new strategies**:
```bash
# Capture current performance
python src/performance_baseline.py

# Test with new configuration
python tests/test_performance_regression.py

# Compare results
python -c "
import json
with open('performance_baseline.json') as f:
    data = json.load(f)
    print(f'Average response time: {data[\"average_response_time\"]:.2f}ms')
"
```

**Set performance alerts**:
```bash
# Add to CI/CD or monitoring
python tests/test_performance_regression.py --max-regression 25
# Fails if performance degrades more than 25%
```

## System Architecture

### Strategy Manager

The system uses a **Strategy Manager** to dynamically control which components are loaded and which tools are available based on your configuration:

#### Component Lifecycle Management
- **Resource Optimization**: Only initializes components for enabled strategies (saves memory and startup time)
- **Automatic Cleanup**: Proper resource management during server shutdown
- **Error Handling**: Graceful fallbacks when components fail to initialize
- **Status Monitoring**: Real-time component health and configuration reporting

#### Dynamic Tool Registration
```bash
# Example: Only base tools available
USE_RERANKING=false
# Tools: crawl_single_page, smart_crawl_url, get_available_sources, perform_rag_query

# Example: Reranking enabled
USE_RERANKING=true  
# Tools: [base tools] + perform_rag_query_with_reranking

# Example: Multiple strategies enabled
USE_RERANKING=true
USE_AGENTIC_RAG=true
# Tools: [base tools] + perform_rag_query_with_reranking + search_code_examples
```

#### Strategy Validation
- **Dependency Checking**: Validates required environment variables for each enabled strategy
- **Configuration Validation**: Ensures strategy combinations are compatible
- **Startup Validation**: Prevents server startup with invalid configurations

#### Status Reporting
```bash
# Check strategy manager status
python -c "
from src.strategies.manager import get_strategy_manager
manager = get_strategy_manager()
if manager:
    status = manager.get_status_report()
    print(f'Enabled strategies: {status[\"enabled_strategies\"]}')
    print(f'Available tools: {len(status[\"available_tools\"])}')
    print(f'Component status: {status[\"components\"]}')
"
```

### Benefits
- **Efficient Resource Usage**: No wasted memory on unused components
- **Clean Interface**: Users only see tools relevant to their configuration  
- **Scalable Design**: Easy to add new strategies without affecting existing functionality
- **Development-Friendly**: Comprehensive testing and status reporting for debugging

## Running the Server

### Using Docker

```bash
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Using Python

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port.

## Manual Crawling

For large crawling jobs that might timeout through the MCP interface, use the manual crawler script with enhanced crawling support:

### Basic Usage

```bash
# Basic crawling (uses environment variable configuration)
python src/manual_crawl.py --url https://docs.example.com

# Force enhanced crawling (overrides environment variables)
python src/manual_crawl.py --url https://docs.n8n.io --enhanced

# Force baseline crawling (overrides environment variables)
python src/manual_crawl.py --url https://example.com --baseline
```

### Enhanced Manual Crawling

When `USE_ENHANCED_CRAWLING=true` is set or `--enhanced` flag is used, the manual crawler includes all enhanced features:

- **Framework Detection**: Automatically detects documentation platforms (Material Design, ReadMe.io, GitBook, etc.)
- **Quality Validation**: Measures content quality and applies fallback strategies when needed
- **Smart CSS Targeting**: Uses framework-specific selectors for better content extraction
- **Navigation Filtering**: Reduces navigation noise from 70-80% to 20-30%
- **Performance Monitoring**: Reports quality metrics and extraction performance

### Advanced Options

```bash
# Customize crawling parameters
python src/manual_crawl.py \
  --url https://docs.example.com \
  --enhanced \
  --max-depth 3 \
  --max-concurrent 10 \
  --chunk-size 5000 \
  --batch-size 20

# Environment variable control (recommended)
USE_ENHANCED_CRAWLING=true \
USE_CONTEXTUAL_EMBEDDINGS=true \
USE_AGENTIC_RAG=true \
python src/manual_crawl.py --url https://docs.example.com
```

### Quality Metrics Output

Enhanced crawling provides detailed quality reporting:

```
✅ Enhanced crawl: https://docs.n8n.io/getting-started/ - Quality: excellent (0.823)
   🔄 Used fallback after 2 attempts
📊 Enhanced crawling summary: 12 pages, avg quality: 0.751
📋 Mode: Enhanced crawling with framework detection and quality validation
```

### Benefits for Documentation Sites

Enhanced manual crawling is particularly effective for:

- **API Documentation**: ReadMe.io, Swagger/OpenAPI sites
- **Technical Documentation**: n8n, Kubernetes, React, Vue.js docs  
- **Knowledge Bases**: Company wikis, GitBook-hosted content
- **Educational Content**: Tutorial sites, course materials

**Quality Improvement Examples**:
- n8n docs: Content ratio improved from 30:70 to 80:20
- VirusTotal API docs: Navigation noise reduced by 60%
- GitHub Pages: Better extraction of main content vs. site navigation

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8051/sse"
>     }
>   }
> }
> ```
>
> **Note for Docker users**: Use `host.docker.internal` instead of `localhost` if your client is running in a different container. This will apply if you are using this MCP server within n8n!

### Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "OPENAI_API_KEY", 
               "-e", "SUPABASE_URL", 
               "-e", "SUPABASE_SERVICE_KEY", 
               "mcp/crawl4ai"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```

## Troubleshooting

### Configuration Issues

#### "Invalid strategy configuration" errors
**Problem**: Strategy validation fails on startup
**Solution**: 
```bash
# Check your .env file for typos
# Ensure boolean values are lowercase: true/false not True/False
USE_RERANKING=true  # ✓ Correct
USE_RERANKING=True  # ✗ Incorrect
```

#### "sentence-transformers not found" errors
**Problem**: Reranking enabled but dependency missing
**Solution**:
```bash
# Install the dependency
uv pip install sentence-transformers>=3.0.0
# Or disable reranking
USE_RERANKING=false
```

#### "Model loading failed" errors
**Problem**: Reranking model fails to load
**Solution**:
```bash
# Check model name in .env
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Try a different model if needed
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2

# Check available disk space (models can be 100-500MB)
```

### Performance Issues

#### Slow search responses
**Check current performance**:
```bash
# Run performance regression test
python tests/test_performance_regression.py
```

**Common causes and solutions**:
- **Multiple strategies enabled**: Disable unused strategies
- **Reranking overhead**: Adjust `RERANKING_MODEL` to a smaller model
- **Database issues**: Check Supabase connection and indexes
- **Network latency**: Use localhost URLs for local testing

#### High memory usage
**Symptoms**: Server crashes or becomes unresponsive
**Solutions**:
```bash
# Disable resource-intensive strategies
USE_RERANKING=false
USE_CONTEXTUAL_EMBEDDINGS=false

# Use smaller reranking models
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  # Smaller
RERANKING_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2  # Smallest
```

### Database Connection Issues

#### "Function hybrid_search_crawled_pages not found"
**Problem**: Database setup incomplete
**Solution**: 
1. Go to Supabase SQL Editor
2. Run the contents of `crawled_pages.sql`
3. Verify the function exists in Database → Functions

#### Connection timeouts
**Problem**: Network issues or incorrect URLs
**Solution**:
```bash
# For local development
SUPABASE_URL=http://localhost:54321  # Local Supabase

# For n8n/Docker integration
SUPABASE_URL=http://host.docker.internal:54321

# For cloud Supabase
SUPABASE_URL=https://your-project.supabase.co
```

### Docker Issues

#### Container won't start
**Check logs**:
```bash
docker logs <container_id>
```

**Common solutions**:
```bash
# Rebuild with latest changes
docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .

# Check environment variables
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag

# Verify .env file format (no spaces around =)
OPENAI_API_KEY=your_key  # ✓ Correct
OPENAI_API_KEY = your_key  # ✗ Incorrect
```

### Getting Help

#### Validate your configuration
```bash
# Test configuration loading
python -c "from src.config import get_strategy_config; print('✓ Configuration valid')"

# Test strategy manager
python -c "from src.strategies.manager import get_strategy_manager; print('✓ Strategy manager ready')"

# Test performance
python tests/test_performance_regression.py --quick

# Run comprehensive tests
uv run pytest tests/ -v

# Run specific test suites
uv run pytest tests/test_config.py -v                     # Configuration system (21 tests)
uv run pytest tests/test_reranking.py -v                  # Reranking functionality (22 tests)
uv run pytest tests/test_contextual_integration.py -v     # Contextual embeddings (16 tests)
uv run pytest tests/test_strategy_manager.py -v           # Strategy Manager (32 tests)
uv run pytest tests/test_code_extraction_pipeline.py -v   # Code extraction pipeline (14 tests)
```

#### Enable debug logging
```bash
# Add to your .env
LOG_LEVEL=DEBUG

# Run with verbose output
uv run src/crawl4ai_mcp.py --verbose
```

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling capabilities. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies
3. Modify the `utils.py` file for any helper functions you need
4. Extend the crawling capabilities by adding more specialized crawlers

# Enhanced Contextual Embeddings Configuration

The enhanced contextual embeddings system now includes adaptive prompting strategies that automatically detect content types and use specialized prompts for different kinds of content.

## Content Type Detection

The system automatically detects 9 content types:

- **Technical**: Documentation, APIs, code guides (e.g., from GitHub, docs sites)
- **Academic**: Research papers, studies, scholarly articles (e.g., from arXiv, PubMed)
- **Forum**: Discussion threads, Q&A posts (e.g., Reddit, StackOverflow)
- **News**: News articles, press releases (e.g., CNN, Reuters, Bloomberg)
- **Blog**: Opinion pieces, personal blogs (e.g., Medium, Substack)
- **Social Media**: Posts, threads, content from social platforms (e.g., Twitter/X, LinkedIn, Instagram)
- **Legal**: Legal documents, contracts, terms of service, statutes (e.g., law sites, court documents)
- **Educational**: Tutorials, how-to guides, online course materials (e.g., Udemy, Coursera, Khan Academy)
- **General**: Fallback for other content types

## Adaptive Prompting Configuration

```bash
# Enable content-type-aware prompting (default: true)
USE_ADAPTIVE_CONTEXTUAL_PROMPTS=true

# Enable automatic content type detection (default: true)
CONTEXTUAL_CONTENT_TYPE_DETECTION=true

# Include content type tags in contextual text (default: true)
CONTEXTUAL_INCLUDE_CONTENT_TYPE_TAG=true
```

## Configuration Examples

### Full Enhanced Setup (Recommended)
```bash
# Core strategies
USE_CONTEXTUAL_EMBEDDINGS=true
USE_RERANKING=true
USE_AGENTIC_RAG=true

# Enhanced contextual embeddings
USE_ADAPTIVE_CONTEXTUAL_PROMPTS=true
CONTEXTUAL_CONTENT_TYPE_DETECTION=true
CONTEXTUAL_INCLUDE_CONTENT_TYPE_TAG=true
CONTEXTUAL_MODEL=gpt-4o-mini
```

### Legacy Compatibility Mode
```bash
# Use enhanced contextual embeddings but with legacy prompting
USE_CONTEXTUAL_EMBEDDINGS=true
USE_ADAPTIVE_CONTEXTUAL_PROMPTS=false
CONTEXTUAL_CONTENT_TYPE_DETECTION=false
```

### Content-Type Detection Only
```bash
# Detect content types but use general prompting
USE_CONTEXTUAL_EMBEDDINGS=true
USE_ADAPTIVE_CONTEXTUAL_PROMPTS=false
CONTEXTUAL_CONTENT_TYPE_DETECTION=true
CONTEXTUAL_INCLUDE_CONTENT_TYPE_TAG=true
```

## Specialized Prompting Strategies

### Technical Content
- Focuses on APIs, procedures, technical concepts
- Preserves technical terminology and code references
- Identifies setup, examples, reference, troubleshooting contexts

### Academic Content  
- Emphasizes research concepts, methodology, findings
- Preserves academic language and theoretical frameworks
- Identifies introduction, methodology, results, discussion sections

### Forum Content
- Captures problems, solutions, discussion points
- Distinguishes questions vs. answers vs. commentary
- Preserves conversational context and key insights

### News Content
- Highlights events, people, developments
- Maintains journalistic objectivity
- Captures temporal context and key facts

### Blog Content
- Focuses on ideas, opinions, experiences
- Preserves author's voice and perspective
- Identifies insights, recommendations, lessons

### Social Media Content
- Captures key messages, announcements, discussion points
- Analyzes tone and engagement style (professional, casual, promotional)
- Preserves hashtags, mentions, and trending topics
- Identifies content purpose (share, promote, discuss, network)

### Legal Content
- Focuses on legal concepts, obligations, and procedural elements
- Preserves precise legal terminology and formal tone
- Identifies key parties, jurisdictions, and legal frameworks
- Distinguishes definitions, obligations, exceptions, and procedures

### Educational Content
- Emphasizes learning objectives, skills, and instructional elements
- Preserves instructional clarity and learning-focused language
- Identifies difficulty levels and target audience context
- Distinguishes tutorials, exercises, examples, and explanations

## Benefits of Enhanced Contextual Embeddings

1. **Better Semantic Understanding**: Content-aware prompts generate more relevant context
2. **Improved Search Quality**: Type-specific context enhances embedding quality
3. **Diverse Content Support**: Works well with technical docs, research, forums, news, blogs
4. **Backward Compatibility**: Can be disabled to use legacy prompting
5. **Debugging Support**: Content type tags help monitor classification accuracy