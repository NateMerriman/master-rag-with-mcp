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

The server provides four essential web crawling and search tools:

1. **`crawl_single_page`**: Quickly crawl a single web page and store its content in the vector database
2. **`smart_crawl_url`**: Intelligently crawl a full website based on the type of URL provided (sitemap, llms-full.txt, or a regular webpage that needs to be crawled recursively)
3. **`get_available_sources`**: Get a list of all available sources (domains) in the database
4. **`perform_rag_query`**: Search for relevant content using semantic search with optional source filtering

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

Before running the server, you need to set up the database with the pgvector extension:

1. Go to the SQL Editor in your Supabase dashboard (create a new project first if necessary)

2. Create a new query and paste the contents of `crawled_pages.sql`

3. Run the query to create the necessary tables and functions

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

#### AI Coding Assistant Setup
```bash
# Core + contextual embeddings + code search + reranking
OPENAI_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_SERVICE_KEY=your_key
USE_CONTEXTUAL_EMBEDDINGS=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
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
**Purpose**: Improves search result quality by reordering the top results.

**What it does**: Uses a specialized AI model to score query-document pairs and reorder results by relevance.

**Benefits**:
- Significantly improves result quality for complex queries
- Works on top of existing hybrid search
- Particularly effective for natural language questions

**Trade-offs**:
- Adds ~200-500ms processing time for reranking
- Requires sentence-transformers dependency (~500MB)
- Uses additional CPU for local inference

**Best for**: Users who prioritize search quality over speed

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
**Purpose**: Specialized capabilities for AI coding assistants.

**What it does**: Extracts and indexes code examples separately, enables code-to-code search.

**Benefits**:
- Specialized code search and extraction
- Better retrieval for programming-related queries
- Supports natural language to code search

**Trade-offs**:
- Additional database tables and complexity
- Code extraction processing overhead
- Most beneficial only for programming content

**Best for**: AI coding assistants, technical documentation with code examples

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

# Test performance
python tests/test_performance_regression.py --quick
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