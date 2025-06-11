"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import functools
import time

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
)
from utils import (
    get_supabase_client,
    add_documents_to_supabase,
    search_documents,
)
from config import get_config, ConfigurationError, StrategyConfig, RAGStrategy
from strategies import StrategyManager
from strategies.manager import (
    initialize_strategy_manager,
    cleanup_strategy_manager,
    get_strategy_manager,
)

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Load and validate configuration at startup
try:
    strategy_config = get_config()
    print(f"✅ {strategy_config}")

    # Log enabled strategies for debugging
    enabled_strategies = strategy_config.get_enabled_strategies()
    if enabled_strategies:
        strategy_names = [s.value for s in enabled_strategies]
        print(f"📊 Enhanced RAG strategies enabled: {', '.join(strategy_names)}")
    else:
        print("📊 Running in baseline mode (no enhanced strategies)")

    # Initialize strategy manager
    strategy_manager = initialize_strategy_manager(strategy_config)
    print(
        f"✅ Strategy manager initialized with {len(strategy_manager.components)} components"
    )

except ConfigurationError as e:
    print(f"❌ Configuration Error: {e}")
    exit(1)
except RuntimeError as e:
    print(f"❌ Strategy Manager Error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected configuration error: {e}")
    exit(1)


# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""

    crawler: AsyncWebCrawler
    supabase_client: Client
    strategy_manager: StrategyManager


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.

    Args:
        server: The FastMCP server instance

    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler, Supabase client, and strategy manager
    """
    # Create browser configuration
    browser_config = BrowserConfig(headless=True, verbose=False)

    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()

    # Initialize Supabase client
    supabase_client = get_supabase_client()

    # Get the strategy manager (already initialized at startup)
    strategy_manager = get_strategy_manager()
    if strategy_manager is None:
        raise RuntimeError("Strategy manager not initialized")

    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            strategy_manager=strategy_manager,
        )
    finally:
        # Clean up the crawler
        await crawler.__aexit__(None, None, None)
        # Strategy manager cleanup happens at application shutdown


# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051"),
)
# Force reload of modules


def conditional_tool(tool_name: str, required_strategies: List[RAGStrategy] = None):
    """
    Decorator for conditional tool registration based on enabled strategies.

    Args:
        tool_name: Name of the tool for availability checking
        required_strategies: List of strategies required for this tool (None for always available)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(ctx: Context, *args, **kwargs):
            # Get strategy manager from context
            strategy_manager = ctx.request_context.lifespan_context.strategy_manager

            # Check if tool should be available
            if not strategy_manager.should_tool_be_available(tool_name):
                # Determine which strategies are missing
                if required_strategies:
                    enabled = strategy_manager.get_enabled_strategies()
                    missing = [s.value for s in required_strategies if s not in enabled]
                    error_msg = (
                        f"Tool '{tool_name}' requires strategies: {', '.join(missing)}"
                    )
                else:
                    error_msg = (
                        f"Tool '{tool_name}' is not available in current configuration"
                    )

                return json.dumps(
                    {
                        "success": False,
                        "error": error_msg,
                        "tool": tool_name,
                        "required_strategies": [s.value for s in required_strategies]
                        if required_strategies
                        else [],
                    },
                    indent=2,
                )

            # Tool is available, proceed with execution
            return await func(ctx, *args, **kwargs)

        return wrapper

    return decorator


def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.

    Args:
        url: URL to check

    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path


def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.

    Args:
        url: URL to check

    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith(".txt")


def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.

    Args:
        sitemap_url: URL of the sitemap

    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall(".//{*}loc")]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls


def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """
    Split text into chunks, respecting code blocks and paragraphs.

    Enhanced version that properly preserves code block integrity.
    Uses the EnhancedMarkdownChunker for robust code block handling.
    """
    try:
        # Try relative import first (when running as part of package)
        from .improved_chunking import smart_chunk_markdown_enhanced
    except ImportError:
        # Fall back to absolute import (when running as script)
        from improved_chunking import smart_chunk_markdown_enhanced

    return smart_chunk_markdown_enhanced(text, chunk_size)


def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.

    Args:
        chunk: Markdown chunk

    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r"^(#+)\s+(.+)$", chunk, re.MULTILINE)
    header_str = "; ".join([f"{h[0]} {h[1]}" for h in headers]) if headers else ""

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split()),
    }


@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.

    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl

    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)

        if result.success and result.markdown:
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)

            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)

            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}

            # Add to Supabase
            add_documents_to_supabase(
                supabase_client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document,
            )

            return json.dumps(
                {
                    "success": True,
                    "url": url,
                    "chunks_stored": len(chunks),
                    "content_length": len(result.markdown),
                    "links_count": {
                        "internal": len(result.links.get("internal", [])),
                        "external": len(result.links.get("external", [])),
                    },
                },
                indent=2,
            )
        else:
            return json.dumps(
                {"success": False, "url": url, "error": result.error_message}, indent=2
            )
    except Exception as e:
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)


@mcp.tool()
async def smart_crawl_url(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.

    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth

    All crawled content is chunked and stored in Supabase for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 1000)

    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Get the crawler and Supabase client from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        crawl_results = []
        crawl_type = "webpage"

        # Detect URL type and use appropriate crawl method
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps(
                    {"success": False, "url": url, "error": "No URLs found in sitemap"},
                    indent=2,
                )
            crawl_results = await crawl_batch(
                crawler, sitemap_urls, max_concurrent=max_concurrent
            )
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(
                crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent
            )
            crawl_type = "webpage"

        if not crawl_results:
            return json.dumps(
                {"success": False, "url": url, "error": "No content found"}, indent=2
            )

        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0

        for doc in crawl_results:
            source_url = doc["url"]
            md = doc["markdown"]
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)

            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = urlparse(source_url).netloc
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)

                chunk_count += 1

        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in crawl_results:
            url_to_full_document[doc["url"]] = doc["markdown"]

        # Add to Supabase
        # IMPORTANT: Adjust this batch size for more speed if you want! Just don't overwhelm your system or the embedding API ;)
        batch_size = 20
        add_documents_to_supabase(
            supabase_client,
            urls,
            chunk_numbers,
            contents,
            metadatas,
            url_to_full_document,
            batch_size=batch_size,
        )

        return json.dumps(
            {
                "success": True,
                "url": url,
                "crawl_type": crawl_type,
                "pages_crawled": len(crawl_results),
                "chunks_stored": chunk_count,
                "urls_crawled": [doc["url"] for doc in crawl_results][:5]
                + (["..."] if len(crawl_results) > 5 else []),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)


async def crawl_markdown_file(
    crawler: AsyncWebCrawler, url: str
) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.

    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{"url": url, "markdown": result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []


async def crawl_batch(
    crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.

    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    results = await crawler.arun_many(
        urls=urls, config=crawl_config, dispatcher=dispatcher
    )
    return [
        {"url": r.url, "markdown": r.markdown}
        for r in results
        if r.success and r.markdown
    ]


async def crawl_recursive_internal_links(
    crawler: AsyncWebCrawler,
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.

    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [
            normalize_url(url)
            for url in current_urls
            if normalize_url(url) not in visited
        ]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(
            urls=urls_to_crawl, config=run_config, dispatcher=dispatcher
        )
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({"url": result.url, "markdown": result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources based on unique source metadata values.

    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database. This is useful for discovering what content is available for querying.

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with the list of available sources
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Use a direct query with the Supabase client
        # This could be more efficient with a direct Postgres query but
        # I don't want to require users to set a DB_URL environment variable as well
        result = (
            supabase_client.from_("crawled_pages")
            .select("metadata")
            .not_.is_("metadata->>source", "null")
            .execute()
        )

        # Use a set to efficiently track unique sources
        unique_sources = set()

        # Extract the source values from the result using a set for uniqueness
        if result.data:
            for item in result.data:
                source = item.get("metadata", {}).get("source")
                if source:
                    unique_sources.add(source)

        # Convert set to sorted list for consistent output
        sources = sorted(list(unique_sources))

        return json.dumps(
            {"success": True, "sources": sources, "count": len(sources)}, indent=2
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def perform_rag_query(
    ctx: Context, query: str, source: str = None, match_count: int = 5
) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.

    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.

    Use the tool to get source domains if the user is asking to use a specific tool or framework.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}

        # Perform the search
        results = search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata,
        )

        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "url": result.get("url"),
                    "content": result.get("content"),
                    "metadata": result.get("metadata"),
                    "rrf_score": result.get("rrf_score"),
                    "full_text_rank": result.get("full_text_rank"),
                    "semantic_rank": result.get("semantic_rank"),
                }
            )

        return json.dumps(
            {
                "success": True,
                "query": query,
                "source_filter": source,
                "results": formatted_results,
                "count": len(formatted_results),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)


# Conditional strategy-specific tools (registered based on enabled strategies)


@mcp.tool()
@conditional_tool("search_code_examples", [RAGStrategy.AGENTIC_RAG])
async def search_code_examples(
    ctx: Context,
    query: str,
    programming_language: str = None,
    complexity_min: int = 1,
    complexity_max: int = 10,
    match_count: int = 5,
) -> str:
    """
    Search for code examples using hybrid search (semantic + full-text).

    This tool searches the code_examples table for relevant code snippets based on
    natural language queries or code-to-code similarity using advanced hybrid search
    with Reciprocal Rank Fusion (RRF) that combines semantic vector search with
    full-text search for optimal results.

    Args:
        ctx: The MCP server provided context
        query: Search query for code examples (natural language or code fragments)
        programming_language: Optional filter by programming language (e.g., 'python', 'javascript')
        complexity_min: Minimum complexity score (1-10, default: 1)
        complexity_max: Maximum complexity score (1-10, default: 10)
        match_count: Maximum number of results to return (default: 5, max: 30)

    Returns:
        JSON string with code search results including content, summaries, and metadata
    """
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Validate parameters
        match_count = min(max(match_count, 1), 30)  # Clamp between 1 and 30
        complexity_min = max(min(complexity_min, 10), 1)  # Clamp between 1 and 10
        complexity_max = max(
            min(complexity_max, 10), complexity_min
        )  # Ensure max >= min

        # Create enhanced query for better code search results
        # Combine the original query with context that helps with code understanding
        enhanced_query = f"Code example: {query}"
        if programming_language:
            enhanced_query += f" in {programming_language}"

        # Create embedding for the enhanced query
        from utils import create_embedding

        query_embedding = create_embedding(enhanced_query)

        # Prepare parameters for the hybrid search function
        search_params = {
            "query_text": query,  # Use original query for full-text search
            "query_embedding": query_embedding,
            "match_count": match_count,
            "language_filter": programming_language if programming_language else None,
            "max_complexity": complexity_max,
        }

        # Execute hybrid search using the RPC function
        response = supabase_client.rpc(
            "hybrid_search_code_examples", search_params
        ).execute()

        if not response.data:
            return json.dumps(
                {
                    "success": True,
                    "query": query,
                    "enhanced_query": enhanced_query,
                    "filters": {
                        "programming_language": programming_language,
                        "complexity_range": [complexity_min, complexity_max],
                        "match_count": match_count,
                    },
                    "results": [],
                    "count": 0,
                    "message": "No code examples found matching your criteria",
                },
                indent=2,
            )

        # Filter results by complexity_min (SQL function only filters by max)
        filtered_results = [
            result
            for result in response.data
            if result.get("complexity_score", 1) >= complexity_min
        ]

        # Format results for optimal readability and usability
        formatted_results = []
        for i, result in enumerate(filtered_results[:match_count]):
            # Extract metadata safely
            metadata = result.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    import json as json_lib

                    metadata = json_lib.loads(metadata)
                except:
                    metadata = {}

            formatted_result = {
                "id": result.get("id"),
                "url": result.get("url"),
                "content": result.get("content", ""),
                "summary": result.get("summary", ""),
                "programming_language": result.get("programming_language"),
                "complexity_score": result.get("complexity_score"),
                "similarity": round(result.get("similarity", 0.0), 4),
                "rrf_score": round(result.get("rrf_score", 0.0), 4),
                "ranking": {
                    "position": i + 1,
                    "semantic_rank": result.get("semantic_rank"),
                    "full_text_rank": result.get("full_text_rank"),
                },
                "metadata": metadata,
            }
            formatted_results.append(formatted_result)

        # Calculate search statistics
        total_found = len(response.data)
        after_complexity_filter = len(filtered_results)
        returned_count = len(formatted_results)

        return json.dumps(
            {
                "success": True,
                "query": query,
                "enhanced_query": enhanced_query,
                "filters": {
                    "programming_language": programming_language,
                    "complexity_range": [complexity_min, complexity_max],
                    "match_count": match_count,
                },
                "results": formatted_results,
                "count": returned_count,
                "search_stats": {
                    "total_found": total_found,
                    "after_complexity_filter": after_complexity_filter,
                    "returned": returned_count,
                    "search_type": "hybrid_rrf",
                },
                "message": f"Found {returned_count} code examples using hybrid search (RRF)",
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "query": query,
                "error": f"Code search failed: {str(e)}",
                "filters": {
                    "programming_language": programming_language,
                    "complexity_range": [complexity_min, complexity_max],
                    "match_count": match_count,
                },
            },
            indent=2,
        )


@mcp.tool()
@conditional_tool("perform_rag_query_with_reranking", [RAGStrategy.RERANKING])
async def perform_rag_query_with_reranking(
    ctx: Context,
    query: str,
    source: str = None,
    match_count: int = 20,
    rerank_top_k: int = 5,
) -> str:
    """
    Perform RAG query with cross-encoder reranking for improved result quality.

    This tool performs hybrid search (combining semantic + full-text with RRF) and then
    applies cross-encoder reranking to improve the quality and relevance of results.
    The hybrid search RRF benefits are preserved as the initial ranking, with reranking
    providing additional quality improvements.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results
        match_count: Number of initial results to retrieve for reranking
        rerank_top_k: Number of top results to return after reranking

    Returns:
        JSON string with reranked search results
    """
    try:
        strategy_manager = ctx.request_context.lifespan_context.strategy_manager
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Get reranker component
        reranker = strategy_manager.get_component("reranker")
        if reranker is None:
            return json.dumps(
                {"success": False, "error": "Reranker component not available"},
                indent=2,
            )

        # Step 1: Perform hybrid search (preserves RRF benefits)
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}

        hybrid_results = search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata,
        )

        if not hybrid_results:
            return json.dumps(
                {
                    "success": True,
                    "query": query,
                    "source_filter": source,
                    "results": [],
                    "count": 0,
                    "reranked": False,
                    "message": "No results found for query",
                },
                indent=2,
            )

        # Step 2: Apply cross-encoder reranking on top of hybrid search results
        start_time = time.time()

        # Convert results to format expected by reranker
        search_results_for_reranking = []
        for result in hybrid_results:
            search_results_for_reranking.append(
                {
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "title": result.get("metadata", {}).get("headers", ""),
                    "chunk_index": result.get("metadata", {}).get("chunk_index", 0),
                    "score": result.get("rrf_score", 0.0),
                    "metadata": result.get("metadata", {}),
                    # Preserve hybrid search scores
                    "rrf_score": result.get("rrf_score"),
                    "full_text_rank": result.get("full_text_rank"),
                    "semantic_rank": result.get("semantic_rank"),
                }
            )

        # Apply reranking
        reranking_result = reranker.rerank_results(query, search_results_for_reranking)

        # Step 3: Format final results preserving both hybrid and reranking scores
        formatted_results = []
        for i, result in enumerate(reranking_result.results[:rerank_top_k]):
            formatted_results.append(
                {
                    "url": result.url,
                    "content": result.content,
                    "metadata": {
                        **result.metadata,
                        # Preserve original hybrid search scores
                        "original_rrf_score": result.original_score,
                        "original_rank": i + 1,
                        # Add reranking information
                        "reranking_score": result.metadata.get("reranking_score", 0.0),
                        "reranked_position": i + 1,
                    },
                    # Keep original hybrid search scores at top level for compatibility
                    "rrf_score": result.original_score,
                    "full_text_rank": search_results_for_reranking[0].get(
                        "full_text_rank"
                    )
                    if search_results_for_reranking
                    else None,
                    "semantic_rank": search_results_for_reranking[0].get(
                        "semantic_rank"
                    )
                    if search_results_for_reranking
                    else None,
                    # Add reranking score at top level
                    "reranking_score": result.metadata.get("reranking_score", 0.0),
                }
            )

        reranking_time = (time.time() - start_time) * 1000

        return json.dumps(
            {
                "success": True,
                "query": query,
                "source_filter": source,
                "results": formatted_results,
                "count": len(formatted_results),
                "reranked": True,
                "reranking_stats": {
                    "model_used": reranking_result.model_used,
                    "reranking_time_ms": reranking_result.reranking_time_ms,
                    "total_scored": reranking_result.total_scored,
                    "fallback_used": reranking_result.fallback_used,
                    "initial_results": len(hybrid_results),
                    "reranked_results": len(formatted_results),
                },
                "message": f"Results enhanced with hybrid search (RRF) + cross-encoder reranking in {reranking_time:.1f}ms",
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
@conditional_tool("perform_contextual_rag_query", [RAGStrategy.CONTEXTUAL_EMBEDDINGS])
async def perform_contextual_rag_query(
    ctx: Context, query: str, source: str = None, match_count: int = 5
) -> str:
    """
    Perform RAG query with enhanced contextual embeddings for improved semantic understanding.

    This tool leverages document-level context to generate better embeddings that capture
    the semantic relationships within and across documents.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results
        match_count: Maximum number of results to return

    Returns:
        JSON string with contextually enhanced search results
    """
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Use the existing search function which already supports contextual embeddings
        # when USE_CONTEXTUAL_EMBEDDINGS is enabled
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}

        results = search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata,
        )

        # Format the results with contextual enhancement indicators
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "url": result.get("url"),
                    "content": result.get("content"),
                    "metadata": result.get("metadata"),
                    "rrf_score": result.get("rrf_score"),
                    "full_text_rank": result.get("full_text_rank"),
                    "semantic_rank": result.get("semantic_rank"),
                    "contextual_enhanced": True,
                }
            )

        return json.dumps(
            {
                "success": True,
                "query": query,
                "source_filter": source,
                "results": formatted_results,
                "count": len(formatted_results),
                "contextual_embeddings": True,
                "message": "Results enhanced with document-level context",
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def get_strategy_status(ctx: Context) -> str:
    """
    Get current strategy configuration and available tools.

    This tool provides information about which RAG strategies are enabled,
    what tools are available, and the status of strategy components.

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with strategy status and available tools
    """
    try:
        strategy_manager = ctx.request_context.lifespan_context.strategy_manager

        # Get comprehensive status report
        status_report = strategy_manager.get_status_report()

        # Add tool availability information
        status_report["tool_descriptions"] = {
            # Base tools (always available)
            "crawl_single_page": "Crawl a single web page and store its content",
            "smart_crawl_url": "Intelligently crawl URLs (sitemaps, txt files, or regular pages)",
            "get_available_sources": "Get all available sources in the database",
            "perform_rag_query": "Perform basic RAG query with hybrid search",
            # Strategy-specific tools
            "search_code_examples": "Search for code examples (requires AGENTIC_RAG)",
            "perform_rag_query_with_reranking": "Enhanced RAG query with reranking (requires RERANKING)",
            "perform_contextual_rag_query": "RAG query with contextual embeddings (requires CONTEXTUAL_EMBEDDINGS)",
        }

        # Add configuration guide
        status_report["configuration_guide"] = {
            "enable_strategies": {
                "USE_CONTEXTUAL_EMBEDDINGS": "Enhanced semantic understanding with document context",
                "USE_RERANKING": "Improved result quality with cross-encoder reranking",
                "USE_AGENTIC_RAG": "Code extraction and specialized code search capabilities",
                "USE_HYBRID_SEARCH_ENHANCED": "Advanced hybrid search algorithms",
            },
            "model_settings": {
                "CONTEXTUAL_MODEL": "LLM model for contextual embeddings (default: gpt-4o-mini-2024-07-18)",
                "RERANKING_MODEL": "Cross-encoder model for reranking (default: ms-marco-MiniLM-L-6-v2)",
            },
        }

        return json.dumps(status_report, indent=2)

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def main():
    transport = os.getenv("TRANSPORT", "sse")
    try:
        if transport == "sse":
            # Run the MCP server with sse transport
            await mcp.run_sse_async()
        else:
            # Run the MCP server with stdio transport
            await mcp.run_stdio_async()
    finally:
        # Clean up strategy manager resources
        print("🧹 Cleaning up strategy manager...")
        cleanup_strategy_manager()


if __name__ == "__main__":
    asyncio.run(main())
