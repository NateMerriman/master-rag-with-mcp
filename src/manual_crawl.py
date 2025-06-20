#!/usr/bin/env python3
"""
Manual crawler / chunker / Supabase loader for big jobs that would time-out
when triggered through the MCP tool layer.

This script supports all enhanced RAG strategies via environment variable configuration.
All enhancements from the MCP server are available during manual crawling:
- Contextual embeddings (USE_CONTEXTUAL_EMBEDDINGS=true)
- Agentic RAG code extraction (USE_AGENTIC_RAG=true)
- Enhanced crawling with framework detection (USE_ENHANCED_CRAWLING=true)
- Cross-encoder reranking (USE_RERANKING=true)
- Enhanced hybrid search capabilities

Enhanced crawling features:
- Framework detection (Material Design, ReadMe.io, GitBook, Docusaurus, etc.)
- Quality validation with automatic fallback mechanisms
- Smart CSS selector targeting for better content extraction
- Navigation noise reduction (70-80% to 20-30%)
- Quality metrics reporting and monitoring

Usage examples:
  # Basic enhanced crawling
  python src/manual_crawl.py --url https://docs.n8n.io --enhanced
  
  # Force baseline crawling
  python src/manual_crawl.py --url https://example.com --baseline
  
  # Use environment variable control
  USE_ENHANCED_CRAWLING=true python src/manual_crawl.py --url https://docs.example.com
"""

import argparse, asyncio, json, datetime, os
from urllib.parse import urlparse
from tqdm import tqdm
import datetime, zoneinfo  # add zoneinfo
from pathlib import Path
from dotenv import load_dotenv
import logging, sys
import importlib

UTC = zoneinfo.ZoneInfo("UTC")  # single timezone object

# Load environment variables
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path, override=True)

# The yellow "import crawl4ai could not be resolved" comes from Pylance on your Mac.
# It happens because the crawl4ai package lives only inside the Docker image, not in your local Python env. The warning is harmless; the import will work once the code runs in the container.
from crawl4ai import AsyncWebCrawler, BrowserConfig

# keep Supabase helpers from utils.py (these now include all enhancements)
from . import utils

importlib.reload(utils)
from .utils import (
    get_supabase_client,
    add_documents_to_supabase,
    add_code_examples_to_supabase,
    extract_code_from_content,
)

# bring the crawl helpers in from crawl4ai_mcp.py
from .crawl4ai_mcp import (
    is_sitemap,
    is_txt,
    parse_sitemap,
    crawl_markdown_file,
    crawl_batch,
    crawl_recursive_internal_links,
    smart_chunk_markdown,
    extract_section_info,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,  # <-- this clears earlier handlers
)
logger = logging.getLogger(__name__)


def safe_get_quality_category(quality_metrics) -> str:
    """
    Safely extract quality category from quality_metrics.
    
    This defensive function handles cases where quality_metrics might be:
    - None
    - A ContentQualityMetrics object (expected)
    - A string (unexpected but should be handled gracefully)
    - Any other type (unexpected)
    
    Returns:
        str: The quality category or a safe default
    """
    if quality_metrics is None:
        return "unknown"
    
    # Check if it has the expected attribute
    if hasattr(quality_metrics, 'quality_category'):
        return quality_metrics.quality_category
    
    # If it's a string, it might be a serialized version - log and return it
    if isinstance(quality_metrics, str):
        logger.warning(f"quality_metrics is unexpectedly a string: {quality_metrics}")
        return "error_string"
    
    # For any other unexpected type
    logger.error(f"quality_metrics has unexpected type {type(quality_metrics)}: {quality_metrics}")
    return "error_type"


# Load and validate configuration for enhanced features
try:
    from .config import get_config, ConfigurationError

    strategy_config = get_config()
    logger.info(f"✅ {strategy_config}")

    # Log enabled strategies for manual crawling
    enabled_strategies = strategy_config.get_enabled_strategies()
    if enabled_strategies:
        strategy_names = [s.value for s in enabled_strategies]
        logger.info(
            f"📊 Manual crawl with enhanced RAG strategies: {', '.join(strategy_names)}"
        )
    else:
        logger.info("📊 Manual crawl in baseline mode (no enhanced strategies)")

except (ConfigurationError, ImportError) as e:
    logger.warning(f"⚠️ Configuration Warning: {e}")
    logger.info("📊 Manual crawl will use baseline functionality only")
    strategy_config = None

# Enhanced crawling integration (requires USE_ENHANCED_CRAWLING=true)
use_enhanced_crawling = os.getenv("USE_ENHANCED_CRAWLING", "false").lower() == "true"

if use_enhanced_crawling:
    try:
        # Add current directory to Python path for imports
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import enhanced crawling modules
        from .smart_crawler_factory import (
            EnhancedCrawler,
            CrawlResult,
            crawl_single_page_enhanced,
            smart_crawl_url_enhanced,
            crawl_recursive_internal_links_enhanced
        )
        logger.info("🚀 Enhanced crawling modules loaded successfully")
        
    except ImportError as e:
        logger.error(f"❌ Enhanced crawling modules not available: {e}")
        logger.info("📊 Manual crawl will use baseline crawling functionality")
        use_enhanced_crawling = False
    except Exception as e:
        logger.error(f"❌ Unexpected error loading enhanced crawling: {e}")
        logger.info("📊 Manual crawl will use baseline crawling functionality") 
        use_enhanced_crawling = False
else:
    logger.info("📊 Enhanced crawling disabled (USE_ENHANCED_CRAWLING=false)")

DEFAULT_CHUNK_SIZE = 5_000
DEFAULT_BATCH_SIZE = 20
DEFAULT_DEPTH = 3
DEFAULT_CONCURRENCY = 10


async def _crawl_and_store_enhanced(
    url: str,
    max_depth: int,
    max_concurrent: int,
    chunk_size: int,
    batch_size: int,
):
    """Enhanced crawling with framework detection and quality validation."""
    supabase = get_supabase_client()

    try:
        # 1️⃣ Enhanced crawling ----------------------------------------------------
        if is_txt(url) or is_sitemap(url):
            # For txt files and sitemaps, use smart crawl enhanced
            crawl_results = await smart_crawl_url_enhanced(url)
        else:
            # For regular URLs, use enhanced recursive crawling
            crawl_results = await crawl_recursive_internal_links_enhanced(
                start_urls=[url],
                max_depth=max_depth,
                max_concurrent=max_concurrent
            )
        
        if not crawl_results or not any(r.success for r in crawl_results):
            logger.warning("Enhanced crawling failed - no successful results")
            return

        # Convert enhanced results to legacy format for compatibility
        pages = []
        total_quality_score = 0
        successful_pages = 0
        
        for result in crawl_results:
            if result.success and result.markdown.strip():
                pages.append({
                    "url": result.url,
                    "markdown": result.markdown
                })
                if result.quality_metrics:
                    total_quality_score += result.quality_metrics.overall_quality_score
                    successful_pages += 1
                    quality_category = safe_get_quality_category(result.quality_metrics)
                    logger.info(f"✅ Enhanced crawl: {result.url} - Quality: {quality_category} ({result.quality_metrics.overall_quality_score:.3f})")
                    if result.used_fallback:
                        logger.info(f"   🔄 Used fallback after {result.extraction_attempts} attempts")
        
        if successful_pages > 0:
            avg_quality = total_quality_score / successful_pages
            logger.info(f"📊 Enhanced crawling summary: {successful_pages} pages, avg quality: {avg_quality:.3f}")

        if not pages:
            logger.warning("No pages with content extracted")
            return

        # 2️⃣ Pre-scan to know how many chunks we'll handle ---------------------------
        total_chunks = 0
        for p in pages:
            total_chunks += len(
                smart_chunk_markdown(p["markdown"], chunk_size=chunk_size)
            )

        page_bar = tqdm(pages, desc="enhanced pages", unit="page")
        chunk_bar = tqdm(total=total_chunks, desc="chunking", unit="chunk")

        # 3️⃣ Chunk + build payload ------------------------------------------
        urls, idxs, contents, metas = [], [], [], []
        for page in page_bar:
            chunks = smart_chunk_markdown(page["markdown"], chunk_size=chunk_size)
            for i, chunk in enumerate(chunks):
                urls.append(page["url"])
                idxs.append(i)
                contents.append(chunk)
                metas.append(
                    {
                        **extract_section_info(chunk),
                        "chunk_index": i,
                        "url": page["url"],
                        "source": urlparse(page["url"]).netloc,
                        "manual_run": True,
                        "enhanced_crawling": True,
                        "crawl_time": datetime.datetime.now(
                            UTC
                        ).isoformat(),
                    }
                )
                chunk_bar.update(1)
        chunk_bar.close()

        # 4️⃣ Store in Supabase ----------------------------------------------
        add_documents_to_supabase(
            supabase,
            urls,
            idxs,
            contents,
            metas,
            {p["url"]: p["markdown"] for p in pages},
            strategy_config,
            batch_size=batch_size,
        )
        logger.info(
            json.dumps(
                {
                    "success": True,
                    "pages_crawled": len(pages),
                    "chunks_stored": len(contents),
                    "enhanced_crawling": True,
                    "avg_quality_score": avg_quality if successful_pages > 0 else None,
                },
            )
        )
    
    except Exception as e:
        logger.error(f"Enhanced crawling failed: {e}")
        logger.info("Falling back to baseline crawling")
        # Fall back to baseline crawling
        await _crawl_and_store_baseline(url, max_depth, max_concurrent, chunk_size, batch_size)


async def _crawl_and_store_baseline(
    url: str,
    max_depth: int,
    max_concurrent: int,
    chunk_size: int,
    batch_size: int,
):
    """Baseline crawling functionality (original implementation)."""
    crawler = AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False))
    await crawler.__aenter__()
    supabase = get_supabase_client()

    try:
        # 1️⃣ Collect pages ----------------------------------------------------
        if is_txt(url):
            pages = await crawl_markdown_file(crawler, url)
        elif is_sitemap(url):
            pages = await crawl_batch(
                crawler, parse_sitemap(url), max_concurrent=max_concurrent
            )
        else:
            pages = await crawl_recursive_internal_links(
                crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent
            )
        if not pages:
            logger.warning("nothing_crawled")
            return

        # 2️⃣ pre-scan to know how many chunks we'll handle ---------------------------
        total_chunks = 0
        for p in pages:
            total_chunks += len(
                smart_chunk_markdown(p["markdown"], chunk_size=chunk_size)
            )

        page_bar = tqdm(pages, desc="baseline pages", unit="page")
        chunk_bar = tqdm(total=total_chunks, desc="chunking", unit="chunk")

        # 3️⃣ Chunk + build payload ------------------------------------------
        urls, idxs, contents, metas = [], [], [], []
        for page in page_bar:
            chunks = smart_chunk_markdown(page["markdown"], chunk_size=chunk_size)
            for i, chunk in enumerate(chunks):
                urls.append(page["url"])
                idxs.append(i)
                contents.append(chunk)
                metas.append(
                    {
                        **extract_section_info(chunk),
                        "chunk_index": i,
                        "url": page["url"],
                        "source": urlparse(page["url"]).netloc,
                        "manual_run": True,
                        "crawl_time": datetime.datetime.now(
                            UTC
                        ).isoformat(),
                    }
                )
                chunk_bar.update(1)
        chunk_bar.close()

        # 4️⃣ Store in Supabase ----------------------------------------------
        add_documents_to_supabase(
            supabase,
            urls,
            idxs,
            contents,
            metas,
            {p["url"]: p["markdown"] for p in pages},
            strategy_config,
            batch_size=batch_size,
        )
        logger.info(
            json.dumps(
                {
                    "success": True,
                    "pages_crawled": len(pages),
                    "chunks_stored": len(contents),
                },
            )
        )
    finally:
        await crawler.__aexit__(None, None, None)


async def _crawl_and_store(
    url: str,
    max_depth: int,
    max_concurrent: int,
    chunk_size: int,
    batch_size: int,
):
    """Main crawl function that dispatches to enhanced or baseline crawling."""
    if use_enhanced_crawling:
        logger.info("🚀 Using enhanced crawling with framework detection")
        await _crawl_and_store_enhanced(url, max_depth, max_concurrent, chunk_size, batch_size)
    else:
        logger.info("📊 Using baseline crawling")
        await _crawl_and_store_baseline(url, max_depth, max_concurrent, chunk_size, batch_size)


def main() -> None:
    p = argparse.ArgumentParser(description="Manual Crawl → Chunk → Supabase")
    p.add_argument("--url", required=True, help="Target URL or sitemap.txt")
    p.add_argument("--max-depth", type=int, default=DEFAULT_DEPTH, 
                   help=f"Max crawl depth (default: {DEFAULT_DEPTH})")
    p.add_argument("--max-concurrent", type=int, default=DEFAULT_CONCURRENCY,
                   help=f"Max concurrent requests (default: {DEFAULT_CONCURRENCY})")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help=f"Chunk size for content splitting (default: {DEFAULT_CHUNK_SIZE})")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Batch size for database insertion (default: {DEFAULT_BATCH_SIZE})")
    
    # Enhanced crawling options
    p.add_argument("--enhanced", action="store_true", 
                   help="Force enable enhanced crawling (overrides USE_ENHANCED_CRAWLING env var)")
    p.add_argument("--baseline", action="store_true", 
                   help="Force disable enhanced crawling (overrides USE_ENHANCED_CRAWLING env var)")
    
    args = p.parse_args()
    
    # Handle enhanced crawling override flags
    global use_enhanced_crawling
    if args.enhanced and args.baseline:
        logger.error("Cannot specify both --enhanced and --baseline flags")
        return
    elif args.enhanced:
        use_enhanced_crawling = True
        logger.info("🚀 Enhanced crawling forced via --enhanced flag")
    elif args.baseline:
        use_enhanced_crawling = False
        logger.info("📊 Baseline crawling forced via --baseline flag")
    
    # Log current configuration
    if use_enhanced_crawling:
        logger.info("📋 Mode: Enhanced crawling with framework detection and quality validation")
    else:
        logger.info("📋 Mode: Baseline crawling (original functionality)")
    
    asyncio.run(
        _crawl_and_store(
            args.url,
            args.max_depth,
            args.max_concurrent,
            args.chunk_size,
            args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
