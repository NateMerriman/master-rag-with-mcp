#!/usr/bin/env python3
"""
Manual crawler / chunker / Supabase loader for big jobs that would time-out
when triggered through the MCP tool layer.

This script supports all enhanced RAG strategies via environment variable configuration.
All enhancements from the MCP server are available during manual crawling:
- Contextual embeddings (USE_CONTEXTUAL_EMBEDDINGS=true)
- Agentic RAG code extraction (USE_AGENTIC_RAG=true) 
- Enhanced hybrid search capabilities
"""

import argparse, asyncio, json, datetime
from urllib.parse import urlparse
from tqdm import tqdm
import datetime, zoneinfo  # add zoneinfo
from pathlib import Path
from dotenv import load_dotenv

UTC = zoneinfo.ZoneInfo("UTC")  # single timezone object

# Load environment variables
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path, override=True)

# The yellow “import crawl4ai could not be resolved” comes from Pylance on your Mac.
# It happens because the crawl4ai package lives only inside the Docker image, not in your local Python env. The warning is harmless; the import will work once the code runs in the container.
from crawl4ai import AsyncWebCrawler, BrowserConfig

# keep Supabase helpers from utils.py (these now include all enhancements)
from utils import get_supabase_client, add_documents_to_supabase

# bring the crawl helpers in from crawl4ai_mcp.py
from crawl4ai_mcp import (
    is_sitemap,
    is_txt,
    parse_sitemap,
    crawl_markdown_file,
    crawl_batch,
    crawl_recursive_internal_links,
    smart_chunk_markdown,
    extract_section_info,
)

import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,  # <-- this clears earlier handlers
)
logger = logging.getLogger(__name__)

# Load and validate configuration for enhanced features
try:
    from config import get_config, ConfigurationError
    strategy_config = get_config()
    logger.info(f"✅ {strategy_config}")
    
    # Log enabled strategies for manual crawling
    enabled_strategies = strategy_config.get_enabled_strategies()
    if enabled_strategies:
        strategy_names = [s.value for s in enabled_strategies]
        logger.info(f"📊 Manual crawl with enhanced RAG strategies: {', '.join(strategy_names)}")
    else:
        logger.info("📊 Manual crawl in baseline mode (no enhanced strategies)")
        
except (ConfigurationError, ImportError) as e:
    logger.warning(f"⚠️ Configuration Warning: {e}")
    logger.info("📊 Manual crawl will use baseline functionality only")
    strategy_config = None


DEFAULT_CHUNK_SIZE = 5_000
DEFAULT_BATCH_SIZE = 20
DEFAULT_DEPTH = 3
DEFAULT_CONCURRENCY = 10


async def _crawl_and_store(
    url: str,
    max_depth: int,
    max_concurrent: int,
    chunk_size: int,
    batch_size: int,
):
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

        # 2️⃣ pre-scan to know how many chunks we’ll handle ---------------------------
        total_chunks = 0
        for p in pages:
            total_chunks += len(
                smart_chunk_markdown(p["markdown"], chunk_size=chunk_size)
            )

        page_bar = tqdm(pages, desc="pages", unit="page")
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
                        ).isoformat(),  # no warning
                    }
                )
                chunk_bar.update(1)  # <-- advance chunk bar
        chunk_bar.close()

        # 4️⃣ Store in Supabase ----------------------------------------------
        add_documents_to_supabase(
            supabase,
            urls,
            idxs,
            contents,
            metas,
            {p["url"]: p["markdown"] for p in pages},
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


def main() -> None:
    p = argparse.ArgumentParser(description="Manual Crawl → Chunk → Supabase")
    p.add_argument("--url", required=True, help="Target URL or sitemap.txt")
    p.add_argument("--max-depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--max-concurrent", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = p.parse_args()
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
