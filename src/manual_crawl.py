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

import argparse, asyncio, json, datetime, os, time, requests
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

# Handle relative imports - try both module and direct execution
try:
    from . import utils
except ImportError:
    # If running directly, add src to path and import without relative
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    import utils

importlib.reload(utils)
try:
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
except ImportError:
    # Direct execution fallback
    from utils import (
        get_supabase_client,
        add_documents_to_supabase,
        add_code_examples_to_supabase,
        extract_code_from_content,
    )
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

# Import AdvancedWebCrawler system
try:
    try:
        from .advanced_web_crawler import (
            AdvancedWebCrawler,
            AdvancedCrawlResult,
            crawl_single_page_advanced,
            batch_crawl_advanced
        )
        from .crawler_quality_validation import (
            ContentQualityValidator,
            validate_crawler_output,
            create_quality_report
        )
    except ImportError:
        # Direct execution fallback
        from advanced_web_crawler import (
            AdvancedWebCrawler,
            AdvancedCrawlResult,
            crawl_single_page_advanced,
            batch_crawl_advanced
        )
        from crawler_quality_validation import (
            ContentQualityValidator,
            validate_crawler_output,
            create_quality_report
        )
    ADVANCED_CRAWLER_AVAILABLE = True
    print("🚀 AdvancedWebCrawler system loaded for manual crawling")
except ImportError as e:
    ADVANCED_CRAWLER_AVAILABLE = False
    print(f"⚠️ AdvancedWebCrawler not available: {e}")

# Import DocumentIngestionPipeline system
try:
    try:
        from .document_ingestion_pipeline import (
            DocumentIngestionPipeline,
            PipelineConfig,
            ChunkingConfig,
            PipelineResult
        )
    except ImportError:
        # Direct execution fallback
        from document_ingestion_pipeline import (
            DocumentIngestionPipeline,
            PipelineConfig,
            ChunkingConfig,
            PipelineResult
        )
    DOCUMENT_PIPELINE_AVAILABLE = True
    print("🚀 DocumentIngestionPipeline system loaded for manual crawling")
except ImportError as e:
    DOCUMENT_PIPELINE_AVAILABLE = False
    print(f"⚠️ DocumentIngestionPipeline not available: {e}")

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
    use_advanced_crawler: bool = False,
    use_pipeline_integration: bool = False,
):
    """Main crawl function that dispatches to appropriate crawling method."""
    if use_advanced_crawler and use_pipeline_integration:
        logger.info("🎯 Using INTEGRATED AdvancedWebCrawler + DocumentIngestionPipeline (Task 14.6)")
        await _crawl_and_store_advanced_with_pipeline(url, max_depth, max_concurrent, chunk_size, batch_size)
    elif use_advanced_crawler:
        logger.info("🎯 Using AdvancedWebCrawler with legacy chunking")
        await _crawl_and_store_advanced_legacy(url, max_depth, max_concurrent, chunk_size, batch_size)
    elif use_enhanced_crawling:
        logger.info("🚀 Using enhanced crawling with framework detection")
        await _crawl_and_store_enhanced(url, max_depth, max_concurrent, chunk_size, batch_size)
    else:
        logger.info("📊 Using baseline crawling")
        await _crawl_and_store_baseline(url, max_depth, max_concurrent, chunk_size, batch_size)


async def _crawl_and_store_advanced_with_pipeline(
    url: str,
    max_depth: int,
    max_concurrent: int,
    chunk_size: int,
    batch_size: int,
):
    """
    Crawl and store using the integrated AdvancedWebCrawler + DocumentIngestionPipeline system.
    
    This is the complete implementation of Task 14.6 - integrating the clean markdown output
    from AdvancedWebCrawler with the DocumentIngestionPipeline for advanced processing.
    """
    
    if not ADVANCED_CRAWLER_AVAILABLE:
        logger.error("AdvancedWebCrawler not available - cannot proceed")
        return
    
    if not DOCUMENT_PIPELINE_AVAILABLE:
        logger.error("DocumentIngestionPipeline not available - falling back to basic storage")
        return await _crawl_and_store_advanced_legacy(url, max_depth, max_concurrent, chunk_size, batch_size)
    
    start_time = time.time()
    
    # Track results for comprehensive reporting
    all_crawler_results = []
    pipeline_results = []
    quality_results = []
    
    try:
        # 1️⃣ Determine URLs to crawl based on input type
        urls_to_crawl = []
        
        if is_sitemap(url):
            logger.info(f"📋 Parsing sitemap: {url}")
            urls_to_crawl = parse_sitemap(url)
            logger.info(f"📋 Found {len(urls_to_crawl)} URLs in sitemap")
        elif is_txt(url):
            logger.info(f"📋 Reading URL list from: {url}")
            resp = requests.get(url)
            if resp.status_code == 200:
                urls_to_crawl = [line.strip() for line in resp.text.split('\n') if line.strip()]
                logger.info(f"📋 Found {len(urls_to_crawl)} URLs in text file")
            else:
                logger.error(f"Failed to fetch text file: {resp.status_code}")
                return
        else:
            # Single URL
            urls_to_crawl = [url]
            logger.info(f"📋 Crawling single URL: {url}")
        
        if not urls_to_crawl:
            logger.error("No URLs to crawl")
            return
        
        # 2️⃣ Setup DocumentIngestionPipeline with optimal configuration
        pipeline_config = PipelineConfig(
            chunking=ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=min(200, chunk_size // 5),  # 20% overlap, max 200
                use_semantic_splitting=True,  # Enable LLM-powered semantic chunking
                max_chunk_size=chunk_size * 2
            ),
            generate_embeddings=True,    # Enable embedding generation
            store_in_database=True,      # Enable direct database storage
            extract_entities=False       # Disable for performance (can be enabled later)
        )
        
        pipeline = DocumentIngestionPipeline(pipeline_config)
        logger.info(f"🚀 DocumentIngestionPipeline initialized with semantic chunking and embedding generation")
        
        # 3️⃣ Crawl content using AdvancedWebCrawler
        if len(urls_to_crawl) > 1:
            logger.info(f"🎯 Starting batch crawl of {len(urls_to_crawl)} URLs with AdvancedWebCrawler")
            batch_results = await batch_crawl_advanced(
                urls_to_crawl, 
                max_concurrent=max_concurrent,
                enable_quality_validation=True
            )
            all_crawler_results.extend(batch_results)
        else:
            logger.info(f"🎯 Starting single URL crawl with AdvancedWebCrawler")
            result = await crawl_single_page_advanced(
                urls_to_crawl[0], 
                enable_quality_validation=True
            )
            all_crawler_results.append(result)
        
        # 4️⃣ Process each successful crawl result through DocumentIngestionPipeline
        total_chunks = 0
        total_embeddings = 0
        successful_crawls = 0
        failed_crawls = 0
        failed_pipeline_processing = 0
        
        for crawler_result in all_crawler_results:
            if not crawler_result.success:
                logger.warning(f"❌ Failed to crawl {crawler_result.url}: {crawler_result.error_message}")
                failed_crawls += 1
                continue
            
            successful_crawls += 1
            
            # Store quality validation result
            if crawler_result.quality_validation:
                quality_results.append(crawler_result.quality_validation)
            
            # Process through DocumentIngestionPipeline
            logger.info(f"🔄 Processing {crawler_result.url} through DocumentIngestionPipeline...")
            
            # Create comprehensive metadata from crawler results
            pipeline_metadata = {
                "crawler_type": "advanced_crawler",
                "framework": crawler_result.framework_detected,
                "extraction_time_ms": crawler_result.extraction_time_ms,
                "has_dynamic_content": crawler_result.has_dynamic_content,
                "content_ratio": crawler_result.content_to_navigation_ratio,
                "manual_run": True,
                "crawl_time": datetime.datetime.now(UTC).isoformat(),
            }
            
            # Add quality validation metrics
            if crawler_result.quality_validation:
                pipeline_metadata.update({
                    "quality_score": crawler_result.quality_score,
                    "quality_category": crawler_result.quality_validation.category,
                    "quality_passed": crawler_result.quality_passed,
                    "html_artifacts": crawler_result.quality_validation.html_artifacts_found,
                    "script_contamination": crawler_result.quality_validation.script_contamination,
                })
            
            try:
                # Process document through the complete pipeline
                pipeline_result = await pipeline.process_document(
                    content=crawler_result.markdown,
                    source_url=crawler_result.url,
                    metadata=pipeline_metadata
                )
                
                pipeline_results.append(pipeline_result)
                
                if pipeline_result.success:
                    total_chunks += pipeline_result.chunks_created
                    total_embeddings += pipeline_result.embeddings_generated
                    
                    logger.info(f"✅ Pipeline processed {crawler_result.url}: "
                              f"{crawler_result.word_count} words → {pipeline_result.chunks_created} chunks "
                              f"→ {pipeline_result.embeddings_generated} embeddings "
                              f"(quality: {crawler_result.quality_score:.3f})")
                else:
                    failed_pipeline_processing += 1
                    logger.error(f"❌ Pipeline failed for {crawler_result.url}: {pipeline_result.errors}")
                    
            except Exception as e:
                failed_pipeline_processing += 1
                logger.error(f"❌ Error processing {crawler_result.url} through pipeline: {str(e)}")
        
        # 5️⃣ Generate comprehensive quality report
        if quality_results:
            logger.info("\n" + "="*60)
            logger.info("📊 QUALITY VALIDATION REPORT")
            logger.info("="*60)
            quality_report = create_quality_report(quality_results)
            print(quality_report)
        
        # 6️⃣ Generate pipeline processing report
        if pipeline_results:
            logger.info("\n" + "="*60)
            logger.info("🚀 DOCUMENT INGESTION PIPELINE REPORT")
            logger.info("="*60)
            
            successful_pipeline = [r for r in pipeline_results if r.success]
            failed_pipeline = [r for r in pipeline_results if not r.success]
            
            if successful_pipeline:
                avg_processing_time = sum(r.processing_time_ms for r in successful_pipeline) / len(successful_pipeline)
                avg_chunks_per_doc = sum(r.chunks_created for r in successful_pipeline) / len(successful_pipeline)
                avg_embeddings_per_doc = sum(r.embeddings_generated for r in successful_pipeline) / len(successful_pipeline)
                
                logger.info(f"✅ Successful pipeline processing: {len(successful_pipeline)}")
                logger.info(f"📊 Average processing time: {avg_processing_time:.1f} ms")
                logger.info(f"📊 Average chunks per document: {avg_chunks_per_doc:.1f}")
                logger.info(f"📊 Average embeddings per document: {avg_embeddings_per_doc:.1f}")
            
            if failed_pipeline:
                logger.info(f"❌ Failed pipeline processing: {len(failed_pipeline)}")
                for result in failed_pipeline:
                    logger.info(f"   - {result.document_id}: {result.errors}")
        
        # 7️⃣ Final comprehensive summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("🎯 INTEGRATED CRAWLER + PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"URLs attempted: {len(all_crawler_results)}")
        logger.info(f"Successful crawls: {successful_crawls}")
        logger.info(f"Failed crawls: {failed_crawls}")
        logger.info(f"Successful pipeline processing: {len([r for r in pipeline_results if r.success])}")
        logger.info(f"Failed pipeline processing: {failed_pipeline_processing}")
        logger.info(f"Total chunks created: {total_chunks}")
        logger.info(f"Total embeddings generated: {total_embeddings}")
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Average time per URL: {elapsed_time/len(all_crawler_results):.2f} seconds")
        
        if quality_results:
            avg_quality = sum(r.score for r in quality_results) / len(quality_results)
            passed_count = sum(1 for r in quality_results if r.passed)
            logger.info(f"Average quality score: {avg_quality:.3f}")
            logger.info(f"Quality validation passed: {passed_count}/{len(quality_results)} ({passed_count/len(quality_results)*100:.1f}%)")
        
        logger.info("🎉 Integrated AdvancedWebCrawler + DocumentIngestionPipeline processing complete!")
        
        # Return summary for external use
        return {
            "success": True,
            "urls_attempted": len(all_crawler_results),
            "successful_crawls": successful_crawls,
            "failed_crawls": failed_crawls,
            "successful_pipeline_processing": len([r for r in pipeline_results if r.success]),
            "failed_pipeline_processing": failed_pipeline_processing,
            "total_chunks": total_chunks,
            "total_embeddings": total_embeddings,
            "processing_time_seconds": elapsed_time,
            "average_quality_score": sum(r.score for r in quality_results) / len(quality_results) if quality_results else None
        }
        
    except Exception as e:
        logger.error(f"❌ Error in integrated crawler + pipeline processing: {str(e)}")
        raise


async def _crawl_and_store_advanced_legacy(
    url: str,
    max_depth: int,
    max_concurrent: int,
    chunk_size: int,
    batch_size: int,
):
    """Crawl and store using the AdvancedWebCrawler system with legacy chunking (fallback)."""
    
    if not ADVANCED_CRAWLER_AVAILABLE:
        logger.error("AdvancedWebCrawler not available - cannot proceed")
        return
    
    start_time = time.time()
    supabase_client = get_supabase_client()
    
    # Track results for quality reporting
    all_results = []
    quality_results = []
    
    try:
        # Determine URLs to crawl based on input type
        urls_to_crawl = []
        
        if is_sitemap(url):
            logger.info(f"📋 Parsing sitemap: {url}")
            urls_to_crawl = parse_sitemap(url)
            logger.info(f"📋 Found {len(urls_to_crawl)} URLs in sitemap")
        elif is_txt(url):
            logger.info(f"📋 Reading URL list from: {url}")
            # Read text file content
            resp = requests.get(url)
            if resp.status_code == 200:
                urls_to_crawl = [line.strip() for line in resp.text.split('\n') if line.strip()]
                logger.info(f"📋 Found {len(urls_to_crawl)} URLs in text file")
            else:
                logger.error(f"Failed to fetch text file: {resp.status_code}")
                return
        else:
            # Single URL
            urls_to_crawl = [url]
            logger.info(f"📋 Crawling single URL: {url}")
        
        if not urls_to_crawl:
            logger.error("No URLs to crawl")
            return
        
        # Use batch crawling for multiple URLs
        if len(urls_to_crawl) > 1:
            logger.info(f"🎯 Starting batch crawl of {len(urls_to_crawl)} URLs with AdvancedWebCrawler")
            
            # Batch crawl with concurrency control
            batch_results = await batch_crawl_advanced(
                urls_to_crawl, 
                max_concurrent=max_concurrent,
                enable_quality_validation=True
            )
            all_results.extend(batch_results)
        else:
            # Single URL crawl
            logger.info(f"🎯 Starting single URL crawl with AdvancedWebCrawler")
            result = await crawl_single_page_advanced(
                urls_to_crawl[0], 
                enable_quality_validation=True
            )
            all_results.append(result)
        
        # Process results and store in Supabase
        total_chunks = 0
        successful_crawls = 0
        failed_crawls = 0
        
        for result in all_results:
            if not result.success:
                logger.warning(f"❌ Failed to crawl {result.url}: {result.error_message}")
                failed_crawls += 1
                continue
            
            successful_crawls += 1
            
            # Store quality validation result
            if result.quality_validation:
                quality_results.append(result.quality_validation)
            
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown, chunk_size=chunk_size)
            
            if not chunks:
                logger.warning(f"⚠️ No chunks created for {result.url}")
                continue
            
            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                urls.append(result.url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Enhanced metadata with AdvancedWebCrawler metrics
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = result.url
                meta["source"] = urlparse(result.url).netloc
                meta["crawl_type"] = "advanced_crawler"
                meta["title"] = result.title
                meta["framework"] = result.framework_detected
                meta["extraction_time_ms"] = result.extraction_time_ms
                meta["has_dynamic_content"] = result.has_dynamic_content
                meta["content_ratio"] = result.content_to_navigation_ratio
                
                # Add quality validation metrics
                if result.quality_validation:
                    meta["quality_score"] = result.quality_score
                    meta["quality_category"] = result.quality_validation.category
                    meta["quality_passed"] = result.quality_passed
                    meta["html_artifacts"] = result.quality_validation.html_artifacts_found
                    meta["script_contamination"] = result.quality_validation.script_contamination
                
                metadatas.append(meta)
            
            # Create url_to_full_document mapping
            url_to_full_document = {result.url: result.markdown}
            
            # Add to Supabase
            logger.info(f"💾 Storing {len(chunks)} chunks for {result.url}")
            add_documents_to_supabase(
                supabase_client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document,
                strategy_config,
                batch_size=batch_size,
            )
            
            total_chunks += len(chunks)
            
            logger.info(f"✅ Processed {result.url}: {result.word_count} words → {len(chunks)} chunks (quality: {result.quality_score:.3f})")
        
        # Generate quality report
        if quality_results:
            logger.info("\n" + "="*60)
            logger.info("📊 QUALITY VALIDATION REPORT")
            logger.info("="*60)
            
            quality_report = create_quality_report(quality_results)
            print(quality_report)
        
        # Final summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("🎯 ADVANCED CRAWLER SUMMARY")
        logger.info("="*60)
        logger.info(f"URLs attempted: {len(all_results)}")
        logger.info(f"Successful crawls: {successful_crawls}")
        logger.info(f"Failed crawls: {failed_crawls}")
        logger.info(f"Total chunks stored: {total_chunks}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Average time per URL: {elapsed_time/len(all_results):.2f} seconds")
        
        if quality_results:
            avg_quality = sum(r.score for r in quality_results) / len(quality_results)
            logger.info(f"Average quality score: {avg_quality:.3f}")
            passed_count = sum(1 for r in quality_results if r.passed)
            logger.info(f"Quality validation passed: {passed_count}/{len(quality_results)} ({passed_count/len(quality_results)*100:.1f}%)")
        
        logger.info("🎉 AdvancedWebCrawler processing complete!")
        
    except Exception as e:
        logger.error(f"❌ Error in advanced crawler processing: {str(e)}")
        raise


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
    p.add_argument("--advanced", action="store_true",
                   help="Use NEW AdvancedWebCrawler with Playwright and TrafilaturaExtractor")
    p.add_argument("--pipeline", action="store_true",
                   help="Use integrated AdvancedWebCrawler + DocumentIngestionPipeline (Task 14.6 implementation)")
    
    args = p.parse_args()
    
    # Handle crawling mode override flags
    global use_enhanced_crawling
    use_advanced_crawler = False
    use_pipeline_integration = False
    
    # Check for conflicting flags
    flags_set = sum([args.enhanced, args.baseline, args.advanced, args.pipeline])
    if flags_set > 1:
        logger.error("Cannot specify multiple crawling mode flags (--enhanced, --baseline, --advanced, --pipeline)")
        return
    
    # Set crawling mode
    if args.pipeline:
        if not ADVANCED_CRAWLER_AVAILABLE:
            logger.error("AdvancedWebCrawler not available - check imports")
            return
        if not DOCUMENT_PIPELINE_AVAILABLE:
            logger.error("DocumentIngestionPipeline not available - check imports")
            return
        use_advanced_crawler = True
        use_pipeline_integration = True
        use_enhanced_crawling = False
        logger.info("🎯 INTEGRATED AdvancedWebCrawler + DocumentIngestionPipeline mode (Task 14.6)")
    elif args.advanced:
        if not ADVANCED_CRAWLER_AVAILABLE:
            logger.error("AdvancedWebCrawler not available - check imports")
            return
        use_advanced_crawler = True
        use_enhanced_crawling = False  # Advanced crawler is separate
        logger.info("🎯 AdvancedWebCrawler with legacy chunking mode")
    elif args.enhanced:
        use_enhanced_crawling = True
        logger.info("🚀 Enhanced crawling forced via --enhanced flag")
    elif args.baseline:
        use_enhanced_crawling = False
        logger.info("📊 Baseline crawling forced via --baseline flag")
    
    # Log current configuration
    if use_advanced_crawler and use_pipeline_integration:
        logger.info("📋 Mode: INTEGRATED AdvancedWebCrawler + DocumentIngestionPipeline with semantic chunking, embeddings, and database storage")
    elif use_advanced_crawler:
        logger.info("📋 Mode: AdvancedWebCrawler with Playwright, TrafilaturaExtractor, and quality validation (legacy chunking)")
    elif use_enhanced_crawling:
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
            use_advanced_crawler,
            use_pipeline_integration,
        )
    )


if __name__ == "__main__":
    main()
