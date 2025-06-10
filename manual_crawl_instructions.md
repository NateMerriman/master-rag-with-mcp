# Manual Crawl – Enhanced RAG Quick‑Start Guide

> **Goal:** run `manual_crawl.py` inside the *crawl4ai‑rag‑mcp* Docker container with full enhanced RAG capabilities and push fresh content into Supabase with all strategies applied.
>
> **✨ Enhanced Features Available:**
> * **Contextual Embeddings** - Document-level context for improved semantic understanding
> * **Agentic RAG Code Extraction** - Automatic code detection, extraction, and specialized storage
> * **Advanced Database Architecture** - Sources table integration and foreign key constraints
> * **Enhanced Hybrid Search** - All content optimized for advanced retrieval strategies
>
> These steps assume you already have:
>
> * The project repo with its `Dockerfile` and `.env` configured with enhanced strategies
> * Supabase CLI stack running on `http://localhost:54321` with enhanced schema
> * `crawl4ai-rag-mcp:latest` image built (or ready to build)
>
> Follow every step **in order**—skip nothing.

---

## 1  Build (or rebuild) the image

```bash
# from the repo root
docker build -t crawl4ai-rag-mcp:latest --build-arg PORT=8051 .
```

**⚠️ Important:** Make sure to rebuild the image if you haven't updated it since enabling enhanced strategies. The new image includes `sentence-transformers` and other enhanced dependencies.

If the build fails, check your Docker disk space (see *Troubleshooting* at the bottom).

---

## 2  Start the container

```bash
# stop/remove any previous instance (safe if none exist)
docker stop  crawl4ai-rag-mcp-container || true
docker rm    crawl4ai-rag-mcp-container || true

# run the server + crawler tooling, publishing port 8051
# adjust paths/ports only if you know why
docker run -d \
  --name crawl4ai-rag-mcp-container \
  --env-file .env \
  -p 8051:8051 \
  crawl4ai-rag-mcp:latest
```


✅ **Verify:**

```bash
docker ps --filter "name=crawl4ai-rag-mcp-container"
```

Should show `0.0.0.0:8051->8051/tcp` and a recent *Up* status.

---

## 3  Open a shell inside the container

```bash
docker exec -it crawl4ai-rag-mcp-container bash
```

The prompt should change to `root@<container-id>:/app#`.

---

## 4  Run `manual_crawl.py` with Enhanced RAG Strategies

The manual crawler now automatically applies all configured RAG strategies from your `.env` file.

Basic syntax:

```bash
python src/manual_crawl.py \
  --url <START_URL> \
  [--max-depth N] \
  [--chunk-size BYTES] \
  [--batch-size M]
```

Where:

* **`--url`** *(required)* – page or site to crawl.
* **`--max-depth`** – how many link‑levels deep to follow (default `3`).

  * `1` ⇒ crawl only the start URL.
* **`--chunk-size`** – split text into chunks of this size (default `5000`).
* **`--batch-size`** – how many chunks to upsert per Supabase request (default `20`).

### Enhanced Processing Features

When you run manual crawling, it automatically applies your configured strategies:

**✅ If `USE_CONTEXTUAL_EMBEDDINGS=true`:**
- Generates contextual summaries for each chunk using your `CONTEXTUAL_MODEL`
- Creates enhanced embeddings with document-level context

**✅ If `USE_AGENTIC_RAG=true`:**
- Automatically detects and extracts code examples (18+ programming languages)
- Stores code with dual embeddings (code content + natural language summary)
- Assigns complexity scores (1-10 scale) and language detection
- Links code examples to sources table via foreign keys

**✅ Database Integration:**
- Automatically populates sources table for new domains
- Maintains foreign key relationships with 100% data integrity
- Optimized for enhanced hybrid search with all new indexing

### 4.1  Variants & examples

| Use‑case                                      | Command                                                                                                          |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Single HTML page** (force exactly one page) | `python src/manual_crawl.py --url https://example.com/post/123 --max-depth 1`                                    |
| **Plain text file** (`llms-full.txt`)         | `python src/manual_crawl.py --url https://modelcontextprotocol.io/llms-full.txt --max-depth 1 --chunk-size 8000` |
| **Full site crawl** (default recursion)       | `python src/manual_crawl.py --url https://www.anthropic.com --chunk-size 4000`                                   |
| **Cap depth to 2 levels**                     | `python src/manual_crawl.py --url https://docs.python.org --max-depth 2`                                         |

> **Tip:** depth `N` crawls pages that are **≤ N–1 clicks** away from the start URL.

### 4.2  What success looks like

You’ll see three phases per URL:

```
[FETCH]... ↓ <url> | ✓ | ⏱: 1.23s
[SCRAPE].. ◆ <url> | ✓ | ⏱: 0.04s
[COMPLETE] ● <url> | ✓ | ⏱: 1.27s
```

Then chunking progress bars, OpenAI embedding calls, and a final JSON summary:

```json
{"success": true, "pages_crawled": 12, "chunks_stored": 98}
```

**Enhanced output when strategies enabled:**
```
INFO:__main__:✅ StrategyConfig(contextual_embeddings=True, reranking=True, agentic_rag=True, hybrid_search_enhanced=True)
INFO:__main__:📊 Manual crawl with enhanced RAG strategies: contextual_embeddings, reranking, agentic_rag, hybrid_search_enhanced
🔧 Extracted 15 code examples from https://docs.example.com
🔧 Extracted 8 code examples from https://api.example.com
```

Leave the container shell:

```bash
exit
```

---

## 5  Confirm Enhanced Data in Supabase

### 5.1 Verify crawled content with enhanced metadata

Run in any psql window or Supabase Studio:

```sql
SELECT url, chunk_number, 
       metadata->>'chunk_index' AS chunk,
       metadata->>'contextual_embedding' AS has_context,
       metadata->>'manual_run' AS is_manual
FROM crawled_pages
ORDER BY id DESC
LIMIT 10;
```

### 5.2 Check extracted code examples (if agentic RAG enabled)

```sql
SELECT ce.programming_language, ce.complexity_score, 
       s.url as source_url, 
       LEFT(ce.code_content, 100) as code_preview
FROM code_examples ce
JOIN sources s ON ce.source_id = s.source_id
ORDER BY ce.created_at DESC
LIMIT 10;
```

### 5.3 Verify sources table integration

```sql
SELECT s.url, s.total_word_count, 
       COUNT(cp.id) as chunks_count,
       COUNT(ce.id) as code_examples_count
FROM sources s
LEFT JOIN crawled_pages cp ON cp.source_id = s.source_id
LEFT JOIN code_examples ce ON ce.source_id = s.source_id
GROUP BY s.source_id, s.url, s.total_word_count
ORDER BY s.created_at DESC
LIMIT 5;
```

You should see your recent crawl data with enhanced features applied.

---

## Troubleshooting

| Symptom                                                                          | Fix                                                                                                               |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **`FATAL: could not write lock file "postmaster.pid": No space left on device`** | Free Docker disk space with `docker system prune -af` *or* enlarge Docker Desktop’s disk image.                   |
| ImportError: `No module named crawler`                                           | Make sure you’re inside the container **and** running `python src/manual_crawl.py` (not `python -m crawler.cli`). |
| Supabase 400 “missing column chunk\_number”                                      | Column is present in new schema; double‑check you ran the latest migration.                                       |

### Enhanced Strategy Troubleshooting

| Enhanced Feature Issue                                                           | Solution                                                                                                          |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **`ModuleNotFoundError: No module named 'sentence_transformers'`**              | Rebuild Docker image - enhanced dependencies not installed. Run `docker build -t crawl4ai-rag-mcp:latest .`      |
| **`No source_id found for URL`** warnings                                        | Normal for new domains - sources table automatically populated on first crawl of each domain.                   |
| **Configuration warnings about missing strategies**                              | Expected if running with baseline config - strategies are optional enhancements.                                 |
| **`Error extracting code from URL`** messages                                    | Normal when `USE_AGENTIC_RAG=false` or content has no detectable code blocks.                                    |
| **`StrategyConfig` import errors**                                               | Configuration system disabled - manual crawler falls back to baseline functionality.                            |

---

## Recap

1. **Build** image → 2. **Run** container → 3. **Exec** shell → 4. **Call** `manual_crawl.py` with the right flags → 5. **Verify** rows.

That’s all you need to keep your `crawled_pages` shelf stocked with enhanced RAG strategies! 🚀✨

**Enhanced Features Now Available:**
- ✅ **Contextual embeddings** for better semantic understanding
- ✅ **Automatic code extraction** with dual embeddings  
- ✅ **Advanced database architecture** with relational integrity
- ✅ **Cross-encoder reranking** ready for MCP queries
