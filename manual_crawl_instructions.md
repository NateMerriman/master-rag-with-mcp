# Manual Crawl – Quick‑Start Guide

> **Goal:** run `manual_crawl.py` inside the *crawl4ai‑rag‑mcp* Docker container and push fresh chunks into the `crawled_pages` table in Supabase.
>
> These steps assume you already have:
>
> * The project repo with its `Dockerfile` and `.env`.
> * Supabase CLI stack running on `http://localhost:54321`.
> * `crawl4ai-rag-mcp:latest` image built (or ready to build).
>
> Follow every step **in order**—skip nothing.

---

## 1  Build (or rebuild) the image

```bash
# from the repo root
docker build -t crawl4ai-rag-mcp:latest .
```

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

## 4  Run `manual_crawl.py`

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
* **`--batch-size`** – how many chunks to upsert per Supabase request (default `10`).

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

Leave the container shell:

```bash
exit
```

---

## 5  Confirm rows in Supabase

Run in any psql window or Supabase Studio:

```sql
SELECT url, chunk_number, metadata->>'chunk_index' AS chunk
FROM crawled_pages
ORDER BY id DESC
LIMIT 10;
```

You should see your recent crawl at the top.

---

## Troubleshooting

| Symptom                                                                          | Fix                                                                                                               |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **`FATAL: could not write lock file "postmaster.pid": No space left on device`** | Free Docker disk space with `docker system prune -af` *or* enlarge Docker Desktop’s disk image.                   |
| ImportError: `No module named crawler`                                           | Make sure you’re inside the container **and** running `python src/manual_crawl.py` (not `python -m crawler.cli`). |
| Supabase 400 “missing column chunk\_number”                                      | Column is present in new schema; double‑check you ran the latest migration.                                       |

---

## Recap

1. **Build** image → 2. **Run** container → 3. **Exec** shell → 4. **Call** `manual_crawl.py` with the right flags → 5. **Verify** rows.

That’s all you need to keep your `crawled_pages` shelf stocked for hybrid RAG 🎉
