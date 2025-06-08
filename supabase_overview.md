# supabase_overview.md

This file provides an overview of my connected Supabase repository.

## Project Overview

This is a self-hosted Supabase Docker setup with hybrid search capabilities. The project provides a complete Supabase stack with three specialized edge functions for semantic search using OpenAI embeddings and Postgres vector similarity.

## Architecture

### Core Components
- **Supabase Services**: Dockerized Supabase stack including Auth, Database, Storage, Realtime, and Studio
- **Edge Functions**: Three Deno-based functions providing semantic search via OpenAI embeddings
- **Postgres RPC Functions**: Backend functions for hybrid search (`hybrid_search`, `hybrid_search_career_docs`, `hybrid_search_crawled_pages`)

### Edge Functions
All edge functions follow the same pattern:
1. Accept HTTP POST requests with `query` and optional `match_count` parameters
2. Generate embeddings using OpenAI's `text-embedding-3-small` model (1536 dimensions)
3. Call corresponding Postgres RPC function with query text and embeddings
4. Return ranked search results

- **hybrid-search**: General purpose semantic search
- **hybrid-search-career**: Career-specific document search
- **hybrid-search-crawled-pages**: Web-crawled content search

## Development Commands

### Starting Services
```bash
# Start all services (Supabase + all edge functions)
./start-supabase-all.sh

# Start only specific edge functions
./start-supabase-all.sh --hybrid-search
./start-supabase-all.sh --hybrid-search-career  
./start-supabase-all.sh --hybrid-search-crawled-pages

# Start multiple specific functions
./start-supabase-all.sh --hybrid-search --hybrid-search-career
```

### Stopping Services
```bash
# Stop all services
./stop-supabase-all.sh

# Stop specific edge functions (same flag pattern as start script)
./stop-supabase-all.sh --hybrid-search-career
```

### Docker Operations
```bash
# Manual Docker control
docker compose up
docker compose down
docker compose -f docker-compose.yml -f ./dev/docker-compose.dev.yml up

# Reset everything
./reset.sh
```

### Supabase CLI
```bash
# Start Supabase services only
supabase start

# Serve edge functions manually
supabase functions serve --env-file ./supabase/.env.local [function-names]

# Stop Supabase services
supabase stop
```

## Configuration

### Environment Variables
Required in `./supabase/.env.local`:
- `OPENAI_API_KEY`: For embedding generation
- `SUPABASE_URL`: Supabase instance URL  
- `SUPABASE_ANON_KEY`: Supabase anonymous key

### Service Ports
- Studio: http://localhost:54323
- API: http://localhost:54321
- Database: localhost:54322
- Inbucket (email testing): http://localhost:54324

## Edge Function Development

### Function Structure
Each edge function in `supabase/functions/[name]/`:
- `index.ts`: Main function code
- `deno.json`: Import map and dependencies
- `deno.lock`: Dependency lockfile

### Adding New Edge Functions
1. Create function directory in `supabase/functions/`
2. Add function configuration to `supabase/config.toml`
3. Update start/stop scripts to include new function
4. Ensure corresponding Postgres RPC function exists

### Testing Edge Functions
```bash
curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/[function-name]' \
  --header 'Authorization: Bearer [anon-key]' \
  --header 'Content-Type: application/json' \
  --data '{"query":"search terms","match_count":10}'
```

## Database Integration

Edge functions call Postgres RPC functions that must exist in the database:
- `hybrid_search(query_text, query_embedding, match_count)`
- `hybrid_search_career_docs(query_text, query_embedding, match_count)`  
- `hybrid_search_crawled_pages(query_text, query_embedding, match_count)`

These functions perform vector similarity search combined with text search on their respective document tables.

## Connection to MCP Crawl4AI RAG Project

This Supabase database serves as the vector storage backend for the MCP Crawl4AI RAG server (separate repository). The connection includes:

- **Primary Table**: `crawled_pages` - stores web content chunks with embeddings, full-text search vectors, and metadata
- **Hybrid Search Function**: `hybrid_search_crawled_pages` - RPC function combining semantic and keyword search using Reciprocal Rank Fusion
- **Data Flow**: The MCP server crawls websites, chunks content, generates OpenAI embeddings, and stores everything here
- **External Integrations**: This database also connects to n8n agentic workflows running in Docker containers

### Schema Dependencies
- Vector embeddings (1536 dimensions for OpenAI text-embedding-3-small)
- PostgreSQL FTS with English language support
- JSONB metadata storage with GIN indexing
- HNSW vector indexing for semantic search

**Important**: Changes to the database schema, indexes, or RPC functions may break the MCP server and connected n8n workflows. Always test modifications against the broader Master RAG Pipeline system.