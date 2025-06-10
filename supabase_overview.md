# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a self-hosted Supabase Docker setup with hybrid search capabilities. The project provides a complete Supabase stack with four specialized edge functions for semantic search using OpenAI embeddings and Postgres vector similarity.

## Architecture

### Core Components
- **Supabase Services**: Dockerized Supabase stack including Auth, Database, Storage, Realtime, and Studio
- **Edge Functions**: Four Deno-based functions providing semantic search via OpenAI embeddings
- **Postgres RPC Functions**: Backend functions for hybrid search (`hybrid_search`, `hybrid_search_career_docs`, `hybrid_search_crawled_pages`, `hybrid_search_code_examples`)

### Edge Functions
All edge functions follow the same pattern:
1. Accept HTTP POST requests with `query` and optional parameters
2. Generate embeddings using OpenAI's `text-embedding-3-small` model (1536 dimensions)
3. Call corresponding Postgres RPC function with query text and embeddings
4. Return ranked search results using Reciprocal Rank Fusion (RRF)

- **hybrid-search**: General purpose semantic search
- **hybrid-search-career**: Career-specific document search  
- **hybrid-search-crawled-pages**: Web-crawled content search
- **hybrid-search-code-examples**: Code examples search with language and complexity filtering

## Development Commands

### Starting Services
```bash
# Start all services (Supabase + all edge functions)
./start-supabase-all.sh

# Start only specific edge functions
./start-supabase-all.sh --hybrid-search
./start-supabase-all.sh --hybrid-search-career  
./start-supabase-all.sh --hybrid-search-crawled-pages
./start-supabase-all.sh --hybrid-search-code-examples

# Start multiple specific functions
./start-supabase-all.sh --hybrid-search --hybrid-search-career
./start-supabase-all.sh --hybrid-search-code-examples --hybrid-search-crawled-pages
```

### Stopping Services
```bash
# Stop all services
./stop-supabase-all.sh

# Stop specific edge functions (same flag pattern as start script)
./stop-supabase-all.sh --hybrid-search-career
./stop-supabase-all.sh --hybrid-search-code-examples
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
- `hybrid_search_code_examples(query_text, query_embedding, match_count, language_filter, max_complexity)`

These functions perform vector similarity search combined with text search on their respective document tables using Reciprocal Rank Fusion (RRF) scoring.

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

## Code Examples Search System

The `hybrid-search-code-examples` function provides specialized search for code examples with advanced filtering capabilities:

### Features
- **Language Filtering**: Filter results by programming language (e.g., "python", "javascript", "rust")
- **Complexity Filtering**: Filter by complexity score (1-10 scale) to find examples appropriate for skill level
- **Dual Search Modes**: Supports both code similarity search and natural language queries
- **RRF Scoring**: Uses Reciprocal Rank Fusion to combine full-text and semantic search results

### Request Format
```json
{
  "query": "sorting algorithm",
  "match_count": 5,
  "language_filter": "python", 
  "max_complexity": 5
}
```

### Parameters
- `query` (required): Search query text
- `match_count` (optional): Number of results to return (default: 10)
- `language_filter` (optional): Programming language filter (default: null)
- `max_complexity` (optional): Maximum complexity score 1-10 (default: 10)

### Example Usage
```bash
curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/hybrid-search-code-examples' \
  --header 'Authorization: Bearer [anon-key]' \
  --header 'Content-Type: application/json' \
  --data '{
    "query": "array sorting algorithm",
    "match_count": 5,
    "language_filter": "python",
    "max_complexity": 5
  }'
```

### Database Schema
The function operates on the `code_examples` table with the following key columns:
- `code_content`: The actual code snippet
- `summary`: Natural language description
- `programming_language`: Language identifier for filtering
- `complexity_score`: Numeric complexity rating (1-10)
- `embedding`: Vector embedding for semantic search
- `fts`: Full-text search vector for keyword matching

### Hybrid Search Crawled Pages Index.ts (code reference)

```
// supabase/functions/hybrid-search-crawled-pages/index.ts
// Edge function for hybrid search on crawled pages using OpenAI embeddings
import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// CORS headers - important for browser invocation
const corsHeaders = {
  "Access-Control-Allow-Origin": "*", // Or specific origin
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS", // Added OPTIONS and POST
};

// Function to get embeddings from OpenAI
async function getOpenAIEmbedding(
  text: string,
  apiKey: string,
): Promise<number[]> {
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      input: text,
      model: "text-embedding-3-small", // Matches your specified model
    }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    console.error("OpenAI API Error Data:", errorData);
    throw new Error(
      `OpenAI API error: ${response.status} ${
        errorData.error?.message || "Unknown error"
      }`,
    );
  }

  const { data } = await response.json();
  if (!data || !data[0] || !data[0].embedding) {
    console.error("OpenAI API Invalid Response Structure:", data);
    throw new Error(
      "Failed to get embedding from OpenAI or embedding format is incorrect",
    );
  }
  return data[0].embedding; // This should be a 1536-dimension array
}

serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const body = await req.json();
    const query = body.query;
    const match_count = body.match_count || 10; // Default match_count to 10

    if (!query) {
      return new Response(
        JSON.stringify({ error: "Missing query parameter" }),
        {
          status: 400,
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        },
      );
    }

    // Get OpenAI API Key from environment variables
    const openAIKey = Deno.env.get("OPENAI_API_KEY");
    if (!openAIKey) {
      console.error("OPENAI_API_KEY environment variable not set");
      return new Response(
        JSON.stringify({
          error: "OPENAI_API_KEY environment variable not set",
        }),
        {
          status: 500,
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        },
      );
    }

    // Generate embedding for the query using OpenAI
    const query_embedding = await getOpenAIEmbedding(query, openAIKey);

    // Create Supabase client
    // Ensure SUPABASE_URL and SUPABASE_ANON_KEY are set as environment variables.
    // For local development, these can be in `supabase/.env.local`
    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const supabaseAnonKey = Deno.env.get("SUPABASE_ANON_KEY");

    if (!supabaseUrl || !supabaseAnonKey) {
      console.error(
        "SUPABASE_URL or SUPABASE_ANON_KEY environment variable not set",
      );
      return new Response(
        JSON.stringify({
          error: "Supabase environment variables not set",
        }),
        {
          status: 500,
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        },
      );
    }

    const supabaseClient = createClient(
      supabaseUrl,
      supabaseAnonKey,
      {
        global: {
          headers: {
            Authorization: req.headers.get("Authorization")!,
          },
        },
      }, // Pass auth header for RLS
    );

    // Call the RPC function - CHANGED TO USE hybrid_search_crawled_pages
    const { data, error } = await supabaseClient.rpc(
      "hybrid_search_crawled_pages",
      {
        query_text: query,
        query_embedding: query_embedding, // This will be the 1536-dim vector from OpenAI
        match_count: match_count,
      },
    );

    if (error) {
      console.error(
        "Error calling hybrid_search_crawled_pages RPC:",
        error,
      );
      return new Response(
        JSON.stringify({
          error: error.message,
          details: error.details || null,
        }),
        {
          status: 500,
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        },
      );
    }

    return new Response(JSON.stringify(data), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });
  } catch (e) {
    console.error("Error in Edge Function:", e);
    // Check if the error is an instance of Error and has a message property
    const errorMessage = e instanceof Error
      ? e.message
      : "An unexpected error occurred";
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500,
    });
  }
});

/* To invoke locally:

  1. Run `supabase start` (see: https://supabase.com/docs/reference/cli/supabase-start)
  2. Make an HTTP request:

  curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/hybrid-search-crawled-pages' \
    --header 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0' \
    --header 'Content-Type: application/json' \
    --data '{"query":"your search query here"}'

*/
```

### Hybrid Search Code Examples Index.ts (code reference)

```
// supabase/functions/hybrid-search-code-examples/index.ts
// Edge function for hybrid search on code examples using OpenAI embeddings
import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// CORS headers - important for browser invocation
const corsHeaders = {
  "Access-Control-Allow-Origin": "*", // Or specific origin
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS", // Added OPTIONS and POST
};

// Function to get embeddings from OpenAI
async function getOpenAIEmbedding(
  text: string,
  apiKey: string,
): Promise<number[]> {
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      input: text,
      model: "text-embedding-3-small", // Matches your specified model
    }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    console.error("OpenAI API Error Data:", errorData);
    throw new Error(
      `OpenAI API error: ${response.status} ${
        errorData.error?.message || "Unknown error"
      }`,
    );
  }

  const { data } = await response.json();
  if (!data || !data[0] || !data[0].embedding) {
    console.error("OpenAI API Invalid Response Structure:", data);
    throw new Error(
      "Failed to get embedding from OpenAI or embedding format is incorrect",
    );
  }
  return data[0].embedding; // This should be a 1536-dimension array
}

serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const body = await req.json();
    const query = body.query;
    const match_count = body.match_count || 10; // Default match_count to 10
    const language_filter = body.language_filter || null; // Optional language filter
    const max_complexity = body.max_complexity || 10; // Default max complexity to 10

    if (!query) {
      return new Response(
        JSON.stringify({ error: "Missing query parameter" }),
        {
          status: 400,
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        },
      );
    }

    // Get OpenAI API Key from environment variables
    const openAIKey = Deno.env.get("OPENAI_API_KEY");
    if (!openAIKey) {
      console.error("OPENAI_API_KEY environment variable not set");
      return new Response(
        JSON.stringify({
          error: "OPENAI_API_KEY environment variable not set",
        }),
        {
          status: 500,
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        },
      );
    }

    // Generate embedding for the query using OpenAI
    const query_embedding = await getOpenAIEmbedding(query, openAIKey);

    // Create Supabase client
    // Ensure SUPABASE_URL and SUPABASE_ANON_KEY are set as environment variables.
    // For local development, these can be in `supabase/.env.local`
    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const supabaseAnonKey = Deno.env.get("SUPABASE_ANON_KEY");

    if (!supabaseUrl || !supabaseAnonKey) {
      console.error(
        "SUPABASE_URL or SUPABASE_ANON_KEY environment variable not set",
      );
      return new Response(
        JSON.stringify({
          error: "Supabase environment variables not set",
        }),
        {
          status: 500,
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        },
      );
    }

    const supabaseClient = createClient(
      supabaseUrl,
      supabaseAnonKey,
      {
        global: {
          headers: {
            Authorization: req.headers.get("Authorization")!,
          },
        },
      }, // Pass auth header for RLS
    );

    // Call the RPC function for code examples
    const { data, error } = await supabaseClient.rpc(
      "hybrid_search_code_examples",
      {
        query_text: query,
        query_embedding: query_embedding, // This will be the 1536-dim vector from OpenAI
        match_count: match_count,
        language_filter: language_filter,
        max_complexity: max_complexity,
      },
    );

    if (error) {
      console.error(
        "Error calling hybrid_search_code_examples RPC:",
        error,
      );
      return new Response(
        JSON.stringify({
          error: error.message,
          details: error.details || null,
        }),
        {
          status: 500,
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
          },
        },
      );
    }

    return new Response(JSON.stringify(data), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });
  } catch (e) {
    console.error("Error in Edge Function:", e);
    // Check if the error is an instance of Error and has a message property
    const errorMessage = e instanceof Error
      ? e.message
      : "An unexpected error occurred";
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 500,
    });
  }
});

/* To invoke locally:

  1. Run `supabase start` (see: https://supabase.com/docs/reference/cli/supabase-start)
  2. Make an HTTP request:

  curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/hybrid-search-code-examples' \
    --header 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0' \
    --header 'Content-Type: application/json' \
    --data '{"query":"array sorting algorithm","language_filter":"python","max_complexity":5}'

*/
```