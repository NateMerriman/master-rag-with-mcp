-- 🟢 1. Storage table
CREATE TABLE crawled_pages (
  id        BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  url       TEXT,
  chunk_number  INT,
  content   TEXT,
  metadata  JSONB,           -- e.g. {"site":"nytimes", "chunk":5}
  fts       TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  embedding VECTOR(1536)
);

-- 🟢 2. Indexes
CREATE INDEX ON crawled_pages USING gin(fts);
CREATE INDEX ON crawled_pages USING hnsw (embedding vector_ip_ops);
CREATE INDEX ON crawled_pages USING gin (metadata jsonb_path_ops);  -- filters on metadata

-- 🟢 3. Helper: semantic-only match (inner-product everywhere)
CREATE OR REPLACE FUNCTION match_crawled_pages (
  query_embedding VECTOR(1536),
  match_count     INT         DEFAULT NULL,
  filter          JSONB       DEFAULT '{}'
) RETURNS TABLE (
  id         BIGINT,
  content    TEXT,
  metadata   JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    p.id,
    p.content,
    p.metadata,
    1 - (p.embedding <#> query_embedding) AS similarity   -- switched to <#>
  FROM crawled_pages AS p
  WHERE p.metadata @> filter
  ORDER BY p.embedding <#> query_embedding
  LIMIT match_count;
END;
$$;

-- 🟢 4. Hybrid search (RRF) for crawled_pages
DROP TYPE IF EXISTS hybrid_search_crawled_pages_result CASCADE;

CREATE TYPE hybrid_search_crawled_pages_result AS (
  doc_id        BIGINT,
  content       TEXT,
  metadata      JSONB,
  full_text_rank BIGINT,
  semantic_rank  BIGINT,
  rrf_score     FLOAT
);

CREATE OR REPLACE FUNCTION hybrid_search_crawled_pages(
  query_text TEXT,
  query_embedding VECTOR(1536),
  match_count INT,
  full_text_weight FLOAT = 1.0,
  semantic_weight FLOAT = 1.0,
  rrf_k INT = 50
)
RETURNS SETOF hybrid_search_crawled_pages_result
LANGUAGE SQL
AS $$
WITH full_text AS (
  SELECT id,
         row_number() OVER (ORDER BY ts_rank_cd(fts, websearch_to_tsquery(query_text)) DESC) AS rank_ix
  FROM crawled_pages
  WHERE fts @@ websearch_to_tsquery(query_text)
  LIMIT LEAST(match_count, 30) * 2
),
semantic AS (
  SELECT id,
         row_number() OVER (ORDER BY embedding <#> query_embedding) AS rank_ix
  FROM crawled_pages
  LIMIT LEAST(match_count, 30) * 2
),
combined AS (
  SELECT COALESCE(full_text.id, semantic.id)           AS doc_id,
         full_text.rank_ix                             AS full_text_rank,
         semantic.rank_ix                              AS semantic_rank,
         (COALESCE(1.0 / (rrf_k + full_text.rank_ix), 0) * full_text_weight +
          COALESCE(1.0 / (rrf_k + semantic.rank_ix), 0) * semantic_weight) AS rrf_score
  FROM full_text
  FULL JOIN semantic ON full_text.id = semantic.id
)
SELECT c.doc_id, p.content, p.metadata,
       c.full_text_rank, c.semantic_rank, c.rrf_score
FROM combined c
JOIN crawled_pages p ON p.id = c.doc_id
ORDER BY c.rrf_score DESC
LIMIT LEAST(match_count, 30);
$$;

-- 🟢 5. RLS – turned off because only the service-role key will access this table
ALTER TABLE crawled_pages DISABLE ROW LEVEL SECURITY;