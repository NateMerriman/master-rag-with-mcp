-- RPC function for hybrid search on code examples using Reciprocal Rank Fusion
-- Adapted from hybrid_search_crawled_pages for code_examples table schema

-- First create the custom composite type for return results
CREATE TYPE hybrid_search_code_examples_result AS (
    id BIGINT,
    source_id INT,
    code_content TEXT,
    summary TEXT,
    programming_language TEXT,
    complexity_score INT,
    similarity FLOAT,
    rrf_score FLOAT,
    semantic_rank INT,
    full_text_rank INT
);

CREATE OR REPLACE FUNCTION hybrid_search_code_examples(
    query_text TEXT,
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 10,
    language_filter TEXT DEFAULT NULL,
    max_complexity INT DEFAULT 10
)
RETURNS SETOF hybrid_search_code_examples_result
LANGUAGE plpgsql
AS $$
DECLARE
    rrf_k INT := 60;
    full_text_weight FLOAT := 1.0;
    semantic_weight FLOAT := 1.0;
BEGIN
    RETURN QUERY
    WITH full_text AS (
        SELECT ce.id,
               row_number() OVER (ORDER BY ts_rank_cd(ce.fts, websearch_to_tsquery(query_text)) DESC) AS rank_ix
        FROM code_examples ce
        WHERE ce.fts @@ websearch_to_tsquery(query_text)
            AND (language_filter IS NULL OR ce.programming_language = language_filter)
            AND ce.complexity_score <= max_complexity
        LIMIT LEAST(match_count, 30) * 2
    ),
    semantic AS (
        SELECT ce.id,
               row_number() OVER (ORDER BY ce.embedding <#> query_embedding) AS rank_ix,
               1 - (ce.embedding <#> query_embedding) AS similarity
        FROM code_examples ce
        WHERE (language_filter IS NULL OR ce.programming_language = language_filter)
            AND ce.complexity_score <= max_complexity
        LIMIT LEAST(match_count, 30) * 2
    ),
    combined AS (
        SELECT COALESCE(full_text.id, semantic.id) AS doc_id,
               full_text.rank_ix AS full_text_rank,
               semantic.rank_ix AS semantic_rank,
               semantic.similarity,
               (COALESCE(1.0 / (rrf_k + full_text.rank_ix), 0) * full_text_weight +
                COALESCE(1.0 / (rrf_k + semantic.rank_ix), 0) * semantic_weight) AS rrf_score
        FROM full_text
        FULL JOIN semantic ON full_text.id = semantic.id
    )
    SELECT c.doc_id,
           ce.source_id,
           ce.code_content,
           ce.summary,
           ce.programming_language,
           ce.complexity_score,
           c.similarity,
           c.rrf_score,
           c.semantic_rank,
           c.full_text_rank
    FROM combined c
    JOIN code_examples ce ON ce.id = c.doc_id
    ORDER BY c.rrf_score DESC
    LIMIT LEAST(match_count, 30);
END;
$$;