-- Find all sources with 'https://docs.n8n.io/' in their URL
-- Run this script in Supabase Studio SQL Editor

-- Basic query to find n8n documentation sources
SELECT 
    source_id,
    url,
    summary,
    total_word_count,
    created_at,
    updated_at
FROM sources 
WHERE url LIKE '%https://docs.n8n.io/%'
ORDER BY url;

-- Optional: Get count of n8n sources
SELECT COUNT(*) as n8n_source_count
FROM sources 
WHERE url LIKE '%https://docs.n8n.io/%';

-- Optional: Get count of associated crawled_pages for n8n sources
SELECT 
    COUNT(*) as total_n8n_chunks,
    COUNT(DISTINCT source_id) as unique_n8n_sources
FROM crawled_pages cp
WHERE cp.source_id IN (
    SELECT source_id 
    FROM sources 
    WHERE url LIKE '%https://docs.n8n.io/%'
);

-- Optional: Detailed breakdown of n8n sources with chunk counts
SELECT 
    s.source_id,
    s.url,
    s.summary,
    COUNT(cp.id) as chunk_count,
    s.total_word_count,
    s.created_at
FROM sources s
LEFT JOIN crawled_pages cp ON s.source_id = cp.source_id
WHERE s.url LIKE '%https://docs.n8n.io/%'
GROUP BY s.source_id, s.url, s.summary, s.total_word_count, s.created_at
ORDER BY chunk_count DESC, s.url;