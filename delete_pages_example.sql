-- Delete all rows from crawled_pages where URL starts with https://docs.crawl4ai.com/
-- Run this script manually in Supabase Studio or psql

-- First, check how many rows will be deleted (optional verification step)
SELECT COUNT(*) as rows_to_delete 
FROM crawled_pages 
WHERE url LIKE 'https://docs.crawl4ai.com/%';

-- Delete the rows
DELETE FROM crawled_pages 
WHERE url LIKE 'https://docs.crawl4ai.com/%';

-- Verify deletion completed
SELECT COUNT(*) as remaining_crawl4ai_rows 
FROM crawled_pages 
WHERE url LIKE 'https://docs.crawl4ai.com/%';