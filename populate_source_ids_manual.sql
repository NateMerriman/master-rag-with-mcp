-- Manual SQL Script: Populate source_id column in crawled_pages
-- Task 2.3.1: Match crawled_pages to sources by URL
-- Date: 2025-06-08
-- 
-- Instructions:
-- 1. Copy this entire script
-- 2. Open Supabase Studio (http://localhost:54323)
-- 3. Go to SQL Editor
-- 4. Paste and run this script
-- 5. Review the results

-- Step 1: Check current state
SELECT 
    'BEFORE MIGRATION' as status,
    COUNT(*) as total_pages,
    COUNT(source_id) as populated_source_ids,
    COUNT(*) - COUNT(source_id) as null_source_ids
FROM crawled_pages;

-- Step 2: Analyze URL matching potential
SELECT 
    'URL MATCHING ANALYSIS' as analysis,
    COUNT(DISTINCT cp.url) as unique_crawled_urls,
    COUNT(DISTINCT s.url) as unique_source_urls,
    COUNT(DISTINCT cp.url) FILTER (WHERE s.url IS NOT NULL) as matching_urls
FROM crawled_pages cp
LEFT JOIN sources s ON cp.url = s.url;

-- Step 3: Show sample URL matches (for verification)
SELECT 
    'SAMPLE MATCHES' as sample_type,
    cp.url as page_url,
    s.source_id,
    COUNT(cp.id) as page_count
FROM crawled_pages cp
INNER JOIN sources s ON cp.url = s.url
GROUP BY cp.url, s.source_id
ORDER BY COUNT(cp.id) DESC
LIMIT 5;

-- Step 4: Show sample unmatched URLs (for debugging)
SELECT 
    'SAMPLE UNMATCHED' as sample_type,
    cp.url as page_url,
    COUNT(cp.id) as page_count
FROM crawled_pages cp
LEFT JOIN sources s ON cp.url = s.url
WHERE s.url IS NULL
GROUP BY cp.url
ORDER BY COUNT(cp.id) DESC
LIMIT 5;

-- Step 5: Perform the actual update
UPDATE crawled_pages 
SET source_id = s.source_id
FROM sources s
WHERE crawled_pages.url = s.url
  AND crawled_pages.source_id IS NULL;

-- Step 6: Check results after update
SELECT 
    'AFTER MIGRATION' as status,
    COUNT(*) as total_pages,
    COUNT(source_id) as populated_source_ids,
    COUNT(*) - COUNT(source_id) as null_source_ids,
    ROUND((COUNT(source_id)::NUMERIC / COUNT(*)::NUMERIC) * 100, 2) as match_percentage
FROM crawled_pages;

-- Step 7: Analyze remaining unmatched pages
SELECT 
    'UNMATCHED ANALYSIS' as analysis,
    COUNT(*) as unmatched_pages,
    COUNT(DISTINCT url) as unique_unmatched_urls
FROM crawled_pages 
WHERE source_id IS NULL;

-- Step 8: Show remaining unmatched URLs (top 10)
SELECT 
    'TOP UNMATCHED URLS' as info,
    url,
    COUNT(*) as page_count
FROM crawled_pages 
WHERE source_id IS NULL 
GROUP BY url 
ORDER BY COUNT(*) DESC 
LIMIT 10;

-- Step 9: Final validation
SELECT 
    CASE 
        WHEN COUNT(source_id) = 0 THEN 'ERROR: No pages matched'
        WHEN (COUNT(source_id)::NUMERIC / COUNT(*)::NUMERIC) < 0.5 THEN 'WARNING: Low match rate'
        WHEN (COUNT(source_id)::NUMERIC / COUNT(*)::NUMERIC) < 0.9 THEN 'OK: Moderate match rate'
        ELSE 'SUCCESS: High match rate'
    END as migration_result,
    COUNT(*) as total_pages,
    COUNT(source_id) as matched_pages,
    ROUND((COUNT(source_id)::NUMERIC / COUNT(*)::NUMERIC) * 100, 2) || '%' as match_percentage
FROM crawled_pages;