-- Migration: Populate source_id column in crawled_pages table
-- Task 2.3.1: Map crawled_pages to sources by matching URLs
-- Date: 2025-06-08

\echo 'Starting source_id population migration...'

-- Step 1: Verify we have sources table and crawled_pages with source_id column
\echo 'Verifying table structure...'

DO $$
BEGIN
    -- Check that sources table exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sources') THEN
        RAISE EXCEPTION 'sources table does not exist. Run 001_create_sources_table.sql first.';
    END IF;
    
    -- Check that crawled_pages has source_id column
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'crawled_pages' AND column_name = 'source_id') THEN
        RAISE EXCEPTION 'crawled_pages table missing source_id column. Run 001_create_sources_table.sql first.';
    END IF;
    
    RAISE NOTICE 'Table structure verified successfully.';
END $$;

-- Step 2: Display current state before migration
\echo 'Current state before migration:'
SELECT 
    'crawled_pages' as table_name,
    COUNT(*) as total_records,
    COUNT(source_id) as populated_source_ids,
    COUNT(*) - COUNT(source_id) as null_source_ids
FROM crawled_pages
UNION ALL
SELECT 
    'sources' as table_name,
    COUNT(*) as total_records,
    COUNT(*) as populated_source_ids,
    0 as null_source_ids
FROM sources;

-- Step 3: Create temporary mapping table for validation
\echo 'Creating URL mapping analysis...'

CREATE TEMP TABLE url_mapping_analysis AS
SELECT 
    cp.url as crawled_url,
    s.source_id,
    s.url as source_url,
    COUNT(cp.id) as page_count
FROM crawled_pages cp
LEFT JOIN sources s ON cp.url = s.url
GROUP BY cp.url, s.source_id, s.url
ORDER BY page_count DESC;

-- Display mapping statistics
\echo 'URL mapping analysis:'
SELECT 
    CASE 
        WHEN source_id IS NOT NULL THEN 'Matched URLs'
        ELSE 'Unmatched URLs'
    END as mapping_status,
    COUNT(*) as unique_urls,
    SUM(page_count) as total_pages
FROM url_mapping_analysis
GROUP BY CASE WHEN source_id IS NOT NULL THEN 'Matched URLs' ELSE 'Unmatched URLs' END;

-- Show sample unmatched URLs for debugging
\echo 'Sample unmatched URLs (first 5):'
SELECT crawled_url, page_count
FROM url_mapping_analysis 
WHERE source_id IS NULL 
ORDER BY page_count DESC 
LIMIT 5;

-- Step 4: Populate source_id column
\echo 'Populating source_id column...'

UPDATE crawled_pages 
SET source_id = s.source_id
FROM sources s
WHERE crawled_pages.url = s.url
  AND crawled_pages.source_id IS NULL;

-- Get update statistics
GET DIAGNOSTICS updated_count = ROW_COUNT;
\echo 'Updated rows: ' || updated_count;

-- Step 5: Verify results after update
\echo 'State after migration:'
SELECT 
    'crawled_pages' as table_name,
    COUNT(*) as total_records,
    COUNT(source_id) as populated_source_ids,
    COUNT(*) - COUNT(source_id) as null_source_ids
FROM crawled_pages
UNION ALL
SELECT 
    'sources' as table_name,
    COUNT(*) as total_records,
    COUNT(*) as populated_source_ids,
    0 as null_source_ids
FROM sources;

-- Step 6: Analyze any remaining unmatched pages
\echo 'Analysis of remaining unmatched pages:'
SELECT 
    COUNT(*) as unmatched_pages,
    COUNT(DISTINCT url) as unique_unmatched_urls
FROM crawled_pages 
WHERE source_id IS NULL;

-- Show sample of remaining unmatched URLs
\echo 'Sample remaining unmatched URLs:'
SELECT url, COUNT(*) as page_count
FROM crawled_pages 
WHERE source_id IS NULL 
GROUP BY url 
ORDER BY COUNT(*) DESC 
LIMIT 5;

-- Step 7: Validation checks
\echo 'Running validation checks...'

DO $$
DECLARE
    total_pages INTEGER;
    matched_pages INTEGER;
    match_percentage NUMERIC;
BEGIN
    SELECT COUNT(*) INTO total_pages FROM crawled_pages;
    SELECT COUNT(*) INTO matched_pages FROM crawled_pages WHERE source_id IS NOT NULL;
    
    match_percentage := ROUND((matched_pages::NUMERIC / total_pages::NUMERIC) * 100, 2);
    
    RAISE NOTICE 'Validation Results:';
    RAISE NOTICE '- Total pages: %', total_pages;
    RAISE NOTICE '- Matched pages: %', matched_pages;
    RAISE NOTICE '- Match percentage: %', match_percentage || '%';
    
    -- Warn if match rate is low
    IF match_percentage < 90 THEN
        RAISE WARNING 'Low match rate (% < 90%). Review unmatched URLs above.', match_percentage;
    ELSE
        RAISE NOTICE 'Good match rate achieved.';
    END IF;
END $$;

\echo 'Source ID population migration completed successfully!'