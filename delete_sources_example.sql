-- Delete all n8n documentation sources and associated crawled_pages via CASCADE
-- ⚠️  WARNING: This will permanently delete data. Review before executing.
-- Run this script in Supabase Studio SQL Editor

-- Step 1: Preview what will be deleted (run this first to confirm)
SELECT 
    'PREVIEW - Sources to be deleted:' as action,
    COUNT(*) as sources_count
FROM sources 
WHERE url LIKE '%https://docs.n8n.io/%';

-- Step 2: Preview associated crawled_pages that will be cascade deleted
SELECT 
    'PREVIEW - Crawled pages to be cascade deleted:' as action,
    COUNT(*) as pages_count
FROM crawled_pages cp
WHERE cp.source_id IN (
    SELECT source_id 
    FROM sources 
    WHERE url LIKE '%https://docs.n8n.io/%'
);

-- Step 3: Show detailed list of sources that will be deleted
SELECT 
    'PREVIEW - Source details:' as action,
    source_id,
    url,
    summary,
    total_word_count
FROM sources 
WHERE url LIKE '%https://docs.n8n.io/%'
ORDER BY url;

-- ═══════════════════════════════════════════════════════════════════
-- ⚠️  DANGER ZONE - ACTUAL DELETION BELOW
-- ═══════════════════════════════════════════════════════════════════

-- Step 4: ACTUAL DELETION - Uncomment the lines below to execute
-- This will delete sources and trigger CASCADE deletion of crawled_pages

/*
-- Begin transaction for safety
BEGIN;

-- Delete n8n sources (this will CASCADE delete associated crawled_pages)
DELETE FROM sources 
WHERE url LIKE '%https://docs.n8n.io/%';

-- Show deletion results
SELECT 
    'Deletion completed' as status,
    'Check row counts to confirm' as next_step;

-- Verify deletion (should return 0)
SELECT COUNT(*) as remaining_n8n_sources
FROM sources 
WHERE url LIKE '%https://docs.n8n.io/%';

-- Commit the transaction (uncomment to make deletion permanent)
-- COMMIT;

-- To rollback instead of commit (if something went wrong):
-- ROLLBACK;
*/

-- ═══════════════════════════════════════════════════════════════════
-- Instructions:
-- 1. First run Steps 1-3 to preview what will be deleted
-- 2. If satisfied with the preview, uncomment the deletion block
-- 3. Run the deletion block
-- 4. Uncomment COMMIT to make changes permanent, or ROLLBACK to undo
-- ═══════════════════════════════════════════════════════════════════