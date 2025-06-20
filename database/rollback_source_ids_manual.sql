-- Manual SQL Script: Rollback source_id population
-- Task 2.3.1 Rollback: Clear source_id values in crawled_pages
-- Date: 2025-06-08
--
-- Instructions:
-- 1. Copy this entire script
-- 2. Open Supabase Studio (http://localhost:54323)
-- 3. Go to SQL Editor
-- 4. Paste and run this script

-- Step 1: Check current state before rollback
SELECT 
    'BEFORE ROLLBACK' as status,
    COUNT(*) as total_pages,
    COUNT(source_id) as populated_source_ids,
    COUNT(*) - COUNT(source_id) as null_source_ids
FROM crawled_pages;

-- Step 2: Clear all source_id values
UPDATE crawled_pages 
SET source_id = NULL 
WHERE source_id IS NOT NULL;

-- Step 3: Verify rollback results
SELECT 
    'AFTER ROLLBACK' as status,
    COUNT(*) as total_pages,
    COUNT(source_id) as populated_source_ids,
    COUNT(*) - COUNT(source_id) as null_source_ids,
    CASE 
        WHEN COUNT(source_id) = 0 THEN 'SUCCESS: All source_id values cleared'
        ELSE 'ERROR: Some source_id values remain'
    END as rollback_result
FROM crawled_pages;