-- Manual SQL Script: Remove foreign key constraint
-- Task 2.3.2 Rollback: Remove FK constraint from crawled_pages
-- Date: 2025-06-08
--
-- Instructions:
-- 1. Copy this entire script
-- 2. Open Supabase Studio (http://localhost:54323)
-- 3. Go to SQL Editor
-- 4. Paste and run this script

-- Step 1: Check current constraint status
SELECT 
    'CURRENT CONSTRAINTS' as info,
    conname as constraint_name,
    contype as constraint_type
FROM pg_constraint 
WHERE conname = 'fk_crawled_pages_source_id';

-- Step 2: Remove the foreign key constraint
ALTER TABLE crawled_pages 
DROP CONSTRAINT IF EXISTS fk_crawled_pages_source_id;

-- Step 3: Verify constraint was removed
SELECT 
    'CONSTRAINT REMOVAL VERIFICATION' as status,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM pg_constraint 
            WHERE conname = 'fk_crawled_pages_source_id'
        ) THEN 'ERROR: Foreign key constraint still exists'
        ELSE 'SUCCESS: Foreign key constraint removed'
    END as removal_status;

-- Step 4: Show remaining constraints on crawled_pages
SELECT 
    'REMAINING CONSTRAINTS' as info,
    tc.constraint_name,
    tc.constraint_type
FROM information_schema.table_constraints tc
WHERE tc.table_name = 'crawled_pages';