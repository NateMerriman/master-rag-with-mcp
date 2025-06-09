-- Manual SQL Script: Add foreign key constraint
-- Task 2.3.2: Add FK constraint between crawled_pages.source_id and sources.source_id
-- Date: 2025-06-08
--
-- Instructions:
-- 1. Copy this entire script
-- 2. Open Supabase Studio (http://localhost:54323)
-- 3. Go to SQL Editor
-- 4. Paste and run this script

-- Step 1: Verify data integrity before adding constraint
SELECT 
    'PRE-CONSTRAINT VALIDATION' as check_type,
    COUNT(*) as total_pages,
    COUNT(source_id) as pages_with_source_id,
    COUNT(*) - COUNT(source_id) as pages_without_source_id
FROM crawled_pages;

-- Step 2: Check for any orphaned source_id values (should be 0)
SELECT 
    'ORPHANED SOURCE_ID CHECK' as check_type,
    COUNT(*) as orphaned_pages
FROM crawled_pages cp
LEFT JOIN sources s ON cp.source_id = s.source_id
WHERE cp.source_id IS NOT NULL 
  AND s.source_id IS NULL;

-- Step 3: Show source_id distribution
SELECT 
    'SOURCE_ID DISTRIBUTION' as info,
    s.source_id,
    s.url,
    COUNT(cp.id) as page_count
FROM sources s
LEFT JOIN crawled_pages cp ON s.source_id = cp.source_id
GROUP BY s.source_id, s.url
ORDER BY COUNT(cp.id) DESC
LIMIT 10;

-- Step 4: Add the foreign key constraint
ALTER TABLE crawled_pages 
ADD CONSTRAINT fk_crawled_pages_source_id 
FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE;

-- Step 5: Verify constraint was added successfully
SELECT 
    'CONSTRAINT VERIFICATION' as info,
    conname as constraint_name,
    contype as constraint_type,
    confupdtype as on_update,
    confdeltype as on_delete
FROM pg_constraint 
WHERE conname = 'fk_crawled_pages_source_id';

-- Step 6: Show all foreign key constraints on crawled_pages
SELECT 
    'ALL FK CONSTRAINTS' as info,
    tc.constraint_name,
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name,
    rc.delete_rule
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu 
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage ccu 
    ON ccu.constraint_name = tc.constraint_name
JOIN information_schema.referential_constraints rc 
    ON tc.constraint_name = rc.constraint_name
WHERE tc.table_name = 'crawled_pages' 
  AND tc.constraint_type = 'FOREIGN KEY';

-- Step 7: Final validation
SELECT 
    'FINAL VALIDATION' as status,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM pg_constraint 
            WHERE conname = 'fk_crawled_pages_source_id'
        ) THEN 'SUCCESS: Foreign key constraint added'
        ELSE 'ERROR: Foreign key constraint not found'
    END as constraint_status;