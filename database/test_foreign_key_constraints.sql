-- Manual SQL Script: Test foreign key constraint enforcement (FIXED)
-- Task 2.3.3: Test constraint enforcement and cascade operations
-- Date: 2025-06-08
--
-- Instructions:
-- 1. Copy this entire script
-- 2. Open Supabase Studio (http://localhost:54323)
-- 3. Go to SQL Editor
-- 4. Paste and run this script
-- 5. Review test results

-- Step 1: Verify constraint exists
SELECT 
    'CONSTRAINT STATUS' as test_type,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM pg_constraint 
            WHERE conname = 'fk_crawled_pages_source_id'
        ) THEN 'PASSED: Constraint exists'
        ELSE 'FAILED: Constraint not found'
    END as result;

-- Step 2: Create test source for constraint testing
INSERT INTO sources (url, summary, total_word_count)
VALUES ('https://test-constraint-enforcement.example.com', 'Test source for FK constraint testing', 100)
ON CONFLICT (url) DO UPDATE SET 
    summary = EXCLUDED.summary,
    total_word_count = EXCLUDED.total_word_count
RETURNING source_id, url;

-- Get the test source_id for use in subsequent tests
WITH test_source AS (
    SELECT source_id 
    FROM sources 
    WHERE url = 'https://test-constraint-enforcement.example.com'
)
SELECT 
    'TEST SOURCE CREATED' as test_type,
    source_id,
    'Test source ready for constraint testing' as result
FROM test_source;

-- Step 3: Test valid foreign key insertion (should succeed)
-- First, clean up any existing test data
DELETE FROM crawled_pages 
WHERE url = 'https://test-constraint-enforcement.example.com';

-- Insert test page with valid source_id
WITH test_source AS (
    SELECT source_id 
    FROM sources 
    WHERE url = 'https://test-constraint-enforcement.example.com'
)
INSERT INTO crawled_pages (url, chunk_number, content, source_id)
SELECT 
    'https://test-constraint-enforcement.example.com',
    1,
    'Test content for constraint validation',
    source_id
FROM test_source;

SELECT 
    'VALID FK INSERT TEST' as test_type,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM crawled_pages 
            WHERE url = 'https://test-constraint-enforcement.example.com'
        ) THEN 'PASSED: Valid foreign key insertion succeeded'
        ELSE 'FAILED: Valid foreign key insertion failed'
    END as result;

-- Step 4: Test invalid foreign key insertion (should fail)
DO $$
BEGIN
    BEGIN
        INSERT INTO crawled_pages (url, chunk_number, content, source_id)
        VALUES ('https://test-invalid-fk.example.com', 1, 'Test content', 99999);
        RAISE NOTICE 'INVALID FK INSERT TEST: FAILED - Invalid foreign key insertion should have been rejected';
    EXCEPTION WHEN foreign_key_violation THEN
        RAISE NOTICE 'INVALID FK INSERT TEST: PASSED - Invalid foreign key insertion correctly rejected';
    END;
END $$;

-- Step 5: Test CASCADE DELETE behavior
-- First, count pages for our test source
WITH test_counts AS (
    SELECT 
        s.source_id,
        s.url,
        COUNT(cp.id) as page_count
    FROM sources s
    LEFT JOIN crawled_pages cp ON s.source_id = cp.source_id
    WHERE s.url = 'https://test-constraint-enforcement.example.com'
    GROUP BY s.source_id, s.url
)
SELECT 
    'PRE-DELETE COUNT' as test_type,
    source_id,
    page_count,
    'Pages associated with test source' as description
FROM test_counts;

-- Delete the test source (should cascade to crawled_pages)
DELETE FROM sources 
WHERE url = 'https://test-constraint-enforcement.example.com';

-- Verify cascade delete worked
SELECT 
    'CASCADE DELETE TEST' as test_type,
    CASE 
        WHEN NOT EXISTS (
            SELECT 1 FROM crawled_pages 
            WHERE url = 'https://test-constraint-enforcement.example.com'
        ) THEN 'PASSED: Cascade delete removed associated pages'
        ELSE 'FAILED: Cascade delete did not remove associated pages'
    END as result;

-- Step 6: Test constraint with NULL source_id (should be allowed)
INSERT INTO crawled_pages (url, chunk_number, content, source_id)
VALUES ('https://test-null-fk.example.com', 1, 'Test content with NULL source_id', NULL);

SELECT 
    'NULL FK TEST' as test_type,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM crawled_pages 
            WHERE url = 'https://test-null-fk.example.com' 
            AND source_id IS NULL
        ) THEN 'PASSED: NULL source_id insertion allowed'
        ELSE 'FAILED: NULL source_id insertion rejected'
    END as result;

-- Step 7: Test UPDATE with invalid foreign key (should fail)
DO $$
DECLARE
    test_page_id INTEGER;
BEGIN
    -- Get a test page ID
    SELECT id INTO test_page_id 
    FROM crawled_pages 
    WHERE url = 'https://test-null-fk.example.com' 
    LIMIT 1;
    
    IF test_page_id IS NOT NULL THEN
        BEGIN
            UPDATE crawled_pages 
            SET source_id = 99999 
            WHERE id = test_page_id;
            RAISE NOTICE 'INVALID FK UPDATE TEST: FAILED - Invalid foreign key update should have been rejected';
        EXCEPTION WHEN foreign_key_violation THEN
            RAISE NOTICE 'INVALID FK UPDATE TEST: PASSED - Invalid foreign key update correctly rejected';
        END;
    END IF;
END $$;

-- Step 8: Clean up test data
DELETE FROM crawled_pages 
WHERE url IN (
    'https://test-constraint-enforcement.example.com',
    'https://test-null-fk.example.com',
    'https://test-invalid-fk.example.com'
);

-- Step 9: Final constraint verification
SELECT 
    'FINAL CONSTRAINT STATUS' as test_type,
    tc.constraint_name,
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name,
    rc.delete_rule,
    'Constraint is active and properly configured' as status
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu 
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage ccu 
    ON ccu.constraint_name = tc.constraint_name
JOIN information_schema.referential_constraints rc 
    ON tc.constraint_name = rc.constraint_name
WHERE tc.table_name = 'crawled_pages' 
  AND tc.constraint_type = 'FOREIGN KEY'
  AND tc.constraint_name = 'fk_crawled_pages_source_id';

-- Step 10: Summary of test results
SELECT 
    'TEST SUMMARY' as summary,
    'All foreign key constraint tests completed' as status,
    'Review results above for any failures' as note;