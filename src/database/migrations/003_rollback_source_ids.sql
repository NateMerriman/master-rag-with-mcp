-- Rollback: Clear source_id column in crawled_pages table
-- Task 2.3.1 Rollback: Reset source_id values to NULL
-- Date: 2025-06-08

\echo 'Starting source_id population rollback...'

-- Step 1: Display current state before rollback
\echo 'Current state before rollback:'
SELECT 
    COUNT(*) as total_pages,
    COUNT(source_id) as populated_source_ids,
    COUNT(*) - COUNT(source_id) as null_source_ids
FROM crawled_pages;

-- Step 2: Clear all source_id values
\echo 'Clearing source_id column...'

UPDATE crawled_pages 
SET source_id = NULL 
WHERE source_id IS NOT NULL;

-- Get update statistics
GET DIAGNOSTICS updated_count = ROW_COUNT;
\echo 'Cleared source_id from rows: ' || updated_count;

-- Step 3: Verify rollback results
\echo 'State after rollback:'
SELECT 
    COUNT(*) as total_pages,
    COUNT(source_id) as populated_source_ids,
    COUNT(*) - COUNT(source_id) as null_source_ids
FROM crawled_pages;

-- Step 4: Validation
DO $$
DECLARE
    remaining_populated INTEGER;
BEGIN
    SELECT COUNT(*) INTO remaining_populated FROM crawled_pages WHERE source_id IS NOT NULL;
    
    IF remaining_populated > 0 THEN
        RAISE WARNING 'Rollback incomplete: % pages still have source_id populated', remaining_populated;
    ELSE
        RAISE NOTICE 'Rollback successful: All source_id values cleared';
    END IF;
END $$;

\echo 'Source ID population rollback completed!'