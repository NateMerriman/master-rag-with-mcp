-- Rollback 001: Remove sources table and related changes
-- This script safely rolls back migration 001_create_sources_table.sql
-- 
-- Rollback Plan:
-- • Removes source_id column from crawled_pages 
-- • Drops sources table and all related objects
-- • Verifies rollback success
-- 
-- Safety: This rollback preserves all original crawled_pages data

BEGIN;

-- 🔴 1. Remove source_id column from crawled_pages
-- Note: This is safe because we haven't populated it with data yet
ALTER TABLE crawled_pages DROP COLUMN IF EXISTS source_id;

-- 🔴 2. Drop sources table and related objects
DROP TRIGGER IF EXISTS update_sources_updated_at ON sources;
DROP TABLE IF EXISTS sources CASCADE;

-- 🔴 3. Drop the trigger function if no other tables use it
-- Note: We check for other uses to avoid breaking other functionality
DO $$
DECLARE
    trigger_usage_count INTEGER;
BEGIN
    -- Check if other tables use the update_updated_at_column function
    SELECT COUNT(*) INTO trigger_usage_count 
    FROM information_schema.triggers 
    WHERE action_statement LIKE '%update_updated_at_column%';
    
    -- Only drop if no other triggers use it
    IF trigger_usage_count = 0 THEN
        DROP FUNCTION IF EXISTS update_updated_at_column();
        RAISE NOTICE 'Dropped update_updated_at_column function (no other usage found)';
    ELSE
        RAISE NOTICE 'Preserved update_updated_at_column function (% other triggers use it)', trigger_usage_count;
    END IF;
END $$;

-- 🔴 4. Verify rollback success
DO $$
DECLARE
    sources_exists BOOLEAN;
    source_id_exists BOOLEAN;
BEGIN
    -- Check if sources table exists
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = 'sources'
    ) INTO sources_exists;
    
    -- Check if source_id column exists in crawled_pages
    SELECT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_name = 'crawled_pages' AND column_name = 'source_id'
    ) INTO source_id_exists;
    
    -- Verify rollback success
    IF sources_exists OR source_id_exists THEN
        RAISE EXCEPTION 'Rollback verification failed: sources table exists=%, source_id column exists=%', 
            sources_exists, source_id_exists;
    END IF;
    
    RAISE NOTICE 'Rollback 001 completed successfully: all changes reverted';
END $$;

COMMIT;