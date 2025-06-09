-- Rollback Migration: Remove code_examples table and related components
-- Task 2.2: Code Examples Table Implementation - ROLLBACK
-- 
-- This script safely removes the code_examples table and all related components
-- in the reverse order of creation to avoid dependency conflicts.

-- Begin transaction for atomic rollback
BEGIN;

-- Drop the trigger first
DROP TRIGGER IF EXISTS trigger_update_code_examples_content_tokens ON code_examples;

-- Drop the trigger function
DROP FUNCTION IF EXISTS update_code_examples_content_tokens();

-- Drop all indexes (they will be dropped automatically with the table, but being explicit)
DROP INDEX IF EXISTS idx_code_examples_embedding_hnsw;
DROP INDEX IF EXISTS idx_code_examples_summary_embedding_hnsw;
DROP INDEX IF EXISTS idx_code_examples_content_tokens_gin;
DROP INDEX IF EXISTS idx_code_examples_source_id;
DROP INDEX IF EXISTS idx_code_examples_programming_language;
DROP INDEX IF EXISTS idx_code_examples_complexity_score;
DROP INDEX IF EXISTS idx_code_examples_created_at;
DROP INDEX IF EXISTS idx_code_examples_language_complexity;

-- Drop the table (this will also drop all dependent objects)
DROP TABLE IF EXISTS code_examples CASCADE;

-- Validate the rollback
DO $$
BEGIN
    -- Check that the table was removed
    IF EXISTS (SELECT 1 FROM information_schema.tables 
               WHERE table_name = 'code_examples') THEN
        RAISE EXCEPTION 'code_examples table still exists after rollback';
    END IF;
    
    -- Check that the trigger function was removed
    IF EXISTS (SELECT 1 FROM information_schema.routines 
               WHERE routine_name = 'update_code_examples_content_tokens') THEN
        RAISE EXCEPTION 'update_code_examples_content_tokens function still exists after rollback';
    END IF;
    
    RAISE NOTICE 'code_examples table rollback completed successfully';
END $$;

-- Commit the rollback transaction
COMMIT;