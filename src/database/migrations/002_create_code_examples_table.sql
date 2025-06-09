-- Migration: Create code_examples table for specialized code storage and search
-- Task 2.2: Code Examples Table Implementation
-- 
-- This migration creates a specialized table for storing extracted code examples
-- with metadata and support for hybrid search (semantic + full-text + RRF scoring).

-- Create the code_examples table
CREATE TABLE IF NOT EXISTS code_examples (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES sources(source_id) ON DELETE CASCADE,
    code_content TEXT NOT NULL,
    summary TEXT,
    programming_language TEXT,
    complexity_score INTEGER CHECK (complexity_score >= 1 AND complexity_score <= 10),
    embedding vector(1536),  -- OpenAI text-embedding-3-small
    summary_embedding vector(1536),  -- For natural language queries about code
    content_tokens tsvector,  -- For full-text search
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for optimal performance

-- HNSW indexes for vector similarity search
CREATE INDEX IF NOT EXISTS idx_code_examples_embedding_hnsw 
ON code_examples USING hnsw (embedding vector_ip_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_code_examples_summary_embedding_hnsw 
ON code_examples USING hnsw (summary_embedding vector_ip_ops)
WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_code_examples_content_tokens_gin
ON code_examples USING gin(content_tokens);

-- Regular indexes for filtering and joining
CREATE INDEX IF NOT EXISTS idx_code_examples_source_id ON code_examples(source_id);
CREATE INDEX IF NOT EXISTS idx_code_examples_programming_language ON code_examples(programming_language);
CREATE INDEX IF NOT EXISTS idx_code_examples_complexity_score ON code_examples(complexity_score);
CREATE INDEX IF NOT EXISTS idx_code_examples_created_at ON code_examples(created_at);

-- Combined index for common filter combinations
CREATE INDEX IF NOT EXISTS idx_code_examples_language_complexity 
ON code_examples(programming_language, complexity_score);

-- Create trigger to automatically update content_tokens from code_content and summary
CREATE OR REPLACE FUNCTION update_code_examples_content_tokens()
RETURNS TRIGGER AS $$
BEGIN
    -- Combine code_content and summary for full-text search
    NEW.content_tokens := to_tsvector('english', 
        COALESCE(NEW.code_content, '') || ' ' || COALESCE(NEW.summary, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_code_examples_content_tokens
    BEFORE INSERT OR UPDATE ON code_examples
    FOR EACH ROW
    EXECUTE FUNCTION update_code_examples_content_tokens();

-- Validate the migration
DO $$
BEGIN
    -- Check that the table was created
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables 
                   WHERE table_name = 'code_examples') THEN
        RAISE EXCEPTION 'code_examples table was not created successfully';
    END IF;
    
    -- Check that all required indexes exist
    IF NOT EXISTS (SELECT 1 FROM pg_indexes 
                   WHERE tablename = 'code_examples' 
                   AND indexname = 'idx_code_examples_embedding_hnsw') THEN
        RAISE EXCEPTION 'Vector index for embedding was not created';
    END IF;
    
    -- Check that the trigger exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.triggers 
                   WHERE trigger_name = 'trigger_update_code_examples_content_tokens') THEN
        RAISE EXCEPTION 'Content tokens trigger was not created';
    END IF;
    
    RAISE NOTICE 'code_examples table migration completed successfully';
END $$;