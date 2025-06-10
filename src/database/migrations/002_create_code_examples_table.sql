-- Migration: Create code_examples table with a refined schema for the Hybrid RAG approach.
-- This script creates a specialized table for storing extracted code examples
-- with a single embedding, metadata, and support for hybrid search.

BEGIN;

-- Drop the table if it exists to ensure a clean slate, along with any dependent objects.
DROP TABLE IF EXISTS code_examples CASCADE;

-- Create the code_examples table with the new schema
CREATE TABLE code_examples (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES sources(source_id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    summary TEXT, -- Added for AI-generated contextual summaries
    programming_language TEXT,
    complexity_score INTEGER CHECK (complexity_score >= 1 AND complexity_score <= 10),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536),  -- Single embedding for OpenAI text-embedding-3-small
    content_tokens TSVECTOR,         -- For full-text search
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Prevent duplicate chunks from the same URL
    CONSTRAINT code_examples_url_chunk_number_key UNIQUE(url, chunk_number)
);

-- Create indexes for optimal performance

-- HNSW index for vector similarity search
CREATE INDEX idx_code_examples_embedding_hnsw 
ON code_examples USING hnsw (embedding vector_ip_ops)
WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX idx_code_examples_content_tokens_gin
ON code_examples USING gin(content_tokens);

-- Regular indexes for filtering and joining
CREATE INDEX idx_code_examples_source_id ON code_examples(source_id);
CREATE INDEX idx_code_examples_programming_language ON code_examples(programming_language);
CREATE INDEX idx_code_examples_complexity_score ON code_examples(complexity_score);
CREATE INDEX idx_code_examples_created_at ON code_examples(created_at);

-- Combined index for common filter combinations
CREATE INDEX idx_code_examples_language_complexity 
ON code_examples(programming_language, complexity_score);

-- Create trigger to automatically update content_tokens from the 'content' column
CREATE OR REPLACE FUNCTION update_code_examples_content_tokens()
RETURNS TRIGGER AS $$
BEGIN
    -- Generate the tsvector from the 'content' and 'summary' columns for full-text search.
    NEW.content_tokens := 
        to_tsvector('english', COALESCE(NEW.content, '') || ' ' || COALESCE(NEW.summary, ''));
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
                   WHERE table_name = 'code_examples' AND table_schema = 'public') THEN
        RAISE EXCEPTION 'code_examples table was not created successfully';
    END IF;
    
    -- Check for the unique constraint
    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                   WHERE table_name = 'code_examples' AND table_schema = 'public'
                   AND constraint_name = 'code_examples_url_chunk_number_key') THEN
        RAISE EXCEPTION 'UNIQUE constraint on (url, chunk_number) was not created';
    END IF;

    -- Check that the trigger exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.triggers 
                   WHERE trigger_name = 'trigger_update_code_examples_content_tokens'
                   AND event_object_table = 'code_examples' AND event_object_schema = 'public') THEN
        RAISE EXCEPTION 'Content tokens trigger was not created';
    END IF;
    
    RAISE NOTICE 'code_examples table migration completed successfully';
END $$;

COMMIT;