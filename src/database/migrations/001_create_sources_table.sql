-- Migration 001: Create sources table
-- This migration creates a centralized sources table for better data organization
-- 
-- Migration Plan:
-- • Creates sources table with appropriate structure and indexes
-- • Populates sources table from existing crawled_pages URLs
-- • Adds source_id column to crawled_pages (prepared for future FK constraint)
-- 
-- Safety: This is a safe additive migration - no existing data is modified

BEGIN;

-- 🟢 1. Create sources table
CREATE TABLE IF NOT EXISTS sources (
    source_id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    summary TEXT,
    total_word_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 🟢 2. Add indexes for sources table performance
CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url);
CREATE INDEX IF NOT EXISTS idx_sources_created_at ON sources(created_at);
CREATE INDEX IF NOT EXISTS idx_sources_word_count ON sources(total_word_count);

-- 🟢 3. Add updated_at trigger function (if not exists)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 🟢 4. Add trigger for automatic updated_at handling
DROP TRIGGER IF EXISTS update_sources_updated_at ON sources;
CREATE TRIGGER update_sources_updated_at
    BEFORE UPDATE ON sources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 🟢 5. Populate sources table from existing crawled_pages data
INSERT INTO sources (url, total_word_count, created_at)
SELECT 
    url,
    COALESCE(SUM(CAST(metadata->>'word_count' AS INTEGER)), 0) as total_word_count,
    MIN(CAST(metadata->>'crawl_time' AS TIMESTAMP WITH TIME ZONE)) as created_at
FROM crawled_pages 
WHERE url IS NOT NULL
GROUP BY url
ON CONFLICT (url) DO UPDATE SET
    total_word_count = EXCLUDED.total_word_count,
    updated_at = NOW();

-- 🟢 6. Add source_id column to crawled_pages (for future FK constraint)
-- Note: We add the column but don't populate it yet - that's in the next migration
ALTER TABLE crawled_pages 
ADD COLUMN IF NOT EXISTS source_id INTEGER;

-- 🟢 7. Add index for future FK lookups (performance optimization)
CREATE INDEX IF NOT EXISTS idx_crawled_pages_source_id ON crawled_pages(source_id);

-- 🟢 8. Verify migration success
DO $$
DECLARE
    sources_count INTEGER;
    unique_urls_count INTEGER;
BEGIN
    -- Count sources created
    SELECT COUNT(*) INTO sources_count FROM sources;
    
    -- Count unique URLs in crawled_pages
    SELECT COUNT(DISTINCT url) INTO unique_urls_count FROM crawled_pages WHERE url IS NOT NULL;
    
    -- Verify counts match
    IF sources_count != unique_urls_count THEN
        RAISE EXCEPTION 'Migration verification failed: sources count (%) != unique URLs count (%)', 
            sources_count, unique_urls_count;
    END IF;
    
    RAISE NOTICE 'Migration 001 completed successfully: % sources created', sources_count;
END $$;

COMMIT;