-- Manual SQL Script: Validate performance impact of foreign key constraints (SIMPLE)
-- Task 2.3.5: Measure performance impact of new constraints
-- Date: 2025-06-08
--
-- Instructions:
-- 1. Copy this entire script
-- 2. Open Supabase Studio (http://localhost:54323)
-- 3. Go to SQL Editor
-- 4. Paste and run this script
-- 5. Compare with baseline performance metrics

-- Step 1: Analyze current database statistics
SELECT 
    'DATABASE STATISTICS' as metric_type,
    schemaname,
    relname as tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples
FROM pg_stat_user_tables 
WHERE relname IN ('crawled_pages', 'sources', 'code_examples')
ORDER BY relname;

-- Step 2: Check index usage and performance
SELECT 
    'INDEX STATISTICS' as metric_type,
    schemaname,
    relname as tablename,
    indexrelname as indexname,
    idx_tup_read as index_reads,
    idx_tup_fetch as index_fetches,
    idx_scan as index_scans
FROM pg_stat_user_indexes 
WHERE relname IN ('crawled_pages', 'sources', 'code_examples')
ORDER BY relname, indexrelname;

-- Step 3: Test query performance (Execution time will be shown in Supabase Studio)

-- Test 1: Simple source_id lookup (should be fast with FK index)
EXPLAIN (ANALYZE, BUFFERS) 
SELECT COUNT(*) 
FROM crawled_pages 
WHERE source_id = 1;

-- Test 2: Join performance with FK relationship
EXPLAIN (ANALYZE, BUFFERS)
SELECT s.url, s.summary, COUNT(cp.id) as page_count
FROM sources s
LEFT JOIN crawled_pages cp ON s.source_id = cp.source_id
WHERE s.source_id BETWEEN 1 AND 10
GROUP BY s.source_id, s.url, s.summary
ORDER BY page_count DESC;

-- Test 3: Complex query with multiple joins (includes code_examples)
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    s.url,
    COUNT(DISTINCT cp.id) as page_count,
    COUNT(DISTINCT ce.id) as code_count
FROM sources s
LEFT JOIN crawled_pages cp ON s.source_id = cp.source_id
LEFT JOIN code_examples ce ON s.source_id = ce.source_id
WHERE s.source_id BETWEEN 1 AND 50
GROUP BY s.source_id, s.url
ORDER BY page_count DESC
LIMIT 20;

-- Test 4: Constraint validation overhead (INSERT with FK check)
EXPLAIN (ANALYZE, BUFFERS)
WITH test_insert AS (
    INSERT INTO crawled_pages (url, chunk_number, content, source_id)
    VALUES ('https://performance-test.example.com', 999, 'Performance test content', 1)
    RETURNING id, source_id
)
SELECT COUNT(*) FROM test_insert;

-- Step 5: Constraint overhead analysis
SELECT 
    'CONSTRAINT OVERHEAD' as analysis,
    conname as constraint_name,
    CASE contype
        WHEN 'f' THEN 'Foreign Key'
        WHEN 'p' THEN 'Primary Key'
        WHEN 'u' THEN 'Unique'
        WHEN 'c' THEN 'Check'
        ELSE 'Other'
    END as constraint_type,
    'Active constraint may add validation overhead' as impact
FROM pg_constraint
WHERE conrelid = 'crawled_pages'::regclass
ORDER BY contype, conname;

-- Step 6: Table size and storage analysis
SELECT 
    'TABLE STORAGE' as metric_type,
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as total_size,
    pg_size_pretty(pg_relation_size(tablename::regclass)) as table_size,
    pg_size_pretty(pg_total_relation_size(tablename::regclass) - pg_relation_size(tablename::regclass)) as index_size
FROM pg_tables 
WHERE tablename IN ('crawled_pages', 'sources', 'code_examples')
ORDER BY pg_total_relation_size(tablename::regclass) DESC;

-- Step 7: Memory and buffer usage
SELECT 
    'BUFFER USAGE' as metric_type,
    name,
    setting,
    unit,
    short_desc
FROM pg_settings 
WHERE name IN (
    'shared_buffers',
    'effective_cache_size',
    'work_mem',
    'maintenance_work_mem'
);

-- Step 8: Foreign key constraint details
SELECT 
    'FK CONSTRAINT DETAILS' as metric_type,
    tc.constraint_name,
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name,
    rc.delete_rule,
    rc.update_rule
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu 
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage ccu 
    ON ccu.constraint_name = tc.constraint_name
JOIN information_schema.referential_constraints rc 
    ON tc.constraint_name = rc.constraint_name
WHERE tc.table_name = 'crawled_pages' 
  AND tc.constraint_type = 'FOREIGN KEY';

-- Step 9: Data integrity validation
SELECT 
    'DATA INTEGRITY CHECK' as check_type,
    COUNT(*) as total_pages,
    COUNT(source_id) as pages_with_source_id,
    COUNT(*) - COUNT(source_id) as pages_without_source_id,
    ROUND((COUNT(source_id)::NUMERIC / COUNT(*)::NUMERIC) * 100, 2) as integrity_percentage
FROM crawled_pages;

-- Step 10: Source distribution analysis
SELECT 
    'SOURCE DISTRIBUTION' as analysis,
    s.source_id,
    LEFT(s.url, 50) || '...' as url_preview,
    COUNT(cp.id) as page_count
FROM sources s
LEFT JOIN crawled_pages cp ON s.source_id = cp.source_id
GROUP BY s.source_id, s.url
ORDER BY COUNT(cp.id) DESC
LIMIT 10;

-- Step 11: Performance recommendations
SELECT 
    'PERFORMANCE RECOMMENDATIONS' as category,
    'Monitor foreign key constraint overhead during high-volume operations' as recommendation
UNION ALL
SELECT 
    'PERFORMANCE RECOMMENDATIONS',
    'Consider batching large INSERT/UPDATE operations to minimize FK validation overhead'
UNION ALL
SELECT 
    'PERFORMANCE RECOMMENDATIONS',
    'Index on source_id should provide good JOIN performance'
UNION ALL
SELECT 
    'PERFORMANCE RECOMMENDATIONS',
    'CASCADE deletes may be expensive for sources with many pages - test carefully';

-- Cleanup test data
DELETE FROM crawled_pages 
WHERE url = 'https://performance-test.example.com' AND chunk_number = 999;