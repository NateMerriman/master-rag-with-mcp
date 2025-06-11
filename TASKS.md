# TASKS.md - Crawl4AI RAG Enhancement Implementation Tasks

## Current Status: Phase 1 Complete ✅ | Phase 2 Complete ✅ | Phase 3 Complete ✅

## ⚡ URGENT: Code Extraction Logic Improvements

### ✅ TASK: Fix Smart Chunking Code Block Preservation - COMPLETED
**Priority: HIGH | Estimated: 4-6 hours | Status: ✅ COMPLETED | Date: 2024-12-19**

**Root Cause Analysis Completed:**
- Diagnosed DeepWiki mem0-mcp page crawl failure
- Identified `smart_chunk_markdown` function as the culprit
- Issue: Function breaks AT code block boundaries instead of AFTER them
- Result: Malformed markdown chunks with orphaned opening/closing tags

**Specific Issues Found & Fixed:**
1. ✅ Code extraction logic works perfectly on properly formatted markdown
2. ✅ Fixed `smart_chunk_markdown()` breaking before closing ```` ``` ```` tags 
3. ✅ No longer creates incomplete chunks like: ```` ```python\n@mcp.tool(\n````
4. ✅ Eliminated empty code blocks: ```` ```\n```\n ```` 
5. ✅ Function signatures remain intact (no merging like `asyncdefadd_coding_preference`)

**Implementation Completed:**

- [x] **Task A: Enhanced Smart Chunking Algorithm**
  - [x] Created `EnhancedMarkdownChunker` class with robust code block detection
  - [x] Implemented complete code block preservation with `find_code_blocks()` method
  - [x] Added safe break point detection with `find_safe_break_point()` logic
  - [x] Implemented fallback logic for oversized code blocks
  - [x] Added comprehensive validation with `validate_chunks()` method

- [x] **Task B: Integration & Validation**
  - [x] Replaced original `smart_chunk_markdown()` with enhanced version
  - [x] Created comprehensive test suite in `test_chunking_only.py`
  - [x] Added integration tests in `test_integration.py`
  - [x] Validated chunk quality with detailed analysis tools

- [x] **Task C: Enhanced Code Block Processing**
  - [x] Fixed infinite loop bug in chunking algorithm
  - [x] Added proper paragraph and sentence break detection
  - [x] Implemented robust boundary detection for natural breaks
  - [x] Created detailed diagnostic and debugging tools

- [x] **Task D: Testing & Documentation**
  - [x] Created `src/improved_chunking.py` with full implementation
  - [x] Added diagnostic tools: `debug_chunking_issue.py`, `diagnose_crawl_issues.py`
  - [x] Comprehensive test coverage with multiple test files
  - [x] Integration with existing crawl pipeline confirmed

**Technical Implementation:**

```python
def smart_chunk_markdown_enhanced(text: str, chunk_size: int = 5000) -> List[str]:
    """Enhanced chunking that preserves complete code blocks."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
            
        # Find code block boundaries with look-ahead
        chunk = text[start:end]
        
        # Look for opening code blocks
        code_start = chunk.rfind("```")
        if code_start != -1:
            # Look ahead for closing tag
            remaining = text[start + code_start:]
            code_end = remaining.find("```", 3)  # Skip the opening ```
            
            if code_end != -1:
                # Complete code block found, include it entirely
                end = start + code_start + code_end + 3
            else:
                # Incomplete code block, break before it
                end = start + code_start
                
        # Continue with existing paragraph/sentence logic...
        # [Rest of enhanced implementation]
```

**Acceptance Criteria: ✅ ALL MET**
- [x] ✅ DeepWiki mem0-mcp page extracts the @mcp.tool code block correctly
- [x] ✅ No malformed code blocks in chunked content (0 validation issues)
- [x] ✅ All function signatures remain intact (no merging)
- [x] ✅ Code extraction success rate improved significantly
- [x] ✅ Existing functionality preserved (backward compatible)
- [x] ✅ Comprehensive test coverage for edge cases
- [x] ✅ Performance maintained (no infinite loops resolved)

**Implementation Results:**
- **Enhanced Algorithm**: `EnhancedMarkdownChunker` with proper code block boundary detection
- **Code Block Detection**: Finds all complete code blocks before chunking decisions
- **Safe Break Points**: Never breaks within code blocks, preserves integrity
- **Validation System**: Built-in chunk validation with issue detection
- **Test Coverage**: Multiple test files with comprehensive edge case coverage
- **Integration**: Drop-in replacement maintaining full backward compatibility

**Performance Impact:**
- Fixed infinite loop bug that was causing performance issues
- Enhanced algorithm has minimal overhead vs original
- Validation shows 0 malformed blocks vs multiple in original
- No regressions in existing functionality

---

## Enhanced Adaptive Prompting Features

### ✅ TASK: Expand Content Type Categories - COMPLETED
**Priority: MEDIUM | Estimated: 2 hours | Actual: 3 hours | Date: 2024-12-19**

- [x] Add three new content type categories for adaptive prompting:
  - [x] **Social Media Posts (non-forum)**: LinkedIn posts, Twitter threads, Instagram content
  - [x] **Legal Documents**: Contracts, terms of service, statutes, legal frameworks
  - [x] **Educational/Instructional Materials (non-academic)**: How-to guides, tutorials, online course materials
- [x] Implement specialized prompting strategies for each new content type:
  - [x] Social media: Focus on engagement, tone, hashtags, and networking purpose
  - [x] Legal: Preserve precise terminology, capture obligations and procedural elements
  - [x] Educational: Emphasize learning objectives, instructional clarity, and skill development
- [x] Add comprehensive URL and content-based detection patterns
- [x] Create extensive unit tests for all new functionality (9 new tests)
- [x] Update documentation (README.md, .env.example) to reflect new content types
- [x] Fix existing test issues and ensure backward compatibility

**Acceptance Criteria: ✅ ALL MET**
- All three new content types detected accurately by URL patterns ✅
- Content-based detection works with appropriate indicators ✅
- Specialized prompting strategies generate appropriate context ✅
- Comprehensive test coverage for new functionality ✅
- Existing functionality preserved and backward compatible ✅
- Documentation updated to reflect expanded capabilities ✅

**Implementation Notes:**
- Extended content type detection system from 6 to 9 total content types
- Added 23 social media indicators, 27 legal indicators, and 28 educational indicators
- Implemented URL pattern matching for major platforms (Twitter/X, LinkedIn, Instagram, Facebook, legal sites, educational platforms)
- Created specialized system messages and user prompts for each content type
- Fixed legal URL pattern detection issue (gov/laws → separate gov and laws patterns)
- Fixed blog content detection to avoid conflicts with forum indicators
- Fixed test mocking issues with global configuration cache by patching config.get_config directly
- Added reset_config() calls to properly clear cached configuration in tests
- All 32 tests in enhanced contextual embeddings test suite now passing

**Technical Details:**
- Updated `_detect_content_type()` function with comprehensive indicator lists
- Enhanced `_get_contextual_prompt_and_system_message()` with three new content type handlers
- Expanded scoring algorithm to handle 8 content types (vs. previous 5)
- Maintained 2-indicator minimum threshold for confident classification
- Preserved fallback to "general" content type for unrecognized content
- Updated environment documentation to list all 9 content types with examples

## Phase 1: Foundation Enhancements (1-2 days)

### ✅ TASK 1.0: Performance Baseline Establishment - COMPLETED
**Priority: HIGH | Estimated: 1 hour | Actual: 3 hours**

- [x] Capture current search performance metrics
  - [x] Average query response time for hybrid search: 790.81ms
  - [x] Memory usage patterns during search operations
  - [x] Database metrics: 9,149 documents across 8 sources
- [x] Document baseline performance in metrics file (`performance_baseline.json`)
- [x] Set up performance monitoring framework for comparison
- [x] Create automated performance regression tests
- [x] **Fixed critical issues during implementation:**
  - [x] Environment variable configuration for local testing vs Docker
  - [x] Supabase Python client edge function compatibility
  - [x] Search function implementation using direct requests

**Acceptance Criteria: ✅ ALL MET**
- Baseline metrics documented and stored ✅
- Performance monitoring framework operational ✅
- Regression tests ready for continuous validation ✅

**Implementation Notes:**
- Added temporary URL override for local testing while preserving n8n Docker configuration
- Fixed `search_documents()` function to use direct requests instead of Supabase client
- Successfully captured real baseline with actual search results (not empty queries)
- Added comprehensive monitoring tools: `performance_baseline.py`, `performance_monitor.py`, `test_performance_regression.py`

### ✅ TASK 1.1: Strategy Configuration System - COMPLETED
**Priority: HIGH | Estimated: 4 hours | Actual: 3 hours**

- [x] Create `src/config.py` with strategy configuration class
  - [x] Load environment variables for all strategy toggles
  - [x] Implement validation logic for strategy combinations
  - [x] Add comprehensive error handling with clear messages
  - [x] Include default values for backward compatibility
- [x] Update `.env.example` with new strategy configuration variables
- [x] Add configuration validation to application startup
- [x] Create unit tests for configuration validation logic

**Acceptance Criteria: ✅ ALL MET**
- All strategy toggles work correctly ✅
- Invalid configurations rejected with helpful error messages ✅
- Backward compatibility maintained (all strategies default false) ✅

**Implementation Notes:**
- Created comprehensive StrategyConfig class with environment loading
- Added RAGStrategy enum for type-safe strategy identification
- Implemented dependency validation for strategy-specific requirements
- Added global configuration caching with reset capability for testing
- Updated .env.example with clear documentation and usage examples
- Integrated configuration validation into application startup with helpful logging

### ✅ TASK 1.2: Sentence Transformers Integration - COMPLETED
**Priority: HIGH | Estimated: 3 hours | Actual: 2 hours**

- [x] Add `sentence-transformers = "^3.0.0"` to pyproject.toml
- [x] Create `src/reranking.py` module
  - [x] Implement CrossEncoder model loading and caching
  - [x] Add batch processing for efficient scoring
  - [x] Include error handling for model loading failures
  - [x] Implement fallback behavior when reranking unavailable
- [x] Add model initialization to application startup (ready for integration)
- [x] Create unit tests for reranking functionality

**Acceptance Criteria: ✅ ALL MET**
- sentence-transformers dependency installed successfully ✅
- Cross-encoder model loads and caches properly ✅
- Batch processing works efficiently ✅
- Graceful fallback when model unavailable ✅

**Implementation Notes:**
- Created comprehensive `ResultReranker` class with singleton pattern
- Added 22 unit tests covering all functionality and edge cases
- Implemented robust error handling and fallback behavior
- Added timeout monitoring and performance tracking
- Global convenience functions for easy integration
- Ready for integration into search pipeline in later tasks

### ✅ TASK 1.3: Enhanced Documentation - COMPLETED
**Priority: MEDIUM | Estimated: 2 hours | Actual: 1.5 hours**

- [x] Update README.md with new strategy explanations
- [x] Add configuration examples for common use cases
- [x] Document each RAG strategy with benefits and trade-offs
- [x] Create troubleshooting guide for configuration issues
- [x] Add performance tuning recommendations

**Acceptance Criteria: ✅ ALL MET**
- Clear documentation for all new strategies ✅
- Configuration examples for different scenarios ✅
- Troubleshooting guide covers common issues ✅

**Implementation Notes:**
- Added comprehensive RAG Strategy Guide with benefits/trade-offs for each strategy
- Created 4 configuration examples: Basic, Enhanced Search Quality, AI Coding Assistant, Maximum Performance
- Added detailed troubleshooting section covering configuration, performance, database, and Docker issues
- Included performance tuning recommendations with specific model choices and expected timings
- Added database optimization SQL and memory optimization tips
- Integrated performance monitoring commands and alerting guidance

## Phase 2: Database Architecture Enhancements (2-3 days)

### ✅ TASK 2.1: Sources Table Implementation - COMPLETED
**Priority: HIGH | Estimated: 5 hours | Actual: 4 hours**

- [x] Create migration script for sources table
  ```sql
  CREATE TABLE sources (
      source_id SERIAL PRIMARY KEY,
      url TEXT UNIQUE NOT NULL,
      summary TEXT,
      total_word_count INTEGER DEFAULT 0,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
      updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
  );
  ```
- [x] Add appropriate indexes for performance
- [x] Create data migration script to populate from existing crawled_pages
- [x] **Create and test rollback migration script**
- [x] **Test rollback procedures with sample data**
- [x] Update database models in `src/database/models.py`
- [x] Test migration with sample data
- [x] **Validate data integrity after migration and rollback**

**Acceptance Criteria: ✅ ALL MET**
- Sources table created with proper structure ✅
- Existing data migrated successfully ✅ (729 sources from unique URLs)
- **Rollback procedures tested and documented** ✅
- Indexes optimize common query patterns ✅
- No data loss during migration or rollback ✅

**Implementation Notes:**
- Successfully created sources table with 729 records extracted from crawled_pages
- Added performance-optimized indexes for URL, created_at, and word_count queries
- Implemented automated trigger for updated_at timestamp maintenance
- Added source_id column to crawled_pages (NULL for now, will be populated in Task 2.3)
- Verified hybrid search functionality remains fully operational after schema changes
- Created comprehensive rollback script with validation checks
- Database models created for both Source and CrawledPage entities

### ✅ TASK 2.2: Code Examples Table Implementation - COMPLETED
**Priority: HIGH | Estimated: 6 hours | Actual: 4 hours**

- [x] Create migration script for code_examples table
  ```sql
  CREATE TABLE code_examples (
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
  ```
- [x] Add HNSW indexes for vector search (both embedding columns)
- [x] Add indexes for language, complexity, and source filtering
- [x] Create database functions for code search (hybrid_search_code_examples RPC)
- [x] Create Supabase edge function (hybrid-search-code-examples)
- [x] **Create and test rollback migration script**
- [x] **Implement code extraction and processing pipeline**
- [x] **Create comprehensive unit tests and validation scripts**
- [x] **Update database models for CodeExample entities**

**Acceptance Criteria: ✅ ALL MET**
- Code examples table supports specialized code storage ✅
- Vector indexes enable efficient similarity search (dual embeddings) ✅
- **Rollback procedures tested and documented** ✅
- Foreign key constraints maintain data integrity ✅
- **Hybrid search functions work correctly** ✅ (RPC + Edge function)
- **Code extraction pipeline operational** ✅ (18+ language support)

**Implementation Notes:**
- Created complete code extraction system with language detection for 18+ programming languages
- Implemented complexity scoring algorithm (1-10 scale) based on nesting, patterns, and line count
- Added dual embedding support: code content + natural language summaries
- Built comprehensive hybrid search with RRF scoring, language filtering, and complexity filtering
- Created both RPC function (`hybrid_search_code_examples`) and edge function (`hybrid-search-code-examples`)
- Added automatic content_tokens generation for full-text search integration
- Implemented CodeExample database models with proper serialization
- Created extensive test suite for code extraction logic (19 unit tests)
- Ready for integration with crawling pipeline and MCP tools

### ✅ TASK 2.3: Foreign Key Constraints - COMPLETED
**Priority: MEDIUM | Estimated: 3 hours | Actual: 2 hours**

- [x] Add source_id column to crawled_pages table
- [x] Populate source_id from existing URL data (100% match rate achieved)
- [x] Add foreign key constraint to crawled_pages
- [x] Test constraint enforcement
- [x] **Create and test rollback procedures for migration issues**
- [x] **Test data integrity validation after constraint addition**
- [x] **Validate performance impact of new constraints**

**Acceptance Criteria: ✅ ALL MET**
- Foreign key relationships established correctly ✅
- **Rollback procedures tested and validated** ✅
- Data integrity maintained across all tables ✅ (100% data integrity)
- Cascade operations work as expected ✅
- **No performance degradation from new constraints** ✅

**Implementation Notes:**
- Successfully populated source_id for all 9,149 crawled_pages with 100% URL match rate
- Created comprehensive SQL scripts for manual execution in Supabase Studio
- FK constraint `fk_crawled_pages_source_id` added with CASCADE DELETE behavior
- All constraint enforcement tests passed (valid FK inserts, invalid FK rejection, CASCADE deletes, NULL handling)
- Performance validation completed with comprehensive analysis and recommendations
- Rollback procedures created and documented for both data population and constraint addition

## Phase 3: Application Features Enhancement (3-5 days)

### ✅ TASK 3.0: Strategy Configuration Integration with Existing Contextual Embeddings - COMPLETED
**Priority: HIGH | Estimated: 3 hours | Actual: 2 hours**

- [x] Update `src/utils.py` to integrate new config system with existing contextual embeddings
  - [x] Check both `MODEL_CHOICE` AND `USE_CONTEXTUAL_EMBEDDINGS` flags
  - [x] Use `CONTEXTUAL_MODEL` from config when `USE_CONTEXTUAL_EMBEDDINGS=true`
  - [x] Maintain backward compatibility with existing `MODEL_CHOICE` behavior
- [x] Update `generate_contextual_embedding()` to use configurable model
- [x] Add strategy configuration validation to MCP server initialization  
- [x] Create tests for configuration integration and backward compatibility
- [x] **Performance validation with new configuration approach**

**Acceptance Criteria: ✅ ALL MET**
- New `USE_CONTEXTUAL_EMBEDDINGS` flag works correctly with existing functionality ✅
- `CONTEXTUAL_MODEL` setting overrides `MODEL_CHOICE` when strategy enabled ✅
- Backward compatibility preserved for existing `MODEL_CHOICE` usage ✅
- Configuration validation prevents invalid combinations ✅

**Implementation Notes:**
- Successfully integrated new configuration system with existing contextual embeddings
- Added `_should_use_contextual_embeddings()` and `_get_contextual_model()` helper functions
- Configuration precedence: `USE_CONTEXTUAL_EMBEDDINGS` + `CONTEXTUAL_MODEL` takes priority over legacy `MODEL_CHOICE`
- Updated config validation to remove MODEL_CHOICE dependency for contextual embeddings
- Created comprehensive test suite with 16 test cases covering all integration scenarios
- Performance validation confirmed no regressions: 761.65ms average (baseline: 790.81ms)
- Graceful fallback to legacy behavior when new configuration system unavailable

### ✅ TASK 3.1: Strategy Manager Implementation - COMPLETED
**Priority: HIGH | Estimated: 6 hours | Actual: 3 hours**

- [x] Create `src/strategies/` directory structure
- [x] Implement StrategyManager class
  - [x] Component initialization based on enabled strategies
  - [x] Runtime strategy validation
  - [x] Resource management for strategy components
- [x] Integrate with FastMCP application initialization
- [x] Add conditional tool registration based on enabled strategies
- [x] Create comprehensive unit tests

**Acceptance Criteria: ✅ ALL MET**
- Strategy manager correctly initializes enabled components ✅
- Tools appear/disappear based on configuration ✅
- Resource usage optimized for enabled strategies only ✅

**Implementation Notes:**
- Created comprehensive StrategyManager class with lifecycle management
- Added 32 unit tests covering all functionality and edge cases
- Integrated with FastMCP server initialization and cleanup
- Added conditional tool registration for strategy-specific functionality
- Implemented robust error handling and status reporting
- Global strategy manager pattern for easy access throughout application
- Ready for integration with actual strategy implementations in future tasks

### ✅ TASK 3.2: Code Extraction Pipeline - COMPLETED
**Priority: HIGH | Estimated: 8 hours | Actual: 4 hours**

- [x] Create `src/code_extraction.py` module (pre-existing from Task 2.2)
- [x] Implement code block pattern matching
  - [x] Fenced code blocks (```language)
  - [x] Indented code blocks
  - [x] Inline code snippets
- [x] Add language detection and validation (18+ programming languages supported)
- [x] Implement code summarization using rule-based approach
- [x] Add complexity scoring algorithms (1-10 scale)
- [x] Integrate with existing content processing pipeline (`add_documents_to_supabase`)
- [x] Create comprehensive tests with various code formats (14 tests passing)
- [x] **Add dual embeddings storage for code examples with hybrid search support**
- [x] **Integrate with agentic RAG configuration system (`USE_AGENTIC_RAG` flag)**
- [x] **Add source_id relationship management and error handling**

**Acceptance Criteria: ✅ ALL MET**
- Code blocks identified accurately across formats ✅
- Language detection works reliably ✅ (Python, JavaScript, SQL, Java, C++, etc.)
- Summaries provide meaningful descriptions ✅
- Integration preserves existing functionality ✅
- **Dual embeddings enable code + natural language search** ✅
- **Pipeline only runs when agentic RAG strategy enabled** ✅

**Implementation Notes:**
- Successfully integrated code extraction into document processing pipeline
- Added `add_code_examples_to_supabase()` function with dual embeddings (code content + summary)
- Created `_should_use_agentic_rag()` configuration helper for strategy detection
- Added `get_source_id_from_url()` function for database relationship management
- Comprehensive error handling for missing source_ids and embedding generation failures
- 14 comprehensive tests covering configuration, storage, integration, and error scenarios
- Ready for specialized code search tools implementation in Task 4.3

### ✅ TASK 3.3: Conditional Tool Registration - COMPLETED
**Priority: MEDIUM | Estimated: 3 hours | Actual: 2 hours**

- [x] Modify existing tool registration to check strategy configuration
- [x] Create strategy-specific tools (code search, enhanced search)
- [x] Add tool documentation generation based on enabled strategies
- [x] Implement error handling for disabled tool access
- [x] Test tool availability with different configurations

**Acceptance Criteria: ✅ ALL MET**
- Tools appear only when relevant strategies enabled ✅
- Clear error messages when accessing disabled tools ✅
- Documentation reflects current configuration ✅

**Implementation Notes:**
- Created `@conditional_tool` decorator for strategy-aware tool registration
- Added strategy-specific tools: `search_code_examples`, `perform_rag_query_with_reranking`, `perform_contextual_rag_query`
- Added `get_strategy_status` tool for dynamic configuration documentation
- Updated StrategyManager's `get_available_tools()` to track all tools
- Comprehensive error handling with informative messages including required strategies
- Created 13 test cases covering all conditional tool scenarios
- Updated existing strategy manager tests to account for new tools

## Phase 4: Advanced RAG Strategies (5-7 days)

### ✅ TASK 4.1: Cross-Encoder Reranking Integration - COMPLETED
**Priority: HIGH | Estimated: 6 hours | Actual: 4 hours**

- [x] Implement ResultReranker class in `src/reranking.py` (already completed in Task 1.2)
- [x] Add batch processing for efficient scoring (already completed in Task 1.2)
- [x] Integrate with existing search pipeline as post-processing step
- [x] Preserve RRF benefits while adding reranking quality
- [x] Add performance monitoring and metrics
- [x] Create comprehensive tests for reranking quality

**Acceptance Criteria: ✅ ALL MET**
- Reranking improves search result quality ✅
- Performance remains within acceptable limits ✅ (~150-500ms overhead)
- Integration preserves existing search functionality ✅ (RRF scores preserved in metadata)

**Implementation Notes:**
- Enhanced `perform_rag_query_with_reranking` tool with full pipeline integration
- Added `measure_reranking_performance()` method to performance monitoring framework
- Created comprehensive integration test suite (`tests/test_reranking_integration.py`)
- Pipeline: Hybrid Search (RRF) → Cross-Encoder Reranking → Final Results
- Graceful fallback when sentence-transformers unavailable
- Updated documentation with integration details and troubleshooting

### TASK 4.2: Cross-Strategy Integration Testing
**Priority: MEDIUM | Estimated: 4 hours**

- [ ] Test multiple strategies enabled simultaneously
  - [ ] Contextual embeddings + reranking
  - [ ] Agentic RAG + contextual embeddings  
  - [ ] All strategies enabled together
- [ ] Validate performance with strategy combinations
- [ ] Test resource usage with multiple strategies
- [ ] Create integration test suite for strategy combinations
- [ ] Document recommended strategy combinations

**Acceptance Criteria:**
- Multiple strategies work together without conflicts
- Performance remains acceptable with combinations
- Resource usage scales appropriately
- Clear documentation for strategy interactions

### ✅ TASK 4.3: Agentic RAG Tools Implementation - COMPLETED
**Priority: HIGH | Estimated: 10 hours | Status: COMPLETED**

- [x] Create specialized code search tools
- [x] Implement code example extraction workflow
- [x] Add code-to-code similarity search
- [x] Create natural language to code search
- [x] Integrate with code_examples table
- [x] Add code search result formatting
- [x] Create comprehensive tests for code search scenarios

**Acceptance Criteria: ✅ ALL MET**
- Code search provides relevant results using hybrid search with RRF ✅
- Multiple search modalities work correctly (semantic + full-text) ✅
- Results formatted appropriately for coding scenarios ✅

**Implementation Summary:**
- Replaced placeholder `search_code_examples` tool with full implementation
- Uses `hybrid_search_code_examples` SQL function for optimal search performance
- Supports programming language filtering and complexity scoring
- Enhanced query generation with context-aware prefixes
- Comprehensive parameter validation and clamping
- Client-side complexity filtering in addition to SQL-side filtering
- JSON metadata parsing with fallback handling
- Complete test suite with 10 comprehensive test cases covering:
  - Successful searches with multiple results
  - Empty result handling
  - Complexity filtering (both SQL-side and client-side)
  - Parameter validation and clamping
  - Metadata parsing from various formats
  - Error handling and graceful degradation
  - Enhanced query generation logic
  - Result ranking preservation
  - Conditional tool availability based on strategy configuration

**Note on Reranking Integration:**
- Current implementation uses hybrid search (RRF) only, NO reranking applied
- Reranking is only available for `crawled_pages` via `perform_rag_query_with_reranking` tool
- Pipeline difference: `crawled_pages` = Hybrid Search → Cross-Encoder Reranking | `code_examples` = Hybrid Search only
- **Future Enhancement Opportunity**: Could create `search_code_examples_with_reranking` tool requiring both `USE_AGENTIC_RAG=true` AND `USE_RERANKING=true`

### **TASK 4.4: Code Search Reranking Integration** - HIGH PRIORITY
  - [ ] Create `search_code_examples_with_reranking` tool combining agentic RAG + reranking strategies
  - [ ] Implement pipeline: `hybrid_search_code_examples` SQL → Cross-Encoder Reranking → Final Results
  - [ ] Require both `USE_AGENTIC_RAG=true` AND `USE_RERANKING=true` for availability
  - [ ] Preserve hybrid search benefits (RRF scores) while adding reranking quality improvements
  - [ ] Add performance monitoring and comprehensive testing for code-specific reranking scenarios
  - [ ] Consider specialized reranking models optimized for code similarity (e.g., CodeBERT-based models)

## Phase 5: Code Examples Refactoring (Hybrid Approach)
**Goal**: Simplify the `code_examples` implementation by adopting a hybrid approach that aligns with the reference repository's simplicity in some areas while preserving advanced features like hybrid search, language detection, and complexity scoring.

### ✅ TASK 5.1 COMPLETE: Database Schema Migration for Code Examples
**Priority: HIGH | Estimated: 4 hours**

- [x] Create a migration script (`002_create_code_examples_table.sql`) to set up the `code_examples` table with its final, correct schema.
  - [x] Rename `code_content` column to `content`.
  - [x] Drop the `summary_embedding` and `summary` columns.
  - [x] Add `url` (TEXT) and `chunk_number` (INTEGER) columns.
  - [x] Add `metadata` (JSONB) column with a default value.
  - [x] Add a `UNIQUE` constraint on `(url, chunk_number)`.
- [x] Create a corresponding rollback script.
- [x] Clean up obsolete migration scripts from the repository.
- [x] Update `README.md` to reflect the simplified database setup.

### ✅ TASK 5.2 COMPLETE: Update Code Extraction and Storage Pipeline
**Priority: HIGH | Estimated: 6 hours**

- [x] Modify the code processing logic in `src/code_extraction.py`.
  - [x] Update `ExtractedCode` dataclass to match the new schema (`content`, `url`, `chunk_number`, `metadata`).
  - [x] Remove the `generate_summary` method.
  - [x] Update `process_code_blocks` to populate the new fields.
- [x] Modify `add_code_examples_to_supabase` in `src/utils.py`.
  - [x] Update function to accept a list of `ExtractedCode` objects.
  - [x] Remove summary embedding logic.
  - [x] Map new fields to the database columns correctly.

### ✅ TASK 5.3 COMPLETE: Adapt Hybrid Search Function
**Priority: HIGH | Estimated: 2 hours**

- [x] Update the `hybrid_search_code_examples` SQL function.
  - [x] Modify the function's return type (`hybrid_search_code_examples_result`) to match the new schema.
  - [x] Remove logic related to `summary_embedding`.
  - [x] Update the query to use `content_tokens` for full-text search.
  - [x] Ensure the final SELECT returns the new fields like `url` and `metadata`.

### ✅ TASK 5.4 COMPLETE: End-to-End Testing and Validation
**Priority: HIGH | Estimated: 4 hours**

- [x] Create a new test file `tests/test_code_examples_flow.py`.
- [x] Write a comprehensive test to validate the full extraction and storage pipeline.
  - [x] Mock external dependencies (Supabase, OpenAI).
  - [x] Simulate adding a document with code blocks.
  - [x] Assert that the data being sent to the database has the correct, new structure.
- [x] Run the tests and ensure they pass.

## Post-Refactoring Enhancements

### ✅ TASK 6.1: Re-introduce AI-Generated Code Summaries - COMPLETED
**Priority: HIGH | Estimated: 2 hours | Status: COMPLETED**

- **Goal**: Restore the AI-generated `summary` for code examples to provide better context. This functionality was previously removed during a refactoring phase but is considered essential.
- [x] **Database Schema Update**:
  - [x] Modified `002_create_code_examples_table.sql` to re-add the `summary` TEXT column.
  - [x] Updated the `update_code_examples_content_tokens` trigger to include the `summary` in the full-text search index.
- [x] **Code Extraction Enhancement**:
  - [x] Modified `src/code_extraction.py` to add a `generate_summary` method to the `CodeExtractor`.
  - [x] This method uses the OpenAI API to generate a contextual summary for each code block.
  - [x] Updated the `ExtractedCode` dataclass to include the `summary`.
- [x] **Data Storage Pipeline Update**:
  - [x] Modified `src/utils.py` in the `add_code_examples_to_supabase` function.
  - [x] The function now passes the `summary` to be stored in the database.
  - [x] The embedding is now generated from a combination of the code `content` and the AI-generated `summary`.
- [x] **Search Function Update**:
    - [x] Modified `create_hybrid_search_code_examples_function.sql` to include the `summary` in the search results.

**Acceptance Criteria: ✅ ALL MET**
- `code_examples` table includes a `summary` column. ✅
- The code extraction pipeline calls the OpenAI API to generate summaries. ✅
- Summaries are stored in the database. ✅
- Embeddings are created from both code content and summary. ✅
- Search results for code examples include the `summary`. ✅

**Implementation Notes:**
- This change deviates from the "Code Examples Refactoring" plan, which originally aimed to remove the summary. The `PLANNING.md` file has been updated to reflect this new direction.
- The implementation uses a single embedding for the combined text, which is consistent with the refactoring's goal of moving away from a dual-embedding system.

## Cross-Strategy Testing Requirements

### Strategy Combination Matrix
- [ ] **Baseline only** (existing functionality)
- [ ] **Contextual embeddings only**
- [ ] **Reranking only** 
- [ ] **Agentic RAG only**
- [ ] **Contextual + Reranking**
- [ ] **Contextual + Agentic RAG**
- [ ] **Reranking + Agentic RAG**
- [ ] **All strategies enabled**

### Validation Criteria for Each Combination
- [ ] **Performance**: Response times within acceptable limits
- [ ] **Quality**: Search results improve or maintain quality
- [ ] **Resource usage**: Memory and CPU usage acceptable
- [ ] **Stability**: No errors or conflicts between strategies
- [ ] **Compatibility**: Works with existing Supabase and n8n integration

## Backlog Items (Future Enhancements)

### Code Examples Contextualization Prompt Enhancements
**Priority**: MEDIUM-HIGH | **Estimated Total**: 12-16 hours | **Impact**: High-quality code search improvements

#### 🚀 RECOMMENDED IMPLEMENTATION PRIORITY

##### ✨ TASK A1: Template-Based Enhancement with Examples - HIGH PRIORITY
**Priority: HIGH | Estimated: 4 hours | Impact: Immediate quality improvement**

- [x] **Goal**: Implement few-shot learning approach for consistent, high-quality code summaries ✅ **COMPLETED 2024-06-10**
- [x] **Implementation**:
  - [x] Create language-specific example libraries in `src/code_extraction.py` ✅ **COMPLETED**
  - [x] Update `generate_summary()` method to use template-based prompts ✅ **COMPLETED**
  - [x] Add examples for Python, JavaScript, SQL, and Java (top 4 languages) ✅ **COMPLETED** - Added TypeScript as well
  - [x] Create configuration flags for enhanced summaries ✅ **COMPLETED**
  - [x] Implement fallback mechanism for unsupported languages ✅ **COMPLETED**
  - [x] Add comprehensive unit tests ✅ **COMPLETED** - 16 tests covering all aspects

**✅ COMPLETED DELIVERABLES:**
- ✅ **Enhanced Code Summary Generation**: Implemented `generate_enhanced_summary()` method with few-shot learning
- ✅ **Language-Specific Examples**: Added high-quality examples for Python, JavaScript, TypeScript, SQL, and Java
- ✅ **Configuration System**: Added 8 new environment variables for code summary enhancement
- ✅ **Robust Fallback System**: Graceful degradation to basic summaries when enhanced mode fails
- ✅ **Comprehensive Testing**: 16 unit tests covering configuration, generation, fallback, and integration
- ✅ **Documentation**: Updated `.env.example` with new configuration options and performance notes

**🎯 IMPLEMENTATION HIGHLIGHTS:**
- **Few-Shot Learning**: Uses language-specific examples to ensure consistent, high-quality summaries
- **Configurable Enhancement**: Can be enabled/disabled via `USE_ENHANCED_CODE_SUMMARIES=true`
- **Smart Context Limiting**: Configurable context character limits (50-1000 chars)
- **Temperature Control**: Lower temperature (0.2) for consistent results vs basic (0.3)
- **Integration**: Works seamlessly with existing code extraction pipeline
- **Performance**: Added performance metrics and recommendations to documentation

##### 🎯 TASK A2: Retrieval-Optimized Summaries - HIGH PRIORITY  
**Priority: HIGH | Estimated: 3 hours | Impact: Better search relevance**

- [ ] **Goal**: Generate summaries specifically optimized for semantic search discovery
- [ ] **Implementation**:
  - [ ] Create `generate_retrieval_optimized_summary()` method
  - [ ] Focus prompt on action keywords, domain terms, use cases, and technical patterns
  - [ ] Update embedding generation to leverage optimized summaries
  - [ ] Add search-focused vocabulary and terminology guidance
- [ ] **Expected Benefits**:
  - [ ] Developers find relevant code examples more easily
  - [ ] Better matching between search queries and code functionality
  - [ ] Enhanced discoverability of similar implementation patterns
- [ ] **Acceptance Criteria**:
  - [ ] Search relevance scores improve in A/B testing
  - [ ] Summaries include actionable keywords for common developer searches
  - [ ] Integration preserves existing embedding generation pipeline
  - [ ] Performance impact remains within acceptable limits

##### 🧠 TASK A3: Domain-Specific Prompt Adaptation - MEDIUM PRIORITY
**Priority: MEDIUM | Estimated: 3 hours | Impact: Content-aware contextualization**

- [ ] **Goal**: Leverage existing content type detection system for adaptive prompts
- [ ] **Implementation**:
  - [ ] Integrate with existing `_detect_content_type()` from `utils.py`
  - [ ] Create domain-specific prompt templates (technical, educational, forum, etc.)
  - [ ] Add language-specific focus areas for each content type
  - [ ] Update `generate_summary()` to use detected content type
- [ ] **Expected Benefits**:
  - [ ] Tutorial code gets educational-focused summaries
  - [ ] API documentation gets integration-focused summaries  
  - [ ] Forum code gets problem-solving focused summaries
- [ ] **Acceptance Criteria**:
  - [ ] Content type detection accuracy validated across test cases
  - [ ] Domain-specific prompts generate appropriate contextual summaries
  - [ ] Integration works seamlessly with existing content type system
  - [ ] No performance degradation in content type detection

##### ⚙️ TASK A4: Configuration-Driven Enhancement System - MEDIUM PRIORITY
**Priority: MEDIUM | Estimated: 2 hours | Impact: Controlled rollout and strategy integration**

- [ ] **Goal**: Integrate code summary enhancements with existing strategy system
- [ ] **Implementation**:
  - [ ] Extend `StrategyConfig` in `config.py` with `CodeSummaryConfig`
  - [ ] Add environment variables for code summary enhancement toggles
  - [ ] Update strategy manager to handle code summary strategy
  - [ ] Add validation for code summary configuration combinations
- [ ] **Expected Benefits**:
  - [ ] Gradual rollout capability for enhanced summaries
  - [ ] Integration with existing configuration validation system
  - [ ] Performance monitoring for different summary strategies
- [ ] **Acceptance Criteria**:
  - [ ] Configuration system supports all enhancement toggles
  - [ ] Strategy combinations validate correctly
  - [ ] Existing configuration tests extended for code summaries
  - [ ] Documentation updated with new environment variables

### Additional Optional Enhancements (Lower Priority)

##### TASK A5: Multi-Layered Context Analysis - OPTIONAL
**Priority: LOW | Estimated: 4 hours | Impact: Rich metadata integration**

- [ ] Enhance prompts with complexity scores, patterns, and identifiers from existing metadata
- [ ] Create technical context sections in summary prompts
- [ ] Integrate with existing `METADATA_ENHANCEMENTS.md` system

##### TASK A6: Semantic Intent Recognition - OPTIONAL  
**Priority: LOW | Estimated: 3 hours | Impact: Purpose-driven summaries**

- [ ] Add pattern matching for API integration, tutorials, configuration, error handling
- [ ] Create intent-specific summary templates
- [ ] Update prompts based on detected code intent

##### TASK A7: Progressive Summarization Strategy - OPTIONAL
**Priority: LOW | Estimated: 4 hours | Impact: Complex code handling**

- [ ] Implement two-stage approach: quick analysis then detailed summary
- [ ] Add complexity-based decision logic for progressive vs. simple summarization
- [ ] Monitor API cost impact of multi-stage approach

### Implementation Notes

#### Integration Strategy
- **Phase 1**: Start with TASK A1 (Template-Based) for immediate quality improvement
- **Phase 2**: Add TASK A2 (Retrieval-Optimized) for search relevance boost
- **Phase 3**: Implement TASK A3 (Domain-Specific) for content-aware adaptation
- **Phase 4**: Complete TASK A4 (Configuration) for strategy system integration
- **Phase 5**: Consider optional enhancements based on impact assessment

#### Testing Requirements
- [ ] A/B testing framework for summary quality comparison
- [ ] Search relevance metrics before/after enhancement implementation
- [ ] Performance impact assessment for each enhancement
- [ ] Unit tests for all new prompt generation methods
- [ ] Integration tests with existing code extraction pipeline

#### Performance Considerations
- **API Cost**: Enhanced prompts may increase OpenAI usage by 20-40%
- **Processing Time**: Template-based approach adds ~50-100ms per code block
- **Memory Usage**: Example libraries require ~5-10MB additional memory
- **Search Quality**: Expected 15-25% improvement in search relevance scores

### Enhanced Hybrid Search
- [ ] Implement more sophisticated RRF algorithms
- [ ] Add query expansion capabilities
- [ ] Optimize weight balancing for different content types

### Row Level Security
- [ ] Add RLS policies for multi-tenant scenarios
- [ ] Implement user-based content filtering
- [ ] Add tenant isolation capabilities

### Performance Optimizations
- [ ] Implement result caching strategies
- [ ] Add query optimization for complex searches
- [ ] Optimize vector index parameters

## Testing Milestones

### Phase 1 Testing
- [ ] **Performance baseline captured and documented**
- [ ] All configuration combinations validated
- [ ] Reranking model loads successfully
- [ ] Documentation covers all new features
- [ ] **Performance monitoring framework operational**

### Phase 2 Testing
- [ ] Database migrations complete without data loss
- [ ] **Rollback procedures tested and validated**
- [ ] All indexes perform optimally
- [ ] Foreign key constraints enforce properly
- [ ] **Data integrity validation across all migration steps**
- [ ] **Performance impact assessment vs baseline**

### Phase 3 Testing
- [ ] Strategy manager handles all configurations
- [ ] **Contextual embeddings enhance accuracy vs baseline**
- [ ] Code extraction identifies relevant examples
- [ ] Tool registration works conditionally
- [ ] **Performance maintained with new features**

### Phase 4 Testing
- [ ] Reranking improves search quality measurably
- [ ] **Cross-strategy combinations work effectively**
- [ ] Agentic RAG provides useful code search
- [ ] **Final performance validation vs baseline**
- [ ] **Integration testing with downstream n8n workflows**

## Discovered Issues / Notes

### 🐛 Bug Fix: Type Mismatch in hybrid_search_code_examples Function
**Date:** 2025-01-09  
**Issue:** MCP server failing with SQL error: "Returned type integer does not match expected type bigint in column 1/11"
**Root Cause:** Multiple type mismatches in `hybrid_search_code_examples_result`:
- `id` was `BIGINT` but table uses `SERIAL` (`INTEGER`)
- `semantic_rank` and `full_text_rank` were `INT` but `row_number()` returns `BIGINT`
**Resolution:** Updated `create_hybrid_search_code_examples_function.sql`:
- Changed `id BIGINT` to `id INT` 
- Changed `semantic_rank INT` to `semantic_rank BIGINT`
- Changed `full_text_rank INT` to `full_text_rank BIGINT`
**Impact:** Code search functionality in n8n MCP integration now works correctly
**Files Modified:** `create_hybrid_search_code_examples_function.sql`

### Task 1.0 - Performance Baseline Issues Resolved
- **Environment Configuration**: Original .env used `host.docker.internal:54321` for Docker networking with n8n, but local testing required `localhost:54321`. Fixed with temporary override in test scripts.
- **Supabase Client Edge Function Issue**: Python supabase client had compatibility issues with edge function calls, returning "Unexpected end of JSON input". Fixed by implementing direct requests approach.
- **Empty Search Results**: Initial baseline capture returned 0 results due to edge function issues. After fixing search function, captured real baseline with 10 results per query.
- **Performance Variation**: Search response times vary (25-50%) between runs, which is normal for hybrid search operations.

## Documentation

### Enhanced Metadata Documentation
- **`METADATA_ENHANCEMENTS.md`** - Comprehensive documentation of the enhanced metadata system for code examples
  - 20+ metadata fields organized into categories (statistics, code analysis, context intelligence, complexity indicators)
  - Language-specific features for Python, JavaScript/TypeScript, SQL, Java, and 18+ programming languages
  - Performance characteristics and size optimization strategies
  - Integration benefits for AI systems and code discovery
  - Example metadata structures and future enhancement possibilities

## Completed Tasks

### ✅ TASK 1.0: Performance Baseline Establishment
**Completed:** 2025-06-05  
**Duration:** 3 hours (estimated 1 hour)  
**Key Deliverables:**
- `src/performance_baseline.py` - Baseline capture script
- `src/performance_monitor.py` - Ongoing monitoring framework  
- `tests/test_performance_regression.py` - Automated regression tests
- `PERFORMANCE.md` - Comprehensive documentation
- `performance_baseline.json` - Real baseline with 790.81ms avg response time
- Fixed search function compatibility with Supabase Docker stack
- Environment variable configuration for local testing vs production

### ✅ TASK 1.1: Strategy Configuration System
**Completed:** 2025-06-06  
**Duration:** 3 hours (estimated 4 hours)  
**Key Deliverables:**
- `src/config.py` - Complete strategy configuration management system
- `tests/test_config.py` - Comprehensive unit tests for configuration validation
- Updated `.env.example` - Clear documentation of all strategy toggles and examples
- Integration with `src/crawl4ai_mcp.py` - Startup validation and logging
- RAGStrategy enum for type-safe strategy identification
- Dependency validation system for strategy-specific requirements
- Global configuration caching with reset capability for testing

### ✅ TASK 1.2: Sentence Transformers Integration
**Completed:** 2025-06-06  
**Duration:** 2 hours (estimated 3 hours)  
**Key Deliverables:**
- `src/reranking.py` - Complete cross-encoder reranking system with ResultReranker class
- `tests/test_reranking.py` - Comprehensive unit tests (22 test cases) covering all functionality
- Updated `pyproject.toml` - Added sentence-transformers>=3.0.0 dependency and pytest dev dependency
- SearchResult and RerankingResult dataclasses for structured data handling
- Global singleton pattern with convenience functions for easy integration
- Robust error handling and graceful fallback when sentence-transformers unavailable
- Batch processing for efficient query-document pair scoring
- Performance monitoring and timeout tracking capabilities
- Ready for integration into search pipeline in future tasks

### ✅ TASK 2.1: Sources Table Implementation
**Completed:** 2025-06-08  
**Duration:** 4 hours (estimated 5 hours)  
**Key Deliverables:**
- `src/database/migrations/001_create_sources_table.sql` - Complete migration script with validation
- `src/database/migrations/001_rollback_sources_table.sql` - Comprehensive rollback script
- `src/database/models.py` - Database models for Source and CrawledPage entities
- `src/database/validate_migration.py` - Migration validation and testing tools
- Sources table with 729 records populated from existing crawled_pages data
- Performance-optimized indexes for URL, created_at, and word_count queries
- Automated updated_at trigger functionality
- source_id column added to crawled_pages (prepared for future FK constraint)
- Verified hybrid search functionality preserved after schema changes
- Complete rollback procedures tested and documented

### ✅ TASK 4.1: Cross-Encoder Reranking Integration
**Completed:** 2025-06-09  
**Duration:** 4 hours (estimated 6 hours)  
**Key Deliverables:**
- Enhanced `perform_rag_query_with_reranking` tool with complete pipeline integration
- `measure_reranking_performance()` method added to performance monitoring framework
- `tests/test_reranking_integration.py` - Comprehensive integration test suite (7 test cases)
- Pipeline implementation: Hybrid Search (RRF) → Cross-Encoder Reranking → Final Results
- Preservation of all hybrid search benefits (RRF scores, metadata) in reranked results
- Graceful fallback behavior when sentence-transformers unavailable
- Performance monitoring with timing metrics and overhead tracking (~150-500ms additional processing)
- Updated README.md with detailed integration documentation and troubleshooting guide
- Strategy-aware tool registration ensures clean tool interface based on configuration

### ✅ TASK 4.3: Agentic RAG Tools Implementation
**Completed:** 2025-01-02  
**Duration:** 4 hours (estimated 10 hours)  
**Key Deliverables:**
- Fully functional `search_code_examples` tool with hybrid search integration
- `tests/test_task_4_3_code_search.py` - Comprehensive test suite (10 test cases)
- Uses `hybrid_search_code_examples` SQL function for optimal search performance
- Enhanced query generation with context-aware prefixes ("Code example: {query} in {language}")
- Complete parameter validation and clamping for complexity, match_count
- Client-side complexity filtering complementing SQL-side filtering
- Robust JSON metadata parsing with graceful fallback handling
- Proper error handling and informative error messages
- Integration with strategy configuration system for conditional tool availability
- Comprehensive test coverage including edge cases, error scenarios, and parameter validation
- Replaces placeholder implementation with production-ready code search functionality

