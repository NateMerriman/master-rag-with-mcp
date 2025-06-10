# PLANNING.md - Crawl4AI RAG MCP Server Enhancement Project

## Project Vision
Enhance existing sophisticated Crawl4AI RAG MCP Server with advanced RAG strategies from latest coleam00-mcp-crawl4ai-rag version while preserving current strengths and avoiding redundancy.

## Current System Strengths (DO NOT MODIFY)
- **Advanced hybrid search** with Reciprocal Rank Fusion (RRF) 
- **Optimized vector operations** using inner product similarity (`<#>` operator)
- **Full-text search** with tsvector and GIN indexing
- **Smart content chunking** respecting code blocks and paragraphs
- **Comprehensive Docker integration** with detailed documentation
- **Manual crawling capabilities** with tqdm progress tracking
- **Production-ready architecture** with proper error handling

## Enhancement Goals
Add 14 targeted enhancements that provide new capabilities without duplicating existing functionality:

### Core Enhancements
1. **Strategy Toggle System** - Runtime RAG strategy configuration via env vars
2. **Cross-Encoder Reranking** - Local result reordering with sentence-transformers
3. **Database Architecture** - Sources table + Code examples table + FK constraints
4. **Agentic RAG** - Code extraction pipeline for AI coding assistant scenarios
5. **Contextual Embeddings** - Document-level context for improved semantic understanding
6. **AI-Generated Code Summaries** - Re-introduction of AI-powered summaries for code examples to enhance contextual understanding and search.

## Technical Architecture

### Technology Stack
- **Backend**: Python 3.11+, FastAPI, FastMCP
- **Database**: PostgreSQL with pgvector extension
- **Vector Search**: HNSW indexes, inner product similarity
- **Text Search**: PostgreSQL FTS with tsvector/GIN
- **Crawling**: Crawl4AI with AsyncWebCrawler
- **Embeddings**: OpenAI embeddings (existing)
- **Reranking**: sentence-transformers cross-encoder models
- **Containerization**: Docker with comprehensive setup

### Database Schema Strategy
```
Current: crawled_pages (single table with hybrid search)
Enhanced: 
  - sources (centralized source management)
  - crawled_pages (enhanced with source_id FK)
  - code_examples (specialized code storage)
```

### RAG Strategy Architecture
```
Base Strategy: Hybrid search (semantic + keyword + RRF)
Enhanced Strategies (configurable):
  - Contextual Embeddings: Document context + chunk content
  - Cross-Encoder Reranking: Local result reordering
  - Agentic RAG: Code extraction + specialized search
  - Enhanced Hybrid: More sophisticated RRF implementation
```

## Implementation Constraints

### Preservation Requirements
- **MUST preserve** existing hybrid search functionality
- **MUST maintain** current Docker setup and documentation
- **MUST keep** manual crawling capabilities
- **MUST retain** performance characteristics
- **MUST ensure** backward compatibility

### Configuration Philosophy
- All new features controlled by environment variables
- Default to disabled (backward compatibility)
- Clear validation and error messages
- Runtime strategy selection without code changes

### Performance Requirements
- **Baseline establishment**: ✅ COMPLETED - Captured 790.81ms avg response time, 9,149 documents
- **No degradation**: Existing search performance must be maintained or improved
- **Reranking performance**: Complete within 500ms for 20 results
- **Database migrations**: Must be non-blocking with comprehensive rollback procedures
- **Memory usage**: Remain reasonable with strategy combinations
- **Cross-strategy validation**: Test performance with multiple strategies enabled

## Development Approach

### Phase Structure
1. **✅ Foundation** (1-2 days): PHASE 1 COMPLETE - Performance baseline + Config system + Reranking infrastructure + Documentation
2. **✅ Database** (2-3 days): PHASE 2 COMPLETE - Schema enhancements + migrations + rollback testing + FK constraints
   - **✅ Task 2.1 Complete**: Sources table implementation (729 sources created)
   - **✅ Task 2.2 Complete**: Code examples table with hybrid search system
   - **✅ Task 2.3 Complete**: Foreign key constraints with 100% data integrity and performance validation
3. **✅ Application** (3-5 days): PHASE 3 COMPLETE - Strategy system + contextual embeddings + code extraction + conditional tools
   - **✅ Task 3.0 Complete**: Contextual embeddings integration with strategy configuration system
   - **✅ Task 3.1 Complete**: Strategy Manager implementation with component lifecycle management
   - **✅ Task 3.2 Complete**: Code extraction pipeline integration with dual embeddings and agentic RAG configuration
   - **✅ Task 3.3 Complete**: Conditional tool registration with strategy-aware availability
4. **Advanced RAG** (5-7 days): ✅ **PHASE 4 PARTIAL COMPLETE** - Reranking integration + agentic RAG tools implementation
5. ✅ **Code Examples Refactoring & Enhancement** (3-5 days): COMPLETE
   - **Goal**: Refactor the `code_examples` functionality to simplify its implementation, while preserving advanced features. A recent enhancement restored AI-powered summaries for better context.
   - **Key Implementation Details**:
     - **Database Schema**:
       - ✅ **Re-introduce `summary` column**: Added a `summary` TEXT column to store AI-generated summaries for each code example.
       - ✅ **Single Combined Embedding**: Use a single `embedding` field for the combined code `content` and `summary`, moving away from a dual-embedding model.
       - ✅ **Granular Tracking**: Use `url` and `chunk_number` columns for better data tracking and to prevent duplicates.
       - ✅ **Preserved Strengths**: Retain `programming_language`, `complexity_score`, `content_tokens` (for FTS), and the efficient integer-based `source_id` foreign key.
     - **Code Extraction & Processing**:
       - ✅ **Restore AI Summary Generation**: The `CodeExtractor` now generates a concise summary for each code block using the OpenAI API.
       - ✅ **Adapt Embedding Strategy**: The embedding process in `src/utils.py` combines the code `content` and the new `summary` to create a single, contextually rich embedding.
       - ✅ **Preserve Advanced Extractor**: The existing `CodeExtractor` with its language detection and complexity scoring capabilities is retained.
     - **Search Functionality**:
       - ✅ **Enhanced Full-Text Search**: The `content_tokens` trigger is updated to index both `content` and `summary` fields, improving keyword search.
       - ✅ **Update Hybrid Search**: The `hybrid_search_code_examples` SQL function has been updated to return the `summary` field in its results, making it available to the application.
       - ✅ **Preserve RRF**: The core Reciprocal Rank Fusion logic for combining semantic and full-text search is maintained.

### Code Organization
```
src/
├── config.py              # ✅ IMPLEMENTED - Strategy configuration management
├── reranking.py           # ✅ IMPLEMENTED - Cross-encoder reranking logic
├── utils.py               # ✅ ENHANCED - Contextual embeddings integrated with strategy configuration
├── code_extraction.py     # Code block identification and processing
├── database/
│   ├── migrations/        # ✅ IMPLEMENTED - Schema migration scripts (001_create_sources_table)
│   └── models.py         # ✅ IMPLEMENTED - Source and CrawledPage models
└── strategies/           # ✅ IMPLEMENTED - RAG strategy implementations
    ├── __init__.py       # ✅ IMPLEMENTED - Strategy package exports
    └── manager.py        # ✅ IMPLEMENTED - StrategyManager with component lifecycle

tests/
├── test_config.py         # ✅ IMPLEMENTED - Configuration system tests (21 tests)
├── test_reranking.py      # ✅ IMPLEMENTED - Reranking functionality tests (22 tests)
├── test_reranking_integration.py # ✅ IMPLEMENTED - Reranking pipeline integration tests (7 tests)
├── test_contextual_integration.py # ✅ IMPLEMENTED - Contextual embeddings integration tests (16 tests)
├── test_strategy_manager.py # ✅ IMPLEMENTED - Strategy Manager functionality tests (32 tests)
├── test_performance_regression.py  # ✅ IMPLEMENTED - Performance monitoring and regression testing
└── ...                   # Additional test files for future components
```

### Testing Strategy
- **Unit tests**: Each new component with comprehensive coverage
- **Integration tests**: All strategy combinations and interactions
- **Performance benchmarks**: Search quality metrics with baseline comparison
- **Migration testing**: Sample data + rollback procedures + data integrity validation
- **Cross-strategy testing**: Multiple strategies enabled simultaneously
- **Regression testing**: Existing functionality preserved after each phase

## Documentation

### Enhanced Code Metadata System
- **`METADATA_ENHANCEMENTS.md`** - Complete documentation of the enhanced metadata system for code examples
  - Advanced code analysis with 20+ metadata fields (identifiers, patterns, complexity indicators)
  - Language-specific features supporting Python, JavaScript/TypeScript, SQL, Java, and 18+ programming languages
  - Context intelligence extracting markdown headers and technical keywords from surrounding content
  - Performance optimization keeping payload sizes under 2KB per code block
  - Integration with hybrid search and AI systems for enhanced code discoverability

## Key Dependencies

### New Dependencies
```toml
sentence-transformers = ">=3.0.0"  # ✅ IMPLEMENTED - For cross-encoder reranking
pytest = ">=8.3.5"  # ✅ IMPLEMENTED - For comprehensive testing
```

### Phase 1, 2, 3 & 4 Completed Deliverables
- ✅ **Performance monitoring framework** (`src/performance_baseline.py`, `src/performance_monitor.py`)
- ✅ **Strategy configuration system** (`src/config.py` with comprehensive validation)
- ✅ **Cross-encoder reranking infrastructure** (`src/reranking.py` with 22 unit tests)
- ✅ **Cross-encoder reranking integration** (`perform_rag_query_with_reranking` tool, hybrid search preservation, 7 integration tests)
- ✅ **Agentic RAG tools implementation** (`search_code_examples` tool with hybrid search integration, 10 comprehensive tests)
  - ⚠️ **Note**: Currently uses hybrid search (RRF) only; reranking NOT integrated for code search
  - 🔄 **Future Enhancement**: Could add `search_code_examples_with_reranking` combining both strategies
- ✅ **Enhanced documentation** (README.md with strategy guides, troubleshooting, performance tuning)
- ✅ **Regression testing** (`tests/test_performance_regression.py`)
- ✅ **Configuration testing** (`tests/test_config.py` with 21 test cases)
- ✅ **Database architecture** (Sources table, code examples table, FK constraints with 100% data integrity)
- ✅ **Contextual embeddings integration** (`src/utils.py` enhanced with strategy configuration, 16 integration tests)
- ✅ **Strategy Manager implementation** (`src/strategies/manager.py` with lifecycle management, 32 unit tests)
- ✅ **Code extraction pipeline integration** (`src/utils.py` enhanced with agentic RAG code processing, 14 comprehensive tests)
- ✅ **Conditional tool registration** (`@conditional_tool` decorator, strategy-aware MCP tool availability, 13 test cases)

### Environment Variables
```bash
# Strategy toggles (all default false)
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH_ENHANCED=false
USE_AGENTIC_RAG=false
USE_RERANKING=false

# Model configuration
CONTEXTUAL_MODEL=gpt-3.5-turbo
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Success Criteria
- All 14 enhancements implemented and tested
- No regression in existing functionality
- Improved search quality with reranking enabled (for `crawled_pages` table)
- Successful code extraction and search for programming content (hybrid search only)
- Clean configuration system with comprehensive validation
- Complete documentation for new features

## Strategy Integration Analysis

### Current Search Tool Coverage:
- **`crawled_pages` table**: Hybrid Search (RRF) + Optional Cross-Encoder Reranking ✅
- **`code_examples` table**: Hybrid Search (RRF) only ⚠️

### Cross-Strategy Opportunities:
1. **Agentic RAG + Reranking**: Not currently integrated
   - Current: `search_code_examples` uses hybrid search only
   - Opportunity: `search_code_examples_with_reranking` combining both pipelines
   - Requirements: `USE_AGENTIC_RAG=true` AND `USE_RERANKING=true`
   
2. **Specialized Code Reranking Models**: 
   - Current reranking uses general-purpose `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Opportunity: Code-specific models like CodeBERT-based cross-encoders for better code similarity

## Risk Mitigation
- **Pre-migration safety**: Comprehensive backup + rollback testing + data integrity checks
- **Feature flags**: Gradual rollout with environment variable controls
- **Performance monitoring**: ✅ IMPLEMENTED - Baseline captured + regression testing framework
- **Rollback procedures**: ✅ IMPLEMENTED - Tested procedures for each enhancement and database change with comprehensive SQL scripts
- **Development environment**: ✅ DOCKER COMPATIBILITY VERIFIED - Fixed local testing vs n8n configuration
- **Strategy interaction testing**: Validation of multiple enabled strategies
- **Production readiness**: ✅ SUPABASE INTEGRATION TESTED - Edge function compatibility resolved

## Integration Points
- FastMCP tool registration (conditional based on enabled strategies)
- Supabase client integration (existing patterns)
- Docker environment configuration
- Existing crawling pipeline (minimal changes)
- Current search tools (enhanced, not replaced)
