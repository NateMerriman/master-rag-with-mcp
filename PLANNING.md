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
2. **🔄 Database** (2-3 days): PHASE 2 IN PROGRESS - Schema enhancements + migrations + rollback testing
   - **✅ Task 2.1 Complete**: Sources table implementation (729 sources created)
3. **Application** (3-5 days): Strategy system + contextual embeddings + code extraction
4. **Advanced RAG** (5-7 days): Reranking integration + cross-strategy testing

### Code Organization
```
src/
├── config.py              # ✅ IMPLEMENTED - Strategy configuration management
├── reranking.py           # ✅ IMPLEMENTED - Cross-encoder reranking logic
├── code_extraction.py     # Code block identification and processing
├── contextual_embeddings.py # Enhanced embedding generation
├── database/
│   ├── migrations/        # ✅ IMPLEMENTED - Schema migration scripts (001_create_sources_table)
│   └── models.py         # ✅ IMPLEMENTED - Source and CrawledPage models
└── strategies/           # RAG strategy implementations

tests/
├── test_config.py         # ✅ IMPLEMENTED - Configuration system tests (21 tests)
├── test_reranking.py      # ✅ IMPLEMENTED - Reranking functionality tests (22 tests)
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

## Key Dependencies

### New Dependencies
```toml
sentence-transformers = ">=3.0.0"  # ✅ IMPLEMENTED - For cross-encoder reranking
pytest = ">=8.3.5"  # ✅ IMPLEMENTED - For comprehensive testing
```

### Phase 1 Completed Deliverables
- ✅ **Performance monitoring framework** (`src/performance_baseline.py`, `src/performance_monitor.py`)
- ✅ **Strategy configuration system** (`src/config.py` with comprehensive validation)
- ✅ **Cross-encoder reranking infrastructure** (`src/reranking.py` with 22 unit tests)
- ✅ **Enhanced documentation** (README.md with strategy guides, troubleshooting, performance tuning)
- ✅ **Regression testing** (`tests/test_performance_regression.py`)
- ✅ **Configuration testing** (`tests/test_config.py` with 21 test cases)

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
- Improved search quality with reranking enabled
- Successful code extraction and search for programming content
- Clean configuration system with comprehensive validation
- Complete documentation for new features

## Risk Mitigation
- **Pre-migration safety**: Comprehensive backup + rollback testing + data integrity checks
- **Feature flags**: Gradual rollout with environment variable controls
- **Performance monitoring**: ✅ IMPLEMENTED - Baseline captured + regression testing framework
- **Rollback procedures**: Tested procedures for each enhancement and database change
- **Development environment**: ✅ DOCKER COMPATIBILITY VERIFIED - Fixed local testing vs n8n configuration
- **Strategy interaction testing**: Validation of multiple enabled strategies
- **Production readiness**: ✅ SUPABASE INTEGRATION TESTED - Edge function compatibility resolved

## Integration Points
- FastMCP tool registration (conditional based on enabled strategies)
- Supabase client integration (existing patterns)
- Docker environment configuration
- Existing crawling pipeline (minimal changes)
- Current search tools (enhanced, not replaced)
