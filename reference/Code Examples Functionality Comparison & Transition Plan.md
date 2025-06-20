# Code Examples Functionality Comparison & Transition Plan

## Executive Summary

This project has evolved significantly beyond the original repository's (`reference-repo.md`) code_examples functionality. While the original repository implements a basic "Agentic RAG" system, this project has developed into a sophisticated code analysis and retrieval system with advanced features like dual embeddings, complexity scoring, language detection, and enhanced hybrid search capabilities.

This document maps out the key differences and provides a transition plan to align your project with the original repository's approach while preserving your advanced hybrid search functionality.

## Table of Contents

1. [Database Schema Differences](#database-schema-differences)
2. [Code Extraction & Processing Differences](#code-extraction--processing-differences)
3. [Search Functionality Differences](#search-functionality-differences)
4. [Architecture & Organization Differences](#architecture--organization-differences)
5. [Transition Plan](#transition-plan)
6. [Recommendations](#recommendations)

---

## Database Schema Differences

### Original Repository: Simple Code Examples Table

```sql
create table code_examples (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,  -- The code example content
    summary text not null,  -- Summary of the code example
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(url, chunk_number),
    foreign key (source_id) references sources(source_id)
);
```

**Key Characteristics:**
- Single embedding field
- Basic metadata structure
- Simple foreign key to sources table (text-based source_id)
- Minimal indexing strategy

### Your Project: Advanced Code Examples Table

```sql
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
```

**Key Characteristics:**
- **Dual embeddings**: Separate embeddings for code content and summaries
- **Language detection**: Automatic programming language identification
- **Complexity scoring**: 1-10 scale complexity assessment
- **Full-text search**: Integrated tsvector for keyword search
- **Enhanced indexing**: HNSW indexes for both embedding types
- **Integer-based foreign keys**: More efficient than text-based references

### Critical Differences Summary

| Feature | Original Repository | Your Project |
|---------|-------------------|--------------|
| **Embeddings** | Single embedding (code + summary) | Dual embeddings (code, summary) |
| **Language Detection** | None | 18+ programming languages |
| **Complexity Analysis** | None | 1-10 complexity scoring |
| **Full-text Search** | None | Integrated tsvector |
| **Foreign Key Type** | Text-based source_id | Integer-based source_id |
| **Field Names** | `content` | `code_content` |
| **Indexing Strategy** | Basic ivfflat | Advanced HNSW + GIN |

---


## Code Extraction & Processing Differences

### Original Repository: Basic Code Block Extraction

**Implementation Location**: `utils.py` - `extract_code_blocks()` function

**Key Features:**
- Simple regex-based extraction of triple-backtick code blocks
- Minimum length filtering (1000 characters default)
- Basic context extraction (1000 chars before/after)
- Single-threaded AI summary generation using OpenAI API
- Combined embedding creation (code + summary)

**Code Structure:**
```python
def extract_code_blocks(markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
    # Simple triple-backtick parsing
    # Basic language detection from first line
    # Context extraction
    return code_blocks

def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    # OpenAI API call for summary generation
    # Simple prompt with limited context
    return summary
```

### Your Project: Advanced Code Analysis System

**Implementation Location**: `src/code_extraction.py` - Dedicated module with comprehensive classes

**Key Features:**
- **Multi-pattern extraction**: Fenced blocks, indented blocks, inline code
- **Advanced language detection**: 18+ programming languages with content analysis
- **Complexity scoring algorithm**: Sophisticated 1-10 scale based on multiple factors
- **Enum-based language management**: Type-safe language handling
- **Rule-based summary generation**: No API dependency for basic summaries
- **Comprehensive validation**: Multiple validation layers for code quality

**Code Structure:**
```python
class ProgrammingLanguage(Enum):
    # 18+ supported languages with aliases

@dataclass
class CodeBlock:
    # Rich metadata structure

class CodeExtractor:
    def extract_code_blocks(self, content: str) -> List[CodeBlock]
    def _detect_language_from_content(self, code: str) -> ProgrammingLanguage
    def calculate_complexity_score(self, code: str, language: ProgrammingLanguage) -> int
    def generate_summary(self, code: str, language: ProgrammingLanguage, context: str = "") -> str
```

### Processing Workflow Differences

#### Original Repository Workflow:
1. Extract code blocks with regex
2. Generate AI summaries (parallel processing)
3. Create single embedding (code + summary)
4. Store in database with basic metadata

#### Your Project Workflow:
1. **Multi-pattern extraction** with language detection
2. **Complexity analysis** using sophisticated algorithms
3. **Rule-based summary generation** (no API dependency)
4. **Dual embedding creation** (code and summary separately)
5. **Full-text indexing** with tsvector generation
6. **Enhanced metadata** with language and complexity

### Critical Processing Differences

| Aspect | Original Repository | Your Project |
|--------|-------------------|--------------|
| **Language Detection** | Basic hint parsing | Content analysis + 18+ languages |
| **Summary Generation** | AI-powered (API cost) | Rule-based (no API cost) |
| **Complexity Analysis** | None | Sophisticated 1-10 scoring |
| **Embedding Strategy** | Single combined | Dual specialized |
| **Validation** | Length-based only | Multi-layer validation |
| **Code Organization** | Single utility function | Dedicated module with classes |

---

## Search Functionality Differences

### Original Repository: Basic Vector Search with Hybrid Option

**Implementation**: Single `search_code_examples` tool with optional hybrid search

**Features:**
- Vector similarity search using single embedding
- Optional hybrid search (vector + keyword)
- Basic reranking with cross-encoder
- Simple query enhancement
- Source filtering support

**Search Function:**
```python
def search_code_examples(client, query, match_count=10, filter_metadata=None, source_id=None):
    # Enhanced query: "Code example for {query}\n\nSummary: Example code showing {query}"
    # Single embedding search
    # Optional hybrid search with ILIKE
    # Optional reranking
```

### Your Project: Advanced Hybrid Search System

**Implementation**: Sophisticated RPC function with specialized database operations

**Features:**
- **Reciprocal Rank Fusion (RRF)** scoring algorithm
- **Language-specific filtering** (programming language)
- **Complexity-based filtering** (1-10 scale)
- **Dual embedding search** (code content vs. summary embeddings)
- **Advanced full-text search** using tsvector
- **Specialized return types** with rich metadata

**Search Function:**
```sql
CREATE OR REPLACE FUNCTION hybrid_search_code_examples(
    query_text TEXT,
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 10,
    language_filter TEXT DEFAULT NULL,
    max_complexity INT DEFAULT 10
)
RETURNS SETOF hybrid_search_code_examples_result
```

**Return Structure:**
```sql
CREATE TYPE hybrid_search_code_examples_result AS (
    id BIGINT,
    source_id INT,
    code_content TEXT,
    summary TEXT,
    programming_language TEXT,
    complexity_score INT,
    similarity FLOAT,
    rrf_score FLOAT,
    semantic_rank INT,
    full_text_rank INT
);
```

### Search Capability Comparison

| Feature | Original Repository | Your Project |
|---------|-------------------|--------------|
| **Search Algorithm** | Vector similarity + basic hybrid | RRF-based hybrid search |
| **Filtering Options** | Source only | Source + Language + Complexity |
| **Embedding Strategy** | Single embedding search | Dual embedding options |
| **Full-text Integration** | ILIKE pattern matching | Native tsvector search |
| **Result Ranking** | Similarity + optional reranking | RRF scoring + multiple ranks |
| **Return Metadata** | Basic fields | Rich metadata with scores |

---

## Architecture & Organization Differences

### Original Repository: Monolithic Approach

**Structure:**
- Single `utils.py` file with all functionality
- Inline code extraction in crawling functions
- Environment variable-based configuration
- Simple tool registration

**Configuration:**
```python
# Simple environment variables
USE_AGENTIC_RAG=true
USE_HYBRID_SEARCH=true
USE_RERANKING=true
```

### Your Project: Modular Architecture

**Structure:**
- **Dedicated modules**: `code_extraction.py`, `config.py`, `reranking.py`
- **Strategy management**: `strategies/manager.py` with conditional tools
- **Database migrations**: Structured migration system
- **Comprehensive testing**: Extensive test suite

**Configuration System:**
```python
@dataclass
class StrategyConfig:
    use_contextual_embeddings: bool = False
    use_hybrid_search_enhanced: bool = False
    use_agentic_rag: bool = False
    use_reranking: bool = False
    
    @classmethod
    def from_environment(cls) -> "StrategyConfig"
```

**Conditional Tool System:**
```python
@conditional_tool("search_code_examples", [RAGStrategy.AGENTIC_RAG])
async def search_code_examples(ctx: Context, query: str, ...):
    # Tool only available when strategy is enabled
```

### Architectural Differences Summary

| Aspect | Original Repository | Your Project |
|--------|-------------------|--------------|
| **Code Organization** | Monolithic utils.py | Modular architecture |
| **Configuration** | Environment variables | Structured config classes |
| **Tool Management** | Static registration | Conditional/dynamic tools |
| **Database Management** | Single SQL file | Migration system |
| **Testing** | Minimal | Comprehensive test suite |
| **Strategy Management** | Simple toggles | Strategy pattern with manager |

---


## Transition Plan

### Phase 1: Database Schema Alignment (High Priority)

#### 1.1 Simplify Code Examples Table Structure

**Goal**: Align with original repository's simpler schema while preserving hybrid search

**Actions Required:**
```sql
-- Create migration to simplify code_examples table
ALTER TABLE code_examples 
  DROP COLUMN programming_language,
  DROP COLUMN complexity_score,
  DROP COLUMN summary_embedding,
  DROP COLUMN content_tokens;

-- Rename columns to match original
ALTER TABLE code_examples 
  RENAME COLUMN code_content TO content;

-- Add missing fields from original
ALTER TABLE code_examples 
  ADD COLUMN url VARCHAR NOT NULL,
  ADD COLUMN chunk_number INTEGER NOT NULL,
  ADD COLUMN metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

-- Add unique constraint
ALTER TABLE code_examples 
  ADD CONSTRAINT unique_url_chunk UNIQUE(url, chunk_number);
```

**Considerations:**
- **Data Migration**: Existing complexity_score and programming_language data will be lost
- **Hybrid Search Impact**: Will need to modify hybrid search to work with simplified schema
- **Backward Compatibility**: Current code extraction pipeline will need updates

#### 1.2 Modify Foreign Key Structure

**Goal**: Align with original text-based source_id approach

**Actions Required:**
```sql
-- Add text-based source_id column
ALTER TABLE code_examples ADD COLUMN source_id_text TEXT;

-- Populate from existing integer source_id
UPDATE code_examples 
SET source_id_text = (SELECT url FROM sources WHERE sources.source_id = code_examples.source_id);

-- Drop integer foreign key
ALTER TABLE code_examples DROP CONSTRAINT code_examples_source_id_fkey;
ALTER TABLE code_examples DROP COLUMN source_id;

-- Rename and add constraint
ALTER TABLE code_examples RENAME COLUMN source_id_text TO source_id;
ALTER TABLE code_examples ADD FOREIGN KEY (source_id) REFERENCES sources(source_id);
```

### Phase 2: Code Extraction Simplification (Medium Priority)

#### 2.1 Replace Advanced Code Extractor

**Goal**: Use original repository's simpler extraction approach

**Actions Required:**

1. **Create simplified extraction function** in `utils.py`:
```python
def extract_code_blocks(markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
    """Simple extraction matching original repository approach."""
    # Implement original regex-based extraction
    # Remove language detection
    # Remove complexity scoring
    # Keep basic context extraction
```

2. **Modify crawling pipeline** to use simplified extraction:
```python
# Replace current code extraction
code_blocks = extract_code_blocks(result.markdown)
# Remove complexity and language processing
# Use AI summary generation (like original)
```

3. **Update embedding strategy**:
```python
# Change from dual embeddings to single combined embedding
combined_text = f"{code_content}\n\nSummary: {summary}"
embedding = create_embedding(combined_text)
```

#### 2.2 Restore AI Summary Generation

**Goal**: Replace rule-based summaries with AI-generated ones (like original)

**Actions Required:**
```python
def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """AI-powered summary generation matching original approach."""
    model_choice = os.getenv("MODEL_CHOICE")
    
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose."""
    
    # Use OpenAI API for summary generation
    return ai_generated_summary
```

### Phase 3: Search Function Alignment (Medium Priority)

#### 3.1 Simplify Search Implementation

**Goal**: Align search functionality with original while preserving hybrid search

**Actions Required:**

1. **Replace RPC function** with simpler approach:
```python
def search_code_examples(client, query, match_count=10, filter_metadata=None, source_id=None):
    """Simplified search matching original repository approach."""
    
    # Enhanced query like original
    enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
    
    # Create embedding
    query_embedding = create_embedding(enhanced_query)
    
    # Use original match_code_examples function
    if use_hybrid_search:
        # Implement hybrid search similar to original
        # Vector search + ILIKE keyword search
        # Combine results with preference for both
    else:
        # Simple vector search
        results = client.rpc('match_code_examples', {
            'query_embedding': query_embedding,
            'match_count': match_count,
            'source_filter': source_id
        })
    
    return results
```

2. **Update database function**:
```sql
-- Replace complex RPC with original approach
CREATE OR REPLACE FUNCTION match_code_examples (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  summary text,
  metadata jsonb,
  source_id text,
  similarity float
)
```

#### 3.2 Preserve Hybrid Search Capability

**Goal**: Maintain your advanced hybrid search while simplifying the interface

**Strategy**: Create a hybrid search wrapper that works with the simplified schema:

```python
def hybrid_search_code_examples_simplified(client, query, match_count=10, source_id=None):
    """Hybrid search adapted for simplified schema."""
    
    # Vector search using simplified match_code_examples
    vector_results = client.rpc('match_code_examples', {
        'query_embedding': create_embedding(query),
        'match_count': match_count * 2,
        'source_filter': source_id
    })
    
    # Keyword search using ILIKE on content and summary
    keyword_results = client.from_('code_examples')\
        .select('id, url, chunk_number, content, summary, metadata, source_id')\
        .or_(f'content.ilike.%{query}%,summary.ilike.%{query}%')\
        .limit(match_count * 2).execute()
    
    # Combine results (preserve your RRF logic)
    return combine_search_results(vector_results, keyword_results, match_count)
```

### Phase 4: Architecture Simplification (Low Priority)

#### 4.1 Consolidate Code Organization

**Goal**: Move towards monolithic approach like original while keeping some modularity

**Actions Required:**

1. **Merge code extraction** into `utils.py`:
   - Move simplified extraction functions to utils.py
   - Remove dedicated `code_extraction.py` module
   - Keep configuration module for strategy management

2. **Simplify tool registration**:
   - Replace conditional tools with simple environment variable checks
   - Maintain strategy manager for configuration but simplify tool registration

3. **Preserve testing infrastructure**:
   - Keep your comprehensive test suite
   - Update tests to work with simplified functionality

### Phase 5: Configuration Alignment (Low Priority)

#### 5.1 Align Environment Variables

**Goal**: Match original repository's configuration approach

**Actions Required:**
```bash
# Original repository variables
USE_AGENTIC_RAG=true
USE_HYBRID_SEARCH=true  # Keep your enhanced version
USE_RERANKING=true
MODEL_CHOICE=gpt-4.1-nano

# Remove your project-specific variables
# USE_HYBRID_SEARCH_ENHANCED -> USE_HYBRID_SEARCH
# Remove complexity and language-specific configs
```

---

## Recommendations

### Option 1: Full Alignment (Recommended for Consistency)

**Pros:**
- Complete compatibility with original repository
- Easier to merge improvements between projects
- Simplified maintenance and debugging
- Lower API costs (no complexity scoring)

**Cons:**
- Loss of advanced features (language detection, complexity scoring)
- Reduced search precision
- Significant development effort for migration

**Timeline:** 2-3 weeks

### Option 2: Hybrid Approach (Recommended for Feature Preservation)

**Pros:**
- Preserves your advanced hybrid search capabilities
- Maintains core compatibility with original
- Keeps valuable features like language detection
- Gradual migration possible

**Cons:**
- Some schema differences remain
- Requires ongoing maintenance of differences
- More complex codebase

**Timeline:** 1-2 weeks

### Option 3: Minimal Changes (Recommended for Short-term)

**Pros:**
- Minimal disruption to current functionality
- Quick implementation
- Preserves all advanced features

**Cons:**
- Maintains significant differences from original
- Harder to merge future improvements
- Continued divergence over time

**Timeline:** 3-5 days

## Immediate Next Steps

1. **Backup Current Implementation**: Create a branch with your current advanced implementation
2. **Choose Migration Strategy**: Decide between full alignment vs. hybrid approach
3. **Start with Database Schema**: Begin with Phase 1 database changes as they have the most impact
4. **Preserve Hybrid Search**: Ensure your hybrid search functionality is maintained throughout the transition
5. **Update Tests**: Modify your comprehensive test suite to work with the simplified approach

## Conclusion

Your project has evolved into a sophisticated code analysis system that significantly exceeds the original repository's capabilities. The transition plan above provides multiple paths to align with the original while preserving your valuable hybrid search functionality. The hybrid approach (Option 2) is recommended as it balances compatibility with feature preservation.

