"""
Database models and schema definitions for the MCP Crawl4AI RAG project.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Source:
    """
    Model for the sources table.
    Represents a crawled source (website/document) with metadata.
    """
    source_id: Optional[int] = None
    url: str = ""
    summary: Optional[str] = None
    total_word_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Supabase insertion."""
        result = {
            "url": self.url,
            "summary": self.summary,
            "total_word_count": self.total_word_count,
        }
        
        if self.source_id is not None:
            result["source_id"] = self.source_id
        if self.created_at is not None:
            result["created_at"] = self.created_at.isoformat()
        if self.updated_at is not None:
            result["updated_at"] = self.updated_at.isoformat()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Source":
        """Create Source from dictionary."""
        return cls(
            source_id=data.get("source_id"),
            url=data.get("url", ""),
            summary=data.get("summary"),
            total_word_count=data.get("total_word_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


@dataclass
class CrawledPage:
    """
    Model for the crawled_pages table.
    Represents a chunk of content from a crawled source.
    """
    id: Optional[int] = None
    url: Optional[str] = None
    chunk_number: Optional[int] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[list] = None
    source_id: Optional[int] = None  # New FK to sources table
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Supabase insertion."""
        result = {}
        
        if self.id is not None:
            result["id"] = self.id
        if self.url is not None:
            result["url"] = self.url
        if self.chunk_number is not None:
            result["chunk_number"] = self.chunk_number
        if self.content is not None:
            result["content"] = self.content
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.embedding is not None:
            result["embedding"] = self.embedding
        if self.source_id is not None:
            result["source_id"] = self.source_id
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrawledPage":
        """Create CrawledPage from dictionary."""
        return cls(
            id=data.get("id"),
            url=data.get("url"),
            chunk_number=data.get("chunk_number"),
            content=data.get("content"),
            metadata=data.get("metadata"),
            embedding=data.get("embedding"),
            source_id=data.get("source_id"),
        )


@dataclass
class CodeExample:
    """
    Model for the code_examples table.
    Represents extracted code blocks with metadata and embeddings.
    """
    id: Optional[int] = None
    source_id: Optional[int] = None
    code_content: str = ""
    summary: Optional[str] = None
    programming_language: Optional[str] = None
    complexity_score: Optional[int] = None
    embedding: Optional[List[float]] = None
    summary_embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Supabase insertion."""
        result = {}
        
        if self.id is not None:
            result["id"] = self.id
        if self.source_id is not None:
            result["source_id"] = self.source_id
        if self.code_content:
            result["code_content"] = self.code_content
        if self.summary is not None:
            result["summary"] = self.summary
        if self.programming_language is not None:
            result["programming_language"] = self.programming_language
        if self.complexity_score is not None:
            result["complexity_score"] = self.complexity_score
        if self.embedding is not None:
            result["embedding"] = self.embedding
        if self.summary_embedding is not None:
            result["summary_embedding"] = self.summary_embedding
        if self.created_at is not None:
            result["created_at"] = self.created_at.isoformat()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeExample":
        """Create CodeExample from dictionary."""
        return cls(
            id=data.get("id"),
            source_id=data.get("source_id"),
            code_content=data.get("code_content", ""),
            summary=data.get("summary"),
            programming_language=data.get("programming_language"),
            complexity_score=data.get("complexity_score"),
            embedding=data.get("embedding"),
            summary_embedding=data.get("summary_embedding"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        )
    
    def to_extracted_code_format(self) -> Dict[str, Any]:
        """Convert to format expected by code extraction pipeline."""
        return {
            "code_content": self.code_content,
            "summary": self.summary or "",
            "programming_language": self.programming_language or "unknown",
            "complexity_score": self.complexity_score or 1,
            "context": ""  # Context not stored in database model
        }


# SQL Schema Definitions for future migrations
SOURCES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    source_id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    summary TEXT,
    total_word_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

SOURCES_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url);",
    "CREATE INDEX IF NOT EXISTS idx_sources_created_at ON sources(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_sources_word_count ON sources(total_word_count);",
]

ADD_SOURCE_ID_COLUMN_SQL = """
ALTER TABLE crawled_pages 
ADD COLUMN IF NOT EXISTS source_id INTEGER;

CREATE INDEX IF NOT EXISTS idx_crawled_pages_source_id ON crawled_pages(source_id);
"""

# Code examples table schema (Task 2.2)
CODE_EXAMPLES_TABLE_SQL = """
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
"""

CODE_EXAMPLES_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_code_examples_embedding_hnsw ON code_examples USING hnsw (embedding vector_ip_ops) WITH (m = 16, ef_construction = 64);",
    "CREATE INDEX IF NOT EXISTS idx_code_examples_summary_embedding_hnsw ON code_examples USING hnsw (summary_embedding vector_ip_ops) WITH (m = 16, ef_construction = 64);",
    "CREATE INDEX IF NOT EXISTS idx_code_examples_content_tokens_gin ON code_examples USING gin(content_tokens);",
    "CREATE INDEX IF NOT EXISTS idx_code_examples_source_id ON code_examples(source_id);",
    "CREATE INDEX IF NOT EXISTS idx_code_examples_programming_language ON code_examples(programming_language);",
    "CREATE INDEX IF NOT EXISTS idx_code_examples_complexity_score ON code_examples(complexity_score);",
    "CREATE INDEX IF NOT EXISTS idx_code_examples_created_at ON code_examples(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_code_examples_language_complexity ON code_examples(programming_language, complexity_score);",
]

# Future migration for FK constraint (Task 2.3)
ADD_FOREIGN_KEY_CONSTRAINT_SQL = """
ALTER TABLE crawled_pages 
ADD CONSTRAINT fk_crawled_pages_source_id 
FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE;
"""