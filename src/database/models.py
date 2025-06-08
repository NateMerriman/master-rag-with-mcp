"""
Database models and schema definitions for the MCP Crawl4AI RAG project.
"""

from typing import Dict, Any, Optional
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

# Future migration for FK constraint (Task 2.3)
ADD_FOREIGN_KEY_CONSTRAINT_SQL = """
ALTER TABLE crawled_pages 
ADD CONSTRAINT fk_crawled_pages_source_id 
FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE;
"""