#!/usr/bin/env python3
"""
Unit tests for the SemanticChunker implementation.

Tests both LLM-powered semantic chunking and rule-based fallback mechanisms.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from document_ingestion_pipeline import (
    SemanticChunker,
    ChunkingConfig,
    DocumentChunk,
    _retry_with_backoff
)


class TestChunkingConfig:
    """Test chunking configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = ChunkingConfig(
            chunk_size=1000,
            chunk_overlap=200,
            max_chunk_size=2000,
            use_semantic_splitting=True
        )
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_chunk_size == 2000
        assert config.use_semantic_splitting is True
    
    def test_invalid_overlap(self):
        """Test validation of chunk overlap."""
        with pytest.raises(ValueError, match="Chunk overlap .* must be less than chunk size"):
            ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=1000  # Same as chunk_size - should fail
            )
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_chunk_size == 2000
        assert config.use_semantic_splitting is True


class TestDocumentChunk:
    """Test DocumentChunk data model."""
    
    def test_chunk_creation(self):
        """Test basic chunk creation."""
        metadata = {"title": "Test Doc", "source": "test.md"}
        chunk = DocumentChunk(
            content="This is test content",
            chunk_index=0,
            start_position=0,
            end_position=20,
            metadata=metadata
        )
        
        assert chunk.content == "This is test content"
        assert chunk.chunk_index == 0
        assert chunk.start_position == 0
        assert chunk.end_position == 20
        assert chunk.metadata == metadata
        assert chunk.token_count == 5  # ~20 chars / 4 = 5 tokens
        assert chunk.embedding is None
    
    def test_token_count_calculation(self):
        """Test automatic token count calculation."""
        chunk = DocumentChunk(
            content="This is a longer piece of content for testing token estimation",
            chunk_index=0,
            start_position=0,
            end_position=63,
            metadata={}
        )
        
        # Should estimate ~15 tokens (63 chars / 4)
        assert chunk.token_count == 15


class TestSemanticChunker:
    """Test SemanticChunker functionality."""
    
    def test_init_with_api_key(self):
        """Test chunker initialization with API key."""
        config = ChunkingConfig()
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            chunker = SemanticChunker(config)
            
            assert chunker.config == config
            assert chunker.openai_api_key == "test-key"
            assert chunker.model == "gpt-4o-mini-2024-07-18"  # Default model
    
    def test_init_without_api_key(self):
        """Test chunker initialization without API key."""
        config = ChunkingConfig()
        
        with patch.dict(os.environ, {}, clear=True):
            chunker = SemanticChunker(config)
            
            assert chunker.openai_api_key is None
    
    def test_split_on_structure(self):
        """Test structural splitting of markdown."""
        config = ChunkingConfig()
        chunker = SemanticChunker(config)
        
        content = """# Main Title

This is the first paragraph under the main title.

This is the second paragraph.

## Section Header

This is content under the section header.

- List item 1
- List item 2

1. Numbered item 1
2. Numbered item 2

```python
def example():
    return "code block"
```

Final paragraph."""
        
        sections = chunker._split_on_structure(content)
        
        # Should create multiple sections
        assert len(sections) > 1
        
        # Check that headers are preserved
        headers = [s for s in sections if s.strip().startswith('#')]
        assert len(headers) >= 2
        
        # Check that code blocks are preserved
        code_sections = [s for s in sections if '```' in s]
        assert len(code_sections) >= 1
    
    def test_simple_split(self):
        """Test simple rule-based splitting."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = SemanticChunker(config)
        
        text = "This is a test sentence. This is another sentence. And one more sentence for testing."
        chunks = chunker._simple_split(text)
        
        assert len(chunks) > 1
        
        # Check that chunks respect size limits
        for chunk in chunks[:-1]:  # All except last
            assert len(chunk) <= config.chunk_size + 10  # Allow some variance for sentence boundaries
    
    @pytest.mark.asyncio
    async def test_chunk_document_empty_content(self):
        """Test chunking empty content."""
        config = ChunkingConfig()
        chunker = SemanticChunker(config)
        
        chunks = await chunker.chunk_document("", "Empty Doc", "empty.md")
        
        assert len(chunks) == 0
    
    @pytest.mark.asyncio
    async def test_chunk_document_short_content(self):
        """Test chunking content shorter than chunk size."""
        config = ChunkingConfig(chunk_size=100)
        chunker = SemanticChunker(config)
        
        content = "This is a short document."
        chunks = await chunker.chunk_document(content, "Short Doc", "short.md")
        
        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].metadata["title"] == "Short Doc"
        assert chunks[0].metadata["source"] == "short.md"
    
    @pytest.mark.asyncio
    async def test_chunk_document_fallback_to_simple(self):
        """Test fallback to simple chunking when semantic fails."""
        config = ChunkingConfig(chunk_size=50, use_semantic_splitting=True)
        chunker = SemanticChunker(config)
        
        # Mock the semantic chunking to fail
        with patch.object(chunker, '_semantic_chunk', side_effect=Exception("LLM failed")):
            content = "This is test content for fallback testing. " * 10
            chunks = await chunker.chunk_document(content, "Fallback Test", "fallback.md")
            
            # Should still return chunks from simple chunking
            assert len(chunks) > 0
            assert chunks[0].metadata["title"] == "Fallback Test"
            assert chunks[0].metadata["chunk_method"] == "simple"
    
    @pytest.mark.asyncio 
    async def test_chunk_document_no_api_key(self):
        """Test chunking when no API key is available."""
        config = ChunkingConfig(chunk_size=50, use_semantic_splitting=True)
        
        with patch.dict(os.environ, {}, clear=True):
            chunker = SemanticChunker(config)
            content = "This is test content that should be chunked. " * 10
            chunks = await chunker.chunk_document(content, "No API Test", "no_api.md")
            
            # Should fall back to simple chunking
            assert len(chunks) > 0
            assert chunks[0].metadata["chunk_method"] == "simple"
    
    @pytest.mark.asyncio
    async def test_split_long_section_with_mock_llm(self):
        """Test LLM-powered section splitting with mocked response."""
        config = ChunkingConfig(chunk_size=100, max_chunk_size=200)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            chunker = SemanticChunker(config)
            
            # Mock OpenAI response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = """First chunk of content that contains the first part.

---CHUNK_SEPARATOR---

Second chunk of content that contains the second part."""
            
            with patch('openai.chat.completions.create', return_value=mock_response):
                long_section = "This is a very long section that needs to be split into multiple parts. " * 10
                chunks = await chunker._split_long_section(long_section)
                
                assert len(chunks) == 2
                assert "First chunk" in chunks[0]
                assert "Second chunk" in chunks[1]
                
                # Check that chunks don't exceed max size
                for chunk in chunks:
                    assert len(chunk) <= config.max_chunk_size


class TestRetryMechanism:
    """Test retry mechanism for LLM calls."""
    
    def test_retry_success_on_first_try(self):
        """Test successful call on first attempt."""
        mock_fn = Mock(return_value="success")
        
        result = _retry_with_backoff(mock_fn, "arg1", "arg2", kwarg1="value1")
        
        assert result == "success"
        assert mock_fn.call_count == 1
        mock_fn.assert_called_with("arg1", "arg2", kwarg1="value1")
    
    def test_retry_rate_limit_then_success(self):
        """Test retry on rate limit error then success."""
        import openai
        
        mock_fn = Mock(side_effect=[
            openai.RateLimitError("Rate limited", response=None, body=None),
            "success"
        ])
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = _retry_with_backoff(mock_fn, max_retries=2, base_delay=0.1)
        
        assert result == "success"
        assert mock_fn.call_count == 2
    
    def test_retry_exhausted(self):
        """Test that retries are exhausted and exception is raised."""
        import openai
        
        mock_fn = Mock(side_effect=openai.RateLimitError("Rate limited", response=None, body=None))
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(openai.RateLimitError):
                _retry_with_backoff(mock_fn, max_retries=2, base_delay=0.1)
        
        assert mock_fn.call_count == 2


class TestIntegration:
    """Integration tests for the chunking system."""
    
    @pytest.mark.asyncio
    async def test_realistic_document_chunking(self):
        """Test chunking a realistic document."""
        config = ChunkingConfig(
            chunk_size=200,
            chunk_overlap=50,
            use_semantic_splitting=False  # Use simple for predictable testing
        )
        chunker = SemanticChunker(config)
        
        content = """# AI Research Paper

## Abstract
This paper presents new findings in artificial intelligence research.
The research focuses on improving machine learning algorithms for
natural language processing tasks.

## Introduction
Artificial intelligence has seen tremendous growth in recent years.
Machine learning models have become increasingly sophisticated and
capable of handling complex tasks that were previously thought to
be impossible for computers.

## Methodology
Our approach combines traditional statistical methods with modern
deep learning techniques. We use a multi-layered neural network
architecture that can process text input and generate meaningful
representations.

### Data Collection
We collected data from multiple sources including academic papers,
web articles, and social media posts. The dataset contains over
1 million examples across different domains.

### Model Architecture
The model consists of three main components:
1. Input embedding layer
2. Transformer blocks
3. Output classification layer

## Results
Our experiments show significant improvements over baseline methods.
The model achieves 95% accuracy on the test dataset, which represents
a 10% improvement over previous state-of-the-art approaches.

## Conclusion
This work demonstrates the effectiveness of combining traditional and
modern techniques for natural language processing. Future work will
explore applications to other domains."""
        
        chunks = await chunker.chunk_document(content, "AI Research", "ai_research.md")
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should have proper metadata
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.metadata["title"] == "AI Research"
            assert chunk.metadata["source"] == "ai_research.md"
            assert chunk.metadata["total_chunks"] == len(chunks)
            assert chunk.token_count > 0
        
        # Content should be preserved
        reconstructed = " ".join(chunk.content for chunk in chunks)
        assert "AI Research Paper" in reconstructed
        assert "Abstract" in reconstructed
        assert "Conclusion" in reconstructed
        
        # Chunks should respect size constraints (with some tolerance)
        for chunk in chunks[:-1]:  # All except last
            assert len(chunk.content) <= config.chunk_size + 50  # Allow some variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])