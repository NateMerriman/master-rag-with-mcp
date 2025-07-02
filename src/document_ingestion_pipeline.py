#!/usr/bin/env python3
"""
Document Ingestion Pipeline

Based on the agentic-rag-knowledge-graph reference implementation, this module
provides a clean separation between content extraction (handled by AdvancedWebCrawler)
and document processing. The pipeline takes clean markdown as input and manages
its entire lifecycle through title/metadata extraction, semantic chunking, 
embedding, and database storage.

Architecture:
    URL → AdvancedWebCrawler → Clean Markdown → DocumentIngestionPipeline → SemanticChunker → Storage

Key Components:
    - DocumentIngestionPipeline: Main orchestrator
    - SemanticChunker: LLM-powered chunking with fallback
    - EmbeddingGenerator: Vector embedding creation
    - DocumentStorage: Database persistence
"""

from typing import Dict, List, Optional, Any, Callable
import asyncio
import time
import logging
import re
import os
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

# Pydantic models for configuration and results
from pydantic import BaseModel, Field, field_validator

# OpenAI for LLM-powered semantic chunking
import openai
from typing import Union

logger = logging.getLogger(__name__)


def _retry_with_backoff(fn, *args, max_retries=3, base_delay=1, **kwargs):
    """
    Retry function with exponential backoff for LLM API calls.
    Lighter version for chunking tasks with fewer retries.
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Rate limit hit, retrying in {delay}s... ({attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"API error: {e}, retrying in {delay}s... ({attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay *= 2


# Configuration Models
class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")
    max_chunk_size: int = Field(default=2000, ge=500, le=10000, description="Maximum chunk size")
    use_semantic_splitting: bool = Field(default=True, description="Enable LLM-powered semantic chunking")
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v


class PipelineConfig(BaseModel):
    """Configuration for the document ingestion pipeline."""
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    extract_entities: bool = Field(default=False, description="Enable entity extraction")
    generate_embeddings: bool = Field(default=True, description="Generate vector embeddings")
    store_in_database: bool = Field(default=True, description="Store chunks in database")


# Data Models
@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata."""
    content: str
    chunk_index: int
    start_position: int
    end_position: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count is None:
            # Rough estimation: ~4 characters per token
            self.token_count = len(self.content) // 4


@dataclass
class PipelineResult:
    """Result of document processing through the pipeline."""
    document_id: str
    title: str
    source_url: str
    chunks_created: int
    entities_extracted: int
    embeddings_generated: int
    processing_time_ms: float
    success: bool
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# Component Interfaces
class SemanticChunker:
    """
    Semantic document chunker using LLM for intelligent splitting.
    
    This implementation follows the reference architecture with:
    - LLM-powered semantic boundary detection
    - Fallback to rule-based chunking
    - Markdown structure awareness
    - Configurable chunk sizes and overlap
    """
    
    def __init__(self, config: ChunkingConfig):
        """Initialize the semantic chunker."""
        self.config = config
        
        # Initialize OpenAI client
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not set - semantic chunking will fall back to rule-based")
        
        # Model for semantic chunking (lightweight model for efficiency)
        self.model = os.getenv("CONTEXTUAL_MODEL", "gpt-4o-mini-2024-07-18")
        
        # Enable OpenAI API key
        openai.api_key = self.openai_api_key
        
    async def chunk_document(
        self, 
        content: str, 
        title: str, 
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk a document into semantically coherent pieces.
        
        Args:
            content: The markdown content to chunk
            title: Document title for metadata
            source: Source URL or identifier
            metadata: Additional metadata to include
            
        Returns:
            List of DocumentChunk objects
        """
        if not content.strip():
            return []
            
        base_metadata = {
            "title": title,
            "source": source,
            **(metadata or {})
        }
        
        # Try semantic chunking if enabled and content is large enough
        if self.config.use_semantic_splitting and len(content) > self.config.chunk_size:
            try:
                semantic_chunks = await self._semantic_chunk(content)
                if semantic_chunks:
                    return self._create_chunk_objects(semantic_chunks, content, base_metadata)
            except Exception as e:
                logger.warning(f"Semantic chunking failed, falling back to rule-based: {e}")
        
        # Fallback to rule-based chunking
        return self._simple_chunk(content, base_metadata)
    
    async def _semantic_chunk(self, content: str) -> List[str]:
        """
        Perform semantic chunking using LLM.
        
        This follows the reference implementation pattern:
        1. Split on natural structural boundaries
        2. Group sections into semantic chunks
        3. Use LLM for complex splitting decisions
        """
        # First, split on natural boundaries (headers, paragraphs, lists)
        sections = self._split_on_structure(content)
        
        # Group sections into semantic chunks
        chunks = []
        current_chunk = ""
        
        for section in sections:
            # Check if adding this section would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + section if current_chunk else section
            
            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                # Current chunk is ready
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Handle oversized sections
                if len(section) > self.config.max_chunk_size:
                    # Split the section semantically using LLM
                    sub_chunks = await self._split_long_section(section)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = section
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _split_on_structure(self, content: str) -> List[str]:
        """Split content on natural structural boundaries."""
        # Split on markdown headers, paragraph breaks, and list items
        patterns = [
            r'^#{1,6}\s+.*$',  # Headers
            r'^\s*[-*+]\s+.*$',  # Unordered lists
            r'^\s*\d+\.\s+.*$',  # Ordered lists
            r'^\s*```.*?```\s*$',  # Code blocks
        ]
        
        # Split on double newlines (paragraph breaks) first
        paragraphs = re.split(r'\n\s*\n', content)
        
        sections = []
        current_section = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if this is a structural element
            is_structural = any(re.match(pattern, paragraph, re.MULTILINE) for pattern in patterns)
            
            if is_structural and current_section:
                # Save current section and start new one
                sections.append(current_section.strip())
                current_section = paragraph
            else:
                # Add to current section
                if current_section:
                    current_section += "\n\n" + paragraph
                else:
                    current_section = paragraph
        
        # Add final section
        if current_section:
            sections.append(current_section.strip())
        
        return sections
    
    async def _split_long_section(self, section: str) -> List[str]:
        """Split a long section using LLM guidance."""
        if not self.openai_api_key:
            logger.info("OpenAI API key not available, using rule-based splitting")
            return self._simple_split(section)
            
        try:
            logger.info(f"Using LLM to split long section ({len(section)} chars)")
            
            prompt = f"""You are an expert document processor. Split the following text into semantically coherent chunks that preserve meaning and context.

REQUIREMENTS:
1. Each chunk should be roughly {self.config.chunk_size} characters long
2. Chunks must not exceed {self.config.max_chunk_size} characters
3. Split at natural semantic boundaries (complete thoughts, paragraph breaks, section breaks)
4. Preserve context - don't split in the middle of sentences or concepts
5. Maintain readability and coherence in each chunk

INSTRUCTIONS:
- Return ONLY the split text chunks
- Separate chunks with exactly "---CHUNK_SEPARATOR---" 
- Do not add any commentary, explanations, or metadata
- Ensure each chunk is a complete, readable unit

TEXT TO SPLIT:
{section}"""

            # Use OpenAI chat completion for semantic splitting with retry logic
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _retry_with_backoff(
                    openai.chat.completions.create,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert document processor that splits text into semantically coherent chunks."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent results
                    max_tokens=4000,  # Enough for most splitting tasks
                )
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the chunks
            chunks = [chunk.strip() for chunk in result.split("---CHUNK_SEPARATOR---") if chunk.strip()]
            
            # Validate chunks don't exceed max size
            validated_chunks = []
            for chunk in chunks:
                if len(chunk) <= self.config.max_chunk_size:
                    validated_chunks.append(chunk)
                else:
                    # If LLM produced oversized chunks, split them further
                    logger.warning(f"LLM produced oversized chunk ({len(chunk)} chars), splitting further")
                    validated_chunks.extend(self._simple_split(chunk))
            
            if validated_chunks:
                logger.info(f"LLM successfully split section into {len(validated_chunks)} chunks")
                return validated_chunks
            else:
                logger.warning("LLM splitting produced no valid chunks, using fallback")
                return self._simple_split(section)
                
        except Exception as e:
            logger.warning(f"LLM section splitting failed: {e}, using fallback")
            return self._simple_split(section)
    
    def _simple_split(self, text: str) -> List[str]:
        """Simple rule-based text splitting."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > self.config.chunk_size and current_chunk:
                # Try to find a good break point (sentence boundary)
                chunk_text = " ".join(current_chunk)
                
                # Look for sentence endings in the last part of the chunk
                sentences = re.split(r'[.!?]+\s+', chunk_text)
                if len(sentences) > 1:
                    # Keep all but the last incomplete sentence
                    complete_sentences = sentences[:-1]
                    chunk_text = ". ".join(complete_sentences) + "."
                    
                    # Put the incomplete sentence back for the next chunk
                    remaining = sentences[-1]
                    current_chunk = remaining.split() if remaining.strip() else []
                    current_length = len(remaining)
                else:
                    current_chunk = []
                    current_length = 0
                
                chunks.append(chunk_text)
            
            current_chunk.append(word)
            current_length += word_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _simple_chunk(self, content: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create chunks using simple rule-based splitting."""
        chunks = self._simple_split(content)
        return self._create_chunk_objects(chunks, content, base_metadata)
    
    def _create_chunk_objects(
        self, 
        chunks: List[str], 
        original_content: str, 
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create DocumentChunk objects from text chunks."""
        chunk_objects = []
        current_pos = 0
        
        for i, chunk_text in enumerate(chunks):
            # Find position in original content
            start_pos = original_content.find(chunk_text, current_pos)
            if start_pos == -1:
                # Fallback if exact match not found
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk_text)
            
            # Create chunk metadata
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_method": "semantic" if self.config.use_semantic_splitting else "simple"
            }
            
            chunk_objects.append(DocumentChunk(
                content=chunk_text,
                chunk_index=i,
                start_position=start_pos,
                end_position=end_pos,
                metadata=chunk_metadata
            ))
            
            current_pos = end_pos
        
        return chunk_objects


class EmbeddingGenerator:
    """
    Component for generating vector embeddings from text chunks.
    
    This integrates with the existing embedding infrastructure
    and handles batch processing for efficiency. Features:
    - Batch processing for optimal performance
    - Automatic chunking of large batches
    - Comprehensive error handling and retry logic
    - Progress tracking and logging
    """
    
    def __init__(self, batch_size: int = 100, max_retries: int = 3):
        """Initialize the embedding generator.
        
        Args:
            batch_size: Number of texts to process in each batch
            max_retries: Maximum number of retry attempts for failed embeddings
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.embedding_dimension = 1536  # OpenAI text-embedding-3-small dimension
        
        # Import existing embedding functions
        try:
            from .utils import create_embeddings_batch, create_embedding
            self.create_embeddings_batch = create_embeddings_batch
            self.create_embedding = create_embedding
            self.available = True
            logger.info(f"EmbeddingGenerator initialized with batch_size={batch_size}")
        except ImportError:
            try:
                from utils import create_embeddings_batch, create_embedding
                self.create_embeddings_batch = create_embeddings_batch
                self.create_embedding = create_embedding
                self.available = True
                logger.info(f"EmbeddingGenerator initialized with batch_size={batch_size}")
            except ImportError:
                logger.error("Could not import embedding functions from utils")
                self.create_embeddings_batch = None
                self.create_embedding = None
                self.available = False
        
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks with comprehensive error handling.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of DocumentChunk objects with embeddings added
        """
        if not chunks:
            return chunks
            
        if not self.available:
            logger.warning("Embedding functions not available, skipping embeddings")
            return chunks
            
        try:
            start_time = time.time()
            
            # Extract text content for embedding
            texts = [chunk.content for chunk in chunks]
            
            # Filter out empty or very short texts
            valid_indices = []
            valid_texts = []
            for i, text in enumerate(texts):
                if text and len(text.strip()) >= 10:  # Minimum viable text length
                    valid_indices.append(i)
                    valid_texts.append(text)
                else:
                    logger.debug(f"Skipping embedding for chunk {i}: text too short or empty")
            
            if not valid_texts:
                logger.warning("No valid texts for embedding generation")
                return chunks
            
            logger.info(f"Generating embeddings for {len(valid_texts)}/{len(chunks)} chunks")
            
            # Generate embeddings with batching and error handling
            embeddings = await self._generate_embeddings_with_batching(valid_texts)
            
            # Attach embeddings to valid chunks
            embedding_count = 0
            for i, embedding in zip(valid_indices, embeddings):
                if embedding and len(embedding) == self.embedding_dimension:
                    chunks[i].embedding = embedding
                    embedding_count += 1
                else:
                    logger.warning(f"Invalid embedding for chunk {i}: dimension {len(embedding) if embedding else 0}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully generated {embedding_count} embeddings in {elapsed_time:.2f}s")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return chunks without embeddings rather than failing completely
            return chunks
    
    async def _generate_embeddings_with_batching(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using batching and retry logic.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        all_embeddings = []
        
        # Process in batches for memory efficiency and rate limiting
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            # Generate embeddings for this batch with retry logic
            batch_embeddings = await self._generate_batch_with_retry(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def _generate_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with retry logic.
        
        Args:
            texts: Batch of text strings to embed
            
        Returns:
            List of embedding vectors for the batch
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Use existing batch embedding function in executor to avoid blocking
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.create_embeddings_batch,
                    texts
                )
                
                # Validate embeddings
                if not embeddings or len(embeddings) != len(texts):
                    raise ValueError(f"Invalid embeddings: expected {len(texts)}, got {len(embeddings) if embeddings else 0}")
                
                # Validate embedding dimensions
                for i, embedding in enumerate(embeddings):
                    if not embedding or len(embedding) != self.embedding_dimension:
                        logger.warning(f"Invalid embedding dimension for text {i}: {len(embedding) if embedding else 0}")
                        # Replace with zero vector as fallback
                        embeddings[i] = [0.0] * self.embedding_dimension
                
                return embeddings
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Embedding generation failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Embedding generation failed after {self.max_retries} attempts: {e}")
        
        # If all retries failed, return zero vectors as fallback
        logger.warning(f"Using zero vectors as fallback for {len(texts)} failed embeddings")
        return [[0.0] * self.embedding_dimension for _ in texts]
    
    async def _generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text (fallback method).
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        if not self.create_embedding:
            logger.warning("Single embedding function not available")
            return [0.0] * self.embedding_dimension
            
        try:
            embedding = await asyncio.get_event_loop().run_in_executor(
                None,
                self.create_embedding,
                text
            )
            
            if not embedding or len(embedding) != self.embedding_dimension:
                logger.warning(f"Invalid single embedding dimension: {len(embedding) if embedding else 0}")
                return [0.0] * self.embedding_dimension
                
            return embedding
            
        except Exception as e:
            logger.error(f"Single embedding generation failed: {e}")
            return [0.0] * self.embedding_dimension


class DocumentStorage:
    """
    Component for storing processed documents and chunks in the database.
    
    This maintains compatibility with the existing Supabase schema
    and integrates with the current storage infrastructure.
    """
    
    def __init__(self):
        """Initialize the document storage component."""
        # Import existing storage functions
        try:
            from .utils import get_supabase_client, add_documents_to_supabase
            self.get_supabase_client = get_supabase_client
            self.add_documents_to_supabase = add_documents_to_supabase
        except ImportError:
            try:
                from utils import get_supabase_client, add_documents_to_supabase
                self.get_supabase_client = get_supabase_client
                self.add_documents_to_supabase = add_documents_to_supabase
            except ImportError:
                logger.error("Could not import storage functions from utils")
                self.get_supabase_client = None
                self.add_documents_to_supabase = None
        
    async def store_document(
        self, 
        title: str,
        source_url: str,
        original_content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Store a document and its chunks in the database.
        
        Args:
            title: Document title
            source_url: Source URL
            original_content: Original markdown content
            chunks: List of processed chunks
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        try:
            # Generate document ID
            document_id = self._generate_document_id(source_url)
            
            # Store chunks using existing infrastructure
            await self._store_chunks(document_id, chunks, original_content)
            
            logger.info(f"Stored document {document_id} with {len(chunks)} chunks")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            raise
    
    def _generate_document_id(self, source_url: str) -> str:
        """Generate a unique document ID from the source URL."""
        # Use URL as base for ID
        parsed = urlparse(source_url)
        return f"{parsed.netloc}{parsed.path}".replace("/", "_")
    
    async def _store_chunks(
        self, 
        document_id: str, 
        chunks: List[DocumentChunk],
        original_content: str
    ):
        """Store chunks in the database using existing infrastructure."""
        if not self.add_documents_to_supabase or not self.get_supabase_client:
            logger.error("Storage functions not available")
            raise RuntimeError("Storage infrastructure not initialized")
            
        try:
            # Get Supabase client
            supabase_client = self.get_supabase_client()
            
            # Prepare data in the format expected by existing infrastructure
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            
            for chunk in chunks:
                urls.append(chunk.metadata.get("source", ""))
                chunk_numbers.append(chunk.chunk_index)
                contents.append(chunk.content)
                
                # Enhance metadata with pipeline information
                enhanced_metadata = {
                    **chunk.metadata,
                    "document_id": document_id,
                    "pipeline_processed": True,
                    "processing_timestamp": datetime.now().isoformat(),
                    "chunk_method": chunk.metadata.get("chunk_method", "semantic"),
                    "token_count": chunk.token_count,
                    "start_position": chunk.start_position,
                    "end_position": chunk.end_position,
                }
                metadatas.append(enhanced_metadata)
            
            # Create URL to full document mapping
            url_to_full_document = {
                chunk.metadata.get("source", ""): original_content
                for chunk in chunks
            }
            
            # Use existing infrastructure to store chunks
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.add_documents_to_supabase,
                supabase_client,
                urls,
                chunk_numbers, 
                contents,
                metadatas,
                url_to_full_document,
                None,  # strategy_config (optional)
                20     # batch_size
            )
            
            logger.info(f"Successfully stored {len(chunks)} chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise


class DocumentIngestionPipeline:
    """
    Main pipeline for ingesting documents into the vector database.
    
    Based on the agentic-rag-knowledge-graph reference implementation,
    this pipeline orchestrates the entire process from clean markdown
    input to database storage.
    
    Architecture:
        Clean Markdown → Title/Metadata Extraction → Semantic Chunking → 
        Embedding Generation → Database Storage → Result
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the document ingestion pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Initialize components
        self.chunker = SemanticChunker(config.chunking)
        self.embedder = EmbeddingGenerator()
        self.storage = DocumentStorage()
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize pipeline components and database connections."""
        if self._initialized:
            return
            
        logger.info("Initializing document ingestion pipeline...")
        
        # Initialize components with actual clients
        try:
            # Check chunker initialization
            if not hasattr(self.chunker, 'config'):
                raise RuntimeError("SemanticChunker not properly initialized")
                
            # Check embedder functions availability
            if self.config.generate_embeddings:
                if not hasattr(self.embedder, 'create_embeddings_batch'):
                    logger.warning("Embedding functions not available - embeddings will be skipped")
                    self.config.generate_embeddings = False
            
            # Check storage functions availability  
            if self.config.store_in_database:
                if not hasattr(self.storage, 'add_documents_to_supabase'):
                    logger.warning("Storage functions not available - database storage will be skipped")
                    self.config.store_in_database = False
            
            logger.info(f"Pipeline initialized with chunking: {self.config.chunking.use_semantic_splitting}, "
                       f"embeddings: {self.config.generate_embeddings}, "
                       f"storage: {self.config.store_in_database}")
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            # Continue with degraded functionality rather than failing completely
            logger.warning("Continuing with degraded functionality")
        
        self._initialized = True
        logger.info("Document ingestion pipeline initialized")
    
    async def process_document(
        self,
        content: str,
        source_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Process a single document through the complete pipeline.
        
        This is the main entry point that takes clean markdown from
        the AdvancedWebCrawler and processes it through all stages.
        
        Args:
            content: Clean markdown content from AdvancedWebCrawler
            source_url: Source URL of the document
            metadata: Additional metadata from the crawler
            
        Returns:
            PipelineResult with processing statistics and any errors
        """
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Step 1: Extract title and enhance metadata
            title = self._extract_title(content)
            document_metadata = self._extract_document_metadata(content, source_url)
            
            # Merge with provided metadata
            if metadata:
                document_metadata.update(metadata)
            
            logger.info(f"Processing document: {title}")
            
            # Step 2: Chunk the document
            chunks = await self.chunker.chunk_document(
                content=content,
                title=title,
                source=source_url,
                metadata=document_metadata
            )
            
            if not chunks:
                logger.warning(f"No chunks created for {title}")
                return PipelineResult(
                    document_id="",
                    title=title,
                    source_url=source_url,
                    chunks_created=0,
                    entities_extracted=0,
                    embeddings_generated=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    success=False,
                    errors=["No chunks created"]
                )
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings (if enabled)
            embeddings_generated = 0
            if self.config.generate_embeddings:
                chunks = await self.embedder.embed_chunks(chunks)
                embeddings_generated = len([c for c in chunks if c.embedding])
                logger.info(f"Generated embeddings for {embeddings_generated} chunks")
            
            # Step 4: Store in database (if enabled)
            document_id = ""
            if self.config.store_in_database:
                document_id = await self.storage.store_document(
                    title=title,
                    source_url=source_url,
                    original_content=content,
                    chunks=chunks,
                    metadata=document_metadata
                )
                logger.info(f"Stored document with ID: {document_id}")
            
            # Step 5: Extract entities (placeholder for future implementation)
            entities_extracted = 0
            if self.config.extract_entities:
                # Placeholder for entity extraction
                logger.info("Entity extraction not yet implemented")
            
            processing_time = (time.time() - start_time) * 1000
            
            return PipelineResult(
                document_id=document_id,
                title=title,
                source_url=source_url,
                chunks_created=len(chunks),
                entities_extracted=entities_extracted,
                embeddings_generated=embeddings_generated,
                processing_time_ms=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Pipeline processing failed: {str(e)}"
            logger.error(error_msg)
            
            return PipelineResult(
                document_id="",
                title=self._extract_title(content) if content else "Unknown",
                source_url=source_url,
                chunks_created=0,
                entities_extracted=0,
                embeddings_generated=0,
                processing_time_ms=processing_time,
                success=False,
                errors=[error_msg]
            )
    
    def _extract_title(self, content: str) -> str:
        """
        Extract title from markdown content.
        
        Follows the reference implementation pattern with enhanced
        title detection for various markdown formats.
        """
        if not content:
            return "Untitled Document"
            
        lines = content.split('\n')
        
        # Look for H1 headers (# Title)
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                title = line[2:].strip()
                if title:
                    # Clean up the title
                    title = re.sub(r'[#*_`]', '', title).strip()
                    return title
        
        # Look for alternative title formats
        for line in lines:
            line = line.strip()
            
            # Setext-style H1 (title followed by ===)
            if line and not line.startswith('#'):
                next_line_idx = lines.index(line) + 1
                if (next_line_idx < len(lines) and 
                    lines[next_line_idx].strip().startswith('=')):
                    title = re.sub(r'[#*_`]', '', line).strip()
                    if title:
                        return title
                        
            # Look for title: or Title: patterns
            if ':' in line and any(word in line.lower() for word in ['title', 'name', 'subject']):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    title = parts[1].strip()
                    if title:
                        return re.sub(r'[#*_`]', '', title).strip()
        
        # Fallback: use first substantial non-header line
        for line in lines:
            line = line.strip()
            if (line and 
                not line.startswith('#') and 
                not line.startswith('```') and
                not line.startswith('---') and
                len(line) > 5):
                # Use first meaningful line, truncated
                title = re.sub(r'[#*_`]', '', line).strip()
                return title[:50] + "..." if len(title) > 50 else title
        
        return "Untitled Document"
    
    def _extract_document_metadata(self, content: str, source_url: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from document content.
        
        This includes content analysis, complexity scoring, and structural
        information that aids in search and retrieval.
        """
        if not content:
            return {
                "source_url": source_url,
                "domain": urlparse(source_url).netloc,
                "word_count": 0,
                "character_count": 0,
                "error": "No content provided"
            }
        
        try:
            # Basic metrics
            word_count = len(content.split())
            character_count = len(content)
            
            # URL analysis
            parsed_url = urlparse(source_url)
            domain = parsed_url.netloc
            
            # Header analysis
            headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            header_count = len(headers)
            max_header_depth = max([len(h[0]) for h in headers]) if headers else 0
            
            # Code block analysis
            code_blocks = re.findall(r'```(\w+)?\n(.*?)```', content, re.DOTALL)
            code_blocks_count = len(code_blocks)
            programming_languages = list(set([block[0] for block in code_blocks if block[0]]))
            
            # Link analysis
            all_links = re.findall(r'\[([^\]]*)\]\(([^)]+)\)', content)
            total_links_count = len(all_links)
            
            # Classify internal vs external links
            internal_links = []
            external_links = []
            
            for link_text, link_url in all_links:
                if link_url.startswith(('http://', 'https://')):
                    link_domain = urlparse(link_url).netloc
                    if link_domain == domain:
                        internal_links.append((link_text, link_url))
                    else:
                        external_links.append((link_text, link_url))
                else:
                    # Relative links are considered internal
                    internal_links.append((link_text, link_url))
            
            # Content type classification (simplified version)
            content_types = self._classify_content_type(content)
            
            # Complexity scoring
            complexity_score = self._calculate_complexity_score(
                word_count, header_count, code_blocks_count, total_links_count
            )
            
            # Estimated reading time (average 200 words per minute)
            estimated_reading_time_minutes = max(1, round(word_count / 200))
            
            return {
                "source_url": source_url,
                "domain": domain,
                "word_count": word_count,
                "character_count": character_count,
                "headers": [h[1] for h in headers],
                "header_count": header_count,
                "max_header_depth": max_header_depth,
                "code_blocks_count": code_blocks_count,
                "programming_languages": programming_languages,
                "total_links_count": total_links_count,
                "internal_links_count": len(internal_links),
                "external_links_count": len(external_links),
                "content_types": content_types,
                "complexity_score": complexity_score,
                "complexity_category": self._get_complexity_category(complexity_score),
                "estimated_reading_time_minutes": estimated_reading_time_minutes,
                "extraction_timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {
                "source_url": source_url,
                "domain": urlparse(source_url).netloc,
                "word_count": len(content.split()) if content else 0,
                "character_count": len(content) if content else 0,
                "error": str(e)
            }
    
    def _classify_content_type(self, content: str) -> List[str]:
        """Classify the content type based on indicators."""
        content_lower = content.lower()
        content_types = []
        
        # Technical documentation
        tech_indicators = ['api', 'function', 'class', 'method', 'parameter', 'usage', '```', 'installation']
        if any(indicator in content_lower for indicator in tech_indicators):
            content_types.append('technical')
        
        # Tutorial/guide
        tutorial_indicators = ['step', 'tutorial', 'guide', 'how to', 'getting started']
        if any(indicator in content_lower for indicator in tutorial_indicators):
            content_types.append('tutorial')
        
        # Reference material
        reference_indicators = ['reference', 'documentation', 'spec', 'specification']
        if any(indicator in content_lower for indicator in reference_indicators):
            content_types.append('reference')
        
        # Academic/research
        academic_indicators = ['abstract', 'methodology', 'research', 'study', 'analysis']
        if any(indicator in content_lower for indicator in academic_indicators):
            content_types.append('academic')
        
        return content_types if content_types else ['general']
    
    def _calculate_complexity_score(self, word_count: int, header_count: int, 
                                   code_blocks_count: int, links_count: int) -> int:
        """Calculate a complexity score from 0-100 based on content metrics."""
        score = 0
        
        # Word count contribution (0-30 points)
        if word_count > 5000:
            score += 30
        elif word_count > 2000:
            score += 20
        elif word_count > 500:
            score += 10
        
        # Structure complexity (0-25 points)
        if header_count > 10:
            score += 25
        elif header_count > 5:
            score += 15
        elif header_count > 2:
            score += 10
        
        # Code complexity (0-25 points)
        if code_blocks_count > 10:
            score += 25
        elif code_blocks_count > 5:
            score += 15
        elif code_blocks_count > 1:
            score += 10
        
        # Link complexity (0-20 points)
        if links_count > 20:
            score += 20
        elif links_count > 10:
            score += 15
        elif links_count > 3:
            score += 10
        
        return min(score, 100)  # Cap at 100
    
    def _get_complexity_category(self, score: int) -> str:
        """Convert complexity score to category."""
        if score >= 70:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"


# Result Models
class PipelineResult(BaseModel):
    """Result of document processing pipeline."""
    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Extracted document title")
    source_url: str = Field(..., description="Source URL")
    chunks_created: int = Field(..., description="Number of chunks created")
    entities_extracted: int = Field(default=0, description="Number of entities extracted")
    embeddings_generated: int = Field(default=0, description="Number of embeddings generated")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    success: bool = Field(..., description="Whether processing succeeded")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")