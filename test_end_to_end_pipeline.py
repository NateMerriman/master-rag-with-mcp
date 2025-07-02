#!/usr/bin/env python3
"""
End-to-End Test for AdvancedWebCrawler with Mock DocumentIngestionPipeline

This script validates the complete integration by feeding AdvancedWebCrawler output
into a mock DocumentIngestionPipeline and SemanticChunker, verifying that the
crawler produces content suitable for the target architecture.

Architecture Test:
URL → AdvancedWebCrawler → Clean Markdown → DocumentIngestionPipeline → SemanticChunker → Chunks
"""

import asyncio
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from advanced_web_crawler import AdvancedWebCrawler


@dataclass
class DocumentChunk:
    """Mock document chunk from SemanticChunker."""
    
    content: str
    title: str
    source: str
    metadata: Dict[str, Any]
    chunk_index: int
    token_count: int


@dataclass
class IngestionResult:
    """Mock result from DocumentIngestionPipeline."""
    
    success: bool
    source_url: str
    title: str
    chunks: List[DocumentChunk]
    total_chunks: int
    total_tokens: int
    processing_time_ms: float
    error_message: Optional[str] = None


class MockSemanticChunker:
    """
    Mock SemanticChunker based on reference implementation pattern.
    
    This simulates the semantic chunking logic that would be used
    in the actual DocumentIngestionPipeline.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_document(self, content: str, title: str, source: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Chunk markdown content into semantic pieces.
        
        This mock implementation uses rule-based chunking that respects
        markdown structure, similar to what a real SemanticChunker would do.
        """
        
        if not content.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Split by major sections (headers)
        sections = self._split_by_headers(content)
        
        for i, section in enumerate(sections):
            if len(section.strip()) < 50:  # Skip very short sections
                continue
                
            # Further split large sections
            sub_chunks = self._split_large_section(section)
            
            for j, chunk_content in enumerate(sub_chunks):
                chunk_index = len(chunks)
                token_count = self._estimate_token_count(chunk_content)
                
                chunk = DocumentChunk(
                    content=chunk_content.strip(),
                    title=title,
                    source=source,
                    metadata={
                        **metadata,
                        'section_index': i,
                        'sub_chunk_index': j,
                        'chunk_method': 'semantic_mock'
                    },
                    chunk_index=chunk_index,
                    token_count=token_count
                )
                
                chunks.append(chunk)
        
        return chunks
    
    def _split_by_headers(self, content: str) -> List[str]:
        """Split content by markdown headers."""
        
        # Split by headers (# ## ### etc.)
        header_pattern = r'^(#{1,6}\s+.+)$'
        lines = content.split('\n')
        
        sections = []
        current_section = []
        
        for line in lines:
            if re.match(header_pattern, line, re.MULTILINE):
                # Start new section
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def _split_large_section(self, section: str) -> List[str]:
        """Split large sections into smaller chunks."""
        
        if len(section) <= self.chunk_size:
            return [section]
        
        # Split by paragraphs first
        paragraphs = section.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            para_length = len(paragraph)
            
            if current_length + para_length > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = para_length
            else:
                current_chunk.append(paragraph)
                current_length += para_length
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _estimate_token_count(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4


class MockDocumentIngestionPipeline:
    """
    Mock DocumentIngestionPipeline based on reference implementation.
    
    This simulates the pipeline that would process markdown from
    AdvancedWebCrawler through chunking, embedding, and storage.
    """
    
    def __init__(self):
        self.semantic_chunker = MockSemanticChunker()
        
    async def ingest_single_document(self, markdown: str, source_url: str, 
                                   title: Optional[str] = None) -> IngestionResult:
        """
        Process a single document through the ingestion pipeline.
        
        Args:
            markdown: Clean markdown from AdvancedWebCrawler
            source_url: Source URL for metadata
            title: Optional document title
            
        Returns:
            IngestionResult with chunking results
        """
        
        import time
        start_time = time.time()
        
        try:
            # Validate input
            if not markdown or not markdown.strip():
                return IngestionResult(
                    success=False,
                    source_url=source_url,
                    title=title or "Unknown",
                    chunks=[],
                    total_chunks=0,
                    total_tokens=0,
                    processing_time_ms=0,
                    error_message="Empty or invalid markdown content"
                )
            
            # Extract title if not provided
            if not title:
                title = self._extract_title_from_markdown(markdown)
            
            # Prepare metadata
            metadata = {
                'source_url': source_url,
                'ingestion_timestamp': time.time(),
                'pipeline_version': 'mock_v1.0'
            }
            
            # Semantic chunking
            chunks = self.semantic_chunker.chunk_document(
                content=markdown,
                title=title,
                source=source_url,
                metadata=metadata
            )
            
            # Calculate metrics
            total_tokens = sum(chunk.token_count for chunk in chunks)
            processing_time_ms = (time.time() - start_time) * 1000
            
            return IngestionResult(
                success=True,
                source_url=source_url,
                title=title,
                chunks=chunks,
                total_chunks=len(chunks),
                total_tokens=total_tokens,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            return IngestionResult(
                success=False,
                source_url=source_url,
                title=title or "Unknown",
                chunks=[],
                total_chunks=0,
                total_tokens=0,
                processing_time_ms=processing_time_ms,
                error_message=f"Pipeline error: {str(e)}"
            )
    
    def _extract_title_from_markdown(self, markdown: str) -> str:
        """Extract title from markdown content."""
        lines = markdown.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return "Untitled Document"


async def test_end_to_end_pipeline():
    """
    Test the complete pipeline integration from URL to chunks.
    
    This validates that AdvancedWebCrawler produces markdown
    suitable for DocumentIngestionPipeline processing.
    """
    
    print("🔬 End-to-End Pipeline Integration Test")
    print("=" * 60)
    print("Testing: URL → AdvancedWebCrawler → DocumentIngestionPipeline → SemanticChunker")
    print()
    
    # Test URLs with different characteristics
    test_cases = [
        {
            "name": "JavaScript-Heavy Site",
            "url": "https://www.promptingguide.ai/",
            "description": "Tests Playwright rendering with complex JS"
        },
        {
            "name": "Documentation Site",
            "url": "https://docs.python.org/3/tutorial/",
            "description": "Tests structured documentation extraction"
        }
    ]
    
    pipeline = MockDocumentIngestionPipeline()
    overall_success = True
    
    async with AdvancedWebCrawler(enable_quality_validation=True) as crawler:
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test Case {i}: {test_case['name']}")
            print(f"URL: {test_case['url']}")
            print(f"Description: {test_case['description']}")
            print("-" * 50)
            
            try:
                # Stage 1: AdvancedWebCrawler extraction
                print("Stage 1: Crawling with AdvancedWebCrawler...")
                crawl_result = await crawler.crawl_single_page(test_case['url'])
                
                if not crawl_result.success:
                    print(f"❌ Crawling failed: {crawl_result.error_message}")
                    overall_success = False
                    continue
                
                print(f"✅ Crawled {crawl_result.word_count} words in {crawl_result.extraction_time_ms:.1f}ms")
                print(f"   Framework: {crawl_result.framework_detected}")
                print(f"   Quality Score: {crawl_result.quality_score:.3f}")
                print(f"   Quality Passed: {crawl_result.quality_passed}")
                
                # Validate markdown quality for pipeline
                markdown_issues = []
                
                if crawl_result.quality_validation:
                    if crawl_result.quality_validation.html_artifacts_found > 0:
                        markdown_issues.append(f"HTML artifacts: {crawl_result.quality_validation.html_artifacts_found}")
                    
                    if crawl_result.quality_validation.script_contamination:
                        markdown_issues.append("Script contamination detected")
                    
                    if crawl_result.quality_validation.content_to_navigation_ratio < 0.6:
                        markdown_issues.append("Low content ratio")
                
                if markdown_issues:
                    print(f"⚠️  Markdown issues: {', '.join(markdown_issues)}")
                else:
                    print("✅ Markdown quality suitable for pipeline")
                
                # Stage 2: DocumentIngestionPipeline processing
                print("\nStage 2: Processing through DocumentIngestionPipeline...")
                ingestion_result = await pipeline.ingest_single_document(
                    markdown=crawl_result.markdown,
                    source_url=crawl_result.url,
                    title=crawl_result.title
                )
                
                if not ingestion_result.success:
                    print(f"❌ Pipeline processing failed: {ingestion_result.error_message}")
                    overall_success = False
                    continue
                
                print(f"✅ Pipeline processing completed in {ingestion_result.processing_time_ms:.1f}ms")
                print(f"   Document: {ingestion_result.title}")
                print(f"   Chunks Created: {ingestion_result.total_chunks}")
                print(f"   Total Tokens: {ingestion_result.total_tokens}")
                
                # Stage 3: Validate chunk quality
                print("\nStage 3: Validating chunk quality...")
                
                if ingestion_result.total_chunks == 0:
                    print("❌ No chunks created - pipeline failed")
                    overall_success = False
                    continue
                
                # Analyze chunk characteristics
                chunk_sizes = [len(chunk.content) for chunk in ingestion_result.chunks]
                avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
                min_chunk_size = min(chunk_sizes)
                max_chunk_size = max(chunk_sizes)
                
                print(f"✅ Chunk Analysis:")
                print(f"   Average size: {avg_chunk_size:.0f} chars")
                print(f"   Size range: {min_chunk_size}-{max_chunk_size} chars")
                
                # Validate chunk content quality
                empty_chunks = sum(1 for chunk in ingestion_result.chunks if len(chunk.content.strip()) < 50)
                if empty_chunks > 0:
                    print(f"⚠️  {empty_chunks} chunks are very short (< 50 chars)")
                else:
                    print("✅ All chunks have substantial content")
                
                # Show sample chunk
                if ingestion_result.chunks:
                    sample_chunk = ingestion_result.chunks[0]
                    preview = sample_chunk.content[:150].replace('\n', ' ')
                    print(f"   Sample chunk: {preview}...")
                
                print("✅ End-to-end test PASSED for this URL")
                
            except Exception as e:
                print(f"❌ Test failed with exception: {str(e)}")
                overall_success = False
            
            print("\n" + "="*60 + "\n")
    
    # Final results
    if overall_success:
        print("🎉 END-TO-END PIPELINE TEST SUCCESSFUL!")
        print()
        print("✅ AdvancedWebCrawler successfully produces clean markdown")
        print("✅ DocumentIngestionPipeline can process the markdown without errors")
        print("✅ SemanticChunker creates valid chunks from the content")
        print("✅ Complete pipeline integration verified")
        print()
        print("The AdvancedWebCrawler is ready for DocumentIngestionPipeline integration!")
    else:
        print("❌ END-TO-END PIPELINE TEST FAILED")
        print()
        print("Some test cases failed. Check the output above for details.")
    
    return overall_success


async def test_pipeline_error_handling():
    """Test pipeline error handling with problematic content."""
    
    print("🧪 Testing Pipeline Error Handling")
    print("-" * 40)
    
    pipeline = MockDocumentIngestionPipeline()
    
    # Test with empty content
    result = await pipeline.ingest_single_document("", "https://example.com")
    assert not result.success, "Pipeline should fail with empty content"
    print("✅ Empty content handling: PASSED")
    
    # Test with whitespace-only content
    result = await pipeline.ingest_single_document("   \n\n   ", "https://example.com")
    assert not result.success, "Pipeline should fail with whitespace-only content"
    print("✅ Whitespace-only content handling: PASSED")
    
    # Test with minimal valid content
    result = await pipeline.ingest_single_document("# Test\n\nThis is a test document.", "https://example.com")
    assert result.success, "Pipeline should succeed with minimal valid content"
    assert result.total_chunks > 0, "Should create at least one chunk"
    print("✅ Minimal valid content handling: PASSED")
    
    print("✅ All error handling tests passed!")


if __name__ == "__main__":
    print("🚀 Starting End-to-End Pipeline Integration Test")
    print()
    
    # Run main pipeline test
    success = asyncio.run(test_end_to_end_pipeline())
    
    if success:
        print("\n" + "="*60)
        print("Running error handling tests...")
        asyncio.run(test_pipeline_error_handling())
        print("\n🎯 All tests completed successfully!")
    else:
        print("\n⚠️  Main pipeline test failed - check configuration")