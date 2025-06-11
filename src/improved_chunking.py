#!/usr/bin/env python3
"""
Enhanced markdown chunking with robust code block preservation.

This module provides improved chunking logic that addresses the critical issues
found in the original smart_chunk_markdown function, specifically:
1. Breaking before closing ``` tags (leaving orphaned opening tags)
2. Creating malformed code blocks
3. Merging function signatures without spaces
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class CodeBlockInfo:
    """Information about a code block within text."""

    def __init__(self, start: int, end: int, language: str = ""):
        self.start = start  # Position of opening ```
        self.end = end  # Position after closing ```
        self.language = language

    @property
    def length(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"CodeBlock({self.start}-{self.end}, lang='{self.language}')"


class EnhancedMarkdownChunker:
    """Enhanced markdown chunker with robust code block preservation."""

    def __init__(self, chunk_size: int = 5000, min_chunk_ratio: float = 0.3):
        """
        Initialize the enhanced chunker.

        Args:
            chunk_size: Target chunk size in characters
            min_chunk_ratio: Minimum chunk size as ratio of target (0.3 = 30%)
        """
        self.chunk_size = chunk_size
        self.min_chunk_size = int(chunk_size * min_chunk_ratio)

    def find_code_blocks(self, text: str) -> List[CodeBlockInfo]:
        """
        Find all complete code blocks in the text.

        Args:
            text: Input text to analyze

        Returns:
            List of CodeBlockInfo objects for complete code blocks
        """
        code_blocks = []
        pos = 0

        while True:
            # Find opening ```
            start = text.find("```", pos)
            if start == -1:
                break

            # Find the end of the first line (language specifier)
            line_end = text.find("\n", start)
            if line_end == -1:
                # No newline after ```, malformed
                pos = start + 3
                continue

            # Extract language specifier
            language = text[start + 3 : line_end].strip()

            # Find closing ```
            end_search_start = line_end + 1
            end = text.find("```", end_search_start)
            if end == -1:
                # No closing ```, incomplete code block
                logger.warning(f"Found unclosed code block at position {start}")
                break

            # Make sure we're at the start of a line or end of text
            if end + 3 < len(text) and text[end + 3] not in ["\n", "\r"]:
                # This might be ``` inside a code block, keep looking
                pos = end + 3
                continue

            # Valid code block found
            code_blocks.append(CodeBlockInfo(start, end + 3, language))
            pos = end + 3

        return code_blocks

    def find_safe_break_point(
        self, text: str, start: int, max_end: int, code_blocks: List[CodeBlockInfo]
    ) -> int:
        """
        Find a safe break point that doesn't split code blocks.

        Args:
            text: Full text
            start: Start position for this chunk
            max_end: Maximum end position (start + chunk_size)
            code_blocks: List of all code blocks in the text

        Returns:
            Safe break position
        """
        # Check if any code blocks would be split
        for block in code_blocks:
            if block.start >= start and block.start < max_end and block.end > max_end:
                # Code block starts in our chunk but ends after max_end
                if block.start > start + self.min_chunk_size:
                    # Break before the code block
                    return self._find_paragraph_break(text, start, block.start)
                else:
                    # Code block starts too early, include the entire block
                    return self._find_paragraph_break(text, start, block.end)

            elif block.start < start and block.end > start and block.end < max_end:
                # Code block starts before our chunk but ends within it
                return self._find_paragraph_break(text, start, block.end)

        # No code block conflicts, find best natural break point
        return self._find_natural_break(text, start, max_end)

    def _find_paragraph_break(self, text: str, start: int, preferred_end: int) -> int:
        """Find paragraph break near the preferred end position."""
        # Look for paragraph breaks (double newlines) around preferred_end
        search_start = max(start + self.min_chunk_size, preferred_end - 200)
        search_end = min(len(text), preferred_end + 200)

        chunk = text[search_start:search_end]

        # Find paragraph breaks
        para_breaks = []
        pos = 0
        while True:
            para_break = chunk.find("\n\n", pos)
            if para_break == -1:
                break
            para_breaks.append(search_start + para_break)
            pos = para_break + 2

        if para_breaks:
            # Find the break closest to preferred_end
            best_break = min(para_breaks, key=lambda x: abs(x - preferred_end))
            return best_break + 2  # After the double newline

        # No paragraph breaks, return preferred_end
        return preferred_end

    def _find_natural_break(self, text: str, start: int, max_end: int) -> int:
        """Find natural break point (paragraph, sentence, etc.)."""
        if max_end >= len(text):
            return len(text)

        chunk = text[start:max_end]

        # Try paragraph break first
        last_para = chunk.rfind("\n\n")
        if last_para != -1 and last_para > len(chunk) * 0.3:
            return start + last_para + 2

        # Try sentence break
        last_sentence = chunk.rfind(". ")
        if last_sentence != -1 and last_sentence > len(chunk) * 0.3:
            return start + last_sentence + 2

        # Try any line break
        last_line = chunk.rfind("\n")
        if last_line != -1 and last_line > len(chunk) * 0.3:
            return start + last_line + 1

        # No good break found, use max_end
        return max_end

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text with enhanced code block preservation.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        # Find all code blocks first
        code_blocks = self.find_code_blocks(text)
        logger.debug(f"Found {len(code_blocks)} code blocks in text")

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate target end position
            target_end = start + self.chunk_size

            if target_end >= text_length:
                # Last chunk
                chunks.append(text[start:].strip())
                break

            # Find safe break point
            actual_end = self.find_safe_break_point(
                text, start, target_end, code_blocks
            )

            # Extract chunk
            chunk = text[start:actual_end].strip()
            if chunk:
                chunks.append(chunk)

            # Safety check to prevent infinite loops
            if actual_end <= start:
                logger.warning(
                    f"Chunking made no progress at position {start}, forcing advance"
                )
                start += self.chunk_size
            else:
                # Move to next chunk
                start = actual_end

        return chunks

    def validate_chunks(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Validate that chunks don't contain malformed code blocks.

        Args:
            chunks: List of text chunks to validate

        Returns:
            Validation report with issues found
        """
        issues = []
        stats = {
            "total_chunks": len(chunks),
            "chunks_with_code": 0,
            "malformed_blocks": 0,
            "orphaned_opening": 0,
            "orphaned_closing": 0,
            "empty_blocks": 0,
        }

        for i, chunk in enumerate(chunks):
            chunk_issues = []

            # Count code block markers
            opening_count = chunk.count("```")
            if opening_count > 0:
                stats["chunks_with_code"] += 1

                # Check for orphaned opening tags
                if opening_count % 2 == 1:
                    # Odd number of ``` means orphaned opening or closing
                    if chunk.rstrip().endswith("```"):
                        stats["orphaned_closing"] += 1
                        chunk_issues.append("Orphaned closing ``` tag")
                    else:
                        stats["orphaned_opening"] += 1
                        chunk_issues.append("Orphaned opening ``` tag")

                # Check for empty code blocks
                empty_block_pattern = r"```\s*\n\s*```"
                if re.search(empty_block_pattern, chunk):
                    stats["empty_blocks"] += 1
                    chunk_issues.append("Empty code block found")

                # Check for merged function signatures
                merged_pattern = r"(async|function|def)([a-zA-Z_][a-zA-Z0-9_]*)"
                if re.search(merged_pattern, chunk):
                    chunk_issues.append("Potential merged function signature")

            if chunk_issues:
                stats["malformed_blocks"] += 1
                issues.append(
                    {
                        "chunk_index": i,
                        "issues": chunk_issues,
                        "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    }
                )

        return {"valid": len(issues) == 0, "stats": stats, "issues": issues}


def smart_chunk_markdown_enhanced(text: str, chunk_size: int = 5000) -> List[str]:
    """
    Enhanced markdown chunking function that preserves code blocks.

    This is a drop-in replacement for the original smart_chunk_markdown function
    with improved code block preservation.

    Args:
        text: Markdown text to chunk
        chunk_size: Target chunk size in characters

    Returns:
        List of markdown chunks with preserved code blocks
    """
    chunker = EnhancedMarkdownChunker(chunk_size)
    chunks = chunker.chunk_text(text)

    # Validate results in debug mode
    if logger.isEnabledFor(logging.DEBUG):
        validation = chunker.validate_chunks(chunks)
        if not validation["valid"]:
            logger.warning(
                f"Chunking validation found {len(validation['issues'])} issues"
            )
            for issue in validation["issues"]:
                logger.debug(f"Chunk {issue['chunk_index']}: {issue['issues']}")

    return chunks


def analyze_chunking_quality(original_text: str, chunks: List[str]) -> Dict[str, Any]:
    """
    Analyze the quality of chunking results.

    Args:
        original_text: Original text before chunking
        chunks: Resulting chunks

    Returns:
        Quality analysis report
    """
    chunker = EnhancedMarkdownChunker()

    # Basic stats
    total_chars = sum(len(chunk) for chunk in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0

    # Code block analysis
    original_blocks = chunker.find_code_blocks(original_text)

    # Validation
    validation = chunker.validate_chunks(chunks)

    return {
        "chunk_count": len(chunks),
        "total_characters": total_chars,
        "average_chunk_size": avg_chunk_size,
        "original_code_blocks": len(original_blocks),
        "validation": validation,
        "character_preservation": total_chars / len(original_text)
        if original_text
        else 0,
    }
