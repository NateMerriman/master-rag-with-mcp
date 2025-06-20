#!/usr/bin/env python3
"""
Test script to validate enhanced chunking fixes the code block preservation issues.
"""

import os
import sys
import json

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.improved_chunking import (
    smart_chunk_markdown_enhanced,
    EnhancedMarkdownChunker,
    analyze_chunking_quality,
)
from src.crawl4ai_mcp import smart_chunk_markdown
from src.code_extraction import CodeExtractor


def get_test_content():
    """Get the problematic DeepWiki content that should contain the @mcp.tool code block."""
    return '''# Adding Coding Preferences

This document details the functionality and implementation of the tools that add coding preferences and memories to the mem0 system. Both the Python and Node.js implementations provide mechanisms to store code snippets, implementation patterns, and programming knowledge for later semantic retrieval.

For information about searching stored preferences, see Searching Coding Preferences. For retrieving all preferences, see Retrieving All Preferences.

## Tool Overview

Both implementations provide tools to add coding preferences to mem0's semantic memory storage:

| Implementation | Tool Name               | Function                      | Purpose                                                            |
| -------------- | ----------------------- | ----------------------------- | ------------------------------------------------------------------ |
| Python         | add\_coding\_preference | Store coding-specific content | Optimized for code snippets, patterns, and technical documentation |
| Node.js        | add-memory              | Store general content         | Generic memory storage that can handle coding preferences          |

The Python implementation includes detailed instructions for comprehensive code storage, while the Node.js version provides a more general-purpose memory addition capability.

## Python Implementation Architecture

The Python implementation uses FastMCP to expose the `add_coding_preference` tool with extensive documentation and context guidance.

**Tool Configuration and Instructions**

The Python implementation includes comprehensive instructions embedded in the tool description and uses custom project instructions for mem0.

## Node.js Implementation Architecture

The Node.js implementation uses the MCP SDK with a more generic approach to memory storage.

**Tool Registration and Handling**

The Node.js implementation registers tools through the MCP SDK's request handler system and processes tool calls through a switch statement.

## Implementation Comparison

### Function Signatures and Parameters

**Python Implementation:**

```python
async def add_coding_preference(text: str) -> str
```

**Node.js Implementation:**

```javascript
async function addMemory(content: string, userId: string)
```

### Message Construction Patterns

Both implementations construct message arrays for mem0, but with different approaches:

**Python Message Format:**

```python
messages = [{"role": "user", "content": text}]
```

**Node.js Message Format:**

```javascript
const messages = [
 { role: 'system', content: 'Memory storage system' },
 { role: 'user', content }
];
```

The Node.js version includes a system message to provide context to mem0.

## Data Flow and Processing

**Processing Steps:**

1. Input validation and parameter extraction
2. Message array construction with role-based content organization
3. mem0 API call with user identification and format specification
4. Response processing and error handling

## Custom Instructions and Context Enhancement

The Python implementation includes detailed custom instructions that guide mem0's processing:

```python
@mcp.tool(
    description="""Add a new coding preference to mem0. This tool stores code snippets, implementation details,
    and coding patterns for future reference. Store every code snippet. When storing code, you should include:
    - Complete code with all necessary imports and dependencies
    - Language/framework version information (e.g., "Python 3.9", "React 18")
    - Full implementation context and any required setup/configuration
    - [...more details...]
    The preference will be indexed for semantic search and can be retrieved later using natural language queries."""
)
async def add_coding_preference(text: str) -> str:
    """Add a new coding preference to mem0.
    [...]
    """
    try:
        messages = [{"role": "user", "content": text}]
        mem0_client.add(messages, user_id=DEFAULT_USER_ID, output_format="v1.1")
        return f"Successfully added preference: {text}"
    except Exception as e:
        return f"Error adding preference: {str(e)}"
```

These instructions are applied to the mem0 project using `mem0_client.update_project(custom_instructions=CUSTOM_INSTRUCTIONS)`.
'''


def test_original_vs_enhanced():
    """Test original chunking vs enhanced chunking."""
    print("🔍 Testing Original vs Enhanced Chunking")
    print("=" * 60)

    content = get_test_content()
    chunk_size = 2000  # Use smaller chunks to force the issue

    print(f"📄 Original content: {len(content)} characters")
    print(f"🎯 Target chunk size: {chunk_size} characters")
    print()

    # Test original chunking
    print("🚫 Original Chunking Results:")
    print("-" * 30)

    original_chunks = smart_chunk_markdown(content, chunk_size)
    print(f"Chunks created: {len(original_chunks)}")

    # Analyze original chunks for issues
    extractor = CodeExtractor()
    original_code_blocks = 0
    original_issues = []

    for i, chunk in enumerate(original_chunks):
        blocks = extractor.extract_code_blocks(chunk)
        original_code_blocks += len(blocks)

        # Check for malformed patterns
        if "```python" in chunk and not chunk.count("```") % 2 == 0:
            original_issues.append(f"Chunk {i}: Unbalanced code block markers")

        if "asyncdef" in chunk or "functionadd" in chunk:
            original_issues.append(f"Chunk {i}: Merged function signature detected")

        if "```\n```" in chunk:
            original_issues.append(f"Chunk {i}: Empty code block detected")

    print(f"Code blocks extracted: {original_code_blocks}")
    print(f"Issues found: {len(original_issues)}")
    for issue in original_issues[:3]:  # Show first 3 issues
        print(f"  ⚠️  {issue}")

    print()

    # Test enhanced chunking
    print("✅ Enhanced Chunking Results:")
    print("-" * 30)

    enhanced_chunks = smart_chunk_markdown_enhanced(content, chunk_size)
    print(f"Chunks created: {len(enhanced_chunks)}")

    # Analyze enhanced chunks
    enhanced_code_blocks = 0
    enhanced_issues = []

    for i, chunk in enumerate(enhanced_chunks):
        blocks = extractor.extract_code_blocks(chunk)
        enhanced_code_blocks += len(blocks)

        # Check for malformed patterns
        if "```python" in chunk and not chunk.count("```") % 2 == 0:
            enhanced_issues.append(f"Chunk {i}: Unbalanced code block markers")

        if "asyncdef" in chunk or "functionadd" in chunk:
            enhanced_issues.append(f"Chunk {i}: Merged function signature detected")

        if "```\n```" in chunk:
            enhanced_issues.append(f"Chunk {i}: Empty code block detected")

    print(f"Code blocks extracted: {enhanced_code_blocks}")
    print(f"Issues found: {len(enhanced_issues)}")
    for issue in enhanced_issues[:3]:  # Show first 3 issues
        print(f"  ⚠️  {issue}")

    print()

    # Compare results
    print("📊 Comparison Summary:")
    print("-" * 25)
    print(
        f"Original chunks: {len(original_chunks)} | Enhanced chunks: {len(enhanced_chunks)}"
    )
    print(
        f"Original code blocks: {original_code_blocks} | Enhanced code blocks: {enhanced_code_blocks}"
    )
    print(
        f"Original issues: {len(original_issues)} | Enhanced issues: {len(enhanced_issues)}"
    )

    improvement = len(original_issues) - len(enhanced_issues)
    if improvement > 0:
        print(
            f"✅ Improvement: {improvement} fewer issues ({improvement / len(original_issues) * 100:.1f}% better)"
        )
    elif improvement == 0:
        print("➡️  No difference in issue count")
    else:
        print(f"❌ Regression: {abs(improvement)} more issues")

    return original_chunks, enhanced_chunks


def test_specific_mcp_tool_extraction():
    """Test that the @mcp.tool code block is correctly preserved and extractable."""
    print("\n🎯 Testing @mcp.tool Code Block Extraction")
    print("=" * 50)

    content = get_test_content()

    # Test enhanced chunking
    enhanced_chunks = smart_chunk_markdown_enhanced(content, chunk_size=2000)

    # Look for the @mcp.tool code block
    extractor = CodeExtractor()
    mcp_tool_found = False

    for i, chunk in enumerate(enhanced_chunks):
        if "@mcp.tool" in chunk:
            print(f"📍 Found @mcp.tool in chunk {i}")

            # Extract code blocks from this chunk
            blocks = extractor.extract_code_blocks(chunk)
            print(f"🔍 Code blocks in this chunk: {len(blocks)}")

            for j, block in enumerate(blocks):
                if "@mcp.tool" in block.content:
                    mcp_tool_found = True
                    print(f"✅ @mcp.tool code block found in block {j}")
                    print(f"   Language: {block.language.value}")
                    print(f"   Length: {len(block.content)} characters")
                    print(f"   Valid: {extractor._is_valid_code_block_improved(block)}")

                    # Show a preview
                    preview = block.content[:200]
                    print(f"   Preview: {preview}...")

                    # Test full processing
                    processed = extractor.process_code_blocks(chunk, "test-url")
                    print(f"   Processed examples: {len(processed)}")

                    break
            break

    if not mcp_tool_found:
        print("❌ @mcp.tool code block NOT found in any chunk")

        # Debug: Show which chunks contain @mcp.tool text
        for i, chunk in enumerate(enhanced_chunks):
            if "@mcp.tool" in chunk:
                print(
                    f"\n🔍 Chunk {i} contains '@mcp.tool' but no extractable code block:"
                )
                lines = chunk.split("\n")
                for line_num, line in enumerate(lines):
                    if "@mcp.tool" in line or "```" in line:
                        context_start = max(0, line_num - 2)
                        context_end = min(len(lines), line_num + 3)
                        print(f"  Lines {context_start}-{context_end}:")
                        for ctx_line_num in range(context_start, context_end):
                            marker = ">>>" if ctx_line_num == line_num else "   "
                            print(
                                f"  {marker} {ctx_line_num:2d}: {lines[ctx_line_num]}"
                            )
                        break
    else:
        print("✅ @mcp.tool code block successfully extracted!")

    return mcp_tool_found


def test_chunking_quality_analysis():
    """Test the chunking quality analysis functionality."""
    print("\n📈 Chunking Quality Analysis")
    print("=" * 35)

    content = get_test_content()

    # Test both approaches
    original_chunks = smart_chunk_markdown(content, chunk_size=2000)
    enhanced_chunks = smart_chunk_markdown_enhanced(content, chunk_size=2000)

    # Analyze quality
    original_quality = analyze_chunking_quality(content, original_chunks)
    enhanced_quality = analyze_chunking_quality(content, enhanced_chunks)

    print("📊 Original Chunking Quality:")
    print(f"  Chunks: {original_quality['chunk_count']}")
    print(f"  Avg size: {original_quality['average_chunk_size']:.0f} chars")
    print(f"  Code blocks: {original_quality['original_code_blocks']}")
    print(f"  Valid: {original_quality['validation']['valid']}")
    print(f"  Issues: {len(original_quality['validation']['issues'])}")

    print("\n📊 Enhanced Chunking Quality:")
    print(f"  Chunks: {enhanced_quality['chunk_count']}")
    print(f"  Avg size: {enhanced_quality['average_chunk_size']:.0f} chars")
    print(f"  Code blocks: {enhanced_quality['original_code_blocks']}")
    print(f"  Valid: {enhanced_quality['validation']['valid']}")
    print(f"  Issues: {len(enhanced_quality['validation']['issues'])}")

    # Show validation details
    if not enhanced_quality["validation"]["valid"]:
        print("\n⚠️  Enhanced Chunking Issues:")
        for issue in enhanced_quality["validation"]["issues"][:3]:
            print(f"  Chunk {issue['chunk_index']}: {issue['issues']}")

    return original_quality, enhanced_quality


if __name__ == "__main__":
    print("🧪 Enhanced Chunking Validation Test")
    print("=" * 40)

    # Run all tests
    test_original_vs_enhanced()
    mcp_found = test_specific_mcp_tool_extraction()
    test_chunking_quality_analysis()

    print("\n" + "=" * 40)
    print("🏁 Test Summary")
    print("=" * 40)

    if mcp_found:
        print("✅ SUCCESS: @mcp.tool code block is properly preserved and extractable")
        print("✅ Enhanced chunking fixes the identified issues")
    else:
        print("❌ FAILURE: @mcp.tool code block still not properly preserved")
        print("❌ Enhanced chunking needs further refinement")
