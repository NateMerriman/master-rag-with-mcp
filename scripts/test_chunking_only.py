#!/usr/bin/env python3
"""
Simplified test script to validate enhanced chunking fixes code block preservation.
This test doesn't require MCP dependencies and focuses only on the chunking logic.
"""

import os
import sys
import re

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.improved_chunking import (
    smart_chunk_markdown_enhanced,
    EnhancedMarkdownChunker,
    analyze_chunking_quality,
)


def get_test_content():
    """Get the problematic DeepWiki content that should contain the @mcp.tool code block."""
    return '''# Adding Coding Preferences

This document details the functionality and implementation of the tools that add coding preferences and memories to the mem0 system. Both the Python and Node.js implementations provide mechanisms to store code snippets, implementation patterns, and programming knowledge for later semantic retrieval.

For information about searching stored preferences, see Searching Coding Preferences. For retrieving all preferences, see Retrieving All Preferences.

## Tool Overview

Both implementations provide tools to add coding preferences to mem0's semantic memory storage:

| Implementation | Tool Name               | Function                      | Purpose                                                            |
| -------------- | ----------------------- | ----------------------------- | ------------------------------------------------------------------ |
| Python         | add_coding_preference | Store coding-specific content | Optimized for code snippets, patterns, and technical documentation |
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


def original_smart_chunk_markdown(text: str, chunk_size: int = 5000):
    """
    Original chunking logic from the main crawl4ai_mcp.py file.
    Reproduced here to test without MCP dependencies.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block  # THIS IS THE BUG - breaks BEFORE closing ```

        # If no code block, try to break at a paragraph
        elif "\n\n" in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind("\n\n")
            if (
                last_break > chunk_size * 0.3
            ):  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif ". " in chunk:
            # Find the last sentence break
            last_period = chunk.rfind(". ")
            if (
                last_period > chunk_size * 0.3
            ):  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks


def analyze_chunks_for_issues(chunks):
    """Analyze chunks for common issues."""
    issues = []
    stats = {
        "total_chunks": len(chunks),
        "chunks_with_code": 0,
        "orphaned_opening": 0,
        "orphaned_closing": 0,
        "empty_blocks": 0,
        "merged_signatures": 0,
    }

    for i, chunk in enumerate(chunks):
        chunk_issues = []

        # Count code block markers
        code_markers = chunk.count("```")
        if code_markers > 0:
            stats["chunks_with_code"] += 1

            # Check for orphaned markers (odd count)
            if code_markers % 2 == 1:
                if chunk.strip().endswith("```"):
                    stats["orphaned_closing"] += 1
                    chunk_issues.append("Orphaned closing ``` tag")
                else:
                    stats["orphaned_opening"] += 1
                    chunk_issues.append("Orphaned opening ``` tag")

        # Check for empty code blocks
        if "```\n```" in chunk or "```\n\n```" in chunk:
            stats["empty_blocks"] += 1
            chunk_issues.append("Empty code block")

        # Check for merged function signatures (the specific issue we found)
        if "asyncdef" in chunk or "functionadd" in chunk or "asyncfunction" in chunk:
            stats["merged_signatures"] += 1
            chunk_issues.append("Merged function signature")

        if chunk_issues:
            issues.append(
                {
                    "chunk_index": i,
                    "issues": chunk_issues,
                    "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                }
            )

    return {"valid": len(issues) == 0, "stats": stats, "issues": issues}


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

    original_chunks = original_smart_chunk_markdown(content, chunk_size)
    original_analysis = analyze_chunks_for_issues(original_chunks)

    print(f"Chunks created: {len(original_chunks)}")
    print(f"Chunks with code: {original_analysis['stats']['chunks_with_code']}")
    print(f"Issues found: {len(original_analysis['issues'])}")

    for issue in original_analysis["issues"][:3]:  # Show first 3 issues
        print(f"  ⚠️  Chunk {issue['chunk_index']}: {', '.join(issue['issues'])}")

    print()

    # Test enhanced chunking
    print("✅ Enhanced Chunking Results:")
    print("-" * 30)

    enhanced_chunks = smart_chunk_markdown_enhanced(content, chunk_size)
    enhanced_analysis = analyze_chunks_for_issues(enhanced_chunks)

    print(f"Chunks created: {len(enhanced_chunks)}")
    print(f"Chunks with code: {enhanced_analysis['stats']['chunks_with_code']}")
    print(f"Issues found: {len(enhanced_analysis['issues'])}")

    for issue in enhanced_analysis["issues"][:3]:  # Show first 3 issues
        print(f"  ⚠️  Chunk {issue['chunk_index']}: {', '.join(issue['issues'])}")

    print()

    # Compare results
    print("📊 Comparison Summary:")
    print("-" * 25)
    print(
        f"Original chunks: {len(original_chunks)} | Enhanced chunks: {len(enhanced_chunks)}"
    )
    print(
        f"Original issues: {len(original_analysis['issues'])} | Enhanced issues: {len(enhanced_analysis['issues'])}"
    )

    improvement = len(original_analysis["issues"]) - len(enhanced_analysis["issues"])
    if improvement > 0:
        if len(original_analysis["issues"]) > 0:
            print(
                f"✅ Improvement: {improvement} fewer issues ({improvement / len(original_analysis['issues']) * 100:.1f}% better)"
            )
        else:
            print("✅ Improvement: Issues fixed!")
    elif improvement == 0:
        print("➡️  No difference in issue count")
    else:
        print(f"❌ Regression: {abs(improvement)} more issues")

    return original_chunks, enhanced_chunks, original_analysis, enhanced_analysis


def test_specific_mcp_tool_preservation():
    """Test that the @mcp.tool code block is correctly preserved."""
    print("\n🎯 Testing @mcp.tool Code Block Preservation")
    print("=" * 50)

    content = get_test_content()

    # Test both approaches
    original_chunks = original_smart_chunk_markdown(content, chunk_size=2000)
    enhanced_chunks = smart_chunk_markdown_enhanced(content, chunk_size=2000)

    # Look for @mcp.tool preservation in both
    original_found = False
    enhanced_found = False

    print("🔍 Original chunking @mcp.tool search:")
    for i, chunk in enumerate(original_chunks):
        if "@mcp.tool" in chunk:
            print(f"  📍 Found @mcp.tool in chunk {i}")

            # Check if it's in a proper code block
            # Count ``` before and after @mcp.tool
            mcp_pos = chunk.find("@mcp.tool")
            before = chunk[:mcp_pos].count("```")
            after = chunk[mcp_pos:].count("```")

            print(f"     ``` count before: {before}, after: {after}")

            # For a properly preserved code block, we need:
            # - Even number of ``` before @mcp.tool (complete blocks)
            # - At least 2 ``` after @mcp.tool (opening and closing)
            if before % 2 == 0 and after >= 2:
                original_found = True
                print(f"     ✅ Properly preserved in code block")
            else:
                print(f"     ❌ Not properly preserved")

    print("\n🔍 Enhanced chunking @mcp.tool search:")
    for i, chunk in enumerate(enhanced_chunks):
        if "@mcp.tool" in chunk:
            print(f"  📍 Found @mcp.tool in chunk {i}")

            # Check if it's in a proper code block
            mcp_pos = chunk.find("@mcp.tool")
            before = chunk[:mcp_pos].count("```")
            after = chunk[mcp_pos:].count("```")

            print(f"     ``` count before: {before}, after: {after}")

            if before % 2 == 0 and after >= 2:
                enhanced_found = True
                print(f"     ✅ Properly preserved in code block")
            else:
                print(f"     ❌ Not properly preserved")

    print(f"\n📊 Results:")
    print(f"Original chunking preserved @mcp.tool: {'✅' if original_found else '❌'}")
    print(f"Enhanced chunking preserved @mcp.tool: {'✅' if enhanced_found else '❌'}")

    return original_found, enhanced_found


def show_detailed_chunk_analysis(chunks, title):
    """Show detailed analysis of chunks containing code blocks."""
    print(f"\n🔍 Detailed {title} Analysis:")
    print("-" * 40)

    for i, chunk in enumerate(chunks):
        if "```" in chunk:
            code_markers = chunk.count("```")
            print(f"Chunk {i}: {len(chunk)} chars, {code_markers} ``` markers")

            # Show code block boundaries
            lines = chunk.split("\n")
            for j, line in enumerate(lines):
                if "```" in line or "@mcp.tool" in line:
                    print(f"  Line {j}: {line}")


if __name__ == "__main__":
    print("🧪 Enhanced Chunking Validation Test")
    print("=" * 40)

    # Run tests
    original_chunks, enhanced_chunks, original_analysis, enhanced_analysis = (
        test_original_vs_enhanced()
    )
    original_preserved, enhanced_preserved = test_specific_mcp_tool_preservation()

    # Show detailed analysis if there are still issues
    if not enhanced_preserved:
        show_detailed_chunk_analysis(enhanced_chunks, "Enhanced Chunks")

    print("\n" + "=" * 40)
    print("🏁 Test Summary")
    print("=" * 40)

    if enhanced_preserved and not original_preserved:
        print("✅ SUCCESS: Enhanced chunking fixes @mcp.tool preservation!")
        print("✅ The improved logic correctly maintains code block integrity")
    elif enhanced_preserved and original_preserved:
        print(
            "✅ BOTH work: Both methods preserve @mcp.tool (test may need smaller chunks)"
        )
    elif not enhanced_preserved and not original_preserved:
        print("❌ BOTH FAIL: @mcp.tool not preserved in either method")
        print("❌ Enhanced chunking needs further refinement")
    else:  # original works but enhanced doesn't
        print("❌ REGRESSION: Enhanced chunking broke @mcp.tool preservation")
        print("❌ Need to debug the enhanced logic")
