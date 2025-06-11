#!/usr/bin/env python3
"""
Debug the specific break point issue in enhanced chunking.
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.improved_chunking import EnhancedMarkdownChunker


def debug_break_point_issue():
    """Debug the specific break point that's causing infinite loop."""

    content = '''# Adding Coding Preferences

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

    print("🔍 Debugging Break Point Issue")
    print("=" * 40)

    chunker = EnhancedMarkdownChunker(chunk_size=2000)
    code_blocks = chunker.find_code_blocks(content)

    # The problematic case: position 1882 (from error message)
    start = 1882
    max_end = start + 2000  # 3882

    print(f"Content length: {len(content)}")
    print(f"Problematic start: {start}")
    print(f"Target max_end: {max_end}")
    print(f"Min chunk size: {chunker.min_chunk_size}")
    print()

    print("Code blocks found:")
    for i, block in enumerate(code_blocks):
        print(
            f"  Block {i}: {block.start}-{block.end} (len={block.length}), lang='{block.language}'"
        )

    print()
    print("Checking for conflicts with code blocks:")

    for i, block in enumerate(code_blocks):
        print(f"\nBlock {i} ({block.start}-{block.end}):")

        # Check conflict Type 1: Block starts in chunk but ends after
        if block.start >= start and block.start < max_end and block.end > max_end:
            print(
                f"  ❌ Conflict Type 1: Block starts in chunk ({block.start}) but ends after max_end ({max_end})"
            )
            if block.start > start + chunker.min_chunk_size:
                preferred_end = block.start
                print(f"     Should break BEFORE block at {preferred_end}")
            else:
                preferred_end = block.end
                print(
                    f"     Block starts too early, should include entire block and break at {preferred_end}"
                )

            # This would call _find_paragraph_break
            actual_break = chunker._find_paragraph_break(content, start, preferred_end)
            print(
                f"     _find_paragraph_break({start}, {preferred_end}) returns: {actual_break}"
            )

        # Check conflict Type 2: Block starts before chunk but ends within
        elif block.start < start and block.end > start and block.end < max_end:
            print(
                f"  ❌ Conflict Type 2: Block starts before chunk ({block.start}) but ends within ({block.end})"
            )
            preferred_end = block.end
            actual_break = chunker._find_paragraph_break(content, start, preferred_end)
            print(f"     Should break at {preferred_end}")
            print(
                f"     _find_paragraph_break({start}, {preferred_end}) returns: {actual_break}"
            )

        else:
            print(f"  ✅ No conflict")

    print()
    print("Testing find_safe_break_point directly:")
    actual_end = chunker.find_safe_break_point(content, start, max_end, code_blocks)
    print(f"find_safe_break_point({start}, {max_end}) returned: {actual_end}")
    print(f"Progress would be: {actual_end - start}")

    if actual_end <= start:
        print("❌ This is the bug!")

        # Let's test _find_natural_break as fallback
        print("\nTesting _find_natural_break as fallback:")
        natural_break = chunker._find_natural_break(content, start, max_end)
        print(f"_find_natural_break({start}, {max_end}) returned: {natural_break}")
        print(f"Natural break progress: {natural_break - start}")


if __name__ == "__main__":
    debug_break_point_issue()
