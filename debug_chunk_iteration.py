#!/usr/bin/env python3
"""
Debug the chunk_text method step by step to understand the infinite loop.
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.improved_chunking import EnhancedMarkdownChunker


def debug_chunk_text_method():
    """Debug the chunk_text method step by step."""

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

    print("🔍 Debugging chunk_text Method Execution")
    print("=" * 50)

    chunker = EnhancedMarkdownChunker(chunk_size=2000)

    # Manually implement the chunk_text logic with debug output
    print(f"Content length: {len(content)}")
    print(f"Chunk size: {chunker.chunk_size}")
    print()

    if not content or len(content) <= chunker.chunk_size:
        print("Content fits in single chunk")
        return

    # Find all code blocks first
    code_blocks = chunker.find_code_blocks(content)
    print(f"Found {len(code_blocks)} code blocks")

    chunks = []
    start = 0
    text_length = len(content)
    iteration = 0

    while start < text_length and iteration < 10:  # Limit iterations for safety
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"Start position: {start}")
        print(f"Remaining text: {text_length - start} chars")

        # Calculate target end position
        target_end = start + chunker.chunk_size
        print(f"Target end: {target_end}")

        if target_end >= text_length:
            # Last chunk
            remaining_chunk = content[start:].strip()
            print(f"Last chunk: {len(remaining_chunk)} chars")
            if remaining_chunk:
                chunks.append(remaining_chunk)
            break

        # Find safe break point
        print(f"Calling find_safe_break_point({start}, {target_end}, ...)")
        actual_end = chunker.find_safe_break_point(
            content, start, target_end, code_blocks
        )
        print(f"Safe break point returned: {actual_end}")
        print(f"Progress: {actual_end - start} chars")

        # Extract chunk
        chunk = content[start:actual_end].strip()
        print(f"Chunk length after strip: {len(chunk)} chars")

        if chunk:
            chunks.append(chunk)
            print(f"Added chunk {len(chunks)}")
        else:
            print("⚠️ Chunk is empty after strip!")

        # Move to next chunk
        old_start = start
        start = actual_end
        print(f"Moving start from {old_start} to {start}")

        # Safety check to prevent infinite loops
        if actual_end <= old_start:
            print(
                f"❌ ERROR: No progress made! (actual_end {actual_end} <= old_start {old_start})"
            )
            print(f"Forcing advance by chunk_size ({chunker.chunk_size})")
            start = old_start + chunker.chunk_size
            print(f"New forced start: {start}")

    print(f"\n🏁 Completed after {iteration} iterations")
    print(f"Created {len(chunks)} chunks")

    # Show chunk summaries
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} chars")


if __name__ == "__main__":
    debug_chunk_text_method()
