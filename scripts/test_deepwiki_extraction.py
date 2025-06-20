#!/usr/bin/env python3
"""
Test script to diagnose code extraction issues with the DeepWiki mem0-mcp page.
"""

import os
import sys
import json

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

from src.code_extraction import CodeExtractor, ProgrammingLanguage
from src.config import get_config, reset_config


def test_deepwiki_extraction():
    """Test code extraction on the actual DeepWiki content that failed."""

    # The problematic content from DeepWiki page
    deepwiki_content = '''
# Adding Coding Preferences

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

    print("🔍 Testing Code Extraction on DeepWiki Content")
    print("=" * 60)

    # Initialize extractor
    extractor = CodeExtractor()

    # Extract code blocks
    code_blocks = extractor.extract_code_blocks(deepwiki_content)

    print(f"📊 Found {len(code_blocks)} code blocks:")
    print()

    for i, block in enumerate(code_blocks, 1):
        print(f"**Block {i}:**")
        print(f"  Language: {block.language.value}")
        print(f"  Type: {block.block_type}")
        print(f"  Lines: {block.start_line}-{block.end_line}")
        print(f"  Length: {len(block.content)} chars")
        print(f"  Valid: {extractor._is_valid_code_block_improved(block)}")
        print(f"  Content preview: {repr(block.content[:100])}")
        print()

    # Test the full processing pipeline
    print("🔧 Testing Full Processing Pipeline:")
    print("-" * 40)

    processed_codes = extractor.process_code_blocks(
        deepwiki_content,
        "https://deepwiki.com/mem0ai/mem0-mcp/6.1-adding-coding-preferences",
    )

    print(f"📈 Processed {len(processed_codes)} code examples:")

    for i, code in enumerate(processed_codes, 1):
        print(f"\n**Processed Code {i}:**")
        print(f"  Language: {code.programming_language}")
        print(f"  Complexity: {code.complexity_score}")
        print(f"  Summary: {code.summary[:100]}...")
        print(f"  Content length: {len(code.content)} chars")
        print(
            f"  Metadata keys: {list(code.metadata.keys()) if code.metadata else 'None'}"
        )

    # Specifically test the missing code block
    print("\n🎯 Testing the Expected @mcp.tool Code Block:")
    print("-" * 50)

    expected_code = '''@mcp.tool(
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
        return f"Error adding preference: {str(e)}"'''

    # Test this specific block
    test_content = f"```python\n{expected_code}\n```"
    test_blocks = extractor.extract_code_blocks(test_content)

    print(f"Expected code block extracted: {len(test_blocks) > 0}")
    if test_blocks:
        block = test_blocks[0]
        print(f"  Language detected: {block.language.value}")
        print(f"  Valid: {extractor._is_valid_code_block_improved(block)}")
        print(f"  Length: {len(block.content)} chars")

        # Check what validation steps fail
        print("\n🔍 Validation Details:")
        content = block.content.strip()
        lines = content.split("\n")
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        print(f"  Length check: {25 <= len(content) <= 10000}")
        print(f"  Line count: {len(lines)} (min 2 required)")
        print(f"  Non-empty lines: {len(non_empty_lines)}")

        # Check for programming constructs
        import re

        programming_indicators = [
            r"\bdef\s+\w+\s*\(",  # Python function
            r"\bfunction\s+\w+\s*\(",  # JavaScript function
            r"\bclass\s+\w+",  # Class definition
            r"\bif\s*\(",  # Conditional
            r"\bfor\s*\(",  # Loop
            r"\bwhile\s*\(",  # Loop
            r"\breturn\s+",  # Return statement
            r"\bimport\s+\w+",  # Import statement
            r"console\.log\(",  # JavaScript log
            r"print\s*\(",  # Print statement
            r"@\w+",  # Decorator/annotation
            r"=>",  # Arrow function
            r"async\s+",  # Async keyword
            r"await\s+",  # Await keyword
        ]

        has_programming_construct = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in programming_indicators
        )
        print(f"  Programming constructs: {has_programming_construct}")

        # Check individual patterns
        for pattern in programming_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                print(f"    Found: {pattern}")
    else:
        print("  ❌ Expected code block was NOT extracted!")


if __name__ == "__main__":
    # Enable agentic RAG for code extraction
    os.environ["USE_AGENTIC_RAG"] = "true"
    reset_config()

    test_deepwiki_extraction()
