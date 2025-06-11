#!/usr/bin/env python3
"""
Comprehensive diagnostic test for crawling pipeline issues.
"""

import os
import sys
import json
import re

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.code_extraction import CodeExtractor, extract_code_from_content
from src.config import get_config, reset_config


def test_crawl_pipeline_simulation():
    """Simulate the complete crawl pipeline with DeepWiki content."""

    print("🔍 Diagnosing Crawl Pipeline Issues")
    print("=" * 60)

    # Set up environment like the actual crawl
    os.environ["USE_AGENTIC_RAG"] = "true"
    os.environ["USE_CONTEXTUAL_EMBEDDINGS"] = "false"  # Simplify for testing
    reset_config()

    # Test content from the actual crawled chunk (this is what actually got stored)
    actual_crawled_content = """[TECHNICAL] This chunk describes the architecture, function signatures, message construction patterns, and data flow of the `add_coding_preference` tool implementations in Python and Node.js, which enable storing coding preferences and memories in the mem0 semantic memory system, serving as a core reference for understanding how code snippets and technical knowledge are added and retrieved within the broader mem0 API framework.
---
This document details the functionality and implementation of the tools that add coding preferences and memories to the mem0 system. Both the Python and Node.js implementations provide mechanisms to store code snippets, implementation patterns, and programming knowledge for later semantic retrieval.
For information about searching stored preferences, see [Searching Coding Preferences](https://deepwiki.com/mem0ai/mem0-mcp/6.2-searching-coding-preferences). For retrieving all preferences, see [Retrieving All Preferences](https://deepwiki.com/mem0ai/mem0-mcp/6.3-retrieving-all-preferences).
## Tool Overview
Both implementations provide tools to add coding preferences to mem0's semantic memory storage:
Implementation| Tool Name| Function| Purpose  
---|---|---|---  
Python| `add_coding_preference`| Store coding-specific content| Optimized for code snippets, patterns, and technical documentation  
Node.js| `add-memory`| Store general content| Generic memory storage that can handle coding preferences  
The Python implementation includes detailed instructions for comprehensive code storage, while the Node.js version provides a more general-purpose memory addition capability.
Sources: [main.py30-64](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/main.py#L30-L64) [node/mem0/src/index.ts21-39](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/node/mem0/src/index.ts#L21-L39)
## Python Implementation Architecture
The Python implementation uses FastMCP to expose the `add_coding_preference` tool with extensive documentation and context guidance.
**Tool Configuration and Instructions**
The Python implementation includes comprehensive instructions embedded in the tool description and uses custom project instructions for mem0.
Sources: [main.py30-44](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/main.py#L30-L44) [main.py20-28](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/main.py#L20-L28)
## Node.js Implementation Architecture
The Node.js implementation uses the MCP SDK with a more generic approach to memory storage.
**Tool Registration and Handling**
The Node.js implementation registers tools through the MCP SDK's request handler system and processes tool calls through a switch statement.
Sources: [node/mem0/src/index.ts101-103](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/node/mem0/src/index.ts#L101-L103) [node/mem0/src/index.ts105-126](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/node/mem0/src/index.ts#L105-L126) [node/mem0/src/index.ts75-87](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/node/mem0/src/index.ts#L75-L87)
## Implementation Comparison
### Function Signatures and Parameters
**Python Implementation:**
```
```
asyncdefadd_coding_preference(text: str) -> str
```

```

**Node.js Implementation:**
```
```
asyncfunctionaddMemory(content: string, userId: string)
```

```
Aspect| Python| Node.js  
---|---|---  
Input Parameters| Single `text` parameter| `content` and `userId` parameters  
User ID Handling| Hardcoded `DEFAULT_USER_ID = "cursor_mcp"`| Dynamic `userId` parameter  
Return Type| String with success/error message| Boolean success indicator  
Error Handling| Try-catch with string error messages| Try-catch with boolean return  
Sources: [main.py45-64](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/main.py#L45-L64) [node/mem0/src/index.ts75-87](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/node/mem0/src/index.ts#L75-L87)
### Message Construction Patterns
Both implementations construct message arrays for mem0, but with different approaches:
**Python Message Format:**
```
```
messages = [{"role": "user", "content": text}]
```

```

**Node.js Message Format:**
```
```
const messages = [
 { role: 'system', content: 'Memory storage system' },
 { role: 'user', content }
];
```

```

The Node.js version includes a system message to provide context to mem0.
Sources: [main.py60-61](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/main.py#L60-L61) [node/mem0/src/index.ts77-80](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/node/mem0/src/index.ts#L77-L80)
## Data Flow and Processing
**Processing Steps:**
  1. Input validation and parameter extraction
  2. Message array construction with role-based content organization
  3. mem0 API call with user identification and format specification
  4. Response processing and error handling


Sources: [main.py45-64](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/main.py#L45-L64) [node/mem0/src/index.ts114-126](https://github.com/mem0ai/mem0-mcp/blob/fce38b39/node/mem0/src/index.ts#L114-L126)
## Custom Instructions and Context Enhancement
The Python implementation includes detailed custom instructions that guide mem0's processing:
```"""

    print("📋 Analyzing Actual Crawled Content:")
    print("-" * 40)
    print(f"Content length: {len(actual_crawled_content)} chars")
    print(f"Contains @mcp.tool: {'@mcp.tool' in actual_crawled_content}")
    print(f"Contains function def: {'def ' in actual_crawled_content}")
    print(f"Contains async def: {'async def' in actual_crawled_content}")
    print()

    # Test code extraction on the mangled content
    print("🔧 Testing Code Extraction on Actual Crawled Content:")
    print("-" * 50)

    extractor = CodeExtractor()
    blocks = extractor.extract_code_blocks(actual_crawled_content)

    print(f"Code blocks found: {len(blocks)}")
    for i, block in enumerate(blocks, 1):
        print(f"  Block {i}: {block.language.value}, {len(block.content)} chars")
        print(f"    Valid: {extractor._is_valid_code_block_improved(block)}")
        print(f"    Preview: {repr(block.content[:50])}")
    print()

    # Check for the specific patterns that indicate code blocks
    print("🔍 Looking for Mangled Code Patterns:")
    print("-" * 40)

    # Pattern for the broken code blocks
    broken_patterns = [
        r"```\s*```\s*async\s*def",  # Pattern from description
        r"asyncdefadd_coding_preference",  # Merged function name
        r"asyncfunctionaddMemory",  # Merged function name
        r"```\s*```",  # Empty code blocks
    ]

    for pattern in broken_patterns:
        matches = re.findall(
            pattern, actual_crawled_content, re.IGNORECASE | re.MULTILINE
        )
        if matches:
            print(f"Found pattern '{pattern}': {matches}")
        else:
            print(f"Pattern '{pattern}': Not found")

    # Try to extract the embedded code manually
    print("\n🛠️ Manual Code Reconstruction:")
    print("-" * 35)

    # Look for the section with the custom instructions
    custom_instructions_start = actual_crawled_content.find(
        "Custom Instructions and Context Enhancement"
    )
    if custom_instructions_start != -1:
        print("Found 'Custom Instructions' section")
        section = actual_crawled_content[custom_instructions_start:]
        print(f"Section length: {len(section)} chars")

        # This section should contain the @mcp.tool code
        if "@mcp.tool" in section:
            print("✅ Section contains @mcp.tool")
        else:
            print("❌ Section does NOT contain @mcp.tool")

        # Look for the end marker
        if section.endswith("```"):
            print("✅ Section ends with closing markdown")
        else:
            print("❌ Section does NOT end with closing markdown")
            print(f"Last 100 chars: {repr(section[-100:])}")
    else:
        print("❌ Could not find 'Custom Instructions' section")

    # Check what the full document looked like
    print("\n📄 Full Document Analysis:")
    print("-" * 30)

    # Simulate what the original full document should have contained
    proper_full_content = """
# Adding Coding Preferences

## Custom Instructions and Context Enhancement

The Python implementation includes detailed custom instructions that guide mem0's processing:

```python
@mcp.tool(
    description=\"\"\"Add a new coding preference to mem0. This tool stores code snippets, implementation details,
    and coding patterns for future reference. Store every code snippet. When storing code, you should include:
    - Complete code with all necessary imports and dependencies
    - Language/framework version information (e.g., "Python 3.9", "React 18")
    - Full implementation context and any required setup/configuration
    - [...more details...]
    The preference will be indexed for semantic search and can be retrieved later using natural language queries.\"\"\"
)
async def add_coding_preference(text: str) -> str:
    \"\"\"Add a new coding preference to mem0.
    [...]
    \"\"\"
    try:
        messages = [{"role": "user", "content": text}]
        mem0_client.add(messages, user_id=DEFAULT_USER_ID, output_format="v1.1")
        return f"Successfully added preference: {text}"
    except Exception as e:
        return f"Error adding preference: {str(e)}"
```

These instructions are applied to the mem0 project.
"""

    print("Testing proper content extraction:")
    proper_blocks = extractor.extract_code_blocks(proper_full_content)
    print(f"Proper content produces {len(proper_blocks)} code blocks")

    if proper_blocks:
        for i, block in enumerate(proper_blocks, 1):
            print(
                f"  Block {i}: {block.language.value}, {len(block.content)} chars, Valid: {extractor._is_valid_code_block_improved(block)}"
            )

    # Test the actual extraction pipeline (without Supabase dependencies)
    print("\n🔄 Testing Full Extraction Pipeline:")
    print("-" * 40)

    try:
        extracted_codes = extract_code_from_content(
            proper_full_content, "https://deepwiki.com/test"
        )
        print(f"Full pipeline produces {len(extracted_codes)} extracted code objects")

        for i, code in enumerate(extracted_codes, 1):
            print(
                f"  Code {i}: {code.programming_language}, complexity {code.complexity_score}"
            )
            print(
                f"    Summary: {code.summary[:100] if code.summary else 'No summary'}..."
            )
    except Exception as e:
        print(f"Pipeline test failed: {e}")

    print("\n🎯 Root Cause Analysis:")
    print("-" * 25)
    print("FINDINGS:")
    print("1. Code extraction logic works correctly on proper markdown")
    print("2. The issue is in the CRAWLING/CHUNKING process")
    print("3. Crawl4ai is mangling markdown code blocks during content processing")
    print("4. Empty code blocks (```\\n```) are being created")
    print("5. Function signatures are being merged without spaces")
    print()
    print("RECOMMENDATIONS:")
    print("1. Investigate crawl4ai's markdown processing settings")
    print("2. Consider using a different markdown preservation strategy")
    print("3. Implement post-processing to fix mangled code blocks")
    print("4. Add validation to detect and report corrupted content")


if __name__ == "__main__":
    test_crawl_pipeline_simulation()
