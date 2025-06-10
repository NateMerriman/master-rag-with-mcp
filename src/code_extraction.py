"""
Code extraction and processing module for the MCP Crawl4AI RAG project.

This module identifies, extracts, and processes code blocks from crawled content
to populate the code_examples table with structured code data.
"""

import re
import ast
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os
import openai

# Use the environment variable for the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)


class ProgrammingLanguage(Enum):
    """Supported programming languages for code extraction."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    PHP = "php"
    RUBY = "ruby"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    SHELL = "shell"
    YAML = "yaml"
    JSON = "json"
    XML = "xml"
    UNKNOWN = "unknown"


@dataclass
class CodeBlock:
    """Represents an extracted code block with metadata."""

    content: str
    language: ProgrammingLanguage
    start_line: int
    end_line: int
    block_type: str  # 'fenced', 'indented', 'inline'
    context_before: str = ""  # Text before the code block for context
    context_after: str = ""  # Text after the code block for context


@dataclass
class ExtractedCode:
    """Represents processed code ready for database storage."""

    content: str
    summary: str  # Added to store the AI-generated summary
    programming_language: str
    complexity_score: int
    url: str
    chunk_number: int
    metadata: Dict[str, Any]


class CodeExtractor:
    """Extracts and processes code blocks from markdown and other text content."""

    # Language mapping for common aliases
    LANGUAGE_ALIASES = {
        "py": ProgrammingLanguage.PYTHON,
        "js": ProgrammingLanguage.JAVASCRIPT,
        "ts": ProgrammingLanguage.TYPESCRIPT,
        "jsx": ProgrammingLanguage.JAVASCRIPT,
        "tsx": ProgrammingLanguage.TYPESCRIPT,
        "c++": ProgrammingLanguage.CPP,
        "cc": ProgrammingLanguage.CPP,
        "cxx": ProgrammingLanguage.CPP,
        "cs": ProgrammingLanguage.CSHARP,
        "sh": ProgrammingLanguage.SHELL,
        "bash": ProgrammingLanguage.SHELL,
        "zsh": ProgrammingLanguage.SHELL,
        "fish": ProgrammingLanguage.SHELL,
        "yml": ProgrammingLanguage.YAML,
        "htm": ProgrammingLanguage.HTML,
    }

    # Patterns for different code block types
    FENCED_CODE_PATTERN = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL | re.MULTILINE)

    INDENTED_CODE_PATTERN = re.compile(
        r"(?:^|\n)((?:    |\t)[^\n]*(?:\n(?:    |\t)[^\n]*)*)", re.MULTILINE
    )

    INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")

    COMPLEX_PATTERNS = {
        ProgrammingLanguage.PYTHON: [
            re.compile(r"\bclass\s"),
            re.compile(r"\btry:"),
            re.compile(r"\basync\s+def\b"),
            re.compile(r"@[a-zA-Z0-9_.]+"),  # Decorators
            re.compile(r"\byield\s"),
            re.compile(r"with\s+"),
        ],
        ProgrammingLanguage.JAVASCRIPT: [
            re.compile(r"\bclass\s"),
            re.compile(r"\btry\s*\{"),
            re.compile(r"\basync\s+function\b"),
            re.compile(r"\bawait\s"),
            re.compile(r"Promise\."),
        ],
        ProgrammingLanguage.TYPESCRIPT: [
            re.compile(r"\bclass\s"),
            re.compile(r"\binterface\s"),
            re.compile(r"\btry\s*\{"),
            re.compile(r"\basync\s+function\b"),
            re.compile(r"\bawait\s"),
            re.compile(r"<[A-Z][a-zA-Z0-9_]*>"),  # Generics
        ],
        ProgrammingLanguage.JAVA: [
            re.compile(r"\bclass\s"),
            re.compile(r"\binterface\s"),
            re.compile(r"\btry\s*\{"),
            re.compile(r"\bfinally\s*\{"),
            re.compile(r"\bsynchronized\b"),
            re.compile(r"<[A-Z][a-zA-Z0-9_]*>"),  # Generics
        ],
        ProgrammingLanguage.SQL: [
            re.compile(r"\bJOIN\b", re.IGNORECASE),
            re.compile(r"\bUNION\b", re.IGNORECASE),
            re.compile(r"\bWITH\b", re.IGNORECASE),
            re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE),
            re.compile(r"\(SELECT", re.IGNORECASE),  # Subquery
        ],
        ProgrammingLanguage.RUST: [
            re.compile(r"\bunsafe\b"),
            re.compile(r"\basync\b"),
            re.compile(r"\bawait\b"),
            re.compile(r"\btrait\b"),
            re.compile(r"<\'"),  # Lifetimes
        ],
    }

    def __init__(self):
        self.min_code_length = 10  # Minimum characters for a valid code block
        self.max_code_length = 10000  # Maximum characters to avoid huge blocks

    def extract_code_blocks(self, content: str) -> List[CodeBlock]:
        """Extract all code blocks from the given content."""
        code_blocks = []
        lines = content.split("\n")

        # Extract fenced code blocks only. This is a stricter approach to avoid false positives.
        code_blocks.extend(self._extract_fenced_blocks(content, lines))

        # Filter valid code blocks
        return [block for block in code_blocks if self._is_valid_code_block(block)]

    def _extract_fenced_blocks(self, content: str, lines: List[str]) -> List[CodeBlock]:
        """Extract fenced code blocks (```language ... ```)."""
        blocks = []

        for match in self.FENCED_CODE_PATTERN.finditer(content):
            language_str = match.group(1) or ""  # Default to empty string if no hint
            code_content = match.group(2).strip()

            if not code_content:
                continue

            language = self._detect_language(language_str, code_content)

            # Find line numbers
            start_pos = match.start()
            start_line = content[:start_pos].count("\n")
            end_line = start_line + code_content.count("\n")

            # Get context
            context_before, context_after = self._get_context(
                lines, start_line, end_line
            )

            blocks.append(
                CodeBlock(
                    content=code_content,
                    language=language,
                    start_line=start_line,
                    end_line=end_line,
                    block_type="fenced",
                    context_before=context_before,
                    context_after=context_after,
                )
            )

        return blocks

    def _extract_indented_blocks(
        self, content: str, lines: List[str]
    ) -> List[CodeBlock]:
        """Extract indented code blocks (4 spaces or tab indentation)."""
        blocks = []

        for match in self.INDENTED_CODE_PATTERN.finditer(content):
            code_content = match.group(1)

            # Remove indentation
            code_lines = code_content.split("\n")
            dedented_lines = []
            for line in code_lines:
                if line.startswith("    "):
                    dedented_lines.append(line[4:])
                elif line.startswith("\t"):
                    dedented_lines.append(line[1:])
                else:
                    dedented_lines.append(line)

            code_content = "\n".join(dedented_lines).strip()

            if not code_content:
                continue

            language = self._detect_language("", code_content)

            # Find line numbers
            start_pos = match.start()
            start_line = content[:start_pos].count("\n")
            end_line = start_line + code_content.count("\n")

            # Get context
            context_before, context_after = self._get_context(
                lines, start_line, end_line
            )

            blocks.append(
                CodeBlock(
                    content=code_content,
                    language=language,
                    start_line=start_line,
                    end_line=end_line,
                    block_type="indented",
                    context_before=context_before,
                    context_after=context_after,
                )
            )

        return blocks

    def _detect_language(
        self, language_hint: str, code_content: str
    ) -> ProgrammingLanguage:
        """Detect programming language from hint and content analysis."""
        # First try the language hint
        if language_hint:
            lang_lower = language_hint.lower()
            if lang_lower in [lang.value for lang in ProgrammingLanguage]:
                return ProgrammingLanguage(lang_lower)
            elif lang_lower in self.LANGUAGE_ALIASES:
                return self.LANGUAGE_ALIASES[lang_lower]

        # Content-based detection
        return self._detect_language_from_content(code_content)

    def _detect_language_from_content(self, code: str) -> ProgrammingLanguage:
        """Detect language based on code patterns."""
        code_lower = code.lower()

        # Python patterns
        if (
            re.search(r"\bdef\s+\w+\s*\(", code)
            or re.search(r"\bimport\s+\w+", code)
            or re.search(r"\bfrom\s+\w+\s+import", code)
            or "print(" in code
        ):
            return ProgrammingLanguage.PYTHON

        # JavaScript/TypeScript patterns
        if (
            re.search(r"\bfunction\s+\w+\s*\(", code)
            or re.search(r"\bconst\s+\w+\s*=", code)
            or re.search(r"\blet\s+\w+\s*=", code)
            or "console.log(" in code
        ):
            if ": " in code and ("interface " in code or "type " in code):
                return ProgrammingLanguage.TYPESCRIPT
            return ProgrammingLanguage.JAVASCRIPT

        # Java patterns
        if (
            re.search(r"\bpublic\s+class\s+\w+", code)
            or re.search(r"\bpublic\s+static\s+void\s+main", code)
            or "System.out.println(" in code
        ):
            return ProgrammingLanguage.JAVA

        # C/C++ patterns
        if re.search(r"#include\s*<", code) or re.search(r"\bint\s+main\s*\(", code):
            if "std::" in code or "cout <<" in code or "cin >>" in code:
                return ProgrammingLanguage.CPP
            return ProgrammingLanguage.C

        # SQL patterns
        if (
            re.search(r"\bSELECT\s+", code_lower)
            or re.search(r"\bINSERT\s+INTO", code_lower)
            or re.search(r"\bCREATE\s+TABLE", code_lower)
        ):
            return ProgrammingLanguage.SQL

        # HTML patterns
        if re.search(r"<\w+[^>]*>", code) and re.search(r"</\w+>", code):
            return ProgrammingLanguage.HTML

        # CSS patterns
        if re.search(r"\w+\s*\{[^}]*\}", code) and ":" in code and ";" in code:
            return ProgrammingLanguage.CSS

        # Shell patterns
        if (
            code.startswith("#!/bin/")
            or re.search(r"\$\w+", code)
            or " && " in code
            or " || " in code
        ):
            return ProgrammingLanguage.SHELL

        # JSON patterns
        if (
            code.strip().startswith("{")
            and code.strip().endswith("}")
            and '"' in code
            and ":" in code
        ):
            return ProgrammingLanguage.JSON

        return ProgrammingLanguage.UNKNOWN

    def _get_context(
        self, lines: List[str], start_line: int, end_line: int
    ) -> Tuple[str, str]:
        """Get the surrounding context for a code block."""
        context_before = "\n".join(lines[max(0, start_line - 10) : start_line])
        context_after = "\n".join(lines[end_line + 1 : end_line + 11])
        return context_before, context_after

    def _is_valid_code_block(self, block: CodeBlock) -> bool:
        """Check if a code block is valid for processing."""
        # Basic length checks
        if not (self.min_code_length <= len(block.content) <= self.max_code_length):
            return False

        # Check for a minimum number of lines
        if block.content.count("\n") < 1:  # At least 2 lines of code
            return False

        # Sanity check for common code characters. This helps filter out plain text lists.
        code_chars = {"(", ")", "{", "}", "[", "]", "=", ":", ";", "<", ">"}
        if not any(char in block.content for char in code_chars):
            return False

        return True

    def calculate_complexity_score(
        self, code: str, language: ProgrammingLanguage
    ) -> int:
        """
        Calculates a complexity score (1-10) for a given code block.
        The score is based on length, nesting, and language-specific keywords.
        """
        score = 1
        lines = code.split("\n")
        num_lines = len(lines)

        # 1. Length-based score (up to 3 points)
        if num_lines > 50:
            score += 3
        elif num_lines > 20:
            score += 2
        elif num_lines > 10:
            score += 1

        # 2. Nesting-based score (up to 3 points)
        max_nesting = 0
        if language == ProgrammingLanguage.PYTHON:
            # Use indentation for Python nesting
            max_indent = 0
            for line in lines:
                stripped_line = line.lstrip()
                if stripped_line:
                    indent = len(line) - len(stripped_line)
                    max_indent = max(max_indent, indent)
            # Assume 4 spaces per indent level
            max_nesting = max_indent // 4
        else:
            # Use curly braces for other languages
            current_nesting = 0
            for char in code:
                if char == "{":
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                elif char == "}":
                    current_nesting = max(0, current_nesting - 1)

        score += min(3, max_nesting)  # Add up to 3 points for nesting

        # 3. Keyword-based score (up to 4 points)
        keyword_score = 0
        if language in self.COMPLEX_PATTERNS:
            for pattern in self.COMPLEX_PATTERNS[language]:
                if pattern.search(code):
                    keyword_score += 1

        score += min(4, keyword_score)

        return min(10, score)

    def generate_summary(
        self,
        code: str,
        language: ProgrammingLanguage,
        context_before: str,
        context_after: str,
    ) -> str:
        """Generate a concise summary for a code block using OpenAI."""
        if not openai.api_key:
            logger.warning("OPENAI_API_KEY not found. Skipping summary generation.")
            return ""

        model_choice = os.getenv("MODEL_CHOICE", "gpt-3.5-turbo")

        prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example language="{language.value}">
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose."""

        try:
            response = openai.chat.completions.create(
                model=model_choice,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert programmer tasked with summarizing code examples.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.2,
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            logger.error(f"Error generating summary for code block: {e}")
            return ""

    def process_code_blocks(self, content: str, source_url: str) -> List[ExtractedCode]:
        """
        Extract, analyze, and process all code blocks from content for database insertion.

        Args:
            content (str): The raw text content to process.
            source_url (str): The URL of the source page.

        Returns:
            List[ExtractedCode]: A list of processed code objects.
        """
        processed_code_list = []
        code_blocks = self.extract_code_blocks(content)

        for i, block in enumerate(code_blocks):
            # Calculate complexity
            complexity = self.calculate_complexity_score(block.content, block.language)

            summary = self.generate_summary(
                block.content, block.language, block.context_before, block.context_after
            )

            # Create the ExtractedCode object matching the new schema
            processed_code = ExtractedCode(
                content=block.content,
                summary=summary,
                programming_language=block.language.value,
                complexity_score=complexity,
                url=source_url,
                chunk_number=i + 1,  # 1-based index for chunks from this page
                metadata={
                    "start_line": block.start_line,
                    "end_line": block.end_line,
                    "block_type": block.block_type,
                    "context_before": block.context_before,
                    "context_after": block.context_after,
                },
            )
            processed_code_list.append(processed_code)

        return processed_code_list


def extract_code_from_content(content: str, source_url: str) -> List[ExtractedCode]:
    """
    High-level function to extract and process code from a string of content.

    Args:
        content (str): The content to extract code from.
        source_url (str): The URL where the content originated.

    Returns:
        A list of ExtractedCode objects ready for storage.
    """
    extractor = CodeExtractor()
    return extractor.process_code_blocks(content, source_url)


def get_supported_languages() -> List[str]:
    """Returns a list of supported programming language identifiers."""
    return [
        lang.value
        for lang in ProgrammingLanguage
        if lang != ProgrammingLanguage.UNKNOWN
    ]
