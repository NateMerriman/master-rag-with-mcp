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

# Optional OpenAI import for AI-powered summaries
try:
    import openai

    # Use the environment variable for the API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import the centralized model configuration function
try:
    from .utils import _get_contextual_model
except ImportError:
    # Fallback if utils import fails
    def _get_contextual_model() -> Optional[str]:
        return os.getenv("CONTEXTUAL_MODEL", "gpt-4.1-nano")


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

    # Language-specific examples for template-based enhancement
    LANGUAGE_EXAMPLES = {
        ProgrammingLanguage.PYTHON: {
            "example_code": """def authenticate_user(username, password):
    if not username or not password:
        return False
    return bcrypt.check_password_hash(stored_hash, password)""",
            "example_summary": "User authentication function that validates credentials using bcrypt hashing. Checks for empty inputs and compares provided password against stored hash for secure login verification. Essential security component for user access control systems.",
        },
        ProgrammingLanguage.JAVASCRIPT: {
            "example_code": """const fetchData = async (url) => {
    try {
        const response = await fetch(url);
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}""",
            "example_summary": "Asynchronous data fetching function using modern async/await syntax and error handling. Retrieves JSON data from specified URL endpoint with comprehensive error logging and exception propagation. Common pattern for API communication in web applications.",
        },
        ProgrammingLanguage.TYPESCRIPT: {
            "example_code": """interface User {
    id: number;
    email: string;
    roles: string[];
}

function validateUser(user: User): boolean {
    return user.id > 0 && user.email.includes('@');
}""",
            "example_summary": "TypeScript user validation system with strict type checking and interface definition. Defines User interface with required fields and implements validation logic for ID and email format. Demonstrates type safety and structural validation patterns.",
        },
        ProgrammingLanguage.SQL: {
            "example_code": """SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2024-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5
ORDER BY total_spent DESC;""",
            "example_summary": "Complex SQL query analyzing user purchasing behavior with aggregation and filtering. Joins users and orders tables to calculate order counts and total spending per user since 2024. Filters for active customers with more than 5 orders and ranks by spending.",
        },
        ProgrammingLanguage.JAVA: {
            "example_code": """public class UserService {
    private final UserRepository repository;
    
    @Autowired
    public UserService(UserRepository repository) {
        this.repository = repository;
    }
    
    @Transactional
    public User createUser(String email, String name) {
        validateEmail(email);
        return repository.save(new User(email, name));
    }
}""",
            "example_summary": "Spring Boot service class implementing user creation with dependency injection and transaction management. Uses constructor injection for UserRepository and validates email before persisting new users. Demonstrates enterprise Java patterns with annotations and proper separation of concerns.",
        },
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
        self.min_code_length = 25  # Lower minimum that allows small functions but filters out very short text
        self.max_code_length = 10000  # Maximum characters to avoid huge blocks

    def extract_code_blocks(self, content: str) -> List[CodeBlock]:
        """Extract all code blocks from the given content."""
        code_blocks = []

        lines = content.split("\n")

        # Extract only fenced code blocks. This is a stricter approach to avoid false positives.
        extracted_blocks = self._extract_fenced_blocks_improved(
            content, lines, 0
        )  # Always start from 0
        code_blocks.extend(extracted_blocks)

        # Filter valid code blocks with improved validation
        valid_blocks = []
        for i, block in enumerate(code_blocks):
            is_valid = self._is_valid_code_block_improved(block)
            if is_valid:
                valid_blocks.append(block)
        return valid_blocks

    def _extract_fenced_blocks_improved(
        self, content: str, lines: List[str], start_offset: int = 0
    ) -> List[CodeBlock]:
        """Extract fenced code blocks using improved logic from reference implementation."""
        blocks = []

        # Find all occurrences of triple backticks
        backtick_positions = []
        pos = start_offset
        while True:
            pos = content.find("```", pos)
            if pos == -1:
                break
            backtick_positions.append(pos)
            pos += 3

        # Process pairs of backticks
        i = 0
        while i < len(backtick_positions) - 1:
            start_pos = backtick_positions[i]
            end_pos = backtick_positions[i + 1]

            # Extract the content between backticks
            code_section = content[start_pos + 3 : end_pos]

            # Check if there's a language specifier on the first line
            lines_in_block = code_section.split("\n", 1)
            if len(lines_in_block) > 1:
                # Check if first line is a language specifier (no spaces, common language names)
                first_line = lines_in_block[0].strip()
                if first_line and not " " in first_line and len(first_line) < 20:
                    language_hint = first_line
                    code_content = (
                        lines_in_block[1].strip() if len(lines_in_block) > 1 else ""
                    )
                else:
                    language_hint = ""
                    code_content = code_section.strip()
            else:
                language_hint = ""
                code_content = code_section.strip()

            if not code_content:
                i += 2  # Move to next pair
                continue

            # Skip if code block is too short (using reference implementation threshold)
            if len(code_content) < self.min_code_length:
                i += 2  # Move to next pair
                continue

            language = self._detect_language(language_hint, code_content)

            # Find line numbers
            start_line = content[:start_pos].count("\n")
            end_line = start_line + code_content.count("\n")

            # Extract context before (1000 chars like reference implementation)
            context_start = max(0, start_pos - 1000)
            context_before = content[context_start:start_pos].strip()

            # Extract context after (1000 chars like reference implementation)
            context_end = min(len(content), end_pos + 3 + 1000)
            context_after = content[end_pos + 3 : context_end].strip()

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

            # Move to next pair (skip the closing backtick we just processed)
            i += 2

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

        # TypeScript patterns (check before JavaScript since it's more specific)
        if (
            re.search(r"\binterface\s+\w+", code)
            or re.search(r"\btype\s+\w+\s*=", code)
            or re.search(r":\s*(string|number|boolean|any)", code)
            or (": " in code and ("interface " in code or "type " in code))
        ):
            return ProgrammingLanguage.TYPESCRIPT

        # Python patterns
        if (
            re.search(r"\bdef\s+\w+\s*\(", code)
            or re.search(r"\bimport\s+\w+", code)
            or re.search(r"\bfrom\s+\w+\s+import", code)
            or "print(" in code
        ):
            return ProgrammingLanguage.PYTHON

        # JavaScript patterns (after TypeScript check)
        if (
            re.search(r"\bfunction\s+\w+\s*\(", code)
            or re.search(r"\bconst\s+\w+\s*=", code)
            or re.search(r"\blet\s+\w+\s*=", code)
            or "console.log(" in code
        ):
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

        # SQL patterns (improved detection)
        if (
            re.search(r"\bSELECT\s+", code, re.IGNORECASE)
            or re.search(r"\bINSERT\s+INTO", code, re.IGNORECASE)
            or re.search(r"\bCREATE\s+TABLE", code, re.IGNORECASE)
            or re.search(r"\bUPDATE\s+", code, re.IGNORECASE)
            or re.search(r"\bDELETE\s+FROM", code, re.IGNORECASE)
            or re.search(r"\bFROM\s+\w+", code, re.IGNORECASE)
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

    def _is_valid_code_block_improved(self, block: CodeBlock) -> bool:
        """Check if a code block is valid for processing with improved validation logic."""
        content = block.content.strip()

        # Basic length checks
        if not (self.min_code_length <= len(content) <= self.max_code_length):
            return False

        # Check for a minimum number of lines
        lines = content.split("\n")
        if len(lines) < 2:  # At least 2 lines of code
            return False

        # Filter out common non-code patterns that appear in documentation

        # 1. Check for tree/directory structures (like your examples)
        tree_indicators = ["├──", "└──", "│", "┌─", "┐", "┘", "└─"]
        if any(indicator in content for indicator in tree_indicators):
            return False

        # 2. Check for simple lists without code characteristics
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        if len(non_empty_lines) >= 3:  # Only check if we have enough lines
            # If most lines are just simple text without code chars, it's likely a list
            code_chars = {
                "(",
                ")",
                "{",
                "}",
                "[",
                "]",
                "=",
                ":",
                ";",
                "<",
                ">",
                "->",
                "=>",
                "+=",
                "-=",
            }
            lines_with_code_chars = 0
            for line in non_empty_lines:
                if any(char in line for char in code_chars):
                    lines_with_code_chars += 1

            # If less than 30% of lines have code characteristics, it's probably not code
            if lines_with_code_chars / len(non_empty_lines) < 0.3:
                return False

        # 3. Check for configuration-like content that might not be actual code
        # If all lines follow pattern "key: value" or "key = value" without complex logic
        config_pattern_count = 0
        for line in non_empty_lines:
            if ":" in line and line.count(":") == 1:
                parts = line.split(":")
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    # Simple key: value pattern
                    if not any(
                        code_indicator in line
                        for code_indicator in [
                            "(",
                            ")",
                            "{",
                            "}",
                            ";",
                            "function",
                            "def ",
                            "class ",
                            "if ",
                            "for ",
                            "while ",
                        ]
                    ):
                        config_pattern_count += 1

        # If more than 80% are simple config lines, it might be config, not code
        if (
            len(non_empty_lines) > 3
            and config_pattern_count / len(non_empty_lines) > 0.8
        ):
            # But allow it if it's a known config language
            if block.language not in [
                ProgrammingLanguage.YAML,
                ProgrammingLanguage.JSON,
                ProgrammingLanguage.XML,
            ]:
                return False

        # 4. Check for flowchart/diagram patterns
        diagram_keywords = [
            "flowchart",
            "graph",
            "sequenceDiagram",
            "classDiagram",
            "gantt",
            "pie",
        ]
        if any(keyword in content.lower() for keyword in diagram_keywords):
            return False

        # 5. Enhanced code characteristic check
        # Must have some programming constructs, not just basic punctuation
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
            r"SELECT\s+.*FROM",  # SQL
            r"INSERT\s+INTO",  # SQL
            r"<\w+[^>]*>.*</\w+>",  # HTML tags
            r"\w+\s*\{[^}]*\}",  # CSS/object syntax
            r"@\w+",  # Decorator/annotation
            r"=>",  # Arrow function
            r"async\s+",  # Async keyword
            r"await\s+",  # Await keyword
        ]

        has_programming_construct = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in programming_indicators
        )

        # If it's not a known data format and has no programming constructs, reject it
        if not has_programming_construct and block.language not in [
            ProgrammingLanguage.JSON,
            ProgrammingLanguage.YAML,
            ProgrammingLanguage.XML,
        ]:
            return False

        # 6. Additional filter for very short "code" that's likely just text
        if len(content) < 200:  # For shorter blocks, be more strict but not too strict
            # Must have multiple code indicators
            basic_code_chars = {"(", ")", "{", "}", "[", "]", "=", ";"}
            if not any(char in content for char in basic_code_chars):
                return False

            # Must not be just a simple list of items (but be more lenient for code)
            # Only reject if ALL lines are very short AND there are no clear code patterns
            if all(len(line.strip().split()) < 5 for line in non_empty_lines):
                # If all lines are extremely short (< 5 words), check for code patterns
                code_patterns_in_short = [
                    r"\bdef\s+",  # Python function definition
                    r"\bfunction\s+",  # JavaScript function
                    r"\breturn\s+",  # Return statement
                    r"\bif\s+",  # Conditional
                    r"\bfor\s+",  # Loop
                    r"\bwhile\s+",  # Loop
                    r"print\s*\(",  # Print function
                    r"console\.",  # Console methods
                    r"=\s*\w+\s*\(",  # Function assignment
                    r"\binterface\s+",  # TypeScript interface
                    r"\bclass\s+",  # Class definition
                    r"\btype\s+\w+\s*=",  # TypeScript type alias
                    r":\s*(string|number|boolean)",  # TypeScript type annotations
                ]

                # If we have clear code patterns, allow it even with short lines
                has_code_patterns = any(
                    re.search(pattern, content, re.IGNORECASE)
                    for pattern in code_patterns_in_short
                )

                if not has_code_patterns:
                    return False

        return True

    def _is_valid_code_block(self, block: CodeBlock) -> bool:
        """Legacy method for backward compatibility - redirects to improved validation."""
        return self._is_valid_code_block_improved(block)

    def calculate_complexity_score(
        self, code: str, language: ProgrammingLanguage
    ) -> int:
        """
        Calculates a complexity score (1-10) for a given code block.
        The score is based on language type, length, nesting, algorithmic complexity, and language-specific keywords.
        Data formats (JSON, YAML, XML) receive lower base scores as they represent structure, not logic.
        """
        # Define language categories for different scoring approaches
        data_formats = {
            ProgrammingLanguage.JSON,
            ProgrammingLanguage.YAML,
            ProgrammingLanguage.XML,
        }
        markup_languages = {ProgrammingLanguage.HTML, ProgrammingLanguage.CSS}
        programming_languages = {
            ProgrammingLanguage.PYTHON,
            ProgrammingLanguage.JAVASCRIPT,
            ProgrammingLanguage.TYPESCRIPT,
            ProgrammingLanguage.JAVA,
            ProgrammingLanguage.CPP,
            ProgrammingLanguage.C,
            ProgrammingLanguage.CSHARP,
            ProgrammingLanguage.PHP,
            ProgrammingLanguage.RUBY,
            ProgrammingLanguage.GO,
            ProgrammingLanguage.RUST,
        }

        lines = code.split("\n")
        num_lines = len(lines)

        # Base score varies by language type
        if language in data_formats:
            score = 0  # Data formats start at 0 - they're structural, not algorithmic
            length_multiplier = 0.5  # Reduce length impact for data formats
            nesting_multiplier = 0.5  # Reduce nesting impact for data formats
        elif language in markup_languages:
            score = 0  # Markup languages start at 0
            length_multiplier = 0.7
            nesting_multiplier = 0.7
        elif language == ProgrammingLanguage.SQL:
            score = 1  # SQL gets normal base score
            length_multiplier = 0.8
            nesting_multiplier = 0.8
        elif language in programming_languages:
            score = 1  # Programming languages get full base score
            length_multiplier = 1.0
            nesting_multiplier = 1.0
        else:  # UNKNOWN or others
            score = 1
            length_multiplier = 0.9
            nesting_multiplier = 0.9

        # 1. Length-based score (up to 3 points, adjusted by multiplier)
        length_score = 0
        if num_lines > 50:
            length_score = 3
        elif num_lines > 30:
            length_score = 2
        elif num_lines > 15:
            length_score = 1

        score += int(length_score * length_multiplier)

        # 2. Nesting-based score (up to 3 points, adjusted by multiplier)
        max_nesting = self._calculate_nesting_level(code, language)
        nesting_score = min(3, max_nesting)
        score += int(nesting_score * nesting_multiplier)

        # 3. Algorithmic complexity score (up to 2 points) - only for programming languages
        if language in programming_languages or language == ProgrammingLanguage.SQL:
            complexity_score = self._calculate_algorithmic_complexity(code, language)
            score += complexity_score

        # 4. Keyword-based score (up to 4 points) - language-specific patterns
        keyword_score = 0
        if language in self.COMPLEX_PATTERNS:
            for pattern in self.COMPLEX_PATTERNS[language]:
                if pattern.search(code):
                    keyword_score += 1

        # Apply different weights for keyword complexity
        if language in data_formats:
            keyword_score = 0  # No keyword complexity for data formats
        elif language in markup_languages:
            keyword_score = min(1, keyword_score)  # Cap at 1 for markup
        else:
            keyword_score = min(
                4, keyword_score
            )  # Full weight for programming languages

        score += keyword_score

        # 5. Data format specific adjustments
        if language == ProgrammingLanguage.JSON:
            score += self._calculate_json_complexity(code)
        elif language == ProgrammingLanguage.YAML:
            score += self._calculate_yaml_complexity(code)

        return min(10, max(1, score))  # Ensure score is between 1-10

    def _calculate_algorithmic_complexity(
        self, code: str, language: ProgrammingLanguage
    ) -> int:
        """
        Calculate algorithmic complexity based on control structures and logic patterns.
        Returns 0-2 points based on the presence of loops, conditionals, and complex logic.
        """
        complexity_score = 0
        code_lower = code.lower()

        # Control flow patterns (up to 1 point)
        control_patterns = [
            r"\b(if|else|elif|switch|case)\b",  # Conditionals
            r"\b(for|while|do|foreach)\b",  # Loops
            r"\b(try|catch|except|finally)\b",  # Error handling
        ]

        for pattern in control_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                complexity_score += 0.5
                if complexity_score >= 1:
                    break

        # Advanced patterns (up to 1 additional point)
        advanced_patterns = [
            r"\b(async|await|promise)\b",  # Asynchronous code
            r"\b(lambda|=>\s*\{|\s*->\s*)",  # Lambda/arrow functions
            r"\b(yield|generator)\b",  # Generators
            r"@\w+",  # Decorators/annotations
            r"\b(interface|abstract|extends|implements)\b",  # OOP advanced concepts
        ]

        advanced_count = 0
        for pattern in advanced_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                advanced_count += 1

        if advanced_count >= 3:
            complexity_score += 1
        elif advanced_count >= 1:
            complexity_score += 0.5

        return min(2, int(complexity_score))

    def _calculate_json_complexity(self, code: str) -> int:
        """
        Calculate complexity specific to JSON structures.
        Returns 0-2 points based on depth and array complexity.
        """
        try:
            import json

            # Try to parse as JSON to validate structure
            data = json.loads(code)

            # Simple heuristics for JSON complexity
            if isinstance(data, dict):
                # Count nested objects and arrays
                nested_objects = code.count("{") - 1  # Subtract outer object
                nested_arrays = code.count("[")

                if nested_objects > 3 or nested_arrays > 2:
                    return 2
                elif nested_objects > 1 or nested_arrays > 0:
                    return 1

        except (json.JSONDecodeError, ValueError):
            # If not valid JSON, treat as regular structured data
            pass

        return 0

    def _calculate_yaml_complexity(self, code: str) -> int:
        """Calculate complexity for YAML content."""
        lines = [line.strip() for line in code.split("\n") if line.strip()]
        complexity = 1

        # Nested structures (indentation levels)
        indent_levels = set()
        for line in lines:
            if ":" in line:
                indent = len(line) - len(line.lstrip())
                indent_levels.add(indent)

        complexity += min(len(indent_levels), 3)

        # Arrays and complex values
        if any("[" in line or "-" in line.lstrip()[:1] for line in lines):
            complexity += 1

        # References and anchors
        if any("&" in line or "*" in line or "<<" in line for line in lines):
            complexity += 2

        # Multiline values
        if any("|" in line or ">" in line for line in lines):
            complexity += 1

        # Multiple documents
        if "---" in code:
            complexity += 1

        return min(complexity, 10)

    def generate_enhanced_summary(
        self,
        code: str,
        language: ProgrammingLanguage,
        context_before: str = "",
        context_after: str = "",
    ) -> str:
        """
        Generate an enhanced AI-powered summary using template-based few-shot learning.

        Args:
            code: The code content to summarize
            language: The detected programming language
            context_before: Context before the code block
            context_after: Context after the code block

        Returns:
            A high-quality, consistent summary of what the code does
        """
        # Get the model from centralized configuration
        model_to_use = _get_contextual_model()

        if not OPENAI_AVAILABLE or not model_to_use:
            # Fallback to rule-based summary
            return self._generate_rule_based_summary(code, language)

        try:
            # Import config here to avoid circular imports
            from .config import get_config

            config = get_config()

            # Get example for this language or fallback to Python
            example = self.LANGUAGE_EXAMPLES.get(
                language, self.LANGUAGE_EXAMPLES[ProgrammingLanguage.PYTHON]
            )

            # Limit context based on configuration
            max_context = config.code_summary_max_context_chars
            context_before_limited = (
                context_before[-max_context:] if context_before else "No prior context"
            )
            context_after_limited = (
                context_after[:max_context] if context_after else "No following context"
            )

            # Build enhanced prompt with few-shot example
            prompt = f"""You are an expert code documentation assistant. Analyze code blocks and provide clear, actionable summaries for developers.

**Example:**
Language: {language.value}
Code:
{example["example_code"]}

Summary: {example["example_summary"]}

**Your Task:**
Language: {language.value}
Context Before: {context_before_limited}

Code:
{code}

Context After: {context_after_limited}

**Requirements:**
- Write exactly 2-3 sentences
- First sentence: Describe what the code DOES (action/purpose)
- Second sentence: Explain HOW it works or what makes it notable
- Third sentence (if needed): Mention its role in broader context or special characteristics
- Use active voice and clear technical terminology
- Focus on practical understanding for developers searching for similar solutions
- Mention specific frameworks/libraries if relevant
- Include key technical patterns or methodologies used

Summary:"""

            response = openai.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical documentation specialist focused on creating search-optimized code summaries. Provide precise, actionable descriptions that help developers understand both functionality and implementation approach.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.2,  # Lower temperature for more consistent results
            )

            summary = response.choices[0].message.content.strip()
            logger.debug(f"Generated enhanced AI summary using model: {model_to_use}")
            return summary

        except Exception as e:
            logger.warning(
                f"Failed to generate enhanced AI summary using model {model_to_use}: {e}"
            )
            # Fallback to basic summary generation
            return self.generate_summary(code, language, context_before, context_after)

    def generate_summary(
        self,
        code: str,
        language: ProgrammingLanguage,
        context_before: str = "",
        context_after: str = "",
    ) -> str:
        """
        Generate an AI-powered summary of the code block using the OpenAI API.

        Args:
            code: The code content to summarize
            language: The detected programming language
            context_before: Context before the code block
            context_after: Context after the code block

        Returns:
            A concise summary of what the code does
        """
        # Check if enhanced summaries are enabled
        try:
            from .config import get_config

            config = get_config()
            if config.use_enhanced_code_summaries:
                return self.generate_enhanced_summary(
                    code, language, context_before, context_after
                )
        except Exception:
            # If config fails, continue with basic summary
            pass

        # Get the model from centralized configuration
        model_to_use = _get_contextual_model()

        if OPENAI_AVAILABLE and model_to_use:
            try:
                # Prepare the prompt with context
                prompt = f"""
                Analyze this {language.value} code and provide a concise 1-2 sentence summary of what it does:

                Context before:
                {context_before[-200:] if context_before else "No context"}

                Code:
                {code}

                Context after:
                {context_after[:200] if context_after else "No context"}

                Provide a clear, concise summary focusing on the main purpose and functionality:
                """

                response = openai.chat.completions.create(
                    model=model_to_use,  # Use the configured model instead of hard-coded
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.3,
                )

                logger.debug(f"Generated AI summary using model: {model_to_use}")
                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(
                    f"Failed to generate AI summary using model {model_to_use}: {e}"
                )
                # Fallback to rule-based summary
                return self._generate_rule_based_summary(code, language)
        else:
            # OpenAI not available or no model configured, use rule-based summary
            if not OPENAI_AVAILABLE:
                logger.debug("OpenAI not available, using rule-based summary")
            elif not model_to_use:
                logger.debug("No contextual model configured, using rule-based summary")
            return self._generate_rule_based_summary(code, language)

    def _generate_rule_based_summary(
        self, code: str, language: ProgrammingLanguage
    ) -> str:
        """Generate a basic rule-based summary as fallback."""
        lines = code.split("\n")
        line_count = len(lines)

        # Extract function/class names
        if language == ProgrammingLanguage.PYTHON:
            function_match = re.search(r"def\s+(\w+)", code)
            class_match = re.search(r"class\s+(\w+)", code)
            if class_match:
                return f"Python class '{class_match.group(1)}' definition with {line_count} lines"
            elif function_match:
                return f"Python function '{function_match.group(1)}' with {line_count} lines"
        elif language in [
            ProgrammingLanguage.JAVASCRIPT,
            ProgrammingLanguage.TYPESCRIPT,
        ]:
            function_match = re.search(r"function\s+(\w+)", code)
            const_match = re.search(r"const\s+(\w+)\s*=", code)
            if function_match:
                return f"{language.value.title()} function '{function_match.group(1)}' with {line_count} lines"
            elif const_match:
                return f"{language.value.title()} constant/function '{const_match.group(1)}' with {line_count} lines"

        return f"{language.value.title()} code block with {line_count} lines"

    def _extract_identifiers(
        self, code: str, language: ProgrammingLanguage
    ) -> List[str]:
        """Extract function names, class names, and other identifiers from code."""
        identifiers = []

        if language == ProgrammingLanguage.PYTHON:
            # Functions
            identifiers.extend(re.findall(r"def\s+(\w+)", code))
            # Classes
            identifiers.extend(re.findall(r"class\s+(\w+)", code))
            # Imports
            identifiers.extend(re.findall(r"import\s+(\w+)", code))
            identifiers.extend(re.findall(r"from\s+(\w+)", code))

        elif language in [
            ProgrammingLanguage.JAVASCRIPT,
            ProgrammingLanguage.TYPESCRIPT,
        ]:
            # Functions
            identifiers.extend(re.findall(r"function\s+(\w+)", code))
            # Constants/variables
            identifiers.extend(re.findall(r"(?:const|let|var)\s+(\w+)", code))
            # Classes
            identifiers.extend(re.findall(r"class\s+(\w+)", code))

        elif language == ProgrammingLanguage.JAVA:
            # Classes
            identifiers.extend(re.findall(r"class\s+(\w+)", code))
            # Methods
            identifiers.extend(
                re.findall(r"(?:public|private|protected)?\s*\w+\s+(\w+)\s*\(", code)
            )

        elif language == ProgrammingLanguage.SQL:
            # Tables in SELECT
            identifiers.extend(re.findall(r"FROM\s+(\w+)", code, re.IGNORECASE))
            # Tables in JOIN
            identifiers.extend(re.findall(r"JOIN\s+(\w+)", code, re.IGNORECASE))

        return identifiers[:10]  # Limit to first 10 identifiers

    def _extract_patterns(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Extract code patterns and features."""
        patterns = []

        # Common patterns across languages
        if "async" in code.lower():
            patterns.append("async")
        if "await" in code.lower():
            patterns.append("await")
        if re.search(r"\btry\b", code, re.IGNORECASE):
            patterns.append("error_handling")
        if re.search(r"\bclass\b", code, re.IGNORECASE):
            patterns.append("oop")
        if re.search(r"\binterface\b", code, re.IGNORECASE):
            patterns.append("interface")
        if re.search(r"\bimport\b", code, re.IGNORECASE):
            patterns.append("imports")

        # Language-specific patterns
        if language == ProgrammingLanguage.PYTHON:
            if "@" in code:
                patterns.append("decorators")
            if "yield" in code:
                patterns.append("generator")
            if "with " in code:
                patterns.append("context_manager")

        elif language in [
            ProgrammingLanguage.JAVASCRIPT,
            ProgrammingLanguage.TYPESCRIPT,
        ]:
            if "Promise" in code:
                patterns.append("promises")
            if "=>" in code:
                patterns.append("arrow_functions")
            if ".map(" in code or ".filter(" in code or ".reduce(" in code:
                patterns.append("functional_programming")

        elif language == ProgrammingLanguage.SQL:
            if re.search(r"\bJOIN\b", code, re.IGNORECASE):
                patterns.append("joins")
            if re.search(r"\bGROUP\s+BY\b", code, re.IGNORECASE):
                patterns.append("aggregation")
            if re.search(r"\bWITH\b", code, re.IGNORECASE):
                patterns.append("cte")

        return patterns[:8]  # Limit to 8 patterns

    def _extract_context_headers(
        self, context_before: str, context_after: str
    ) -> List[str]:
        """Extract markdown headers from context for better categorization."""
        headers = []

        # Look for markdown headers in context
        for context in [context_before, context_after]:
            if context:
                # Clean up the context text - remove extra whitespace and normalize line endings
                cleaned_context = re.sub(r"\s+", " ", context.strip())
                # Split by lines and rejoin to get proper line structure
                lines = [
                    line.strip() for line in context.strip().split("\n") if line.strip()
                ]
                context_cleaned = "\n".join(lines)

                # Look for headers in both the original and cleaned context
                for text in [context, context_cleaned]:
                    header_matches = re.findall(r"^(#+)\s+(.+)$", text, re.MULTILINE)
                    for level, text_content in header_matches:
                        headers.append(f"h{len(level)}: {text_content.strip()}")

        return headers[:5]  # Limit to 5 headers

    def _extract_keywords_from_context(
        self, context_before: str, context_after: str
    ) -> List[str]:
        """Extract relevant keywords from surrounding context."""
        keywords = []

        # Common technical keywords to look for
        technical_terms = [
            "api",
            "function",
            "method",
            "class",
            "example",
            "usage",
            "tutorial",
            "configuration",
            "setup",
            "installation",
            "documentation",
            "guide",
            "implementation",
            "algorithm",
            "pattern",
            "framework",
            "library",
            "utility",
            "helper",
            "tool",
            "service",
            "component",
            "module",
        ]

        combined_context = f"{context_before} {context_after}".lower()

        for term in technical_terms:
            if term in combined_context:
                keywords.append(term)

        return keywords[:6]  # Limit to 6 keywords

    def _generate_enhanced_metadata(self, block: CodeBlock) -> Dict[str, Any]:
        """Generate comprehensive metadata for a code block."""
        code = block.content
        language = block.language

        # Basic statistics
        lines = code.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        # Extract identifiers and patterns
        identifiers = self._extract_identifiers(code, language)
        patterns = self._extract_patterns(code, language)
        headers = self._extract_context_headers(
            block.context_before, block.context_after
        )
        keywords = self._extract_keywords_from_context(
            block.context_before, block.context_after
        )

        # Limit context length to prevent bloated metadata (500 chars each)
        context_before_limited = (
            block.context_before[-500:] if block.context_before else ""
        )
        context_after_limited = block.context_after[:500] if block.context_after else ""

        metadata = {
            # Original fields
            "start_line": block.start_line,
            "end_line": block.end_line,
            "block_type": block.block_type,
            "context_before": context_before_limited,
            "context_after": context_after_limited,
            # Enhanced statistics
            "char_count": len(code),
            "word_count": len(code.split()),
            "line_count": len(lines),
            "non_empty_line_count": len(non_empty_lines),
            "code_density": round(len(non_empty_lines) / len(lines), 2) if lines else 0,
            # Code analysis
            "identifiers": identifiers,
            "patterns": patterns,
            "has_comments": bool(re.search(r'#|//|/\*|\*/|\'\'\'|"""', code)),
            "has_strings": bool(re.search(r'["\']', code)),
            "has_loops": bool(re.search(r"\b(for|while|do)\b", code, re.IGNORECASE)),
            "has_conditions": bool(
                re.search(r"\b(if|else|elif|switch|case)\b", code, re.IGNORECASE)
            ),
            # Context analysis
            "surrounding_headers": headers,
            "context_keywords": keywords,
            "estimated_reading_time": max(
                1, len(code.split()) // 200
            ),  # Roughly 200 words per minute
            # Quality indicators
            "complexity_indicators": {
                "nesting_level": self._calculate_nesting_level(code, language),
                "cyclomatic_complexity": self._estimate_cyclomatic_complexity(code),
                "unique_identifiers": len(set(identifiers)),
            },
        }

        return metadata

    def _calculate_nesting_level(self, code: str, language: ProgrammingLanguage) -> int:
        """Calculate the maximum nesting level in the code."""
        max_nesting = 0

        if language == ProgrammingLanguage.PYTHON:
            # Use indentation for Python
            lines = code.split("\n")
            max_indent = 0
            for line in lines:
                stripped_line = line.lstrip()
                if stripped_line:
                    indent = len(line) - len(stripped_line)
                    max_indent = max(max_indent, indent)
            max_nesting = max_indent // 4  # Assume 4 spaces per level
        else:
            # Use braces for other languages
            current_nesting = 0
            for char in code:
                if char == "{":
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                elif char == "}":
                    current_nesting = max(0, current_nesting - 1)

        return max_nesting

    def _estimate_cyclomatic_complexity(self, code: str) -> int:
        """Estimate cyclomatic complexity by counting decision points."""
        complexity = 1  # Base complexity

        # Count decision points
        decision_keywords = [
            r"\bif\b",
            r"\belse\b",
            r"\belif\b",
            r"\bwhile\b",
            r"\bfor\b",
            r"\btry\b",
            r"\bcatch\b",
            r"\bexcept\b",
            r"\bfinally\b",
            r"\bswitch\b",
            r"\bcase\b",
            r"\band\b",
            r"\bor\b",
            r"&&",
            r"\|\|",
        ]

        for keyword in decision_keywords:
            complexity += len(re.findall(keyword, code, re.IGNORECASE))

        return min(complexity, 20)  # Cap at 20 for sanity

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

            # Generate AI-powered summary
            summary = self.generate_summary(
                block.content, block.language, block.context_before, block.context_after
            )

            # Generate enhanced metadata
            metadata = self._generate_enhanced_metadata(block)

            # Create the ExtractedCode object matching the new schema
            processed_code = ExtractedCode(
                content=block.content,
                summary=summary,
                programming_language=block.language.value,
                complexity_score=complexity,
                url=source_url,
                chunk_number=i + 1,  # 1-based index for chunks from this page
                metadata=metadata,
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
