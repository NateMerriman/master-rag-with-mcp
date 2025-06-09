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
    context_after: str = ""   # Text after the code block for context


@dataclass
class ExtractedCode:
    """Represents processed code ready for database storage."""
    code_content: str
    summary: str
    programming_language: str
    complexity_score: int
    context: str  # Combined context for better understanding


class CodeExtractor:
    """Extracts and processes code blocks from markdown and other text content."""
    
    # Language mapping for common aliases
    LANGUAGE_ALIASES = {
        'py': ProgrammingLanguage.PYTHON,
        'js': ProgrammingLanguage.JAVASCRIPT,
        'ts': ProgrammingLanguage.TYPESCRIPT,
        'jsx': ProgrammingLanguage.JAVASCRIPT,
        'tsx': ProgrammingLanguage.TYPESCRIPT,
        'c++': ProgrammingLanguage.CPP,
        'cc': ProgrammingLanguage.CPP,
        'cxx': ProgrammingLanguage.CPP,
        'cs': ProgrammingLanguage.CSHARP,
        'sh': ProgrammingLanguage.SHELL,
        'bash': ProgrammingLanguage.SHELL,
        'zsh': ProgrammingLanguage.SHELL,
        'fish': ProgrammingLanguage.SHELL,
        'yml': ProgrammingLanguage.YAML,
        'htm': ProgrammingLanguage.HTML,
    }
    
    # Patterns for different code block types
    FENCED_CODE_PATTERN = re.compile(
        r'```(\w+)?\n(.*?)```',
        re.DOTALL | re.MULTILINE
    )
    
    INDENTED_CODE_PATTERN = re.compile(
        r'(?:^|\n)((?:    |\t)[^\n]*(?:\n(?:    |\t)[^\n]*)*)',
        re.MULTILINE
    )
    
    INLINE_CODE_PATTERN = re.compile(
        r'`([^`\n]+)`'
    )
    
    def __init__(self):
        self.min_code_length = 10  # Minimum characters for a valid code block
        self.max_code_length = 10000  # Maximum characters to avoid huge blocks
        
    def extract_code_blocks(self, content: str) -> List[CodeBlock]:
        """Extract all code blocks from the given content."""
        code_blocks = []
        lines = content.split('\n')
        
        # Extract fenced code blocks
        code_blocks.extend(self._extract_fenced_blocks(content, lines))
        
        # Extract indented code blocks (only if no fenced blocks found)
        if not code_blocks:
            code_blocks.extend(self._extract_indented_blocks(content, lines))
        
        # Filter valid code blocks
        return [block for block in code_blocks if self._is_valid_code_block(block)]
    
    def _extract_fenced_blocks(self, content: str, lines: List[str]) -> List[CodeBlock]:
        """Extract fenced code blocks (```language ... ```)."""
        blocks = []
        
        for match in self.FENCED_CODE_PATTERN.finditer(content):
            language_str = match.group(1) or 'unknown'
            code_content = match.group(2).strip()
            
            if not code_content:
                continue
                
            language = self._detect_language(language_str, code_content)
            
            # Find line numbers
            start_pos = match.start()
            start_line = content[:start_pos].count('\n')
            end_line = start_line + code_content.count('\n')
            
            # Get context
            context_before, context_after = self._get_context(lines, start_line, end_line)
            
            blocks.append(CodeBlock(
                content=code_content,
                language=language,
                start_line=start_line,
                end_line=end_line,
                block_type='fenced',
                context_before=context_before,
                context_after=context_after
            ))
        
        return blocks
    
    def _extract_indented_blocks(self, content: str, lines: List[str]) -> List[CodeBlock]:
        """Extract indented code blocks (4 spaces or tab indentation)."""
        blocks = []
        
        for match in self.INDENTED_CODE_PATTERN.finditer(content):
            code_content = match.group(1)
            
            # Remove indentation
            code_lines = code_content.split('\n')
            dedented_lines = []
            for line in code_lines:
                if line.startswith('    '):
                    dedented_lines.append(line[4:])
                elif line.startswith('\t'):
                    dedented_lines.append(line[1:])
                else:
                    dedented_lines.append(line)
            
            code_content = '\n'.join(dedented_lines).strip()
            
            if not code_content:
                continue
            
            language = self._detect_language('', code_content)
            
            # Find line numbers
            start_pos = match.start()
            start_line = content[:start_pos].count('\n')
            end_line = start_line + code_content.count('\n')
            
            # Get context
            context_before, context_after = self._get_context(lines, start_line, end_line)
            
            blocks.append(CodeBlock(
                content=code_content,
                language=language,
                start_line=start_line,
                end_line=end_line,
                block_type='indented',
                context_before=context_before,
                context_after=context_after
            ))
        
        return blocks
    
    def _detect_language(self, language_hint: str, code_content: str) -> ProgrammingLanguage:
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
        if (re.search(r'\bdef\s+\w+\s*\(', code) or 
            re.search(r'\bimport\s+\w+', code) or
            re.search(r'\bfrom\s+\w+\s+import', code) or
            'print(' in code):
            return ProgrammingLanguage.PYTHON
        
        # JavaScript/TypeScript patterns
        if (re.search(r'\bfunction\s+\w+\s*\(', code) or
            re.search(r'\bconst\s+\w+\s*=', code) or
            re.search(r'\blet\s+\w+\s*=', code) or
            'console.log(' in code):
            if ': ' in code and ('interface ' in code or 'type ' in code):
                return ProgrammingLanguage.TYPESCRIPT
            return ProgrammingLanguage.JAVASCRIPT
        
        # Java patterns
        if (re.search(r'\bpublic\s+class\s+\w+', code) or
            re.search(r'\bpublic\s+static\s+void\s+main', code) or
            'System.out.println(' in code):
            return ProgrammingLanguage.JAVA
        
        # C/C++ patterns
        if (re.search(r'#include\s*<', code) or
            re.search(r'\bint\s+main\s*\(', code)):
            if ('std::' in code or 'cout <<' in code or 'cin >>' in code):
                return ProgrammingLanguage.CPP
            return ProgrammingLanguage.C
        
        # SQL patterns
        if (re.search(r'\bSELECT\s+', code_lower) or
            re.search(r'\bINSERT\s+INTO', code_lower) or
            re.search(r'\bCREATE\s+TABLE', code_lower)):
            return ProgrammingLanguage.SQL
        
        # HTML patterns
        if (re.search(r'<\w+[^>]*>', code) and
            re.search(r'</\w+>', code)):
            return ProgrammingLanguage.HTML
        
        # CSS patterns
        if (re.search(r'\w+\s*\{[^}]*\}', code) and
            ':' in code and ';' in code):
            return ProgrammingLanguage.CSS
        
        # Shell patterns
        if (code.startswith('#!/bin/') or
            re.search(r'\$\w+', code) or
            ' && ' in code or ' || ' in code):
            return ProgrammingLanguage.SHELL
        
        # JSON patterns
        if (code.strip().startswith('{') and code.strip().endswith('}') and
            '"' in code and ':' in code):
            return ProgrammingLanguage.JSON
        
        return ProgrammingLanguage.UNKNOWN
    
    def _get_context(self, lines: List[str], start_line: int, end_line: int) -> Tuple[str, str]:
        """Get context lines before and after the code block."""
        context_lines = 2  # Number of lines to include as context
        
        # Context before
        before_start = max(0, start_line - context_lines)
        context_before = '\n'.join(lines[before_start:start_line]).strip()
        
        # Context after
        after_end = min(len(lines), end_line + context_lines + 1)
        context_after = '\n'.join(lines[end_line + 1:after_end]).strip()
        
        return context_before, context_after
    
    def _is_valid_code_block(self, block: CodeBlock) -> bool:
        """Check if a code block is valid for extraction."""
        if len(block.content) < self.min_code_length:
            return False
        
        if len(block.content) > self.max_code_length:
            return False
        
        # Skip blocks that are mostly comments
        lines = block.content.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith(('#', '//', '/*', '*')))
        if comment_lines > len(lines) * 0.8:
            return False
        
        return True
    
    def calculate_complexity_score(self, code: str, language: ProgrammingLanguage) -> int:
        """Calculate complexity score from 1-10 based on code characteristics."""
        score = 1
        
        # Line count factor
        line_count = len(code.split('\n'))
        if line_count > 50:
            score += 3
        elif line_count > 20:
            score += 2
        elif line_count > 10:
            score += 1
        
        # Nesting level (approximated by indentation)
        max_indent = 0
        for line in code.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent // 4)  # Assuming 4-space indentation
        
        if max_indent > 4:
            score += 2
        elif max_indent > 2:
            score += 1
        
        # Language-specific complexity indicators
        complexity_patterns = {
            ProgrammingLanguage.PYTHON: [
                r'\bclass\s+\w+', r'\bdef\s+\w+', r'\btry:', r'\bexcept:', r'\bwith\s+',
                r'\bfor\s+\w+\s+in', r'\bwhile\s+', r'\bif\s+.*:', r'\belif\s+.*:'
            ],
            ProgrammingLanguage.JAVASCRIPT: [
                r'\bfunction\s+\w+', r'\bclass\s+\w+', r'\btry\s*\{', r'\bcatch\s*\(',
                r'\bfor\s*\(', r'\bwhile\s*\(', r'\bif\s*\(', r'\belse\s+if\s*\('
            ],
            ProgrammingLanguage.JAVA: [
                r'\bclass\s+\w+', r'\bpublic\s+\w+', r'\bprivate\s+\w+', r'\btry\s*\{',
                r'\bcatch\s*\(', r'\bfor\s*\(', r'\bwhile\s*\(', r'\bif\s*\('
            ]
        }
        
        patterns = complexity_patterns.get(language, [])
        complexity_indicators = sum(1 for pattern in patterns if re.search(pattern, code))
        
        if complexity_indicators > 5:
            score += 2
        elif complexity_indicators > 2:
            score += 1
        
        # Ensure score is within bounds
        return min(10, max(1, score))
    
    def generate_summary(self, code: str, language: ProgrammingLanguage, context: str = "") -> str:
        """Generate a summary of what the code does."""
        # This is a simple rule-based summary generator
        # In a real implementation, you might use an LLM for better summaries
        
        code_lower = code.lower()
        
        # Function detection
        if language == ProgrammingLanguage.PYTHON:
            func_match = re.search(r'def\s+(\w+)\s*\(([^)]*)\)', code)
            if func_match:
                func_name = func_match.group(1)
                params = func_match.group(2)
                return f"Python function '{func_name}' with parameters: {params}"
        
        elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
            func_match = re.search(r'function\s+(\w+)\s*\(([^)]*)\)', code)
            if func_match:
                func_name = func_match.group(1)
                params = func_match.group(2)
                return f"JavaScript function '{func_name}' with parameters: {params}"
        
        # Class detection
        class_match = re.search(r'class\s+(\w+)', code)
        if class_match:
            class_name = class_match.group(1)
            return f"{language.value.title()} class definition for '{class_name}'"
        
        # SQL operations
        if language == ProgrammingLanguage.SQL:
            if 'select' in code_lower:
                return "SQL SELECT query for data retrieval"
            elif 'insert' in code_lower:
                return "SQL INSERT statement for data insertion"
            elif 'create table' in code_lower:
                return "SQL table creation statement"
        
        # Generic summary based on content
        line_count = len(code.split('\n'))
        if context:
            return f"{language.value.title()} code block ({line_count} lines) - {context[:100]}..."
        else:
            return f"{language.value.title()} code block with {line_count} lines"
    
    def process_code_blocks(self, content: str, source_url: str = "") -> List[ExtractedCode]:
        """Process content and return extracted code ready for database storage."""
        code_blocks = self.extract_code_blocks(content)
        processed_codes = []
        
        for block in code_blocks:
            # Combine context for better understanding
            context = f"{block.context_before}\n\n{block.context_after}".strip()
            
            # Generate summary
            summary = self.generate_summary(block.content, block.language, context)
            
            # Calculate complexity
            complexity = self.calculate_complexity_score(block.content, block.language)
            
            processed_codes.append(ExtractedCode(
                code_content=block.content,
                summary=summary,
                programming_language=block.language.value,
                complexity_score=complexity,
                context=context
            ))
        
        return processed_codes


# Convenience functions for easy integration
def extract_code_from_content(content: str) -> List[ExtractedCode]:
    """Extract and process code blocks from content."""
    extractor = CodeExtractor()
    return extractor.process_code_blocks(content)


def get_supported_languages() -> List[str]:
    """Get list of supported programming languages."""
    return [lang.value for lang in ProgrammingLanguage if lang != ProgrammingLanguage.UNKNOWN]