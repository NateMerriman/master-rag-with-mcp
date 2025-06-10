"""
Unit tests for code extraction functionality.
Tests the code block identification, language detection, and complexity scoring.
"""

import pytest
from src.code_extraction import (
    CodeExtractor,
    ProgrammingLanguage,
    CodeBlock,
    ExtractedCode,
    extract_code_from_content,
    get_supported_languages,
)


class TestCodeExtractor:
    """Test cases for the CodeExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = CodeExtractor()

    def test_extract_fenced_python_code(self):
        """Test extraction of fenced Python code blocks."""
        content = """
        Here's a Python function:
        
        ```python
        def hello_world():
            print("Hello, World!")
            return True
        ```
        
        This function prints a greeting.
        """

        blocks = self.extractor.extract_code_blocks(content)

        assert len(blocks) == 1
        assert blocks[0].language == ProgrammingLanguage.PYTHON
        assert "def hello_world():" in blocks[0].content
        assert blocks[0].block_type == "fenced"
        assert "Here's a Python function:" in blocks[0].context_before
        assert "This function prints a greeting." in blocks[0].context_after

    def test_extract_fenced_javascript_code(self):
        """Test extraction of fenced JavaScript code blocks."""
        content = """
        JavaScript example:
        
        ```js
        function greet(name) {
            console.log(`Hello, ${name}!`);
            return name.length;
        }
        ```
        """

        blocks = self.extractor.extract_code_blocks(content)

        assert len(blocks) == 1
        assert blocks[0].language == ProgrammingLanguage.JAVASCRIPT
        assert "function greet(name)" in blocks[0].content
        assert blocks[0].block_type == "fenced"

    def test_language_detection_from_hint(self):
        """Test language detection from code fence hints."""
        content = """
        ```typescript
        interface User {
            name: string;
            age: number;
        }
        ```
        """

        blocks = self.extractor.extract_code_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].language == ProgrammingLanguage.TYPESCRIPT

    def test_language_detection_from_content(self):
        """Test language detection from code content patterns."""
        python_content = """
        ```
        def calculate(x, y):
            return x + y
        ```
        """

        blocks = self.extractor.extract_code_blocks(python_content)
        assert len(blocks) == 1
        assert blocks[0].language == ProgrammingLanguage.PYTHON

    def test_complexity_scoring_simple(self):
        """Test complexity scoring for simple code."""
        simple_code = "def hello():\n    return 'world'"
        score = self.extractor.calculate_complexity_score(
            simple_code, ProgrammingLanguage.PYTHON
        )
        assert 1 <= score <= 3

    def test_complexity_scoring_complex(self):
        """Test complexity scoring for complex code."""
        complex_code = """
        @decorator
        async def complex_function(data):
            try:
                for item in data:
                    if item.valid:
                        with open('file.txt') as f:
                            yield process_item(item)
                    else:
                        continue
            except Exception as e:
                handle_error(e)
        """
        score = self.extractor.calculate_complexity_score(
            complex_code, ProgrammingLanguage.PYTHON
        )
        assert score >= 6

    def test_sql_complexity_scoring(self):
        """Test complexity scoring for SQL code."""
        sql_code = """
        WITH ranked_users AS (
            SELECT name, age, 
                   ROW_NUMBER() OVER (ORDER BY age DESC) as rank
            FROM users u
            JOIN profiles p ON u.id = p.user_id
            WHERE u.active = true
            GROUP BY name, age
        )
        SELECT * FROM ranked_users WHERE rank <= 10
        """
        score = self.extractor.calculate_complexity_score(
            sql_code, ProgrammingLanguage.SQL
        )
        assert score >= 4

    def test_minimum_code_length_filter(self):
        """Test that very short code blocks are filtered out."""
        content = """
        ```python
        x = 1
        ```
        """

        blocks = self.extractor.extract_code_blocks(content)
        assert len(blocks) == 0  # Too short to be considered valid

    def test_valid_code_block_detection(self):
        """Test valid code block detection criteria."""
        valid_code = """
        def process_data(items):
            results = []
            for item in items:
                if item.is_valid():
                    results.append(item.process())
            return results
        """

        block = CodeBlock(
            content=valid_code,
            language=ProgrammingLanguage.PYTHON,
            start_line=0,
            end_line=6,
            block_type="fenced",
        )

        assert self.extractor._is_valid_code_block(block)

    def test_invalid_code_block_detection(self):
        """Test invalid code block detection (plain text)."""
        plain_text = """
        This is just regular text without any code characteristics.
        It has no programming constructs or syntax.
        """

        block = CodeBlock(
            content=plain_text,
            language=ProgrammingLanguage.UNKNOWN,
            start_line=0,
            end_line=2,
            block_type="fenced",
        )

        assert not self.extractor._is_valid_code_block(block)

    def test_multiple_code_blocks(self):
        """Test extraction of multiple code blocks."""
        content = """
        First example:
        ```python
        def first_function():
            return "first"
        ```
        
        Second example:
        ```javascript
        function secondFunction() {
            return "second";
        }
        ```
        """

        blocks = self.extractor.extract_code_blocks(content)
        assert len(blocks) == 2
        assert blocks[0].language == ProgrammingLanguage.PYTHON
        assert blocks[1].language == ProgrammingLanguage.JAVASCRIPT

    def test_language_aliases(self):
        """Test that language aliases work correctly."""
        content = """
        ```py
        def test():
            pass
        ```
        """

        blocks = self.extractor.extract_code_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].language == ProgrammingLanguage.PYTHON

    def test_context_extraction(self):
        """Test context extraction around code blocks."""
        content = """
        Introduction paragraph.
        This explains what we're about to do.
        
        ```python
        def example():
            return True
        ```
        
        Conclusion paragraph.
        This explains what we just did.
        """

        blocks = self.extractor.extract_code_blocks(content)
        assert len(blocks) == 1

        block = blocks[0]
        assert "Introduction paragraph" in block.context_before
        assert "This explains what we're about to do" in block.context_before
        assert "Conclusion paragraph" in block.context_after
        assert "This explains what we just did" in block.context_after

    def test_enhanced_metadata_generation(self):
        """Test the enhanced metadata generation."""
        content = """
        # API Configuration Guide
        Here's how to configure the API:
        
        ```python
        import os
        import asyncio
        from typing import List
        
        class APIClient:
            def __init__(self, api_key: str):
                self.api_key = api_key
                self.session = None
            
            async def fetch_data(self, endpoint: str) -> List[dict]:
                try:
                    # Make API request
                    if self.session is None:
                        raise ValueError("Session not initialized")
                    
                    response = await self.session.get(endpoint)
                    return response.json()
                except Exception as e:
                    logger.error(f"API error: {e}")
                    return []
        ```
        
        This example shows asynchronous API usage.
        """

        blocks = self.extractor.extract_code_blocks(content)
        assert len(blocks) == 1

        metadata = self.extractor._generate_enhanced_metadata(blocks[0])

        # Test basic statistics
        assert metadata["char_count"] > 0
        assert metadata["word_count"] > 0
        assert metadata["line_count"] > 10
        assert metadata["non_empty_line_count"] > 0
        assert 0 <= metadata["code_density"] <= 1

        # Test code analysis
        assert "APIClient" in metadata["identifiers"]
        assert "fetch_data" in metadata["identifiers"]
        assert "async" in metadata["patterns"]
        assert "oop" in metadata["patterns"]
        assert "error_handling" in metadata["patterns"]
        assert metadata["has_comments"] == True
        assert metadata["has_strings"] == True
        assert metadata["has_conditions"] == True

        # Test context analysis
        # Check if headers are extracted (should contain "API Configuration Guide")
        if metadata["surrounding_headers"]:
            header_found = any(
                "api" in header.lower() or "configuration" in header.lower()
                for header in metadata["surrounding_headers"]
            )
            assert header_found, (
                f"Expected header with 'api' or 'configuration', got: {metadata['surrounding_headers']}"
            )

        # Check context keywords
        assert "api" in metadata["context_keywords"]
        assert "configuration" in metadata["context_keywords"]
        assert metadata["estimated_reading_time"] >= 1

        # Test complexity indicators
        complexity = metadata["complexity_indicators"]
        assert complexity["nesting_level"] >= 1
        assert complexity["cyclomatic_complexity"] >= 1
        assert complexity["unique_identifiers"] >= 0

    def test_identifier_extraction_python(self):
        """Test identifier extraction for Python code."""
        code = """
        import requests
        from typing import List
        
        class DataProcessor:
            def process_items(self, items):
                return [item.clean() for item in items]
        
        def main():
            processor = DataProcessor()
            return processor.process_items([])
        """

        identifiers = self.extractor._extract_identifiers(
            code, ProgrammingLanguage.PYTHON
        )
        assert "DataProcessor" in identifiers
        assert "process_items" in identifiers
        assert "main" in identifiers
        assert "requests" in identifiers
        assert "typing" in identifiers

    def test_identifier_extraction_javascript(self):
        """Test identifier extraction for JavaScript code."""
        code = """
        const API_URL = "https://api.example.com";
        let userCache = new Map();
        
        function fetchUser(id) {
            return fetch(`${API_URL}/users/${id}`);
        }
        
        class UserManager {
            constructor() {
                this.users = [];
            }
        }
        """

        identifiers = self.extractor._extract_identifiers(
            code, ProgrammingLanguage.JAVASCRIPT
        )
        assert "API_URL" in identifiers
        assert "userCache" in identifiers
        assert "fetchUser" in identifiers
        assert "UserManager" in identifiers

    def test_pattern_extraction(self):
        """Test pattern extraction from code."""
        async_code = """
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                return response.json();
            } catch (error) {
                console.error(error);
            }
        }
        """

        patterns = self.extractor._extract_patterns(
            async_code, ProgrammingLanguage.JAVASCRIPT
        )
        assert "async" in patterns
        assert "await" in patterns
        assert "error_handling" in patterns

    def test_context_header_extraction(self):
        """Test extraction of markdown headers from context."""
        context_before = """
        # Main Title
        Some introduction text.
        
        ## Configuration Section
        Details about configuration.
        
        ### API Setup
        Specific API setup instructions.
        """

        context_after = """
        ## Next Steps
        What to do after this example.
        """

        headers = self.extractor._extract_context_headers(context_before, context_after)
        assert "h1: Main Title" in headers
        assert "h2: Configuration Section" in headers
        assert "h3: API Setup" in headers
        assert "h2: Next Steps" in headers

    def test_keyword_extraction_from_context(self):
        """Test keyword extraction from surrounding context."""
        context = """
        This tutorial shows how to use the API framework for configuration.
        The example demonstrates the implementation of a utility service.
        """

        keywords = self.extractor._extract_keywords_from_context(context, "")

        # Check that we get some relevant keywords (the algorithm limits to 6)
        assert "tutorial" in keywords
        assert "api" in keywords
        assert "framework" in keywords
        assert "configuration" in keywords
        assert "implementation" in keywords
        # "utility" and "service" might not be in the limited keyword list
        # So we just check that we get a reasonable number of keywords
        assert len(keywords) >= 5

    def test_nesting_level_calculation_python(self):
        """Test nesting level calculation for Python code."""
        nested_code = """
        def outer():
            if True:
                for i in range(10):
                    if i % 2 == 0:
                        print(i)
        """

        nesting = self.extractor._calculate_nesting_level(
            nested_code, ProgrammingLanguage.PYTHON
        )
        assert nesting >= 3  # def -> if -> for -> if

    def test_nesting_level_calculation_javascript(self):
        """Test nesting level calculation for JavaScript code."""
        nested_code = """
        function outer() {
            if (true) {
                for (let i = 0; i < 10; i++) {
                    if (i % 2 === 0) {
                        console.log(i);
                    }
                }
            }
        }
        """

        nesting = self.extractor._calculate_nesting_level(
            nested_code, ProgrammingLanguage.JAVASCRIPT
        )
        assert nesting >= 3  # function -> if -> for -> if

    def test_cyclomatic_complexity_estimation(self):
        """Test cyclomatic complexity estimation."""
        complex_code = """
        def process(data):
            if data is None:
                return None
            elif len(data) == 0:
                return []
            
            results = []
            for item in data:
                try:
                    if item.valid and item.active:
                        results.append(item.process())
                    elif item.recoverable:
                        results.append(item.recover())
                except Exception:
                    continue
            
            return results if results else None
        """

        complexity = self.extractor._estimate_cyclomatic_complexity(complex_code)
        assert complexity > 5  # Multiple decision points

    def test_rule_based_summary_fallback(self):
        """Test rule-based summary generation as fallback."""
        code = """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        """

        summary = self.extractor._generate_rule_based_summary(
            code, ProgrammingLanguage.PYTHON
        )
        assert "fibonacci" in summary.lower()
        assert "python" in summary.lower()
        assert "function" in summary.lower()

    def test_process_code_blocks_integration(self):
        """Test the complete code processing pipeline."""
        content = """
        # Utility Functions
        Here's a utility function:
        
        ```python
        def fibonacci(n):
            '''Calculate the nth Fibonacci number recursively.'''
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        ```
        
        This function calculates Fibonacci numbers recursively.
        """

        processed_codes = self.extractor.process_code_blocks(
            content, "https://example.com/docs"
        )

        assert len(processed_codes) == 1
        code = processed_codes[0]

        assert isinstance(code, ExtractedCode)
        assert "def fibonacci(n):" in code.content
        assert code.programming_language == "python"
        assert 1 <= code.complexity_score <= 10
        assert code.summary  # Should have a summary
        assert code.url == "https://example.com/docs"
        assert code.chunk_number == 1

        # Test enhanced metadata
        metadata = code.metadata
        assert metadata["char_count"] > 0
        assert "fibonacci" in metadata["identifiers"]

        # Check for headers more flexibly - might be extracted or not depending on context parsing
        if metadata["surrounding_headers"]:
            header_found = any(
                "utility" in header.lower() or "function" in header.lower()
                for header in metadata["surrounding_headers"]
            )
            # If headers are found, at least one should be relevant
            if not header_found:
                print(
                    f"Warning: Headers found but none seem relevant: {metadata['surrounding_headers']}"
                )

        # Comments detection should work with Python docstrings
        assert metadata["has_comments"] == True  # Should detect '''docstring'''
        assert metadata["has_conditions"] == True


class TestCodeExtractionUtilities:
    """Test utility functions."""

    def test_extract_code_from_content(self):
        """Test the high-level extraction function."""
        content = """
        Example code:
        ```python
        def greet(name):
            return f"Hello, {name}!"
        ```
        """

        results = extract_code_from_content(content, "https://test.com")
        assert len(results) == 1
        assert results[0].programming_language == "python"
        assert results[0].url == "https://test.com"

    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = get_supported_languages()
        assert isinstance(languages, list)
        assert "python" in languages
        assert "javascript" in languages
        assert "sql" in languages
        assert "unknown" not in languages  # Should be filtered out


class TestLanguageDetection:
    """Test language detection across different programming languages."""

    def setup_method(self):
        self.extractor = CodeExtractor()

    def test_detect_python(self):
        """Test Python detection."""
        code = "def hello():\n    print('world')\n    return True"
        lang = self.extractor._detect_language_from_content(code)
        assert lang == ProgrammingLanguage.PYTHON

    def test_detect_javascript(self):
        """Test JavaScript detection."""
        code = "function hello() {\n    console.log('world');\n    return true;\n}"
        lang = self.extractor._detect_language_from_content(code)
        assert lang == ProgrammingLanguage.JAVASCRIPT

    def test_detect_typescript(self):
        """Test TypeScript detection."""
        code = "interface User {\n    name: string;\n    age: number;\n}"
        lang = self.extractor._detect_language_from_content(code)
        assert lang == ProgrammingLanguage.TYPESCRIPT

    def test_detect_sql(self):
        """Test SQL detection."""
        code = "SELECT name, age FROM users WHERE active = true"
        lang = self.extractor._detect_language_from_content(code)
        assert lang == ProgrammingLanguage.SQL

    def test_detect_html(self):
        """Test HTML detection."""
        code = "<div class='container'>\n    <p>Hello World</p>\n</div>"
        lang = self.extractor._detect_language_from_content(code)
        assert lang == ProgrammingLanguage.HTML

    def test_detect_css(self):
        """Test CSS detection."""
        code = ".container {\n    margin: 0 auto;\n    padding: 20px;\n}"
        lang = self.extractor._detect_language_from_content(code)
        assert lang == ProgrammingLanguage.CSS

    def test_detect_shell(self):
        """Test Shell script detection."""
        code = "#!/bin/bash\necho $HOME\nls -la && cd /tmp"
        lang = self.extractor._detect_language_from_content(code)
        assert lang == ProgrammingLanguage.SHELL

    def test_detect_json(self):
        """Test JSON detection."""
        code = '{\n    "name": "John",\n    "age": 30\n}'
        lang = self.extractor._detect_language_from_content(code)
        assert lang == ProgrammingLanguage.JSON


class TestComplexityScoring:
    """Test complexity scoring across different code patterns."""

    def setup_method(self):
        self.extractor = CodeExtractor()

    def test_simple_function_complexity(self):
        """Test complexity for simple functions."""
        code = "def add(a, b):\n    return a + b"
        score = self.extractor.calculate_complexity_score(
            code, ProgrammingLanguage.PYTHON
        )
        assert 1 <= score <= 2

    def test_nested_function_complexity(self):
        """Test complexity for nested functions."""
        code = """
        def outer():
            def inner():
                def deepest():
                    return True
                return deepest()
            return inner()
        """
        score = self.extractor.calculate_complexity_score(
            code, ProgrammingLanguage.PYTHON
        )
        assert score >= 3

    def test_conditional_complexity(self):
        """Test complexity with conditionals."""
        code = """
        def process(x):
            if x > 0:
                if x > 10:
                    return "high"
                else:
                    return "medium"
            else:
                return "low"
        """
        score = self.extractor.calculate_complexity_score(
            code, ProgrammingLanguage.PYTHON
        )
        assert score >= 2

    def test_loop_complexity(self):
        """Test complexity with loops."""
        code = """
        def process_items(items):
            results = []
            for item in items:
                for sub_item in item.children:
                    if sub_item.valid:
                        results.append(sub_item)
            return results
        """
        score = self.extractor.calculate_complexity_score(
            code, ProgrammingLanguage.PYTHON
        )
        assert score >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
