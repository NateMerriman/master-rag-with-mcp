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
    get_supported_languages
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
        assert blocks[0].block_type == 'fenced'
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
        assert blocks[0].block_type == 'fenced'
    
    def test_extract_multiple_code_blocks(self):
        """Test extraction of multiple code blocks."""
        content = """
        First, here's Python:
        
        ```python
        def add(a, b):
            return a + b
        ```
        
        Now some JavaScript:
        
        ```javascript
        const multiply = (a, b) => a * b;
        ```
        
        End of examples.
        """
        
        blocks = self.extractor.extract_code_blocks(content)
        
        assert len(blocks) == 2
        assert blocks[0].language == ProgrammingLanguage.PYTHON
        assert blocks[1].language == ProgrammingLanguage.JAVASCRIPT
        assert "def add(a, b):" in blocks[0].content
        assert "const multiply = (a, b)" in blocks[1].content
    
    def test_extract_indented_code_blocks(self):
        """Test extraction of indented code blocks."""
        content = """
        Here's some code:
        
            def simple_function():
                x = 1
                y = 2
                return x + y
        
        That was a simple example.
        """
        
        blocks = self.extractor.extract_code_blocks(content)
        
        assert len(blocks) == 1
        assert blocks[0].language == ProgrammingLanguage.PYTHON
        assert blocks[0].block_type == 'indented'
        assert "def simple_function():" in blocks[0].content
        # Check that indentation was removed
        lines = blocks[0].content.split('\n')
        assert not lines[0].startswith('    ')
    
    def test_language_detection_from_content(self):
        """Test automatic language detection from code content."""
        test_cases = [
            ("def test(): pass", ProgrammingLanguage.PYTHON),
            ("function test() { return true; }", ProgrammingLanguage.JAVASCRIPT),
            ("console.log('hello');", ProgrammingLanguage.JAVASCRIPT),
            ("public class Test { }", ProgrammingLanguage.JAVA),
            ("SELECT * FROM users", ProgrammingLanguage.SQL),
            ("#include <iostream>", ProgrammingLanguage.C),
            ("std::cout << 'hello';", ProgrammingLanguage.CPP),
            ("<div>Hello</div>", ProgrammingLanguage.HTML),
            ("body { color: red; }", ProgrammingLanguage.CSS),
            ("#!/bin/bash\necho 'hello'", ProgrammingLanguage.SHELL),
        ]
        
        for code, expected_lang in test_cases:
            detected = self.extractor._detect_language_from_content(code)
            assert detected == expected_lang, f"Failed for code: {code}"
    
    def test_language_aliases(self):
        """Test language alias mapping."""
        test_cases = [
            ("py", ProgrammingLanguage.PYTHON),
            ("js", ProgrammingLanguage.JAVASCRIPT),
            ("ts", ProgrammingLanguage.TYPESCRIPT),
            ("c++", ProgrammingLanguage.CPP),
            ("sh", ProgrammingLanguage.SHELL),
            ("bash", ProgrammingLanguage.SHELL),
        ]
        
        for alias, expected_lang in test_cases:
            detected = self.extractor._detect_language(alias, "")
            assert detected == expected_lang, f"Failed for alias: {alias}"
    
    def test_complexity_scoring(self):
        """Test code complexity scoring algorithm."""
        # Simple code (score should be low)
        simple_code = "x = 1\nprint(x)"
        simple_score = self.extractor.calculate_complexity_score(simple_code, ProgrammingLanguage.PYTHON)
        assert 1 <= simple_score <= 3
        
        # Complex code (score should be higher)
        complex_code = """
        class ComplexClass:
            def __init__(self):
                self.data = []
            
            def process_data(self, items):
                try:
                    for item in items:
                        if item.is_valid():
                            while item.has_children():
                                for child in item.children:
                                    if child.process():
                                        self.data.append(child)
                                    elif child.retry_count < 3:
                                        child.retry()
                                    else:
                                        child.mark_failed()
                except Exception as e:
                    self.handle_error(e)
                    return False
                return True
        """
        complex_score = self.extractor.calculate_complexity_score(complex_code, ProgrammingLanguage.PYTHON)
        assert complex_score > simple_score
        assert 1 <= complex_score <= 10
    
    def test_summary_generation(self):
        """Test automatic summary generation."""
        # Python function
        python_code = """
        def calculate_area(radius):
            return 3.14159 * radius * radius
        """
        summary = self.extractor.generate_summary(python_code, ProgrammingLanguage.PYTHON)
        assert "function" in summary.lower()
        assert "calculate_area" in summary
        
        # JavaScript function
        js_code = """
        function validateEmail(email) {
            const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return regex.test(email);
        }
        """
        summary = self.extractor.generate_summary(js_code, ProgrammingLanguage.JAVASCRIPT)
        assert "function" in summary.lower()
        assert "validateEmail" in summary
        
        # Class definition
        class_code = """
        class UserManager:
            def __init__(self):
                self.users = {}
        """
        summary = self.extractor.generate_summary(class_code, ProgrammingLanguage.PYTHON)
        assert "class" in summary.lower()
        assert "UserManager" in summary
    
    def test_filter_invalid_code_blocks(self):
        """Test filtering of invalid code blocks."""
        # Too short
        short_content = "```python\nx = 1\n```"
        blocks = self.extractor.extract_code_blocks(short_content)
        assert len(blocks) == 0  # Should be filtered out
        
        # Mostly comments
        comment_heavy = """
        ```python
        # This is a comment
        # Another comment
        # Yet another comment
        # Final comment
        x = 1  # Only one line of actual code
        ```
        """
        blocks = self.extractor.extract_code_blocks(comment_heavy)
        assert len(blocks) == 0  # Should be filtered out
    
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
    
    def test_process_code_blocks_integration(self):
        """Test the complete code processing pipeline."""
        content = """
        Here's a utility function:
        
        ```python
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        ```
        
        This function calculates Fibonacci numbers recursively.
        """
        
        processed_codes = self.extractor.process_code_blocks(content)
        
        assert len(processed_codes) == 1
        code = processed_codes[0]
        
        assert isinstance(code, ExtractedCode)
        assert "def fibonacci(n):" in code.code_content
        assert code.programming_language == "python"
        assert 1 <= code.complexity_score <= 10
        assert code.summary  # Should have a summary
        assert code.context  # Should have context


class TestConvenienceFunctions:
    """Test convenience functions for easy integration."""
    
    def test_extract_code_from_content(self):
        """Test the main convenience function."""
        content = """
        ```python
        def test_function():
            return "Hello, World!"
        ```
        """
        
        codes = extract_code_from_content(content)
        
        assert len(codes) == 1
        assert isinstance(codes[0], ExtractedCode)
        assert codes[0].programming_language == "python"
    
    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        languages = get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "python" in languages
        assert "javascript" in languages
        assert "unknown" not in languages  # Should exclude UNKNOWN


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = CodeExtractor()
    
    def test_empty_content(self):
        """Test processing empty content."""
        blocks = self.extractor.extract_code_blocks("")
        assert len(blocks) == 0
        
        processed = self.extractor.process_code_blocks("")
        assert len(processed) == 0
    
    def test_no_code_blocks(self):
        """Test content with no code blocks."""
        content = "This is just regular text with no code examples."
        
        blocks = self.extractor.extract_code_blocks(content)
        assert len(blocks) == 0
        
        processed = self.extractor.process_code_blocks(content)
        assert len(processed) == 0
    
    def test_malformed_fenced_blocks(self):
        """Test handling of malformed fenced code blocks."""
        content = """
        ```python
        def incomplete_function(
        # Missing closing backticks
        """
        
        blocks = self.extractor.extract_code_blocks(content)
        assert len(blocks) == 0  # Should not extract malformed blocks
    
    def test_nested_code_blocks(self):
        """Test handling of nested or overlapping code patterns."""
        content = """
        ```markdown
        Here's how to write Python:
        
        ```python
        def hello():
            print("Hello!")
        ```
        
        That was a Python example.
        ```
        """
        
        blocks = self.extractor.extract_code_blocks(content)
        # Should handle this gracefully, exact behavior may vary
        assert isinstance(blocks, list)
    
    def test_very_long_code_block(self):
        """Test handling of very long code blocks."""
        # Create a very long code block
        long_code = "def long_function():\n" + "    x = 1\n" * 1000
        content = f"```python\n{long_code}\n```"
        
        blocks = self.extractor.extract_code_blocks(content)
        # Should be filtered out due to length
        assert len(blocks) == 0
    
    def test_complexity_score_bounds(self):
        """Test that complexity scores are always within bounds."""
        test_codes = [
            "x = 1",  # Very simple
            "def f(): return 1",  # Simple function
            "class A:\n    def __init__(self): pass",  # Simple class
        ]
        
        for code in test_codes:
            score = self.extractor.calculate_complexity_score(code, ProgrammingLanguage.PYTHON)
            assert 1 <= score <= 10, f"Score {score} out of bounds for code: {code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])