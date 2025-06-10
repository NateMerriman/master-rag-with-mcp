"""
Tests for TASK A1: Template-Based Enhancement with Examples

This test suite validates the template-based few-shot learning approach
for generating consistent, high-quality code summaries.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import tempfile

# Import modules under test
try:
    from src.code_extraction import CodeExtractor, ProgrammingLanguage, ExtractedCode
    from src.config import StrategyConfig, get_config, reset_config, ConfigurationError
except ImportError:
    from code_extraction import CodeExtractor, ProgrammingLanguage, ExtractedCode
    from config import StrategyConfig, get_config, reset_config, ConfigurationError


class TestTemplateBasedEnhancement:
    """Test template-based code summary enhancement functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        reset_config()
        self.extractor = CodeExtractor()

    def teardown_method(self):
        """Clean up after each test."""
        reset_config()

    def test_language_examples_exist(self):
        """Test that language examples are properly defined for core languages."""
        # Test that examples exist for top 4 languages
        core_languages = [
            ProgrammingLanguage.PYTHON,
            ProgrammingLanguage.JAVASCRIPT,
            ProgrammingLanguage.TYPESCRIPT,
            ProgrammingLanguage.SQL,
            ProgrammingLanguage.JAVA,
        ]

        for language in core_languages:
            assert language in self.extractor.LANGUAGE_EXAMPLES
            example = self.extractor.LANGUAGE_EXAMPLES[language]

            # Validate example structure
            assert "example_code" in example
            assert "example_summary" in example
            assert isinstance(example["example_code"], str)
            assert isinstance(example["example_summary"], str)
            assert len(example["example_code"]) > 10
            assert len(example["example_summary"]) > 20

    def test_enhanced_summary_configuration_detection(self):
        """Test that enhanced summary configuration is properly detected."""
        # Test with enhanced summaries disabled (default)
        with patch.dict(
            os.environ,
            {"USE_ENHANCED_CODE_SUMMARIES": "false", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()
            config = get_config()
            assert not config.use_enhanced_code_summaries

        # Test with enhanced summaries enabled
        with patch.dict(
            os.environ,
            {"USE_ENHANCED_CODE_SUMMARIES": "true", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()
            config = get_config()
            assert config.use_enhanced_code_summaries

    def test_enhanced_summary_requires_agentic_rag(self):
        """Test that enhanced summaries require agentic RAG to be enabled."""
        with patch.dict(
            os.environ,
            {"USE_ENHANCED_CODE_SUMMARIES": "true", "USE_AGENTIC_RAG": "false"},
        ):
            reset_config()

            # Should raise ConfigurationError
            with pytest.raises(ConfigurationError) as exc_info:
                get_config()

            assert "USE_ENHANCED_CODE_SUMMARIES requires USE_AGENTIC_RAG" in str(
                exc_info.value
            )

    @patch("openai.chat.completions.create")
    @patch("src.code_extraction.OPENAI_AVAILABLE", True)
    def test_enhanced_summary_generation_with_examples(self, mock_openai):
        """Test enhanced summary generation with language-specific examples."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = "Enhanced test summary with detailed analysis."
        mock_openai.return_value = mock_response

        # Configure for enhanced summaries
        with patch.dict(
            os.environ,
            {
                "USE_ENHANCED_CODE_SUMMARIES": "true",
                "USE_AGENTIC_RAG": "true",
                "CONTEXTUAL_MODEL": "gpt-4o-mini-2024-07-18",
                "CODE_SUMMARY_MAX_CONTEXT_CHARS": "250",
            },
        ):
            reset_config()

            code = "def test_function():\n    return 'hello world'"
            language = ProgrammingLanguage.PYTHON
            context_before = "This is a simple test function for demonstration."
            context_after = "It returns a greeting message."

            summary = self.extractor.generate_enhanced_summary(
                code, language, context_before, context_after
            )

            assert summary == "Enhanced test summary with detailed analysis."
            assert mock_openai.called

            # Verify the prompt includes the Python example
            call_args = mock_openai.call_args
            prompt = call_args[1]["messages"][1]["content"]

            # Check that prompt contains Python example
            python_example = self.extractor.LANGUAGE_EXAMPLES[
                ProgrammingLanguage.PYTHON
            ]
            assert python_example["example_code"] in prompt
            assert python_example["example_summary"] in prompt

            # Check structured requirements
            assert "**Requirements:**" in prompt
            assert "Write exactly 2-3 sentences" in prompt
            assert "Use active voice" in prompt

    @patch("openai.chat.completions.create")
    @patch("src.code_extraction.OPENAI_AVAILABLE", True)
    def test_enhanced_summary_context_limiting(self, mock_openai):
        """Test that context is properly limited based on configuration."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test summary."
        mock_openai.return_value = mock_response

        long_context = "A" * 500  # 500 characters

        with patch.dict(
            os.environ,
            {
                "USE_ENHANCED_CODE_SUMMARIES": "true",
                "USE_AGENTIC_RAG": "true",
                "CODE_SUMMARY_MAX_CONTEXT_CHARS": "100",
            },
        ):
            reset_config()

            self.extractor.generate_enhanced_summary(
                "test code",
                ProgrammingLanguage.PYTHON,
                context_before=long_context,
                context_after=long_context,
            )

            # Verify context was limited
            call_args = mock_openai.call_args
            prompt = call_args[1]["messages"][1]["content"]

            # Context should be limited to 100 chars
            # Before context is taken from the end, so should be "A" * 100
            assert "A" * 100 in prompt
            # Should not contain the full 500 character string
            assert "A" * 400 not in prompt

    @patch("openai.chat.completions.create")
    @patch("src.code_extraction.OPENAI_AVAILABLE", True)
    def test_enhanced_summary_fallback_example(self, mock_openai):
        """Test fallback to Python example for unsupported languages."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test summary for unknown language."
        mock_openai.return_value = mock_response

        with patch.dict(
            os.environ,
            {"USE_ENHANCED_CODE_SUMMARIES": "true", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()

            # Use an unsupported language
            self.extractor.generate_enhanced_summary(
                "test code",
                ProgrammingLanguage.RUST,  # Not in our examples
                "",
                "",
            )

            # Should use Python example as fallback
            call_args = mock_openai.call_args
            prompt = call_args[1]["messages"][1]["content"]

            python_example = self.extractor.LANGUAGE_EXAMPLES[
                ProgrammingLanguage.PYTHON
            ]
            assert python_example["example_code"] in prompt

    @patch("openai.chat.completions.create")
    @patch("src.code_extraction.OPENAI_AVAILABLE", True)
    def test_enhanced_summary_error_fallback(self, mock_openai):
        """Test fallback to basic summary when enhanced generation fails."""
        # Mock OpenAI to raise an exception
        mock_openai.side_effect = Exception("API Error")

        with patch.dict(
            os.environ,
            {"USE_ENHANCED_CODE_SUMMARIES": "true", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()

            # Mock the basic generate_summary method to avoid infinite recursion
            with patch.object(
                self.extractor,
                "generate_summary",
                return_value="Basic fallback summary",
            ):
                summary = self.extractor.generate_enhanced_summary(
                    "test code", ProgrammingLanguage.PYTHON, "", ""
                )

                assert summary == "Basic fallback summary"

    @patch("openai.chat.completions.create")
    @patch("src.code_extraction.OPENAI_AVAILABLE", True)
    def test_generate_summary_routes_to_enhanced(self, mock_openai):
        """Test that generate_summary routes to enhanced version when configured."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Enhanced summary via routing."
        mock_openai.return_value = mock_response

        with patch.dict(
            os.environ,
            {"USE_ENHANCED_CODE_SUMMARIES": "true", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()

            summary = self.extractor.generate_summary(
                "test code", ProgrammingLanguage.PYTHON, "", ""
            )

            assert summary == "Enhanced summary via routing."

            # Verify enhanced prompt was used (contains example)
            call_args = mock_openai.call_args
            prompt = call_args[1]["messages"][1]["content"]
            assert "**Example:**" in prompt

    @patch("openai.chat.completions.create")
    @patch("src.code_extraction.OPENAI_AVAILABLE", True)
    def test_generate_summary_uses_basic_when_disabled(self, mock_openai):
        """Test that generate_summary uses basic version when enhanced is disabled."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Basic summary."
        mock_openai.return_value = mock_response

        with patch.dict(
            os.environ,
            {"USE_ENHANCED_CODE_SUMMARIES": "false", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()

            summary = self.extractor.generate_summary(
                "test code", ProgrammingLanguage.PYTHON, "", ""
            )

            assert summary == "Basic summary."

            # Verify basic prompt was used (no examples)
            call_args = mock_openai.call_args
            prompt = call_args[1]["messages"][0][
                "content"
            ]  # Basic uses single user message
            assert "**Example:**" not in prompt

    def test_enhanced_summary_without_openai(self):
        """Test enhanced summary falls back to rule-based when OpenAI unavailable."""
        with patch("src.code_extraction.OPENAI_AVAILABLE", False):
            summary = self.extractor.generate_enhanced_summary(
                "def test(): pass", ProgrammingLanguage.PYTHON, "", ""
            )

            # Should fall back to rule-based summary
            assert "Python function 'test'" in summary

    def test_configuration_validation_summary_style(self):
        """Test validation of code summary style configuration."""
        # Test valid styles
        for style in ["practical", "academic", "tutorial"]:
            with patch.dict(
                os.environ, {"CODE_SUMMARY_STYLE": style, "USE_AGENTIC_RAG": "true"}
            ):
                reset_config()
                config = get_config()
                errors = config.validate()
                style_errors = [e for e in errors if "CODE_SUMMARY_STYLE" in e]
                assert len(style_errors) == 0

        # Test invalid style - should raise ConfigurationError
        with patch.dict(
            os.environ,
            {"CODE_SUMMARY_STYLE": "invalid_style", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()
            with pytest.raises(ConfigurationError) as exc_info:
                get_config()
            assert "CODE_SUMMARY_STYLE must be one of" in str(exc_info.value)

    def test_configuration_validation_context_chars(self):
        """Test validation of context character limits."""
        # Test valid range
        with patch.dict(
            os.environ,
            {"CODE_SUMMARY_MAX_CONTEXT_CHARS": "300", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()
            config = get_config()
            errors = config.validate()
            context_errors = [e for e in errors if "CONTEXT_CHARS" in e]
            assert len(context_errors) == 0

        # Test too small - should raise ConfigurationError
        with patch.dict(
            os.environ,
            {"CODE_SUMMARY_MAX_CONTEXT_CHARS": "0", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()
            with pytest.raises(ConfigurationError) as exc_info:
                get_config()
            assert "CODE_SUMMARY_MAX_CONTEXT_CHARS must be positive" in str(
                exc_info.value
            )

        # Test too large - should raise ConfigurationError
        with patch.dict(
            os.environ,
            {"CODE_SUMMARY_MAX_CONTEXT_CHARS": "2000", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()
            with pytest.raises(ConfigurationError) as exc_info:
                get_config()
            assert "should not exceed 1000" in str(exc_info.value)

    @patch("openai.chat.completions.create")
    @patch("src.code_extraction.OPENAI_AVAILABLE", True)
    def test_enhanced_summary_temperature_and_tokens(self, mock_openai):
        """Test that enhanced summaries use appropriate temperature and token limits."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test summary."
        mock_openai.return_value = mock_response

        with patch.dict(
            os.environ,
            {"USE_ENHANCED_CODE_SUMMARIES": "true", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()

            self.extractor.generate_enhanced_summary(
                "test code", ProgrammingLanguage.PYTHON, "", ""
            )

            # Verify OpenAI call parameters
            call_args = mock_openai.call_args
            assert call_args[1]["max_tokens"] == 200  # Enhanced uses 200 vs basic 150
            assert (
                call_args[1]["temperature"] == 0.2
            )  # Enhanced uses lower temperature for consistency

    def test_language_example_quality_python(self):
        """Test that Python language example demonstrates good practices."""
        python_example = self.extractor.LANGUAGE_EXAMPLES[ProgrammingLanguage.PYTHON]

        code = python_example["example_code"]
        summary = python_example["example_summary"]

        # Code should demonstrate authentication patterns
        assert "authenticate" in code.lower()
        assert "bcrypt" in code.lower()
        assert "password" in code.lower()

        # Summary should be 2-3 sentences and action-focused
        sentences = summary.split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        assert 2 <= len(sentences) <= 3

        # Should start with action verb
        first_word = summary.split()[0].lower()
        assert first_word in [
            "user",
            "authentication",
            "validates",
            "checks",
            "compares",
        ]

    def test_language_example_quality_javascript(self):
        """Test that JavaScript language example demonstrates modern patterns."""
        js_example = self.extractor.LANGUAGE_EXAMPLES[ProgrammingLanguage.JAVASCRIPT]

        code = js_example["example_code"]
        summary = js_example["example_summary"]

        # Code should demonstrate async/await patterns
        assert "async" in code
        assert "await" in code
        assert "fetch" in code
        assert "try" in code and "catch" in code

        # Summary should mention async patterns
        assert "async" in summary.lower() or "asynchronous" in summary.lower()

    def test_integration_with_existing_metadata(self):
        """Test that enhanced summaries work with existing metadata generation."""
        # This tests the integration point where enhanced summaries would be
        # called from the existing code extraction pipeline

        code = """def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"""

        with patch.dict(
            os.environ,
            {"USE_ENHANCED_CODE_SUMMARIES": "true", "USE_AGENTIC_RAG": "true"},
        ):
            reset_config()

            # Test that enhanced summary can be generated alongside metadata
            complexity = self.extractor.calculate_complexity_score(
                code, ProgrammingLanguage.PYTHON
            )

            with patch("openai.chat.completions.create") as mock_openai:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[
                    0
                ].message.content = (
                    "Fibonacci calculation function with recursive implementation."
                )
                mock_openai.return_value = mock_response

                summary = self.extractor.generate_summary(
                    code, ProgrammingLanguage.PYTHON
                )

                # Both should work together
                assert complexity > 1  # Recursive function should have complexity
                assert "fibonacci" in summary.lower()
                assert mock_openai.called


if __name__ == "__main__":
    pytest.main([__file__])
