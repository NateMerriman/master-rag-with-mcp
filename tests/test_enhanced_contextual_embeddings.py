"""
Tests for enhanced contextual embeddings with adaptive prompting.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    _detect_content_type,
    _get_contextual_prompt_and_system_message,
    generate_contextual_embedding,
)
from config import reset_config


class TestContentTypeDetection:
    """Test content type detection functionality."""

    def test_technical_content_detection_by_url(self):
        """Test detection of technical content by URL patterns."""
        content = "This is some content about configuration."

        # GitHub documentation
        assert (
            _detect_content_type(content, "https://github.com/user/repo") == "technical"
        )

        # Documentation sites
        assert (
            _detect_content_type(content, "https://docs.example.com/api") == "technical"
        )
        assert (
            _detect_content_type(content, "https://api.example.com/reference")
            == "technical"
        )
        assert (
            _detect_content_type(content, "https://developer.example.com/guide")
            == "technical"
        )

    def test_forum_content_detection_by_url(self):
        """Test detection of forum content by URL patterns."""
        content = "This is a discussion about programming."

        assert (
            _detect_content_type(content, "https://reddit.com/r/programming") == "forum"
        )
        assert (
            _detect_content_type(content, "https://stackoverflow.com/questions/123")
            == "forum"
        )
        assert (
            _detect_content_type(content, "https://discourse.example.com/topic")
            == "forum"
        )
        assert (
            _detect_content_type(content, "https://forum.example.com/thread") == "forum"
        )

    def test_academic_content_detection_by_url(self):
        """Test detection of academic content by URL patterns."""
        content = "This paper presents a novel approach to machine learning."

        assert (
            _detect_content_type(content, "https://arxiv.org/abs/2301.12345")
            == "academic"
        )
        assert (
            _detect_content_type(content, "https://scholar.google.com/paper")
            == "academic"
        )
        assert (
            _detect_content_type(content, "https://pubmed.ncbi.nlm.nih.gov/12345")
            == "academic"
        )
        assert (
            _detect_content_type(content, "https://jstor.org/article/123") == "academic"
        )

    def test_news_content_detection_by_url(self):
        """Test detection of news content by URL patterns."""
        content = "Breaking news about technology trends."

        assert (
            _detect_content_type(content, "https://news.example.com/article") == "news"
        )
        assert _detect_content_type(content, "https://cnn.com/2024/tech-news") == "news"
        assert (
            _detect_content_type(content, "https://bbc.com/news/technology") == "news"
        )
        assert _detect_content_type(content, "https://reuters.com/technology") == "news"

    def test_blog_content_detection_by_url(self):
        """Test detection of blog content by URL patterns."""
        content = "I think this is a great approach to solving problems."

        assert (
            _detect_content_type(content, "https://medium.com/@author/post") == "blog"
        )
        assert (
            _detect_content_type(content, "https://author.substack.com/post") == "blog"
        )
        assert _detect_content_type(content, "https://blog.example.com/post") == "blog"

    def test_technical_content_detection_by_content(self):
        """Test detection of technical content by content analysis."""
        technical_content = """
        To install the package, run the following command:
        ```bash
        npm install example-package
        ```
        This function takes parameters and returns a response.
        The API endpoint accepts requests and sends responses.
        """

        assert _detect_content_type(technical_content) == "technical"

    def test_academic_content_detection_by_content(self):
        """Test detection of academic content by content analysis."""
        academic_content = """
        Abstract: This study examines the methodology for analyzing data.
        Our research findings demonstrate that the hypothesis is supported.
        The experiment was conducted using a controlled methodology.
        Discussion of results shows significant correlation.
        """

        assert _detect_content_type(academic_content) == "academic"

    def test_forum_content_detection_by_content(self):
        """Test detection of forum content by content analysis."""
        forum_content = """
        Posted by user123: I have a question about this problem.
        Reply: This issue can be solved by following these steps.
        Comment: In my experience, this approach works well.
        Upvoted answer: The solution is marked as solved.
        """

        assert _detect_content_type(forum_content) == "forum"

    def test_news_content_detection_by_content(self):
        """Test detection of news content by content analysis."""
        news_content = """
        Published today by our reporter: Breaking news about the announcement.
        According to sources, the company issued a statement.
        The press release confirmed the update about the development.
        """

        assert _detect_content_type(news_content) == "news"

    def test_blog_content_detection_by_content(self):
        """Test detection of blog content by content indicators."""
        content = "I think this blog post written by the author reflects my personal thoughts on technology."

        assert _detect_content_type(content) == "blog"

        # Test with multiple indicators
        content_multiple = "In my opinion, this blog post written by the author shares thoughts on technology."
        assert _detect_content_type(content_multiple) == "blog"

    def test_social_media_content_detection_by_url(self):
        """Test detection of social media content by URL patterns."""
        content = "This is a post about technology trends."

        # Social media platforms
        assert (
            _detect_content_type(content, "https://twitter.com/user/status/123")
            == "social_media"
        )
        assert (
            _detect_content_type(content, "https://x.com/user/status/123")
            == "social_media"
        )
        assert (
            _detect_content_type(content, "https://linkedin.com/posts/user-123")
            == "social_media"
        )
        assert (
            _detect_content_type(content, "https://instagram.com/p/abc123")
            == "social_media"
        )
        assert (
            _detect_content_type(content, "https://facebook.com/user/posts/123")
            == "social_media"
        )

    def test_social_media_content_detection_by_content(self):
        """Test detection of social media content by content indicators."""
        content = "Just shared this amazing article! #technology @username retweet if you agree."

        assert _detect_content_type(content) == "social_media"

        # Test with multiple indicators
        content_multiple = (
            "This LinkedIn post is trending with lots of engagement from followers."
        )
        assert _detect_content_type(content_multiple) == "social_media"

    def test_legal_content_detection_by_url(self):
        """Test detection of legal content by URL patterns."""
        content = "This document outlines the terms and conditions."

        # Legal sites
        assert (
            _detect_content_type(content, "https://law.example.com/statute") == "legal"
        )
        assert (
            _detect_content_type(content, "https://legal.company.com/terms") == "legal"
        )
        assert _detect_content_type(content, "https://court.gov/decisions") == "legal"
        assert (
            _detect_content_type(content, "https://gov.state.us/laws/section")
            == "legal"
        )

    def test_legal_content_detection_by_content(self):
        """Test detection of legal content by content indicators."""
        content = "Whereas the plaintiff hereby agrees pursuant to section 12 of this contract."

        assert _detect_content_type(content) == "legal"

        # Test with multiple indicators
        content_multiple = "The defendant's attorney shall provide counsel regarding liability and damages in this jurisdiction."
        assert _detect_content_type(content_multiple) == "legal"

    def test_educational_content_detection_by_url(self):
        """Test detection of educational content by URL patterns."""
        content = (
            "This tutorial will teach you step by step how to complete the exercise."
        )

        # Educational platforms
        assert (
            _detect_content_type(content, "https://tutorial.example.com/guide")
            == "educational"
        )
        assert (
            _detect_content_type(content, "https://course.university.edu/lesson")
            == "educational"
        )
        assert (
            _detect_content_type(content, "https://learn.company.com/module")
            == "educational"
        )
        assert (
            _detect_content_type(content, "https://udemy.com/course/python")
            == "educational"
        )
        assert (
            _detect_content_type(content, "https://coursera.org/learn/data")
            == "educational"
        )
        assert (
            _detect_content_type(content, "https://khanacademy.org/math")
            == "educational"
        )

    def test_educational_content_detection_by_content(self):
        """Test detection of educational content by content indicators."""
        content = "This tutorial lesson will guide beginners through step by step instructions."

        assert _detect_content_type(content) == "educational"

        # Test with multiple indicators
        content_multiple = "Important: Remember to practice this exercise in the training module for intermediate skill development."
        assert _detect_content_type(content_multiple) == "educational"

    def test_general_content_fallback(self):
        """Test fallback to general content type."""
        general_content = "This is some neutral content without specific indicators."

        assert _detect_content_type(general_content) == "general"

    def test_insufficient_indicators_fallback(self):
        """Test fallback when not enough indicators are present."""
        content_with_few_indicators = (
            "This content mentions API once but not much else."
        )

        # Should fallback to general since only 1 indicator
        assert _detect_content_type(content_with_few_indicators) == "general"


class TestAdaptivePrompting:
    """Test adaptive prompting functionality."""

    def test_technical_prompting_strategy(self):
        """Test technical content prompting strategy."""
        user_prompt, system_message = _get_contextual_prompt_and_system_message(
            "technical", "Full technical document", "Code example chunk"
        )

        assert "technical" in user_prompt.lower()
        assert "APIs" in user_prompt or "procedures" in user_prompt
        assert "technical documentation specialist" in system_message.lower()
        assert "API" in system_message
        assert "commands" in system_message

    def test_academic_prompting_strategy(self):
        """Test academic content prompting strategy."""
        user_prompt, system_message = _get_contextual_prompt_and_system_message(
            "academic", "Full research paper", "Methodology section"
        )

        assert "academic" in user_prompt.lower() or "research" in user_prompt.lower()
        assert "academic content specialist" in system_message.lower()
        assert "methodology" in system_message.lower()
        assert "research" in system_message.lower()

    def test_forum_prompting_strategy(self):
        """Test forum content prompting strategy."""
        user_prompt, system_message = _get_contextual_prompt_and_system_message(
            "forum", "Full discussion thread", "Answer to question"
        )

        assert "forum" in user_prompt.lower() or "discussion" in user_prompt.lower()
        assert "discussion thread specialist" in system_message.lower()
        assert "question" in system_message.lower()
        assert "answer" in system_message.lower()

    def test_news_prompting_strategy(self):
        """Test news content prompting strategy."""
        user_prompt, system_message = _get_contextual_prompt_and_system_message(
            "news", "Full news article", "Key development paragraph"
        )

        assert "news" in user_prompt.lower()
        assert "events" in user_prompt.lower() or "developments" in user_prompt.lower()
        assert "news content specialist" in system_message.lower()
        assert "event" in system_message.lower()

    def test_blog_prompting_strategy(self):
        """Test blog content prompting strategy."""
        user_prompt, system_message = _get_contextual_prompt_and_system_message(
            "blog", "Full blog post", "Opinion paragraph"
        )

        assert "blog" in user_prompt.lower()
        assert "ideas" in user_prompt.lower() or "opinions" in user_prompt.lower()
        assert "blog content specialist" in system_message.lower()
        assert "opinion" in system_message.lower()

    def test_social_media_prompting_strategy(self):
        """Test social media content prompting strategy."""
        user_prompt, system_message = _get_contextual_prompt_and_system_message(
            "social_media", "Full social media thread", "Individual post"
        )

        assert "social media" in user_prompt.lower()
        assert "message" in user_prompt.lower() or "engagement" in user_prompt.lower()
        assert "social media content specialist" in system_message.lower()
        assert "engagement" in system_message.lower()
        assert "hashtag" in system_message.lower()

    def test_legal_prompting_strategy(self):
        """Test legal content prompting strategy."""
        user_prompt, system_message = _get_contextual_prompt_and_system_message(
            "legal", "Full legal document", "Contract clause"
        )

        assert "legal" in user_prompt.lower()
        assert "concepts" in user_prompt.lower() or "obligations" in user_prompt.lower()
        assert "legal document specialist" in system_message.lower()
        assert "legal" in system_message.lower()
        assert "terminology" in system_message.lower()

    def test_educational_prompting_strategy(self):
        """Test educational content prompting strategy."""
        user_prompt, system_message = _get_contextual_prompt_and_system_message(
            "educational", "Full tutorial guide", "Step-by-step instructions"
        )

        assert "educational" in user_prompt.lower()
        assert (
            "learning" in user_prompt.lower() or "instructional" in user_prompt.lower()
        )
        assert "educational content specialist" in system_message.lower()
        assert "learning" in system_message.lower()
        assert "instructional" in system_message.lower()

    def test_general_prompting_strategy(self):
        """Test general content prompting strategy."""
        user_prompt, system_message = _get_contextual_prompt_and_system_message(
            "general", "Full document", "Content chunk"
        )

        assert "content" in user_prompt.lower()
        assert "concepts" in user_prompt.lower() or "themes" in user_prompt.lower()
        assert "general content specialist" in system_message.lower()
        assert "concept" in system_message.lower()


class TestEnhancedContextualEmbedding:
    """Test the enhanced contextual embedding generation."""

    @patch("utils.openai.chat.completions.create")
    @patch("config.get_config")
    def test_adaptive_prompting_enabled(self, mock_get_config, mock_openai):
        """Test contextual embedding with adaptive prompting enabled."""
        # Reset global config cache
        reset_config()

        # Mock configuration
        mock_config = MagicMock()
        mock_config.contextual_content_type_detection = True
        mock_config.use_adaptive_contextual_prompts = True
        mock_config.contextual_include_content_type_tag = True
        mock_get_config.return_value = mock_config

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[
            0
        ].message.content = "This is contextual information about the API."
        mock_openai.return_value = mock_response

        # Mock model choice
        with patch("utils._get_contextual_model", return_value="gpt-4o-mini"):
            full_doc = "This is a technical document about API usage and configuration."
            chunk = "The API endpoint accepts GET requests with parameters."
            url = "https://docs.example.com/api"

            result, success = generate_contextual_embedding(full_doc, chunk, url)

            assert success
            assert "[TECHNICAL]" in result  # Content type tag should be included
            assert "This is contextual information about the API." in result
            assert chunk in result

            # Verify OpenAI was called with appropriate prompts
            mock_openai.assert_called_once()
            call_args = mock_openai.call_args
            messages = call_args.kwargs["messages"]

            # Should use technical specialist system message
            assert (
                "technical documentation specialist" in messages[0]["content"].lower()
            )

    @patch("utils.openai.chat.completions.create")
    @patch("config.get_config")
    def test_legacy_prompting_mode(self, mock_get_config, mock_openai):
        """Test contextual embedding with legacy prompting (backward compatibility)."""
        # Reset global config cache
        reset_config()

        # Mock configuration for legacy mode
        mock_config = MagicMock()
        mock_config.contextual_content_type_detection = False
        mock_config.use_adaptive_contextual_prompts = False
        mock_config.contextual_include_content_type_tag = False
        mock_get_config.return_value = mock_config

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Legacy contextual information."
        mock_openai.return_value = mock_response

        # Mock model choice
        with patch("utils._get_contextual_model", return_value="gpt-4o-mini"):
            full_doc = "This is a document."
            chunk = "This is a chunk."
            url = "https://example.com"

            result, success = generate_contextual_embedding(full_doc, chunk, url)

            assert success
            assert "[TECHNICAL]" not in result  # No content type tag
            assert "Legacy contextual information." in result
            assert chunk in result

            # Verify OpenAI was called with legacy prompts
            mock_openai.assert_called_once()
            call_args = mock_openai.call_args
            messages = call_args.kwargs["messages"]

            # Should use legacy system message
            assert "concise technical summarizer" in messages[0]["content"].lower()

    @patch("utils.openai.chat.completions.create")
    @patch("config.get_config")
    def test_content_type_detection_only(self, mock_get_config, mock_openai):
        """Test content type detection without adaptive prompting."""
        # Reset global config cache
        reset_config()

        # Mock configuration
        mock_config = MagicMock()
        mock_config.contextual_content_type_detection = True
        mock_config.use_adaptive_contextual_prompts = False  # Disabled
        mock_config.contextual_include_content_type_tag = True
        mock_get_config.return_value = mock_config

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "General contextual information."
        mock_openai.return_value = mock_response

        # Mock model choice
        with patch("utils._get_contextual_model", return_value="gpt-4o-mini"):
            full_doc = "This is a technical document about API usage."
            chunk = "The API endpoint accepts parameters."
            url = "https://docs.example.com/api"

            result, success = generate_contextual_embedding(full_doc, chunk, url)

            assert success
            assert "[TECHNICAL]" in result  # Content type tag should be included
            assert "General contextual information." in result

            # Should use legacy prompting even though content type was detected
            call_args = mock_openai.call_args
            messages = call_args.kwargs["messages"]
            assert "concise technical summarizer" in messages[0]["content"].lower()

    @patch("utils._get_contextual_model", return_value=None)
    def test_no_model_configured(self, mock_get_contextual_model):
        """Test behavior when no contextual model is configured."""
        # Reset global config cache
        reset_config()

        full_doc = "This is a document."
        chunk = "This is a chunk."
        url = "https://example.com"

        result, success = generate_contextual_embedding(full_doc, chunk, url)

        assert not success
        assert result == chunk  # Should return original chunk unchanged

    @patch("utils.openai.chat.completions.create")
    @patch("config.get_config")
    def test_error_handling(self, mock_get_config, mock_openai):
        """Test error handling in contextual embedding generation."""
        # Reset global config cache
        reset_config()

        # Mock configuration
        mock_config = MagicMock()
        mock_config.contextual_content_type_detection = True
        mock_config.use_adaptive_contextual_prompts = True
        mock_config.contextual_include_content_type_tag = True
        mock_get_config.return_value = mock_config

        # Mock OpenAI to raise an exception
        mock_openai.side_effect = Exception("API Error")

        # Mock model choice
        with patch("utils._get_contextual_model", return_value="gpt-4o-mini"):
            full_doc = "This is a document."
            chunk = "This is a chunk."
            url = "https://example.com"

            result, success = generate_contextual_embedding(full_doc, chunk, url)

            assert not success
            assert result == chunk  # Should return original chunk on error


if __name__ == "__main__":
    pytest.main([__file__])
