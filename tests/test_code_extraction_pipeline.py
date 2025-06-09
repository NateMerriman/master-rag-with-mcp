"""
Test suite for code extraction pipeline integration.

This module tests the integration between code extraction functionality
and the document processing pipeline, including dual embeddings storage
and agentic RAG strategy integration.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import the modules being tested
try:
    from src.utils import (
        add_code_examples_to_supabase,
        get_source_id_from_url,
        _should_use_agentic_rag,
        add_documents_to_supabase
    )
    from src.code_extraction import extract_code_from_content, ExtractedCode
    from src.database.models import CodeExample
    from src.config import StrategyConfig
except ImportError:
    from utils import (
        add_code_examples_to_supabase,
        get_source_id_from_url,
        _should_use_agentic_rag,
        add_documents_to_supabase
    )
    from code_extraction import extract_code_from_content, ExtractedCode
    from database.models import CodeExample
    from config import StrategyConfig


class TestAgenticRAGConfiguration:
    """Test agentic RAG configuration detection."""
    
    def test_should_use_agentic_rag_with_config(self):
        """Test agentic RAG detection with StrategyConfig."""
        with patch('src.utils.get_config') as mock_get_config:
            mock_config = StrategyConfig(use_agentic_rag=True)
            mock_get_config.return_value = mock_config
            
            assert _should_use_agentic_rag() is True
    
    def test_should_use_agentic_rag_disabled(self):
        """Test agentic RAG detection when disabled."""
        with patch('src.utils.get_config') as mock_get_config:
            mock_config = StrategyConfig(use_agentic_rag=False)
            mock_get_config.return_value = mock_config
            
            assert _should_use_agentic_rag() is False
    
    def test_should_use_agentic_rag_env_fallback(self):
        """Test agentic RAG detection with environment variable fallback."""
        with patch('src.utils.get_config', side_effect=Exception("Config not available")):
            with patch.dict(os.environ, {'USE_AGENTIC_RAG': 'true'}):
                assert _should_use_agentic_rag() is True
            
            with patch.dict(os.environ, {'USE_AGENTIC_RAG': 'false'}):
                assert _should_use_agentic_rag() is False


class TestSourceIdRetrieval:
    """Test source_id retrieval from URLs."""
    
    def test_get_source_id_from_url_success(self):
        """Test successful source_id retrieval."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [{"source_id": 123}]
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        result = get_source_id_from_url(mock_client, "https://example.com")
        
        assert result == 123
        mock_client.table.assert_called_with("sources")
    
    def test_get_source_id_from_url_not_found(self):
        """Test source_id retrieval when URL not found."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = []
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_response
        
        result = get_source_id_from_url(mock_client, "https://nonexistent.com")
        
        assert result is None
    
    def test_get_source_id_from_url_error(self):
        """Test source_id retrieval with database error."""
        mock_client = Mock()
        mock_client.table.side_effect = Exception("Database error")
        
        result = get_source_id_from_url(mock_client, "https://example.com")
        
        assert result is None


class TestCodeExamplesStorage:
    """Test code examples storage with dual embeddings."""
    
    def test_add_code_examples_to_supabase_success(self):
        """Test successful code examples insertion."""
        mock_client = Mock()
        mock_client.table.return_value.insert.return_value.execute.return_value = Mock()
        
        code_examples = [
            {
                "code_content": "def hello():\n    print('Hello, World!')",
                "summary": "Python function that prints Hello World",
                "programming_language": "python",
                "complexity_score": 2,
            }
        ]
        
        with patch('src.utils.create_embeddings_batch') as mock_embeddings:
            mock_embeddings.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding
            
            add_code_examples_to_supabase(mock_client, code_examples, source_id=123)
            
            # Verify embeddings were created for both code and summary
            assert mock_embeddings.call_count == 2
            
            # Verify insertion was called
            mock_client.table.assert_called_with("code_examples")
    
    def test_add_code_examples_empty_list(self):
        """Test code examples insertion with empty list."""
        mock_client = Mock()
        
        add_code_examples_to_supabase(mock_client, [], source_id=123)
        
        # Should not attempt any database operations
        mock_client.table.assert_not_called()
    
    def test_add_code_examples_embedding_error(self):
        """Test code examples insertion with embedding generation error."""
        mock_client = Mock()
        
        code_examples = [
            {
                "code_content": "def test(): pass",
                "summary": "Test function",
                "programming_language": "python",
                "complexity_score": 1,
            }
        ]
        
        with patch('src.utils.create_embeddings_batch', side_effect=Exception("API error")):
            add_code_examples_to_supabase(mock_client, code_examples, source_id=123)
            
            # Should not attempt insertion due to embedding error
            mock_client.table.assert_not_called()


class TestCodeExtractionIntegration:
    """Test integration between code extraction and document processing."""
    
    def test_extract_code_from_content_integration(self):
        """Test code extraction integration with document content."""
        content = """
        # Python Example
        Here's a simple Python function:
        
        ```python
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        ```
        
        This function calculates the nth Fibonacci number.
        """
        
        extracted_codes = extract_code_from_content(content)
        
        assert len(extracted_codes) == 1
        assert extracted_codes[0].programming_language == "python"
        assert "fibonacci" in extracted_codes[0].code_content
        assert extracted_codes[0].complexity_score >= 1
    
    def test_extract_multiple_languages(self):
        """Test extraction of multiple programming languages."""
        content = """
        # Mixed Code Examples
        
        Python function:
        ```python
        def greet(name):
            return f"Hello, {name}!"
        ```
        
        JavaScript function:
        ```javascript
        function greet(name) {
            return `Hello, ${name}!`;
        }
        ```
        
        SQL query:
        ```sql
        SELECT name, email FROM users WHERE active = true;
        ```
        """
        
        extracted_codes = extract_code_from_content(content)
        
        assert len(extracted_codes) == 3
        languages = [code.programming_language for code in extracted_codes]
        assert "python" in languages
        assert "javascript" in languages
        assert "sql" in languages


class TestDocumentProcessingPipelineIntegration:
    """Test integration with the main document processing pipeline."""
    
    @patch('src.utils._should_use_agentic_rag')
    @patch('src.utils.extract_code_from_content')
    @patch('src.utils.get_source_id_from_url')
    @patch('src.utils.add_code_examples_to_supabase')
    @patch('src.utils.create_embeddings_batch')
    def test_add_documents_with_code_extraction(
        self,
        mock_embeddings,
        mock_add_code_examples,
        mock_get_source_id,
        mock_extract_code,
        mock_should_use_agentic
    ):
        """Test that add_documents_to_supabase extracts code when agentic RAG is enabled."""
        # Setup mocks
        mock_should_use_agentic.return_value = True
        mock_get_source_id.return_value = 123
        mock_embeddings.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock extracted code
        mock_extracted_code = ExtractedCode(
            code_content="def test(): pass",
            summary="Test function",
            programming_language="python",
            complexity_score=1,
            context=""
        )
        mock_extract_code.return_value = [mock_extracted_code]
        
        # Mock Supabase client
        mock_client = Mock()
        mock_client.table.return_value.delete.return_value.in_.return_value.execute.return_value = Mock()
        mock_client.table.return_value.insert.return_value.execute.return_value = Mock()
        
        # Test data
        urls = ["https://example.com"]
        chunk_numbers = [1]
        contents = ["Here's some content with ```python\ndef test(): pass\n```"]
        metadatas = [{"test": "metadata"}]
        url_to_full_document = {"https://example.com": contents[0]}
        
        # Call the function
        add_documents_to_supabase(
            mock_client,
            urls,
            chunk_numbers,
            contents,
            metadatas,
            url_to_full_document
        )
        
        # Verify code extraction was called
        mock_extract_code.assert_called_once()
        mock_get_source_id.assert_called_with(mock_client, "https://example.com")
        mock_add_code_examples.assert_called_once()
    
    @patch('src.utils._should_use_agentic_rag')
    @patch('src.utils.extract_code_from_content')
    def test_add_documents_without_code_extraction(
        self,
        mock_extract_code,
        mock_should_use_agentic
    ):
        """Test that add_documents_to_supabase skips code extraction when agentic RAG is disabled."""
        # Setup mocks
        mock_should_use_agentic.return_value = False
        
        # Mock Supabase client
        mock_client = Mock()
        mock_client.table.return_value.delete.return_value.in_.return_value.execute.return_value = Mock()
        mock_client.table.return_value.insert.return_value.execute.return_value = Mock()
        
        # Test data
        urls = ["https://example.com"]
        chunk_numbers = [1]
        contents = ["Content without code"]
        metadatas = [{"test": "metadata"}]
        url_to_full_document = {"https://example.com": contents[0]}
        
        with patch('src.utils.create_embeddings_batch', return_value=[[0.1, 0.2, 0.3]]):
            # Call the function
            add_documents_to_supabase(
                mock_client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document
            )
        
        # Verify code extraction was NOT called
        mock_extract_code.assert_not_called()


class TestErrorHandling:
    """Test error handling in code extraction pipeline."""
    
    @patch('src.utils._should_use_agentic_rag')
    @patch('src.utils.extract_code_from_content')
    @patch('src.utils.get_source_id_from_url')
    def test_missing_source_id_handling(
        self,
        mock_get_source_id,
        mock_extract_code,
        mock_should_use_agentic
    ):
        """Test handling when source_id is not found."""
        # Setup mocks
        mock_should_use_agentic.return_value = True
        mock_get_source_id.return_value = None  # No source_id found
        
        # Mock Supabase client
        mock_client = Mock()
        mock_client.table.return_value.delete.return_value.in_.return_value.execute.return_value = Mock()
        mock_client.table.return_value.insert.return_value.execute.return_value = Mock()
        
        # Test data
        urls = ["https://example.com"]
        chunk_numbers = [1]
        contents = ["Content with ```python\ndef test(): pass\n```"]
        metadatas = [{"test": "metadata"}]
        url_to_full_document = {"https://example.com": contents[0]}
        
        with patch('src.utils.create_embeddings_batch', return_value=[[0.1, 0.2, 0.3]]):
            # Call the function
            add_documents_to_supabase(
                mock_client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document
            )
        
        # Verify that code extraction was not called due to missing source_id
        mock_extract_code.assert_not_called()
        mock_get_source_id.assert_called_with(mock_client, "https://example.com")


if __name__ == "__main__":
    pytest.main([__file__])