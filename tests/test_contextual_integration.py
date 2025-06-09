"""
Tests for contextual embeddings configuration integration.

This module tests the integration between the new configuration system
and existing contextual embeddings functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Tuple

# Import functions to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import _should_use_contextual_embeddings, _get_contextual_model, generate_contextual_embedding
from config import StrategyConfig, reset_config


class TestContextualEmbeddingsIntegration:
    """Test the integration between new config system and existing contextual embeddings."""
    
    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_config()
    
    def test_should_use_contextual_embeddings_new_config_enabled(self):
        """Test that new USE_CONTEXTUAL_EMBEDDINGS=true enables contextual embeddings."""
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "CONTEXTUAL_MODEL": "gpt-4",
            "OPENAI_API_KEY": "test-key",
            "MODEL_CHOICE": "",  # Clear legacy setting
        }, clear=False):
            reset_config()
            assert _should_use_contextual_embeddings() is True
    
    def test_should_use_contextual_embeddings_new_config_disabled(self):
        """Test that new USE_CONTEXTUAL_EMBEDDINGS=false disables contextual embeddings."""
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "false",
            "MODEL_CHOICE": "",  # Clear legacy setting
        }, clear=False):
            reset_config()
            assert _should_use_contextual_embeddings() is False
    
    def test_should_use_contextual_embeddings_legacy_model_choice(self):
        """Test backward compatibility with MODEL_CHOICE environment variable."""
        with patch.dict(os.environ, {
            "MODEL_CHOICE": "gpt-3.5-turbo",
            "USE_CONTEXTUAL_EMBEDDINGS": "",  # Clear new setting
        }, clear=False):
            reset_config()
            assert _should_use_contextual_embeddings() is True
    
    def test_should_use_contextual_embeddings_no_config(self):
        """Test that no configuration disables contextual embeddings."""
        with patch.dict(os.environ, {
            "MODEL_CHOICE": "",
            "USE_CONTEXTUAL_EMBEDDINGS": "",
        }, clear=False):
            reset_config()
            assert _should_use_contextual_embeddings() is False
    
    def test_should_use_contextual_embeddings_new_overrides_legacy(self):
        """Test that new config takes precedence over legacy when both are set."""
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "CONTEXTUAL_MODEL": "gpt-4",
            "OPENAI_API_KEY": "test-key",
            "MODEL_CHOICE": "gpt-3.5-turbo",  # Legacy setting should be ignored
        }, clear=False):
            reset_config()
            assert _should_use_contextual_embeddings() is True
    
    def test_get_contextual_model_new_config(self):
        """Test that new CONTEXTUAL_MODEL is used when USE_CONTEXTUAL_EMBEDDINGS=true."""
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "CONTEXTUAL_MODEL": "gpt-4",
            "OPENAI_API_KEY": "test-key",
            "MODEL_CHOICE": "gpt-3.5-turbo",  # Should be ignored
        }, clear=False):
            reset_config()
            assert _get_contextual_model() == "gpt-4"
    
    def test_get_contextual_model_legacy_fallback(self):
        """Test fallback to MODEL_CHOICE when new config not enabled."""
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "false",
            "MODEL_CHOICE": "gpt-3.5-turbo",
        }, clear=False):
            reset_config()
            assert _get_contextual_model() == "gpt-3.5-turbo"
    
    def test_get_contextual_model_none_when_disabled(self):
        """Test that None is returned when contextual embeddings disabled."""
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "false",
            "MODEL_CHOICE": "",
        }, clear=False):
            reset_config()
            assert _get_contextual_model() is None
    
    @patch('utils._retry_with_backoff')
    def test_generate_contextual_embedding_with_new_config(self, mock_retry):
        """Test contextual embedding generation with new configuration."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This section covers API authentication methods."
        mock_retry.return_value = mock_response
        
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "CONTEXTUAL_MODEL": "gpt-4",
            "OPENAI_API_KEY": "test-key",
        }, clear=False):
            reset_config()
            
            full_doc = "# API Documentation\n\nThis document covers authentication..."
            chunk = "Use Bearer tokens for authentication."
            
            result, success = generate_contextual_embedding(full_doc, chunk)
            
            assert success is True
            assert "This section covers API authentication methods." in result
            assert chunk in result
            
            # Verify _retry_with_backoff was called
            mock_retry.assert_called_once()
            # Check that the model parameter was passed correctly
            call_args = mock_retry.call_args
            # The model should be in the kwargs passed to the OpenAI function
            assert 'model' in str(call_args) and 'gpt-4' in str(call_args)
    
    def test_generate_contextual_embedding_no_model_configured(self):
        """Test that original chunk is returned when no model configured."""
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "false",
            "MODEL_CHOICE": "",
        }, clear=False):
            reset_config()
            
            full_doc = "# API Documentation"
            chunk = "Use Bearer tokens for authentication."
            
            result, success = generate_contextual_embedding(full_doc, chunk)
            
            assert success is False
            assert result == chunk
    
    @patch('utils._retry_with_backoff')
    def test_generate_contextual_embedding_legacy_compatibility(self, mock_retry):
        """Test that legacy MODEL_CHOICE still works."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Legacy context generation."
        mock_retry.return_value = mock_response
        
        with patch.dict(os.environ, {
            "MODEL_CHOICE": "gpt-3.5-turbo",
            "USE_CONTEXTUAL_EMBEDDINGS": "",  # Not set
            "OPENAI_API_KEY": "test-key",
        }, clear=False):
            reset_config()
            
            full_doc = "# Legacy Documentation"
            chunk = "Legacy authentication method."
            
            result, success = generate_contextual_embedding(full_doc, chunk)
            
            assert success is True
            assert "Legacy context generation." in result
            
            # Verify _retry_with_backoff was called with legacy model
            mock_retry.assert_called_once()
            call_args = mock_retry.call_args
            assert 'gpt-3.5-turbo' in str(call_args)
    
    @patch('utils._retry_with_backoff')
    def test_generate_contextual_embedding_error_handling(self, mock_retry):
        """Test error handling in contextual embedding generation."""
        # Mock OpenAI to raise an exception
        mock_retry.side_effect = Exception("API Error")
        
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "CONTEXTUAL_MODEL": "gpt-4",
            "OPENAI_API_KEY": "test-key",
        }, clear=False):
            reset_config()
            
            full_doc = "# API Documentation"
            chunk = "Use Bearer tokens for authentication."
            
            result, success = generate_contextual_embedding(full_doc, chunk)
            
            assert success is False
            assert result == chunk  # Should fall back to original chunk
    
    def test_config_precedence_order(self):
        """Test that configuration precedence works correctly."""
        # Test 1: New config enabled takes precedence
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "CONTEXTUAL_MODEL": "gpt-4",
            "OPENAI_API_KEY": "test-key",
            "MODEL_CHOICE": "gpt-3.5-turbo",
        }, clear=False):
            reset_config()
            assert _should_use_contextual_embeddings() is True
            assert _get_contextual_model() == "gpt-4"
        
        # Test 2: New config disabled, fall back to legacy
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "false",
            "MODEL_CHOICE": "gpt-3.5-turbo",
        }, clear=False):
            reset_config()
            assert _should_use_contextual_embeddings() is True  # Legacy enables it
            assert _get_contextual_model() == "gpt-3.5-turbo"
        
        # Test 3: Both disabled
        with patch.dict(os.environ, {
            "USE_CONTEXTUAL_EMBEDDINGS": "false",
            "MODEL_CHOICE": "",
        }, clear=False):
            reset_config()
            assert _should_use_contextual_embeddings() is False
            assert _get_contextual_model() is None


class TestBackwardCompatibility:
    """Test backward compatibility with existing setups."""
    
    def setup_method(self):
        """Reset configuration before each test."""
        reset_config()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_config()
    
    def test_existing_model_choice_continues_working(self):
        """Test that existing MODEL_CHOICE environment variable continues to work."""
        with patch.dict(os.environ, {
            "MODEL_CHOICE": "gpt-3.5-turbo",
            # Don't set any new configuration variables
        }, clear=False):
            reset_config()
            
            assert _should_use_contextual_embeddings() is True
            assert _get_contextual_model() == "gpt-3.5-turbo"
    
    def test_no_configuration_disables_feature(self):
        """Test that no configuration properly disables contextual embeddings."""
        with patch.dict(os.environ, {
            "MODEL_CHOICE": "",
            "USE_CONTEXTUAL_EMBEDDINGS": "",
        }, clear=False):
            reset_config()
            
            assert _should_use_contextual_embeddings() is False
            assert _get_contextual_model() is None
    
    def test_gradual_migration_path(self):
        """Test gradual migration from old to new configuration."""
        # Step 1: Current state with MODEL_CHOICE
        with patch.dict(os.environ, {
            "MODEL_CHOICE": "gpt-3.5-turbo",
        }, clear=False):
            reset_config()
            assert _should_use_contextual_embeddings() is True
            assert _get_contextual_model() == "gpt-3.5-turbo"
        
        # Step 2: Add new config alongside old (new takes precedence)
        with patch.dict(os.environ, {
            "MODEL_CHOICE": "gpt-3.5-turbo",
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "CONTEXTUAL_MODEL": "gpt-4",
            "OPENAI_API_KEY": "test-key",
        }, clear=False):
            reset_config()
            assert _should_use_contextual_embeddings() is True
            assert _get_contextual_model() == "gpt-4"  # New config wins
        
        # Step 3: Remove old config (new config still works)
        with patch.dict(os.environ, {
            "MODEL_CHOICE": "",
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "CONTEXTUAL_MODEL": "gpt-4",
            "OPENAI_API_KEY": "test-key",
        }, clear=False):
            reset_config()
            assert _should_use_contextual_embeddings() is True
            assert _get_contextual_model() == "gpt-4"