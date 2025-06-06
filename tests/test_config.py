"""
Unit tests for the configuration management system.

Tests environment variable loading, validation logic, and error handling
for the RAG strategy configuration system.
"""

import os
import pytest
from unittest.mock import patch
from src.config import (
    StrategyConfig,
    RAGStrategy,
    ConfigurationError,
    load_and_validate_config,
    get_config,
    reset_config
)


class TestStrategyConfig:
    """Test cases for StrategyConfig class."""

    def test_default_values(self):
        """Test that default configuration has all strategies disabled."""
        config = StrategyConfig()
        
        assert not config.use_contextual_embeddings
        assert not config.use_hybrid_search_enhanced
        assert not config.use_agentic_rag
        assert not config.use_reranking
        
        assert config.contextual_model == "gpt-3.5-turbo"
        assert config.reranking_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.max_reranking_results == 20
        assert config.reranking_timeout_ms == 500

    def test_get_enabled_strategies_none_enabled(self):
        """Test get_enabled_strategies with no strategies enabled."""
        config = StrategyConfig()
        enabled = config.get_enabled_strategies()
        assert enabled == set()

    def test_get_enabled_strategies_all_enabled(self):
        """Test get_enabled_strategies with all strategies enabled."""
        config = StrategyConfig(
            use_contextual_embeddings=True,
            use_hybrid_search_enhanced=True,
            use_agentic_rag=True,
            use_reranking=True
        )
        enabled = config.get_enabled_strategies()
        expected = {
            RAGStrategy.CONTEXTUAL_EMBEDDINGS,
            RAGStrategy.HYBRID_SEARCH_ENHANCED,
            RAGStrategy.AGENTIC_RAG,
            RAGStrategy.RERANKING
        }
        assert enabled == expected

    def test_is_strategy_enabled(self):
        """Test is_strategy_enabled method."""
        config = StrategyConfig(use_reranking=True)
        
        assert config.is_strategy_enabled(RAGStrategy.RERANKING)
        assert not config.is_strategy_enabled(RAGStrategy.CONTEXTUAL_EMBEDDINGS)

    def test_str_representation_no_strategies(self):
        """Test string representation with no strategies enabled."""
        config = StrategyConfig()
        assert "No strategies enabled (baseline mode)" in str(config)

    def test_str_representation_with_strategies(self):
        """Test string representation with strategies enabled."""
        config = StrategyConfig(use_reranking=True, use_agentic_rag=True)
        result = str(config)
        assert "reranking" in result
        assert "agentic_rag" in result


class TestEnvironmentLoading:
    """Test cases for loading configuration from environment variables."""

    def test_from_environment_defaults(self):
        """Test loading with no environment variables set."""
        with patch.dict(os.environ, {}, clear=True):
            config = StrategyConfig.from_environment()
            
            assert not config.use_contextual_embeddings
            assert not config.use_hybrid_search_enhanced
            assert not config.use_agentic_rag
            assert not config.use_reranking

    def test_from_environment_true_values(self):
        """Test loading with true values in various formats."""
        env_vars = {
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "USE_HYBRID_SEARCH_ENHANCED": "1",
            "USE_AGENTIC_RAG": "yes",
            "USE_RERANKING": "on"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = StrategyConfig.from_environment()
            
            assert config.use_contextual_embeddings
            assert config.use_hybrid_search_enhanced
            assert config.use_agentic_rag
            assert config.use_reranking

    def test_from_environment_false_values(self):
        """Test loading with false values in various formats."""
        env_vars = {
            "USE_CONTEXTUAL_EMBEDDINGS": "false",
            "USE_HYBRID_SEARCH_ENHANCED": "0",
            "USE_AGENTIC_RAG": "no",
            "USE_RERANKING": "off"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = StrategyConfig.from_environment()
            
            assert not config.use_contextual_embeddings
            assert not config.use_hybrid_search_enhanced
            assert not config.use_agentic_rag
            assert not config.use_reranking

    def test_from_environment_model_config(self):
        """Test loading model configuration from environment."""
        env_vars = {
            "CONTEXTUAL_MODEL": "gpt-4",
            "RERANKING_MODEL": "custom-reranker",
            "MAX_RERANKING_RESULTS": "50",
            "RERANKING_TIMEOUT_MS": "1000"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = StrategyConfig.from_environment()
            
            assert config.contextual_model == "gpt-4"
            assert config.reranking_model == "custom-reranker"
            assert config.max_reranking_results == 50
            assert config.reranking_timeout_ms == 1000


class TestValidation:
    """Test cases for configuration validation."""

    def test_validate_empty_models(self):
        """Test validation fails with empty model names."""
        config = StrategyConfig(contextual_model="", reranking_model="  ")
        errors = config.validate()
        
        assert len(errors) == 2
        assert any("CONTEXTUAL_MODEL cannot be empty" in error for error in errors)
        assert any("RERANKING_MODEL cannot be empty" in error for error in errors)

    def test_validate_negative_values(self):
        """Test validation fails with negative performance settings."""
        config = StrategyConfig(
            max_reranking_results=-1,
            reranking_timeout_ms=0
        )
        errors = config.validate()
        
        assert len(errors) == 2
        assert any("MAX_RERANKING_RESULTS must be positive" in error for error in errors)
        assert any("RERANKING_TIMEOUT_MS must be positive" in error for error in errors)

    def test_validate_reranking_limit(self):
        """Test validation warns about high reranking results."""
        config = StrategyConfig(
            use_reranking=True,
            max_reranking_results=150
        )
        errors = config.validate()
        
        assert len(errors) == 1
        assert "should not exceed 100 for performance reasons" in errors[0]

    def test_validate_contextual_embeddings_dependency(self):
        """Test validation requires MODEL_CHOICE for contextual embeddings."""
        config = StrategyConfig(use_contextual_embeddings=True)
        
        with patch.dict(os.environ, {}, clear=True):
            errors = config.validate()
            assert any("requires MODEL_CHOICE environment variable" in error for error in errors)

    def test_validate_dependencies(self):
        """Test dependency validation for enabled strategies."""
        config = StrategyConfig(
            use_contextual_embeddings=True,
            use_agentic_rag=True
        )
        
        with patch.dict(os.environ, {}, clear=True):
            errors = config.validate_dependencies()
            
            # Should have errors for missing OPENAI_API_KEY, MODEL_CHOICE, and SUPABASE_URL
            assert len(errors) > 0
            assert any("OPENAI_API_KEY" in error for error in errors)
            assert any("MODEL_CHOICE" in error for error in errors)
            assert any("SUPABASE_URL" in error for error in errors)

    def test_validate_dependencies_satisfied(self):
        """Test dependency validation passes when all dependencies present."""
        config = StrategyConfig(use_reranking=True)  # No external dependencies
        
        errors = config.validate_dependencies()
        assert len(errors) == 0


class TestConfigurationError:
    """Test cases for ConfigurationError exception."""

    def test_configuration_error_formatting(self):
        """Test ConfigurationError formats errors correctly."""
        errors = ["Error 1", "Error 2"]
        exc = ConfigurationError(errors)
        
        assert exc.errors == errors
        assert "Configuration validation failed:" in str(exc)
        assert "Error 1" in str(exc)
        assert "Error 2" in str(exc)


class TestGlobalConfig:
    """Test cases for global configuration management."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_config()

    def test_load_and_validate_config_success(self):
        """Test successful configuration loading and validation."""
        env_vars = {
            "USE_RERANKING": "true",
            "RERANKING_MODEL": "test-model"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_and_validate_config()
            assert config.use_reranking
            assert config.reranking_model == "test-model"

    def test_load_and_validate_config_failure(self):
        """Test configuration loading fails with validation errors."""
        env_vars = {
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "CONTEXTUAL_MODEL": ""
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                load_and_validate_config()
            
            assert "CONTEXTUAL_MODEL cannot be empty" in str(exc_info.value)

    def test_get_config_caching(self):
        """Test that get_config() caches the configuration."""
        env_vars = {"USE_RERANKING": "true"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            config1 = get_config()
            config2 = get_config()
            
            # Should be the same instance (cached)
            assert config1 is config2

    def test_reset_config(self):
        """Test that reset_config() clears the cached configuration."""
        env_vars = {"USE_RERANKING": "true"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            config1 = get_config()
            reset_config()
            config2 = get_config()
            
            # Should be different instances after reset
            assert config1 is not config2
            # But should have same values
            assert config1.use_reranking == config2.use_reranking


if __name__ == "__main__":
    pytest.main([__file__])