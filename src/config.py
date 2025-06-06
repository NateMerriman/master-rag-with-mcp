"""
Configuration management for RAG strategy toggles and settings.

This module provides centralized configuration management for all RAG enhancement
strategies, with environment variable loading, validation, and error handling.
"""

import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum


class RAGStrategy(Enum):
    """Enumeration of available RAG strategies."""
    CONTEXTUAL_EMBEDDINGS = "contextual_embeddings"
    HYBRID_SEARCH_ENHANCED = "hybrid_search_enhanced"
    AGENTIC_RAG = "agentic_rag"
    RERANKING = "reranking"


@dataclass
class StrategyConfig:
    """Configuration for RAG strategies and related settings."""
    
    # Strategy toggles (all default to False for backward compatibility)
    use_contextual_embeddings: bool = False
    use_hybrid_search_enhanced: bool = False
    use_agentic_rag: bool = False
    use_reranking: bool = False
    
    # Model configuration
    contextual_model: str = "gpt-3.5-turbo"
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Performance settings
    max_reranking_results: int = 20
    reranking_timeout_ms: int = 500
    
    @classmethod
    def from_environment(cls) -> "StrategyConfig":
        """Load configuration from environment variables."""
        
        def str_to_bool(value: str) -> bool:
            """Convert string environment variable to boolean."""
            return value.lower() in ("true", "1", "yes", "on")
        
        return cls(
            use_contextual_embeddings=str_to_bool(
                os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false")
            ),
            use_hybrid_search_enhanced=str_to_bool(
                os.getenv("USE_HYBRID_SEARCH_ENHANCED", "false")
            ),
            use_agentic_rag=str_to_bool(
                os.getenv("USE_AGENTIC_RAG", "false")
            ),
            use_reranking=str_to_bool(
                os.getenv("USE_RERANKING", "false")
            ),
            contextual_model=os.getenv("CONTEXTUAL_MODEL", "gpt-3.5-turbo"),
            reranking_model=os.getenv(
                "RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            max_reranking_results=int(
                os.getenv("MAX_RERANKING_RESULTS", "20")
            ),
            reranking_timeout_ms=int(
                os.getenv("RERANKING_TIMEOUT_MS", "500")
            ),
        )
    
    def get_enabled_strategies(self) -> Set[RAGStrategy]:
        """Get set of currently enabled strategies."""
        enabled = set()
        
        if self.use_contextual_embeddings:
            enabled.add(RAGStrategy.CONTEXTUAL_EMBEDDINGS)
        if self.use_hybrid_search_enhanced:
            enabled.add(RAGStrategy.HYBRID_SEARCH_ENHANCED)
        if self.use_agentic_rag:
            enabled.add(RAGStrategy.AGENTIC_RAG)
        if self.use_reranking:
            enabled.add(RAGStrategy.RERANKING)
            
        return enabled
    
    def is_strategy_enabled(self, strategy: RAGStrategy) -> bool:
        """Check if a specific strategy is enabled."""
        return strategy in self.get_enabled_strategies()
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of error messages.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate model names
        if not self.contextual_model or not self.contextual_model.strip():
            errors.append("CONTEXTUAL_MODEL cannot be empty")
        
        if not self.reranking_model or not self.reranking_model.strip():
            errors.append("RERANKING_MODEL cannot be empty")
        
        # Validate performance settings
        if self.max_reranking_results <= 0:
            errors.append("MAX_RERANKING_RESULTS must be positive")
        
        if self.reranking_timeout_ms <= 0:
            errors.append("RERANKING_TIMEOUT_MS must be positive")
        
        # Strategy-specific validations
        if self.use_reranking and self.max_reranking_results > 100:
            errors.append(
                "MAX_RERANKING_RESULTS should not exceed 100 for performance reasons"
            )
        
        if self.use_contextual_embeddings and "MODEL_CHOICE" not in os.environ:
            errors.append(
                "USE_CONTEXTUAL_EMBEDDINGS requires MODEL_CHOICE environment variable"
            )
        
        return errors
    
    def get_strategy_dependencies(self) -> Dict[RAGStrategy, List[str]]:
        """
        Get environment variable dependencies for each strategy.
        
        Returns:
            Dictionary mapping strategies to required environment variables
        """
        return {
            RAGStrategy.CONTEXTUAL_EMBEDDINGS: ["MODEL_CHOICE", "OPENAI_API_KEY"],
            RAGStrategy.HYBRID_SEARCH_ENHANCED: ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"],
            RAGStrategy.AGENTIC_RAG: ["OPENAI_API_KEY", "SUPABASE_URL"],
            RAGStrategy.RERANKING: [],  # Uses local models, no API keys needed
        }
    
    def validate_dependencies(self) -> List[str]:
        """
        Validate that required environment variables are present for enabled strategies.
        
        Returns:
            List of missing dependency error messages
        """
        errors = []
        enabled_strategies = self.get_enabled_strategies()
        dependencies = self.get_strategy_dependencies()
        
        for strategy in enabled_strategies:
            required_vars = dependencies.get(strategy, [])
            for var in required_vars:
                if not os.getenv(var):
                    errors.append(
                        f"Strategy {strategy.value} requires {var} environment variable"
                    )
        
        return errors
    
    def __str__(self) -> str:
        """String representation of configuration."""
        enabled = self.get_enabled_strategies()
        if not enabled:
            return "StrategyConfig: No strategies enabled (baseline mode)"
        
        strategy_names = [s.value for s in enabled]
        return f"StrategyConfig: Enabled strategies: {', '.join(strategy_names)}"


class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        super().__init__(error_msg)


def load_and_validate_config() -> StrategyConfig:
    """
    Load configuration from environment and validate it.
    
    Returns:
        Validated StrategyConfig instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    config = StrategyConfig.from_environment()
    
    # Collect all validation errors
    validation_errors = config.validate()
    dependency_errors = config.validate_dependencies()
    all_errors = validation_errors + dependency_errors
    
    if all_errors:
        raise ConfigurationError(all_errors)
    
    return config


# Global configuration instance (loaded lazily)
_config: Optional[StrategyConfig] = None


def get_config() -> StrategyConfig:
    """
    Get the global configuration instance.
    
    Returns:
        StrategyConfig instance loaded from environment
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    global _config
    if _config is None:
        _config = load_and_validate_config()
    return _config


def reset_config():
    """Reset the global configuration (mainly for testing)."""
    global _config
    _config = None