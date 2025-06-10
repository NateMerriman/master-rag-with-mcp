"""
Configuration management for RAG strategy toggles and settings.

This module provides centralized configuration management for all RAG enhancement
strategies, with environment variable loading, validation, and error handling.
"""

import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv

    # Try to load .env file from current directory or parent directory
    dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=False)  # Don't override existing env vars
    else:
        # Fallback: try loading from current working directory
        if os.path.exists(".env"):
            load_dotenv(".env", override=False)
except ImportError:
    # python-dotenv not available, continue with system environment variables
    pass


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
    contextual_model: str = "gpt-4o-mini-2024-07-18"
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Contextual embedding configuration
    use_adaptive_contextual_prompts: bool = True  # Enable content-type-aware prompting
    contextual_content_type_detection: bool = (
        True  # Enable automatic content type detection
    )
    contextual_include_content_type_tag: bool = (
        True  # Include [TYPE] tag in contextual text
    )

    # Code summary enhancement configuration
    use_enhanced_code_summaries: bool = False  # Enable template-based code summaries
    use_progressive_code_summarization: bool = False  # Enable two-stage summarization
    use_domain_specific_code_prompts: bool = True  # Enable content-aware code prompts
    use_code_intent_detection: bool = True  # Enable code intent recognition
    use_retrieval_optimized_code_summaries: bool = (
        False  # Enable search-optimized summaries
    )

    # Code summary configuration
    code_summary_style: str = "practical"  # "practical", "academic", "tutorial"
    code_summary_max_context_chars: int = (
        300  # Max context characters for code summaries
    )
    code_summary_include_complexity: bool = True  # Include complexity in summaries

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
            use_agentic_rag=str_to_bool(os.getenv("USE_AGENTIC_RAG", "false")),
            use_reranking=str_to_bool(os.getenv("USE_RERANKING", "false")),
            contextual_model=os.getenv("CONTEXTUAL_MODEL", "gpt-4o-mini-2024-07-18"),
            reranking_model=os.getenv(
                "RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            use_adaptive_contextual_prompts=str_to_bool(
                os.getenv("USE_ADAPTIVE_CONTEXTUAL_PROMPTS", "true")
            ),
            contextual_content_type_detection=str_to_bool(
                os.getenv("CONTEXTUAL_CONTENT_TYPE_DETECTION", "true")
            ),
            contextual_include_content_type_tag=str_to_bool(
                os.getenv("CONTEXTUAL_INCLUDE_CONTENT_TYPE_TAG", "true")
            ),
            # Code summary enhancement configuration
            use_enhanced_code_summaries=str_to_bool(
                os.getenv("USE_ENHANCED_CODE_SUMMARIES", "false")
            ),
            use_progressive_code_summarization=str_to_bool(
                os.getenv("USE_PROGRESSIVE_CODE_SUMMARIZATION", "false")
            ),
            use_domain_specific_code_prompts=str_to_bool(
                os.getenv("USE_DOMAIN_SPECIFIC_CODE_PROMPTS", "true")
            ),
            use_code_intent_detection=str_to_bool(
                os.getenv("USE_CODE_INTENT_DETECTION", "true")
            ),
            use_retrieval_optimized_code_summaries=str_to_bool(
                os.getenv("USE_RETRIEVAL_OPTIMIZED_CODE_SUMMARIES", "false")
            ),
            code_summary_style=os.getenv("CODE_SUMMARY_STYLE", "practical"),
            code_summary_max_context_chars=int(
                os.getenv("CODE_SUMMARY_MAX_CONTEXT_CHARS", "300")
            ),
            code_summary_include_complexity=str_to_bool(
                os.getenv("CODE_SUMMARY_INCLUDE_COMPLEXITY", "true")
            ),
            max_reranking_results=int(os.getenv("MAX_RERANKING_RESULTS", "20")),
            reranking_timeout_ms=int(os.getenv("RERANKING_TIMEOUT_MS", "500")),
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

        # Code summary configuration validation
        valid_summary_styles = ["practical", "academic", "tutorial"]
        if self.code_summary_style not in valid_summary_styles:
            errors.append(
                f"CODE_SUMMARY_STYLE must be one of: {', '.join(valid_summary_styles)}"
            )

        if self.code_summary_max_context_chars <= 0:
            errors.append("CODE_SUMMARY_MAX_CONTEXT_CHARS must be positive")

        if self.code_summary_max_context_chars > 1000:
            errors.append(
                "CODE_SUMMARY_MAX_CONTEXT_CHARS should not exceed 1000 for performance reasons"
            )

        # Enhanced code summaries validation
        if self.use_enhanced_code_summaries and not self.use_agentic_rag:
            errors.append(
                "USE_ENHANCED_CODE_SUMMARIES requires USE_AGENTIC_RAG to be enabled"
            )

        # Remove the MODEL_CHOICE requirement since we now have CONTEXTUAL_MODEL
        # Contextual embeddings validation is handled by dependency checks

        return errors

    def get_strategy_dependencies(self) -> Dict[RAGStrategy, List[str]]:
        """
        Get environment variable dependencies for each strategy.

        Returns:
            Dictionary mapping strategies to required environment variables
        """
        return {
            RAGStrategy.CONTEXTUAL_EMBEDDINGS: ["OPENAI_API_KEY"],
            RAGStrategy.HYBRID_SEARCH_ENHANCED: [
                "SUPABASE_URL",
                "SUPABASE_SERVICE_KEY",
            ],
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
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
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
