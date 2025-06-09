"""
Strategy Manager for RAG enhancement strategies.

This module provides centralized management of RAG strategies, including component
initialization, resource management, and runtime validation.
"""

import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum

try:
    from ..config import StrategyConfig, RAGStrategy, ConfigurationError
except ImportError:
    # Fallback for when running outside package context
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import StrategyConfig, RAGStrategy, ConfigurationError


class ComponentStatus(Enum):
    """Status of strategy components."""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


@dataclass
class ComponentInfo:
    """Information about a strategy component."""
    name: str
    status: ComponentStatus
    instance: Optional[Any] = None
    error_message: Optional[str] = None


class StrategyManager:
    """
    Manages RAG strategy components and their lifecycle.
    
    The StrategyManager is responsible for:
    - Initializing only the components needed for enabled strategies
    - Managing component lifecycle and resources
    - Providing access to strategy-specific functionality
    - Validating strategy combinations and dependencies
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the StrategyManager with a configuration.
        
        Args:
            config: StrategyConfig instance with enabled strategies
        """
        self.config = config
        self.enabled_strategies = config.get_enabled_strategies()
        self.components: Dict[str, ComponentInfo] = {}
        self.logger = logging.getLogger(__name__)
        
        # Track initialization state
        self._is_initialized = False
        self._initialization_errors: List[str] = []
    
    @property
    def is_initialized(self) -> bool:
        """Check if the manager has been initialized."""
        return self._is_initialized
    
    @property
    def initialization_errors(self) -> List[str]:
        """Get any errors that occurred during initialization."""
        return self._initialization_errors.copy()
    
    def get_enabled_strategies(self) -> Set[RAGStrategy]:
        """Get the set of enabled strategies."""
        return self.enabled_strategies.copy()
    
    def is_strategy_enabled(self, strategy: RAGStrategy) -> bool:
        """Check if a specific strategy is enabled."""
        return strategy in self.enabled_strategies
    
    def get_component(self, component_name: str) -> Optional[Any]:
        """
        Get a component instance by name.
        
        Args:
            component_name: Name of the component to retrieve
            
        Returns:
            Component instance if available and ready, None otherwise
        """
        component_info = self.components.get(component_name)
        if component_info and component_info.status == ComponentStatus.READY:
            return component_info.instance
        return None
    
    def get_component_status(self, component_name: str) -> ComponentStatus:
        """
        Get the status of a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            ComponentStatus of the component
        """
        component_info = self.components.get(component_name)
        return component_info.status if component_info else ComponentStatus.NOT_INITIALIZED
    
    def initialize_components(self) -> bool:
        """
        Initialize all components for enabled strategies.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._is_initialized:
            self.logger.warning("StrategyManager already initialized")
            return True
        
        self.logger.info(f"Initializing components for strategies: {[s.value for s in self.enabled_strategies]}")
        
        # Initialize components based on enabled strategies
        success = True
        
        if RAGStrategy.RERANKING in self.enabled_strategies:
            success &= self._initialize_reranking_component()
        
        if RAGStrategy.CONTEXTUAL_EMBEDDINGS in self.enabled_strategies:
            success &= self._initialize_contextual_embeddings_component()
        
        if RAGStrategy.AGENTIC_RAG in self.enabled_strategies:
            success &= self._initialize_agentic_rag_component()
        
        if RAGStrategy.HYBRID_SEARCH_ENHANCED in self.enabled_strategies:
            success &= self._initialize_enhanced_search_component()
        
        self._is_initialized = success
        
        if success:
            self.logger.info("All strategy components initialized successfully")
        else:
            self.logger.error(f"Component initialization failed: {self._initialization_errors}")
        
        return success
    
    def _initialize_reranking_component(self) -> bool:
        """Initialize the reranking component."""
        component_name = "reranker"
        self.components[component_name] = ComponentInfo(
            name=component_name,
            status=ComponentStatus.INITIALIZING
        )
        
        try:
            # Import and initialize reranking component
            try:
                from ..reranking import get_reranker
            except ImportError:
                from reranking import get_reranker
            
            reranker = get_reranker()
            
            self.components[component_name] = ComponentInfo(
                name=component_name,
                status=ComponentStatus.READY,
                instance=reranker
            )
            
            self.logger.info(f"Reranking component initialized with model: {self.config.reranking_model}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize reranking component: {str(e)}"
            self.components[component_name] = ComponentInfo(
                name=component_name,
                status=ComponentStatus.ERROR,
                error_message=error_msg
            )
            self._initialization_errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def _initialize_contextual_embeddings_component(self) -> bool:
        """Initialize the contextual embeddings component."""
        component_name = "contextual_embeddings"
        self.components[component_name] = ComponentInfo(
            name=component_name,
            status=ComponentStatus.INITIALIZING
        )
        
        try:
            # Contextual embeddings functionality is already integrated in utils.py
            # We just need to validate that the OpenAI client is available
            import openai
            import os
            
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required for contextual embeddings")
            
            # Test that we can create a client (don't make actual API calls here)
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            self.components[component_name] = ComponentInfo(
                name=component_name,
                status=ComponentStatus.READY,
                instance={"model": self.config.contextual_model, "client": client}
            )
            
            self.logger.info(f"Contextual embeddings component initialized with model: {self.config.contextual_model}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize contextual embeddings component: {str(e)}"
            self.components[component_name] = ComponentInfo(
                name=component_name,
                status=ComponentStatus.ERROR,
                error_message=error_msg
            )
            self._initialization_errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def _initialize_agentic_rag_component(self) -> bool:
        """Initialize the agentic RAG component."""
        component_name = "agentic_rag"
        self.components[component_name] = ComponentInfo(
            name=component_name,
            status=ComponentStatus.INITIALIZING
        )
        
        try:
            # For now, we'll mark as ready since code extraction is already implemented
            # In future tasks, this will initialize code search tools and extraction pipeline
            
            self.components[component_name] = ComponentInfo(
                name=component_name,
                status=ComponentStatus.READY,
                instance={"status": "ready", "features": ["code_extraction", "code_search"]}
            )
            
            self.logger.info("Agentic RAG component initialized (code extraction ready)")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize agentic RAG component: {str(e)}"
            self.components[component_name] = ComponentInfo(
                name=component_name,
                status=ComponentStatus.ERROR,
                error_message=error_msg
            )
            self._initialization_errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def _initialize_enhanced_search_component(self) -> bool:
        """Initialize the enhanced hybrid search component."""
        component_name = "enhanced_search"
        self.components[component_name] = ComponentInfo(
            name=component_name,
            status=ComponentStatus.INITIALIZING
        )
        
        try:
            # Enhanced search builds on existing hybrid search functionality
            # For now, we'll mark as ready since the foundation is there
            
            self.components[component_name] = ComponentInfo(
                name=component_name,
                status=ComponentStatus.READY,
                instance={"status": "ready", "features": ["enhanced_rrf", "query_expansion"]}
            )
            
            self.logger.info("Enhanced search component initialized")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize enhanced search component: {str(e)}"
            self.components[component_name] = ComponentInfo(
                name=component_name,
                status=ComponentStatus.ERROR,
                error_message=error_msg
            )
            self._initialization_errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of tools that should be available based on enabled strategies.
        
        Returns:
            List of tool names that should be registered
        """
        available_tools = [
            # Base tools always available
            "crawl_single_page",
            "smart_crawl_url", 
            "get_available_sources",
            "perform_rag_query"
        ]
        
        # Add strategy-specific tools
        if self.is_strategy_enabled(RAGStrategy.AGENTIC_RAG):
            available_tools.extend([
                "search_code_examples",
                "extract_code_from_content"
            ])
        
        if self.is_strategy_enabled(RAGStrategy.RERANKING):
            available_tools.extend([
                "perform_rag_query_with_reranking"
            ])
        
        if self.is_strategy_enabled(RAGStrategy.CONTEXTUAL_EMBEDDINGS):
            available_tools.extend([
                "perform_contextual_rag_query"
            ])
        
        return available_tools
    
    def should_tool_be_available(self, tool_name: str) -> bool:
        """
        Check if a tool should be available based on current configuration.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool should be available, False otherwise
        """
        return tool_name in self.get_available_tools()
    
    def cleanup(self):
        """Clean up resources used by strategy components."""
        self.logger.info("Cleaning up strategy components")
        
        for component_name, component_info in self.components.items():
            if component_info.status == ComponentStatus.READY and component_info.instance:
                try:
                    # If the component has a cleanup method, call it
                    if hasattr(component_info.instance, 'cleanup'):
                        component_info.instance.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up component {component_name}: {e}")
        
        self.components.clear()
        self._is_initialized = False
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive status report of all components.
        
        Returns:
            Dictionary with status information
        """
        return {
            "is_initialized": self._is_initialized,
            "enabled_strategies": [s.value for s in self.enabled_strategies],
            "components": {
                name: {
                    "status": info.status.value,
                    "error_message": info.error_message
                }
                for name, info in self.components.items()
            },
            "available_tools": self.get_available_tools(),
            "initialization_errors": self._initialization_errors
        }


# Global strategy manager instance
_strategy_manager: Optional[StrategyManager] = None


def get_strategy_manager() -> Optional[StrategyManager]:
    """
    Get the global strategy manager instance.
    
    Returns:
        StrategyManager instance if initialized, None otherwise
    """
    return _strategy_manager


def initialize_strategy_manager(config: StrategyConfig) -> StrategyManager:
    """
    Initialize the global strategy manager with the given configuration.
    
    Args:
        config: StrategyConfig instance
        
    Returns:
        Initialized StrategyManager instance
        
    Raises:
        RuntimeError: If initialization fails
    """
    global _strategy_manager
    
    if _strategy_manager is not None:
        logging.warning("Strategy manager already initialized, replacing...")
        _strategy_manager.cleanup()
    
    _strategy_manager = StrategyManager(config)
    
    if not _strategy_manager.initialize_components():
        errors = _strategy_manager.initialization_errors
        raise RuntimeError(f"Strategy manager initialization failed: {errors}")
    
    return _strategy_manager


def cleanup_strategy_manager():
    """Clean up the global strategy manager."""
    global _strategy_manager
    
    if _strategy_manager is not None:
        _strategy_manager.cleanup()
        _strategy_manager = None