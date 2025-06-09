"""
Unit tests for StrategyManager functionality.

Tests the strategy manager's component initialization, lifecycle management,
and tool availability logic.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from typing import Set

from src.config import StrategyConfig, RAGStrategy
from src.strategies.manager import (
    StrategyManager, 
    ComponentStatus, 
    ComponentInfo,
    initialize_strategy_manager,
    cleanup_strategy_manager,
    get_strategy_manager
)


class TestStrategyManager:
    """Test cases for StrategyManager class."""
    
    def test_init_with_no_strategies(self):
        """Test StrategyManager initialization with no strategies enabled."""
        config = StrategyConfig()
        manager = StrategyManager(config)
        
        assert manager.config == config
        assert len(manager.enabled_strategies) == 0
        assert not manager.is_initialized
        assert len(manager.components) == 0
    
    def test_init_with_strategies_enabled(self):
        """Test StrategyManager initialization with strategies enabled."""
        config = StrategyConfig(
            use_contextual_embeddings=True,
            use_reranking=True
        )
        manager = StrategyManager(config)
        
        expected_strategies = {RAGStrategy.CONTEXTUAL_EMBEDDINGS, RAGStrategy.RERANKING}
        assert manager.enabled_strategies == expected_strategies
        assert not manager.is_initialized
    
    def test_get_enabled_strategies(self):
        """Test getting enabled strategies."""
        config = StrategyConfig(use_agentic_rag=True)
        manager = StrategyManager(config)
        
        enabled = manager.get_enabled_strategies()
        assert enabled == {RAGStrategy.AGENTIC_RAG}
        
        # Ensure we get a copy, not the original
        enabled.add(RAGStrategy.RERANKING)
        assert manager.enabled_strategies == {RAGStrategy.AGENTIC_RAG}
    
    def test_is_strategy_enabled(self):
        """Test checking if specific strategies are enabled."""
        config = StrategyConfig(
            use_contextual_embeddings=True,
            use_reranking=False
        )
        manager = StrategyManager(config)
        
        assert manager.is_strategy_enabled(RAGStrategy.CONTEXTUAL_EMBEDDINGS)
        assert not manager.is_strategy_enabled(RAGStrategy.RERANKING)
        assert not manager.is_strategy_enabled(RAGStrategy.AGENTIC_RAG)
    
    def test_get_component_not_initialized(self):
        """Test getting component when not initialized."""
        config = StrategyConfig()
        manager = StrategyManager(config)
        
        assert manager.get_component("reranker") is None
        assert manager.get_component_status("reranker") == ComponentStatus.NOT_INITIALIZED
    
    def test_get_component_ready(self):
        """Test getting component when ready."""
        config = StrategyConfig()
        manager = StrategyManager(config)
        
        # Manually add a ready component
        mock_instance = Mock()
        manager.components["test_component"] = ComponentInfo(
            name="test_component",
            status=ComponentStatus.READY,
            instance=mock_instance
        )
        
        assert manager.get_component("test_component") == mock_instance
        assert manager.get_component_status("test_component") == ComponentStatus.READY
    
    def test_get_component_error_state(self):
        """Test getting component in error state."""
        config = StrategyConfig()
        manager = StrategyManager(config)
        
        manager.components["error_component"] = ComponentInfo(
            name="error_component",
            status=ComponentStatus.ERROR,
            error_message="Test error"
        )
        
        assert manager.get_component("error_component") is None
        assert manager.get_component_status("error_component") == ComponentStatus.ERROR


class TestStrategyManagerInitialization:
    """Test cases for strategy component initialization."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch("openai.OpenAI")
    def test_initialize_contextual_embeddings_success(self, mock_openai):
        """Test successful contextual embeddings initialization."""
        config = StrategyConfig(use_contextual_embeddings=True)
        manager = StrategyManager(config)
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        success = manager.initialize_components()
        
        assert success
        assert manager.is_initialized
        assert "contextual_embeddings" in manager.components
        assert manager.get_component_status("contextual_embeddings") == ComponentStatus.READY
        
        component = manager.get_component("contextual_embeddings")
        assert component is not None
        assert component["model"] == config.contextual_model
        assert component["client"] == mock_client
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_initialize_contextual_embeddings_missing_key(self):
        """Test contextual embeddings initialization with missing API key."""
        config = StrategyConfig(use_contextual_embeddings=True)
        manager = StrategyManager(config)
        
        success = manager.initialize_components()
        
        assert not success
        assert not manager.is_initialized
        assert len(manager.initialization_errors) > 0
        assert "OPENAI_API_KEY is required" in manager.initialization_errors[0]
    
    def test_initialize_reranking_success(self):
        """Test successful reranking component initialization."""
        config = StrategyConfig(use_reranking=True)
        manager = StrategyManager(config)
        
        # Mock reranker
        mock_reranker = Mock()
        
        with patch.object(manager, '_initialize_reranking_component') as mock_init:
            mock_init.return_value = True
            
            # Manually set up the component to simulate successful initialization
            manager.components["reranker"] = ComponentInfo(
                name="reranker",
                status=ComponentStatus.READY,
                instance=mock_reranker
            )
            
            success = manager.initialize_components()
            
            assert success
            assert manager.is_initialized
            assert "reranker" in manager.components
            assert manager.get_component_status("reranker") == ComponentStatus.READY
            assert manager.get_component("reranker") == mock_reranker
    
    def test_initialize_reranking_failure(self):
        """Test reranking component initialization failure."""
        config = StrategyConfig(use_reranking=True)
        manager = StrategyManager(config)
        
        with patch.object(manager, '_initialize_reranking_component') as mock_init:
            mock_init.return_value = False
            manager._initialization_errors = ["Failed to initialize reranking component: Test error"]
            
            success = manager.initialize_components()
            
            assert not success
            assert not manager.is_initialized
            assert len(manager.initialization_errors) > 0
            assert "Failed to initialize reranking component" in manager.initialization_errors[0]
    
    def test_initialize_agentic_rag(self):
        """Test agentic RAG component initialization."""
        config = StrategyConfig(use_agentic_rag=True)
        manager = StrategyManager(config)
        
        success = manager.initialize_components()
        
        assert success
        assert manager.is_initialized
        assert "agentic_rag" in manager.components
        assert manager.get_component_status("agentic_rag") == ComponentStatus.READY
        
        component = manager.get_component("agentic_rag")
        assert component is not None
        assert "code_extraction" in component["features"]
    
    def test_initialize_enhanced_search(self):
        """Test enhanced search component initialization."""
        config = StrategyConfig(use_hybrid_search_enhanced=True)
        manager = StrategyManager(config)
        
        success = manager.initialize_components()
        
        assert success
        assert manager.is_initialized
        assert "enhanced_search" in manager.components
        assert manager.get_component_status("enhanced_search") == ComponentStatus.READY
    
    def test_initialize_multiple_strategies(self):
        """Test initialization with multiple strategies enabled."""
        config = StrategyConfig(
            use_agentic_rag=True,
            use_hybrid_search_enhanced=True
        )
        manager = StrategyManager(config)
        
        success = manager.initialize_components()
        
        assert success
        assert manager.is_initialized
        assert len(manager.components) == 2
        assert "agentic_rag" in manager.components
        assert "enhanced_search" in manager.components
    
    def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        config = StrategyConfig()
        manager = StrategyManager(config)
        
        # First initialization
        success1 = manager.initialize_components()
        assert success1
        assert manager.is_initialized
        
        # Second initialization should return True but log warning
        with patch.object(manager, 'logger') as mock_logger:
            success2 = manager.initialize_components()
            assert success2
            mock_logger.warning.assert_called_once()


class TestStrategyManagerToolManagement:
    """Test cases for tool availability management."""
    
    def test_get_available_tools_baseline(self):
        """Test available tools with no strategies enabled."""
        config = StrategyConfig()
        manager = StrategyManager(config)
        
        tools = manager.get_available_tools()
        expected_baseline_tools = [
            "crawl_single_page",
            "smart_crawl_url", 
            "get_available_sources",
            "perform_rag_query"
        ]
        
        assert all(tool in tools for tool in expected_baseline_tools)
        assert len(tools) == len(expected_baseline_tools)
    
    def test_get_available_tools_agentic_rag(self):
        """Test available tools with agentic RAG enabled."""
        config = StrategyConfig(use_agentic_rag=True)
        manager = StrategyManager(config)
        
        tools = manager.get_available_tools()
        assert "search_code_examples" in tools
        assert "extract_code_from_content" in tools
    
    def test_get_available_tools_reranking(self):
        """Test available tools with reranking enabled."""
        config = StrategyConfig(use_reranking=True)
        manager = StrategyManager(config)
        
        tools = manager.get_available_tools()
        assert "perform_rag_query_with_reranking" in tools
    
    def test_get_available_tools_contextual_embeddings(self):
        """Test available tools with contextual embeddings enabled."""
        config = StrategyConfig(use_contextual_embeddings=True)
        manager = StrategyManager(config)
        
        tools = manager.get_available_tools()
        assert "perform_contextual_rag_query" in tools
    
    def test_get_available_tools_all_strategies(self):
        """Test available tools with all strategies enabled."""
        config = StrategyConfig(
            use_contextual_embeddings=True,
            use_hybrid_search_enhanced=True,
            use_agentic_rag=True,
            use_reranking=True
        )
        manager = StrategyManager(config)
        
        tools = manager.get_available_tools()
        
        # Should have baseline tools plus all strategy-specific tools
        expected_tools = [
            "crawl_single_page",
            "smart_crawl_url", 
            "get_available_sources",
            "perform_rag_query",
            "search_code_examples",
            "extract_code_from_content",
            "perform_rag_query_with_reranking",
            "perform_contextual_rag_query"
        ]
        
        assert all(tool in tools for tool in expected_tools)
    
    def test_should_tool_be_available(self):
        """Test checking if specific tools should be available."""
        config = StrategyConfig(use_agentic_rag=True)
        manager = StrategyManager(config)
        
        # Baseline tools should always be available
        assert manager.should_tool_be_available("perform_rag_query")
        
        # Strategy-specific tools
        assert manager.should_tool_be_available("search_code_examples")
        assert not manager.should_tool_be_available("perform_rag_query_with_reranking")
        
        # Non-existent tool
        assert not manager.should_tool_be_available("non_existent_tool")


class TestStrategyManagerCleanup:
    """Test cases for strategy manager cleanup."""
    
    def test_cleanup_components(self):
        """Test component cleanup."""
        config = StrategyConfig()
        manager = StrategyManager(config)
        
        # Add mock component with cleanup method
        mock_component = Mock()
        mock_component.cleanup = Mock()
        
        manager.components["test"] = ComponentInfo(
            name="test",
            status=ComponentStatus.READY,
            instance=mock_component
        )
        
        manager._is_initialized = True
        
        manager.cleanup()
        
        mock_component.cleanup.assert_called_once()
        assert len(manager.components) == 0
        assert not manager.is_initialized
    
    def test_cleanup_component_without_cleanup_method(self):
        """Test cleanup of component without cleanup method."""
        config = StrategyConfig()
        manager = StrategyManager(config)
        
        # Add mock component without cleanup method
        mock_component = Mock(spec=[])  # No cleanup method
        
        manager.components["test"] = ComponentInfo(
            name="test",
            status=ComponentStatus.READY,
            instance=mock_component
        )
        
        # Should not raise exception
        manager.cleanup()
        assert len(manager.components) == 0
    
    def test_cleanup_component_cleanup_exception(self):
        """Test cleanup when component cleanup raises exception."""
        config = StrategyConfig()
        manager = StrategyManager(config)
        
        # Add mock component with failing cleanup
        mock_component = Mock()
        mock_component.cleanup.side_effect = Exception("Cleanup failed")
        
        manager.components["test"] = ComponentInfo(
            name="test",
            status=ComponentStatus.READY,
            instance=mock_component
        )
        
        with patch.object(manager, 'logger') as mock_logger:
            manager.cleanup()
            
            # Should log warning but not raise exception
            mock_logger.warning.assert_called_once()
            assert len(manager.components) == 0


class TestStrategyManagerStatus:
    """Test cases for strategy manager status reporting."""
    
    def test_get_status_report_uninitialized(self):
        """Test status report for uninitialized manager."""
        config = StrategyConfig()
        manager = StrategyManager(config)
        
        status = manager.get_status_report()
        
        assert not status["is_initialized"]
        assert status["enabled_strategies"] == []
        assert status["components"] == {}
        assert status["initialization_errors"] == []
        assert len(status["available_tools"]) == 4  # Baseline tools only
    
    def test_get_status_report_initialized(self):
        """Test status report for initialized manager."""
        config = StrategyConfig(use_agentic_rag=True)
        manager = StrategyManager(config)
        
        success = manager.initialize_components()
        assert success
        
        status = manager.get_status_report()
        
        assert status["is_initialized"]
        assert "agentic_rag" in status["enabled_strategies"]
        assert "agentic_rag" in status["components"]
        assert status["components"]["agentic_rag"]["status"] == "ready"
        assert "search_code_examples" in status["available_tools"]
    
    def test_get_status_report_with_errors(self):
        """Test status report when initialization has errors."""
        config = StrategyConfig(use_contextual_embeddings=True)
        manager = StrategyManager(config)
        
        # Force an error
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            success = manager.initialize_components()
            assert not success
        
        status = manager.get_status_report()
        
        assert not status["is_initialized"]
        assert len(status["initialization_errors"]) > 0
        assert "contextual_embeddings" in status["components"]
        assert status["components"]["contextual_embeddings"]["status"] == "error"


class TestGlobalStrategyManager:
    """Test cases for global strategy manager functions."""
    
    def test_initialize_strategy_manager_success(self):
        """Test global strategy manager initialization."""
        config = StrategyConfig()
        
        manager = initialize_strategy_manager(config)
        
        assert manager is not None
        assert manager.is_initialized
        assert get_strategy_manager() == manager
    
    def test_initialize_strategy_manager_failure(self):
        """Test global strategy manager initialization failure."""
        config = StrategyConfig(use_contextual_embeddings=True)
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with pytest.raises(RuntimeError, match="Strategy manager initialization failed"):
                initialize_strategy_manager(config)
    
    def test_initialize_strategy_manager_replace_existing(self):
        """Test replacing existing global strategy manager."""
        config1 = StrategyConfig()
        config2 = StrategyConfig(use_agentic_rag=True)
        
        manager1 = initialize_strategy_manager(config1)
        
        with patch.object(manager1, 'cleanup') as mock_cleanup:
            manager2 = initialize_strategy_manager(config2)
            
            mock_cleanup.assert_called_once()
            assert get_strategy_manager() == manager2
            assert manager2 != manager1
    
    def test_cleanup_strategy_manager(self):
        """Test global strategy manager cleanup."""
        config = StrategyConfig()
        manager = initialize_strategy_manager(config)
        
        assert get_strategy_manager() is not None
        
        with patch.object(manager, 'cleanup') as mock_cleanup:
            cleanup_strategy_manager()
            
            mock_cleanup.assert_called_once()
            assert get_strategy_manager() is None
    
    def test_cleanup_strategy_manager_none(self):
        """Test cleanup when no global manager exists."""
        cleanup_strategy_manager()  # Should not raise exception
        assert get_strategy_manager() is None