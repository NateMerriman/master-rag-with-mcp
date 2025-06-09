"""
Tests for conditional tool registration and availability.

This module tests that tools appear and disappear based on strategy configuration,
and that appropriate error messages are returned when accessing disabled tools.
"""

import pytest
import json
import os
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Optional

from src.config import StrategyConfig, RAGStrategy
from src.strategies.manager import StrategyManager


@dataclass
class MockContext:
    """Mock context for testing tool functions."""
    strategy_manager: StrategyManager


@dataclass
class MockLifespanContext:
    """Mock lifespan context."""
    strategy_manager: StrategyManager
    supabase_client: Optional[object] = None


@dataclass
class MockRequestContext:
    """Mock request context."""
    lifespan_context: MockLifespanContext


@dataclass
class MockFullContext:
    """Mock full context for tool testing."""
    request_context: MockRequestContext


class TestConditionalToolRegistration:
    """Test conditional tool registration functionality."""
    
    def test_strategy_manager_available_tools_baseline(self):
        """Test available tools with no strategies enabled."""
        config = StrategyConfig()  # All strategies disabled by default
        manager = StrategyManager(config)
        
        available_tools = manager.get_available_tools()
        
        # Should only have base tools
        expected_base_tools = [
            "crawl_single_page",
            "smart_crawl_url", 
            "get_available_sources",
            "perform_rag_query",
            "get_strategy_status"
        ]
        
        assert set(available_tools) == set(expected_base_tools)
        
        # Should not have strategy-specific tools
        strategy_tools = [
            "search_code_examples",
            "perform_rag_query_with_reranking", 
            "perform_contextual_rag_query"
        ]
        
        for tool in strategy_tools:
            assert tool not in available_tools
    
    def test_strategy_manager_available_tools_agentic_rag(self):
        """Test available tools with agentic RAG enabled."""
        config = StrategyConfig(use_agentic_rag=True)
        manager = StrategyManager(config)
        
        available_tools = manager.get_available_tools()
        
        # Should have base tools plus agentic RAG tools
        assert "search_code_examples" in available_tools
        assert "perform_rag_query_with_reranking" not in available_tools
        assert "perform_contextual_rag_query" not in available_tools
    
    def test_strategy_manager_available_tools_reranking(self):
        """Test available tools with reranking enabled."""
        config = StrategyConfig(use_reranking=True)
        manager = StrategyManager(config)
        
        available_tools = manager.get_available_tools()
        
        # Should have base tools plus reranking tools
        assert "perform_rag_query_with_reranking" in available_tools
        assert "search_code_examples" not in available_tools
        assert "perform_contextual_rag_query" not in available_tools
    
    def test_strategy_manager_available_tools_contextual_embeddings(self):
        """Test available tools with contextual embeddings enabled."""
        config = StrategyConfig(use_contextual_embeddings=True)
        manager = StrategyManager(config)
        
        available_tools = manager.get_available_tools()
        
        # Should have base tools plus contextual embeddings tools
        assert "perform_contextual_rag_query" in available_tools
        assert "search_code_examples" not in available_tools
        assert "perform_rag_query_with_reranking" not in available_tools
    
    def test_strategy_manager_available_tools_multiple_strategies(self):
        """Test available tools with multiple strategies enabled."""
        config = StrategyConfig(
            use_agentic_rag=True,
            use_reranking=True,
            use_contextual_embeddings=True
        )
        manager = StrategyManager(config)
        
        available_tools = manager.get_available_tools()
        
        # Should have all tools
        assert "search_code_examples" in available_tools
        assert "perform_rag_query_with_reranking" in available_tools
        assert "perform_contextual_rag_query" in available_tools
    
    def test_should_tool_be_available(self):
        """Test tool availability checking."""
        config = StrategyConfig(use_agentic_rag=True)
        manager = StrategyManager(config)
        
        # Base tools should always be available
        assert manager.should_tool_be_available("crawl_single_page")
        assert manager.should_tool_be_available("perform_rag_query")
        
        # Strategy-specific tools should be conditional
        assert manager.should_tool_be_available("search_code_examples")
        assert not manager.should_tool_be_available("perform_rag_query_with_reranking")
        assert not manager.should_tool_be_available("perform_contextual_rag_query")


class TestConditionalToolDecorator:
    """Test the conditional tool decorator functionality."""
    
    @pytest.fixture
    def mock_context_with_strategy(self):
        """Create mock context with strategy manager."""
        config = StrategyConfig(use_agentic_rag=True)
        manager = StrategyManager(config)
        
        lifespan_ctx = MockLifespanContext(strategy_manager=manager)
        request_ctx = MockRequestContext(lifespan_context=lifespan_ctx)
        return MockFullContext(request_context=request_ctx)
    
    @pytest.fixture 
    def mock_context_no_strategy(self):
        """Create mock context without strategies enabled."""
        config = StrategyConfig()  # No strategies enabled
        manager = StrategyManager(config)
        
        lifespan_ctx = MockLifespanContext(strategy_manager=manager)
        request_ctx = MockRequestContext(lifespan_context=lifespan_ctx)
        return MockFullContext(request_context=request_ctx)
    
    @pytest.mark.asyncio
    async def test_conditional_tool_allowed(self, mock_context_with_strategy):
        """Test tool execution when strategy is enabled."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.crawl4ai_mcp import conditional_tool
        from src.config import RAGStrategy
        
        @conditional_tool("search_code_examples", [RAGStrategy.AGENTIC_RAG])
        async def test_tool(ctx, query: str):
            return json.dumps({"success": True, "query": query})
        
        result = await test_tool(mock_context_with_strategy, "test query")
        response = json.loads(result)
        
        assert response["success"] is True
        assert response["query"] == "test query"
    
    @pytest.mark.asyncio
    async def test_conditional_tool_blocked(self, mock_context_no_strategy):
        """Test tool blocking when strategy is disabled."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.crawl4ai_mcp import conditional_tool
        from src.config import RAGStrategy
        
        @conditional_tool("search_code_examples", [RAGStrategy.AGENTIC_RAG])
        async def test_tool(ctx, query: str):
            return json.dumps({"success": True, "query": query})
        
        result = await test_tool(mock_context_no_strategy, "test query")
        response = json.loads(result)
        
        assert response["success"] is False
        assert "requires strategies" in response["error"]
        assert response["tool"] == "search_code_examples"
        assert RAGStrategy.AGENTIC_RAG.value in response["required_strategies"]
    
    @pytest.mark.asyncio
    async def test_conditional_tool_multiple_strategies(self, mock_context_no_strategy):
        """Test tool with multiple required strategies."""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.crawl4ai_mcp import conditional_tool
        from src.config import RAGStrategy
        
        @conditional_tool("advanced_tool", [RAGStrategy.AGENTIC_RAG, RAGStrategy.RERANKING])
        async def test_tool(ctx, query: str):
            return json.dumps({"success": True, "query": query})
        
        result = await test_tool(mock_context_no_strategy, "test query")
        response = json.loads(result)
        
        assert response["success"] is False
        assert "requires strategies" in response["error"]
        assert RAGStrategy.AGENTIC_RAG.value in response["required_strategies"]
        assert RAGStrategy.RERANKING.value in response["required_strategies"]


class TestToolErrorHandling:
    """Test error handling for disabled tools."""
    
    def test_error_message_format(self):
        """Test that error messages are properly formatted."""
        config = StrategyConfig()  # No strategies enabled
        manager = StrategyManager(config)
        
        # Test basic availability check
        assert not manager.should_tool_be_available("search_code_examples")
        assert not manager.should_tool_be_available("perform_rag_query_with_reranking")
        
        # Base tools should still be available
        assert manager.should_tool_be_available("crawl_single_page")
        assert manager.should_tool_be_available("get_strategy_status")
    
    def test_tool_configuration_guide(self):
        """Test that configuration guidance is available."""
        config = StrategyConfig()
        manager = StrategyManager(config)
        
        status_report = manager.get_status_report()
        
        # Should include enabled strategies (empty)
        assert "enabled_strategies" in status_report
        assert status_report["enabled_strategies"] == []
        
        # Should include available tools
        assert "available_tools" in status_report
        base_tools = status_report["available_tools"]
        assert "crawl_single_page" in base_tools
        assert "search_code_examples" not in base_tools


class TestToolDocumentation:
    """Test dynamic tool documentation based on configuration."""
    
    def test_tool_descriptions_available(self):
        """Test that tool descriptions are provided."""
        config = StrategyConfig(use_agentic_rag=True, use_reranking=True)
        manager = StrategyManager(config)
        
        available_tools = manager.get_available_tools()
        
        # Should have appropriate tools based on enabled strategies
        assert "search_code_examples" in available_tools
        assert "perform_rag_query_with_reranking" in available_tools
        assert "perform_contextual_rag_query" not in available_tools
        
        # Check status report format
        status_report = manager.get_status_report()
        assert "enabled_strategies" in status_report
        assert "agentic_rag" in status_report["enabled_strategies"]
        assert "reranking" in status_report["enabled_strategies"]
    
    def test_configuration_validation(self):
        """Test that configuration validation works."""
        # Valid configuration
        config = StrategyConfig(use_agentic_rag=True)
        errors = config.validate()
        assert len(errors) == 0
        
        # Invalid configuration
        config = StrategyConfig(
            contextual_model="",  # Empty model name
            max_reranking_results=-1  # Invalid value
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("CONTEXTUAL_MODEL cannot be empty" in error for error in errors)
        assert any("MAX_RERANKING_RESULTS must be positive" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__])