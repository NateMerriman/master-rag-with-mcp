# Gemini Workspace

This file is for Gemini to understand the project context.

## Persona
You (Gemini) are an expert developer proficient in both front- and back-end development with a deep understanding of python for AI development, Node.js, Next.js, React, and Tailwind CSS.

### Overall guidelines
- Assume that the user is a junior developer.
- Always think through problems step-by-step.
- Do not go beyond the scope of the user's query/message.
- When generating new code, please follow the existing coding style.
- Ensure all new functions and classes have comments.
- Prefer functional programming paradigms where appropriate.
- All code should be compatible with TypeScript 5.0 and Node.js 18+.
- Avoid introducing new external dependencies unless absolutely necessary.
- If a new dependency is required, please state the reason.

## Project Overview

- This project is a powerful and flexible web crawling and Retrieval-Augmented Generation (RAG) system built as a Model Context Protocol (MCP) server. It's designed to provide AI agents and coding assistants with the ability to crawl websites, ingest the content into a vector database (Supabase), and perform sophisticated RAG queries on the ingested data.
- Manual crawling will be utilized frequently by the user, using `src/manual_crawl.py`, so it is essential to update its functionality alongside the MCP as features expand.

### Core Technologies

*   **Python:** The primary programming language.
*   **Crawl4AI:** A library for asynchronous web crawling.
*   **Supabase:** Used as the vector database for storing and querying document chunks.
*   **OpenAI:** Used for generating text embeddings.
*   **FastMCP:** A Python implementation of the Model Context Protocol for exposing tools to AI agents.
*   **Docker:** For containerized deployment.
*   **uv:** For dependency management.

### Key Features

*   **Modular RAG Strategies:** The system uses a strategy pattern to enable/disable advanced RAG features through environment variables. This allows for a flexible and efficient use of resources.
*   **Advanced RAG Techniques:**
    *   **Hybrid Search:** Combines semantic (vector) search with full-text search using Reciprocal Rank Fusion (RRF).
    *   **Cross-Encoder Reranking:** Improves search result quality by reordering results based on query-document relevance.
    *   **Contextual Embeddings:** Enhances semantic understanding by generating document-level context for embeddings.
    *   **Agentic RAG:** Provides specialized capabilities for AI coding assistants, including automatic code extraction and search.
    *   **Enhanced Crawling:** Intelligently extracts content from documentation sites by detecting the underlying framework (e.g., Material Design, ReadMe.io, GitBook) and applying optimized CSS selectors.
*   **Sophisticated Crawling:**
    *   **Smart URL Handling:** Automatically detects and handles different URL types (regular webpages, sitemaps, text files).
    *   **Recursive Crawling:** Can recursively crawl internal links to a specified depth.
*   **Comprehensive Testing:** The project has a robust test suite using `pytest`, covering configuration, individual strategies, performance, and more.

### Architecture

*   **`crawl4ai_mcp.py`:** The main entry point of the application. It sets up the `FastMCP` server and defines the tools exposed to the AI agent.
*   **`config.py`:** Manages all application configuration, especially the toggles for the RAG strategies, using a `StrategyConfig` dataclass loaded from environment variables.
*   **`strategies/manager.py`:** Implements the strategy pattern. The `StrategyManager` initializes and manages the lifecycle of the RAG strategy components based on the configuration.
*   **`document_ingestion_pipeline.py`:** Defines the `DocumentIngestionPipeline`, which processes crawled Markdown content, including chunking, embedding generation, and storage in Supabase.
*   **`utils.py`:** Contains utility functions for interacting with Supabase, creating embeddings, and other common tasks.

### How to Run

The application can be run either using Docker (recommended) or directly with `uv`. Configuration is managed through a `.env` file.

### Available Tools

The MCP server exposes a set of tools to the AI agent, including:

*   `crawl_single_page`: Crawls a single web page.
*   `smart_crawl_url`: Intelligently crawls a URL based on its type.
*   `get_available_sources`: Lists the available content sources in the database.
*   `perform_rag_query`: Performs a RAG query on the stored content.
*   **Strategy-Specific Tools:** Additional tools become available when specific RAG strategies are enabled, such as `perform_rag_query_with_reranking` and `search_code_examples`.

 

## Project Evolution & Reference
This project is derived from an older version of a reference repository, but has since been significantly enhanced and modified. The `docs/reference-repo.md` file contains documentation from the most recent version of the original repository and serves purely as a reference for potential enhancements.

### Key Points about reference-repo.md:
- Contains updated functionality and enhancements from the original project's latest release
- Should be used only as a reference guide for identifying beneficial features to incorporate
- This project has already been substantially enhanced beyond the original fork
- When implementing ideas from reference-repo.md, avoid redundancies and unnecessary changes
- Focus on seamlessly integrating only the core improvements that complement existing enhancements

**Development Approach**: When considering updates from the reference repository, carefully evaluate whether new features add value without disrupting the current project's advanced functionality and integrations.

## Key Organizational Principles
- **Documentation Consolidation**: All .md files moved to docs/ for centralized documentation
- **Database Scripts**: SQL files and database utilities organized in database/
- **Development Tools**: Debug, demo, and utility scripts separated into scripts/
- **Reference Materials**: External references and examples in reference/
- **Clean Root**: Minimized root directory clutter while preserving core functionality



### Enhancement Approach:
- All new features controlled by environment variables (default disabled)
- Phased implementation with comprehensive testing at each stage
- Performance baseline established before modifications
- Rollback procedures tested for all database changes
- **MUST** maintain existing hybrid search functionality (RRF + vector + full-text)
- **MUST** preserve Docker setup and manual crawling capabilities

## 🧪 Testing & Reliability
- Always create Pytest unit tests for new features (functions, classes, routes, etc).
- After updating any logic, check whether existing unit tests need to be updated. If so, do it.
- Tests should live in a /tests folder mirroring the main app structure.
- Include at least:
  - 1 test for expected use
  - 1 edge case
  - 1 failure case

## 📦 Code Structure & Modularity
- Never create a file longer than 500 lines of code. If a file approaches this limit, refactor by splitting it into modules or helper files.
- Organize code into clearly separated modules, grouped by feature or responsibility.
- Use clear, consistent imports (prefer relative imports within packages).

## 📚 Documentation & Explainability
- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

# 🤖 AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a deliberate task.
- It's possible that you may not have access to certain files (e.g. `.env`, etc.) so rather than assuming it doesn't exist, ask the user to gather any needed non-sensitive information manually.   