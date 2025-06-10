import pytest
import json
from unittest.mock import MagicMock, patch, ANY
import os

# Set dummy env vars for testing before imports, allowing utils to load
os.environ["SUPABASE_URL"] = "http://dummy.url"
os.environ["SUPABASE_SERVICE_KEY"] = "dummy.key"
os.environ["USE_AGENTIC_RAG"] = "true"

from src.utils import add_documents_to_supabase
from supabase import Client

# Sample content with code blocks to be used in tests
SAMPLE_URL = "https://example.com/docs/python-guide"
SAMPLE_CONTENT = """
# Python Guide

Here is a simple function in Python.

```python
def hello_world():
    print("Hello, World!")
```

And here is some JavaScript code.

```javascript
function greet() {
    console.log('Greetings, Program!');
}
```

This is the end of the document.
"""


@pytest.fixture
def mock_supabase_client_tuple():
    """Provides a mock Supabase client with specific mocks for each table."""
    mock_client = MagicMock(spec=Client)

    # Mocks for different tables, returned by the router
    mock_crawled_pages_table = MagicMock()
    mock_sources_table = MagicMock()
    mock_code_examples_table = MagicMock()

    def table_router(table_name):
        if table_name == "crawled_pages":
            return mock_crawled_pages_table
        if table_name == "sources":
            return mock_sources_table
        if table_name == "code_examples":
            return mock_code_examples_table
        return MagicMock()

    mock_client.table.side_effect = table_router

    # Mock the get_or_create_source logic, which interacts with the 'sources' table
    mock_sources_table.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []  # No existing source
    mock_sources_table.upsert.return_value.execute.return_value.data = [
        {"source_id": 1}
    ]  # Return a new source_id

    return mock_client, mock_code_examples_table


@patch("src.utils.create_embeddings_batch")
def test_end_to_end_code_extraction_flow(
    mock_create_embeddings, mock_supabase_client_tuple
):
    """
    Tests the full flow from adding a document to extracting and storing code examples.
    """
    # --- Arrange ---
    mock_client, mock_code_examples_table = mock_supabase_client_tuple

    # Mock the return value of the embedding function. It gets called multiple times.
    mock_create_embeddings.side_effect = [
        [[0.1] * 1536],  # Embedding for the main document content
        [[0.2] * 1536, [0.3] * 1536],  # Embeddings for the two code blocks
    ]

    # Use a patch for get_supabase_client to inject our mock during the test
    with patch("src.utils.get_supabase_client", return_value=mock_client):
        # --- Act ---
        # This function orchestrates the entire process we want to test.
        add_documents_to_supabase(
            client=mock_client,
            urls=[SAMPLE_URL],
            chunk_numbers=[1],
            contents=[SAMPLE_CONTENT],
            metadatas=[{"source": "test"}],
            url_to_full_document={SAMPLE_URL: SAMPLE_CONTENT},
        )

    # --- Assert ---
    # 1. Assert that the `code_examples` table's insert method was called once.
    mock_code_examples_table.insert.assert_called_once()

    # 2. Get the list of records that were passed to the insert method.
    final_insert_args = mock_code_examples_table.insert.call_args[0][0]

    # 3. Verify the correct number of code blocks were extracted and prepared for insertion.
    assert len(final_insert_args) == 2, "Expected two code blocks to be inserted"

    # 4. Check the first code block (Python) for correctness.
    python_record = final_insert_args[0]
    assert python_record["source_id"] == 1
    assert python_record["url"] == SAMPLE_URL
    assert python_record["chunk_number"] == 1
    assert "def hello_world():" in python_record["content"]
    assert python_record["programming_language"] == "python"
    assert isinstance(python_record["complexity_score"], int)
    assert python_record["embedding"] == [0.2] * 1536

    metadata = json.loads(python_record["metadata"])
    assert "start_line" in metadata
    assert metadata["block_type"] == "fenced"

    # 5. Check the second code block (JavaScript) for correctness.
    js_record = final_insert_args[1]
    assert js_record["source_id"] == 1
    assert js_record["url"] == SAMPLE_URL
    assert js_record["chunk_number"] == 2
    assert "function greet()" in js_record["content"]
    assert js_record["programming_language"] == "javascript"
    assert js_record["embedding"] == [0.3] * 1536

    metadata = json.loads(js_record["metadata"])
    assert "start_line" in metadata
    assert metadata["block_type"] == "fenced"
