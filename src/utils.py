"""
Utility functions for the Crawl4AI MCP server.
"""

import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
from supabase import create_client, Client
from urllib.parse import urlparse
import openai
import time

try:
    from .config import get_config, RAGStrategy
    from .code_extraction import extract_code_from_content, CodeExtractor
    from .database.models import CodeExample
except ImportError:
    try:
        from config import get_config, RAGStrategy
        from code_extraction import extract_code_from_content, CodeExtractor
        from database.models import CodeExample
    except ImportError:
        # Fallback for when config is not available (backward compatibility)
        get_config = None
        RAGStrategy = None
        extract_code_from_content = None
        CodeExtractor = None
        CodeExample = None


# --- added helper function to retry on rate limit errors ---
def _retry_with_backoff(fn, *args, max_retries=6, base_delay=2, **kwargs):
    """
    Wrap an OpenAI call and back-off on 429s.
    Retries at 2 s, 4 s, 8 s, … (max ~1 min).
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            print(
                f"Rate-limit hit, retrying in {delay}s … ({attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
            delay *= 2


# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")


def _should_use_contextual_embeddings() -> bool:
    """
    Determine if contextual embeddings should be used based on configuration.
    
    Checks both the new USE_CONTEXTUAL_EMBEDDINGS flag and the existing MODEL_CHOICE
    behavior for backward compatibility.
    
    Returns:
        bool: True if contextual embeddings should be used
    """
    # Check new configuration system first
    if get_config is not None:
        try:
            config = get_config()
            if config.use_contextual_embeddings:
                return True
        except Exception:
            # Fall through to legacy check if config fails
            pass
    
    # Backward compatibility: check MODEL_CHOICE
    model_choice = os.getenv("MODEL_CHOICE")
    return bool(model_choice)


def _get_contextual_model() -> Optional[str]:
    """
    Get the model to use for contextual embeddings.
    
    Returns:
        str: Model name to use, or None if contextual embeddings disabled
    """
    # Check new configuration system first
    if get_config is not None:
        try:
            config = get_config()
            if config.use_contextual_embeddings:
                return config.contextual_model
        except Exception:
            # Fall through to legacy check if config fails
            pass
    
    # Backward compatibility: use MODEL_CHOICE
    model_choice = os.getenv("MODEL_CHOICE")
    return model_choice if model_choice else None


def _should_use_agentic_rag() -> bool:
    """
    Determine if agentic RAG (code extraction) should be used based on configuration.
    
    Returns:
        bool: True if agentic RAG should be used
    """
    if get_config is not None:
        try:
            config = get_config()
            return config.use_agentic_rag
        except Exception:
            pass
    
    # Fallback to environment variable
    return os.getenv("USE_AGENTIC_RAG", "false").lower() in ("true", "1", "yes", "on")


def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.

    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables"
        )

    return create_client(url, key)


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.

    Args:
        texts: List of texts to create embeddings for

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []

    try:
        response = _retry_with_backoff(
            openai.embeddings.create,
            model="text-embedding-3-small",  # Hardcoding embedding model for now, will change this later to be more dynamic
            input=texts,
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error creating batch embeddings: {e}")
        # Return empty embeddings if there's an error
        return [[0.0] * 1536 for _ in range(len(texts))]


def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.

    Args:
        text: Text to create an embedding for

    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536


def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.

    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    # Determine which model to use based on configuration
    model_choice = _get_contextual_model()
    
    # Return original chunk if no model configured
    if not model_choice:
        return chunk, False

    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = _retry_with_backoff(
            openai.chat.completions.create,
            model=model_choice,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise technical summarizer.\n"
                        "Given a markdown chunk from a documentation page, return **one or two plain-English sentences** that capture:\n\n"
                        "• the main concept conveyed in the chunk  \n"
                        "• any APIs, commands, or parameter names mentioned  \n"
                        "• the chunk’s role in the wider doc set (e.g. “prerequisite”, “example”, “configuration reference”)\n\n"
                        "Avoid marketing language or personal opinions.\n"
                        "Retain original terminology and code style.\n"
                        "Output only the summary text—nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        # Extract the generated context
        context = response.choices[0].message.content.strip()

        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"

        return contextual_text, True

    except Exception as e:
        print(
            f"Error generating contextual embedding: {e}. Using original chunk instead."
        )
        return chunk, False


def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.

    Args:
        args: Tuple containing (url, content, full_document)

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)


def add_documents_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20,
) -> None:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.

    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))

    # Delete existing records for these URLs in a single operation
    try:
        if unique_urls:
            # Use the .in_() filter to delete all records with matching URLs
            client.table("crawled_pages").delete().in_("url", unique_urls).execute()
    except Exception as e:
        print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        # Fallback: delete records one by one
        for url in unique_urls:
            try:
                client.table("crawled_pages").delete().eq("url", url).execute()
            except Exception as inner_e:
                print(f"Error deleting record for URL {url}: {inner_e}")
                # Continue with the next URL even if one fails

    # Check if contextual embeddings should be used
    use_contextual_embeddings = _should_use_contextual_embeddings()

    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))

        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]

        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))

            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks and collect results
                future_to_idx = {
                    executor.submit(process_chunk_with_context, arg): idx
                    for idx, arg in enumerate(process_args)
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents.append(result)
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents.append(batch_contents[idx])

            # Sort results back into original order if needed
            if len(contextual_contents) != len(batch_contents):
                print(
                    f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}"
                )
                # Use original contents as fallback
                contextual_contents = batch_contents
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents

        # Create embeddings for the entire batch at once
        batch_embeddings = create_embeddings_batch(contextual_contents)

        batch_data = []
        for j in range(len(contextual_contents)):
            # Extract metadata fields
            chunk_size = len(contextual_contents[j])

            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": contextual_contents[j],  # Store original content
                "metadata": {"chunk_size": chunk_size, **batch_metadatas[j]},
                "embedding": batch_embeddings[
                    j
                ],  # Use embedding from contextual content
            }

            batch_data.append(data)

        # Insert batch into Supabase
        try:
            client.table("crawled_pages").insert(batch_data).execute()
        except Exception as e:
            print(f"Error inserting batch into Supabase: {e}")

    # Extract and store code examples if agentic RAG is enabled
    if _should_use_agentic_rag() and extract_code_from_content is not None:
        # Group content by URL to extract code from full documents
        url_to_content = {}
        for i, url in enumerate(urls):
            if url not in url_to_content:
                url_to_content[url] = []
            url_to_content[url].append(contents[i])
        
        # Process each URL's content for code extraction
        for url, url_contents in url_to_content.items():
            try:
                # Get source_id for this URL
                source_id = get_source_id_from_url(client, url)
                if source_id is None:
                    print(f"⚠️ No source_id found for URL {url}, skipping code extraction")
                    continue
                
                # Combine all chunks for this URL to get full document content
                full_content = "\n\n".join(url_contents)
                
                # Extract code examples from the full content
                extracted_codes = extract_code_from_content(full_content)
                
                if extracted_codes:
                    # Convert to format expected by add_code_examples_to_supabase
                    code_examples = []
                    for code in extracted_codes:
                        code_examples.append({
                            "code_content": code.code_content,
                            "summary": code.summary,
                            "programming_language": code.programming_language,
                            "complexity_score": code.complexity_score,
                        })
                    
                    # Store code examples with dual embeddings
                    add_code_examples_to_supabase(client, code_examples, source_id)
                    print(f"🔧 Extracted {len(code_examples)} code examples from {url}")
                
            except Exception as e:
                print(f"Error extracting code from URL {url}: {e}")
                continue


def add_code_examples_to_supabase(
    client: Client,
    code_examples: List[Dict[str, Any]],
    source_id: int,
    batch_size: int = 10,
) -> None:
    """
    Add extracted code examples to the Supabase code_examples table with dual embeddings.
    
    Args:
        client: Supabase client
        code_examples: List of extracted code examples from code_extraction.py
        source_id: The source_id from the sources table
        batch_size: Size of each batch for insertion
    """
    if not code_examples or CodeExample is None:
        return
    
    # Process in batches to avoid memory and API rate limit issues
    for i in range(0, len(code_examples), batch_size):
        batch_end = min(i + batch_size, len(code_examples))
        batch_examples = code_examples[i:batch_end]
        
        # Prepare texts for batch embedding generation
        code_contents = [example["code_content"] for example in batch_examples]
        summaries = [example["summary"] for example in batch_examples]
        
        # Create embeddings in batches
        try:
            code_embeddings = create_embeddings_batch(code_contents)
            summary_embeddings = create_embeddings_batch(summaries)
        except Exception as e:
            print(f"Error creating embeddings for code examples: {e}")
            continue
        
        # Prepare batch data for insertion
        batch_data = []
        for j, example in enumerate(batch_examples):
            # Create CodeExample model instance
            code_example = CodeExample(
                source_id=source_id,
                code_content=example["code_content"],
                summary=example["summary"],
                programming_language=example["programming_language"],
                complexity_score=example["complexity_score"],
                embedding=code_embeddings[j] if j < len(code_embeddings) else None,
                summary_embedding=summary_embeddings[j] if j < len(summary_embeddings) else None,
            )
            
            batch_data.append(code_example.to_dict())
        
        # Insert batch into Supabase
        try:
            client.table("code_examples").insert(batch_data).execute()
            print(f"✅ Inserted {len(batch_data)} code examples from source_id {source_id}")
        except Exception as e:
            print(f"Error inserting code examples batch into Supabase: {e}")


def get_source_id_from_url(client: Client, url: str) -> Optional[int]:
    """
    Get the source_id for a given URL from the sources table.
    
    Args:
        client: Supabase client
        url: The URL to look up
        
    Returns:
        source_id if found, None otherwise
    """
    try:
        response = client.table("sources").select("source_id").eq("url", url).limit(1).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]["source_id"]
    except Exception as e:
        print(f"Error getting source_id for URL {url}: {e}")
    
    return None


def search_documents(
    client: Client,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using hybrid search edge function.

    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter

    Returns:
        List of matching documents
    """
    # Build the parameter payload for the edge function
    params = {
        "query": query,
        "match_count": match_count,
    }
    if filter_metadata:
        params["filter"] = filter_metadata

    # Call the hybrid search edge function using direct requests
    # (workaround for Supabase Python client issue)
    try:
        import requests
        
        # Get connection details from client
        supabase_url = os.getenv("SUPABASE_URL", "http://localhost:54321")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not supabase_key:
            print("[search_documents] Missing SUPABASE_SERVICE_KEY")
            return []
        
        # Make direct request to edge function
        url = f"{supabase_url}/functions/v1/hybrid-search-crawled-pages"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {supabase_key}"
        }
        
        response = requests.post(url, json=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            rows = response.json()
        else:
            print(f"[search_documents] Edge function failed with status {response.status_code}: {response.text}")
            return []
            
    except Exception as e:
        print(f"[search_documents] Edge function failed: {e}")
        return []

    # Normalize output so the rest of your code stays unchanged
    out = []
    if isinstance(rows, list):
        for r in rows:
            out.append(
                {
                    "url": r["metadata"].get("url"),
                    "content": r["content"],
                    "metadata": r["metadata"],
                    # expose hybrid-ranking numbers
                    "rrf_score": r["rrf_score"],
                    "full_text_rank": r["full_text_rank"],
                    "semantic_rank": r["semantic_rank"],
                }
            )
    return out
