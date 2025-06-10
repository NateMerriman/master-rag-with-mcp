print("--- LOADING LATEST VERSION OF UTILS.PY ---")

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


def _detect_content_type(content: str, url: str = "") -> str:
    """
    Detect the type of content to apply appropriate contextual embedding strategy.

    Args:
        content: The text content to analyze
        url: Optional URL for additional context

    Returns:
        Content type category as string
    """
    content_lower = content.lower()
    url_lower = url.lower()

    # Technical documentation indicators
    tech_indicators = [
        "api",
        "function",
        "class",
        "method",
        "parameter",
        "returns",
        "example:",
        "installation",
        "configuration",
        "setup",
        "import",
        "usage",
        "```",
        "endpoint",
        "request",
        "response",
        "library",
        "framework",
    ]

    # Academic/research indicators
    academic_indicators = [
        "abstract",
        "methodology",
        "hypothesis",
        "conclusion",
        "references",
        "study",
        "research",
        "analysis",
        "findings",
        "literature review",
        "experiment",
        "data",
        "results",
        "discussion",
        "journal",
    ]

    # Forum/discussion indicators
    forum_indicators = [
        "posted by",
        "reply",
        "comment",
        "question:",
        "answer:",
        "solved",
        "upvoted",
        "downvoted",
        "thread",
        "forum",
        "discussion",
        "feedback",
        "opinion",
        "experience",
        "issue",
        "problem",
        "help",
    ]

    # News/article indicators
    news_indicators = [
        "published",
        "editor",
        "reporter",
        "breaking",
        "update",
        "announced",
        "statement",
        "press release",
        "interview",
        "according to",
        "sources say",
        "reuters",
        "bloomberg",
        "cnn",
    ]

    # Blog/opinion indicators
    blog_indicators = [
        "i think",
        "in my opinion",
        "personally",
        "i believe",
        "my experience",
        "thoughts on",
        "reflection",
        "perspective",
        "author:",
        "written by",
        "blog",
        "post",
    ]

    # Social media indicators (non-forum)
    social_media_indicators = [
        "shared",
        "retweet",
        "like",
        "follow",
        "followers",
        "trending",
        "hashtag",
        "#",
        "@",
        "posted on",
        "social media",
        "tweet",
        "linkedin post",
        "instagram",
        "facebook",
        "twitter",
        "thread",
        "viral",
        "engagement",
        "influencer",
    ]

    # Legal document indicators
    legal_indicators = [
        "whereas",
        "therefore",
        "hereby",
        "pursuant to",
        "section",
        "subsection",
        "clause",
        "contract",
        "agreement",
        "terms and conditions",
        "legal",
        "law",
        "statute",
        "regulation",
        "court",
        "jurisdiction",
        "plaintiff",
        "defendant",
        "attorney",
        "counsel",
        "liability",
        "damages",
        "breach",
        "compliance",
        "shall",
        "may not",
        "provided that",
    ]

    # Educational/instructional indicators (non-academic)
    educational_indicators = [
        "tutorial",
        "how to",
        "step by step",
        "lesson",
        "guide",
        "course",
        "learn",
        "instruction",
        "exercise",
        "practice",
        "assignment",
        "module",
        "chapter",
        "beginner",
        "intermediate",
        "advanced",
        "skill",
        "training",
        "workshop",
        "walkthrough",
        "demo",
        "example",
        "tip:",
        "note:",
        "important:",
        "remember:",
        "quick start",
        "getting started",
    ]

    # URL-based detection
    if any(
        domain in url_lower for domain in ["github.com", "docs.", "api.", "developer."]
    ):
        return "technical"
    elif any(
        domain in url_lower
        for domain in ["reddit.com", "stackoverflow", "forum", "discourse"]
    ):
        return "forum"
    elif any(
        domain in url_lower
        for domain in ["arxiv.org", "scholar.google", "pubmed", "jstor"]
    ):
        return "academic"
    elif any(
        domain in url_lower for domain in ["news.", "cnn.com", "bbc.com", "reuters"]
    ):
        return "news"
    elif any(domain in url_lower for domain in ["medium.com", "substack", "blog"]):
        return "blog"
    elif any(
        domain in url_lower
        for domain in [
            "twitter.com",
            "linkedin.com",
            "instagram.com",
            "facebook.com",
            "x.com",
        ]
    ):
        return "social_media"
    elif any(
        domain in url_lower
        for domain in ["law.", "legal", "court", "gov", "laws", "legislation"]
    ):
        return "legal"
    elif any(
        domain in url_lower
        for domain in [
            "tutorial",
            "course",
            "learn",
            "education",
            "training",
            "udemy",
            "coursera",
            "khan",
        ]
    ):
        return "educational"

    # Content-based detection
    tech_score = sum(1 for indicator in tech_indicators if indicator in content_lower)
    academic_score = sum(
        1 for indicator in academic_indicators if indicator in content_lower
    )
    forum_score = sum(1 for indicator in forum_indicators if indicator in content_lower)
    news_score = sum(1 for indicator in news_indicators if indicator in content_lower)
    blog_score = sum(1 for indicator in blog_indicators if indicator in content_lower)
    social_media_score = sum(
        1 for indicator in social_media_indicators if indicator in content_lower
    )
    legal_score = sum(1 for indicator in legal_indicators if indicator in content_lower)
    educational_score = sum(
        1 for indicator in educational_indicators if indicator in content_lower
    )

    scores = {
        "technical": tech_score,
        "academic": academic_score,
        "forum": forum_score,
        "news": news_score,
        "blog": blog_score,
        "social_media": social_media_score,
        "legal": legal_score,
        "educational": educational_score,
    }

    # Return the category with highest score, default to 'general'
    max_score = max(scores.values())
    if max_score >= 2:  # Require at least 2 indicators for confident classification
        return max(scores, key=scores.get)

    return "general"


def _get_contextual_prompt_and_system_message(
    content_type: str, full_document: str, chunk: str
) -> tuple[str, str]:
    """
    Generate appropriate prompt and system message based on content type.

    Args:
        content_type: Detected content type
        full_document: Full document content
        chunk: Individual chunk content

    Returns:
        Tuple of (user_prompt, system_message)
    """
    base_prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk>"""

    if content_type == "technical":
        user_prompt = f"""{base_prompt}
Please provide a concise context that situates this technical chunk within the overall document for better search retrieval. Focus on the technical concepts, APIs, or procedures discussed."""

        system_message = (
            "You are a technical documentation specialist.\n"
            "Given a chunk from technical content, return 1-2 sentences that capture:\n\n"
            "• The main technical concept, API, or procedure\n"
            "• How it relates to the broader technical context\n"
            "• Key terms, commands, or parameters mentioned\n"
            "• The chunk's role (setup, example, reference, troubleshooting)\n\n"
            "Preserve technical terminology and be precise.\n"
            "Output only the contextual summary—nothing else."
        )

    elif content_type == "academic":
        user_prompt = f"""{base_prompt}
Please provide a concise context that situates this academic/research chunk within the overall document for better search retrieval. Focus on the research concepts, methodology, or findings discussed."""

        system_message = (
            "You are an academic content specialist.\n"
            "Given a chunk from academic or research content, return 1-2 sentences that capture:\n\n"
            "• The main research concept, hypothesis, or finding\n"
            "• How it fits into the overall research narrative\n"
            "• Key methodology, data, or theoretical frameworks\n"
            "• The chunk's role (introduction, methodology, results, discussion)\n\n"
            "Use precise academic language and preserve key terms.\n"
            "Output only the contextual summary—nothing else."
        )

    elif content_type == "forum":
        user_prompt = f"""{base_prompt}
Please provide a concise context that situates this forum/discussion chunk within the overall thread for better search retrieval. Focus on the problem, solution, or discussion point being addressed."""

        system_message = (
            "You are a discussion thread specialist.\n"
            "Given a chunk from forum or discussion content, return 1-2 sentences that capture:\n\n"
            "• The main problem, question, or discussion point\n"
            "• Whether it's a question, answer, or commentary\n"
            "• Key solutions, insights, or opinions expressed\n"
            "• How it relates to the broader discussion thread\n\n"
            "Preserve the conversational context and key insights.\n"
            "Output only the contextual summary—nothing else."
        )

    elif content_type == "news":
        user_prompt = f"""{base_prompt}
Please provide a concise context that situates this news chunk within the overall article for better search retrieval. Focus on the key events, people, or developments discussed."""

        system_message = (
            "You are a news content specialist.\n"
            "Given a chunk from news content, return 1-2 sentences that capture:\n\n"
            "• The main event, development, or news angle\n"
            "• Key people, organizations, or locations involved\n"
            "• The temporal context (when, sequence of events)\n"
            "• How it fits into the broader news story\n\n"
            "Maintain journalistic objectivity and preserve key facts.\n"
            "Output only the contextual summary—nothing else."
        )

    elif content_type == "blog":
        user_prompt = f"""{base_prompt}
Please provide a concise context that situates this blog chunk within the overall post for better search retrieval. Focus on the main ideas, opinions, or experiences shared."""

        system_message = (
            "You are a blog content specialist.\n"
            "Given a chunk from blog or opinion content, return 1-2 sentences that capture:\n\n"
            "• The main idea, opinion, or experience shared\n"
            "• The author's perspective or argument\n"
            "• Key insights, recommendations, or lessons\n"
            "• How it fits into the overall narrative or argument\n\n"
            "Preserve the author's voice and key insights.\n"
            "Output only the contextual summary—nothing else."
        )

    elif content_type == "social_media":
        user_prompt = f"""{base_prompt}
Please provide a concise context that situates this social media content within the overall post or thread for better search retrieval. Focus on the key message, engagement, or discussion points."""

        system_message = (
            "You are a social media content specialist.\n"
            "Given a chunk from social media content, return 1-2 sentences that capture:\n\n"
            "• The main message, announcement, or discussion point\n"
            "• The tone and engagement style (professional, casual, promotional)\n"
            "• Key hashtags, mentions, or trending topics referenced\n"
            "• The content's purpose (share, promote, discuss, network)\n\n"
            "Preserve the social context and engagement elements.\n"
            "Output only the contextual summary—nothing else."
        )

    elif content_type == "legal":
        user_prompt = f"""{base_prompt}
Please provide a concise context that situates this legal content within the overall document for better search retrieval. Focus on the legal concepts, obligations, or procedural elements discussed."""

        system_message = (
            "You are a legal document specialist.\n"
            "Given a chunk from legal content, return 1-2 sentences that capture:\n\n"
            "• The main legal concept, obligation, or right described\n"
            "• Key parties, jurisdictions, or legal frameworks involved\n"
            "• Important terms, conditions, or procedural requirements\n"
            "• The chunk's role (definition, obligation, exception, procedure)\n\n"
            "Preserve precise legal terminology and maintain formal tone.\n"
            "Output only the contextual summary—nothing else."
        )

    elif content_type == "educational":
        user_prompt = f"""{base_prompt}
Please provide a concise context that situates this educational content within the overall material for better search retrieval. Focus on the learning objectives, skills, or instructional elements covered."""

        system_message = (
            "You are an educational content specialist.\n"
            "Given a chunk from instructional content, return 1-2 sentences that capture:\n\n"
            "• The main skill, concept, or learning objective taught\n"
            "• The instructional approach (tutorial, exercise, example, explanation)\n"
            "• Key steps, procedures, or techniques demonstrated\n"
            "• The difficulty level and target audience context\n\n"
            "Preserve instructional clarity and learning-focused language.\n"
            "Output only the contextual summary—nothing else."
        )

    else:  # general content
        user_prompt = f"""{base_prompt}
Please provide a concise context that situates this content chunk within the overall document for better search retrieval. Focus on the main concepts, themes, or information discussed."""

        system_message = (
            "You are a general content specialist.\n"
            "Given a chunk of content, return 1-2 sentences that capture:\n\n"
            "• The main concept, theme, or information presented\n"
            "• How it relates to the broader document context\n"
            "• Key terms, names, or ideas mentioned\n"
            "• The chunk's role in the overall narrative\n\n"
            "Be concise and preserve important terminology.\n"
            "Output only the contextual summary—nothing else."
        )

    return user_prompt, system_message


def generate_contextual_embedding(
    full_document: str, chunk: str, url: str = ""
) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    Now includes content-type detection for adaptive prompting strategies.

    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        url: Optional URL for additional content type detection context

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
        # Import here to avoid circular imports
        from config import get_config

        config = get_config()

        # Detect content type for adaptive prompting (if enabled)
        if config.contextual_content_type_detection:
            content_type = _detect_content_type(chunk, url)
        else:
            content_type = "general"

        # Get appropriate prompt and system message for content type (if adaptive prompts enabled)
        if config.use_adaptive_contextual_prompts:
            user_prompt, system_message = _get_contextual_prompt_and_system_message(
                content_type, full_document, chunk
            )
        else:
            # Use legacy prompt for backward compatibility
            user_prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

            system_message = (
                "You are a concise technical summarizer.\n"
                "Given a chunk from a document, return 1-2 plain-English sentences that capture:\n\n"
                "• the main concept conveyed in the chunk\n"
                "• key terms, names, or ideas mentioned\n"
                "• the chunk's role in the wider document\n\n"
                "Avoid marketing language or personal opinions.\n"
                "Retain original terminology and be precise.\n"
                "Output only the summary text—nothing else."
            )

        # Call the OpenAI API to generate contextual information
        response = _retry_with_backoff(
            openai.chat.completions.create,
            model=model_choice,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        # Extract the generated context
        context = response.choices[0].message.content.strip()

        # Combine the context with the original chunk
        if (
            config.contextual_include_content_type_tag
            and config.contextual_content_type_detection
        ):
            contextual_text = f"[{content_type.upper()}] {context}\n---\n{chunk}"
        else:
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
    return generate_contextual_embedding(full_document, content, url)


def add_documents_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    strategy_config: Optional["StrategyConfig"],
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
        strategy_config: Strategy configuration for agentic RAG
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

    # Create or get source entries for all unique URLs
    url_to_source_id = {}
    for url in unique_urls:
        # Get all content chunks for this URL to calculate word count
        url_contents = [contents[i] for i, u in enumerate(urls) if u == url]
        url_metadatas = [metadatas[i] for i, u in enumerate(urls) if u == url]

        source_id = get_or_create_source(client, url, url_contents, url_metadatas)
        if source_id is not None:
            url_to_source_id[url] = source_id
        else:
            print(f"⚠️ Failed to create/get source for URL {url}")

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
                "source_id": url_to_source_id.get(
                    batch_urls[j]
                ),  # Set source_id from our mapping
            }

            batch_data.append(data)

        # Insert batch into Supabase
        try:
            client.table("crawled_pages").insert(batch_data).execute()
        except Exception as e:
            print(f"Error inserting batch into Supabase: {e}")

    # Extract and store code examples if agentic RAG is enabled
    should_extract_code = (
        strategy_config
        and strategy_config.use_agentic_rag
        and extract_code_from_content is not None
    )

    if should_extract_code:
        # Group content by URL to extract code from full documents
        url_to_content = {}
        for i, url in enumerate(urls):
            if url not in url_to_content:
                url_to_content[url] = []
            url_to_content[url].append(contents[i])

        # Process each URL's content for code extraction
        for url, url_contents in url_to_content.items():
            try:
                # Get source_id from our pre-created mapping
                source_id = url_to_source_id.get(url)
                if source_id is None:
                    print(
                        f"⚠️ No source_id found for URL {url}, skipping code extraction"
                    )
                    continue

                # Combine all chunks for this URL to get full document content
                full_content = "\n\n".join(url_contents)

                # Extract code examples from the full content, providing the source_url
                extracted_codes = extract_code_from_content(full_content, url)

                if extracted_codes:
                    # The 'extracted_codes' are already the correct ExtractedCode objects.
                    # No need to convert them.
                    add_code_examples_to_supabase(client, extracted_codes, source_id)
                    print(
                        f"🔧 Extracted and stored {len(extracted_codes)} code examples from {url}"
                    )

            except Exception as e:
                print(f"Error extracting code from URL {url}: {e}")
                continue


def add_code_examples_to_supabase(
    client: Client,
    code_examples: List["ExtractedCode"],
    source_id: int,
    batch_size: int = 10,
) -> None:
    """
    Add extracted code examples to the 'code_examples' table in Supabase.

    Args:
        client: The Supabase client.
        code_examples: A list of ExtractedCode objects.
        source_id: The ID of the source document.
        batch_size: The number of items to process in each batch.
    """
    if not code_examples:
        return

    records_to_insert = []

    # Generate embeddings for all code examples in a single batch
    # Combine content and summary for a single, powerful embedding
    combined_texts = [f"{ex.content}\n\nSummary: {ex.summary}" for ex in code_examples]
    embeddings = create_embeddings_batch(combined_texts)

    for i, example in enumerate(code_examples):
        embedding = embeddings[i]

        record = {
            "source_id": source_id,
            "url": example.url,
            "chunk_number": example.chunk_number,
            "content": example.content,
            "summary": example.summary,
            "programming_language": example.programming_language,
            "complexity_score": example.complexity_score,
            "embedding": embedding,
            "metadata": json.dumps(example.metadata or {}),
        }
        records_to_insert.append(record)

    # Insert records in batches
    for i in range(0, len(records_to_insert), batch_size):
        batch = records_to_insert[i : i + batch_size]
        try:
            response = client.table("code_examples").insert(batch).execute()
            if hasattr(response, "error") and response.error:
                # More detailed error logging
                print(
                    f"Error inserting code examples batch into Supabase: {response.error}"
                )
            else:
                print(f"Successfully inserted {len(batch)} code examples.")
        except Exception as e:
            print(f"An exception occurred during code example insertion: {e}")


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
        response = (
            client.table("sources")
            .select("source_id")
            .eq("url", url)
            .limit(1)
            .execute()
        )
        if response.data and len(response.data) > 0:
            return response.data[0]["source_id"]
    except Exception as e:
        print(f"Error getting source_id for URL {url}: {e}")

    return None


def get_or_create_source(
    client: Client, url: str, contents: List[str], metadatas: List[Dict[str, Any]]
) -> Optional[int]:
    """
    Get existing source_id for a URL or create a new source entry.

    Args:
        client: Supabase client
        url: The URL to get or create source for
        contents: List of content chunks for this URL (for word count calculation)
        metadatas: List of metadata dicts for this URL

    Returns:
        source_id if successful, None if error
    """
    # First try to get existing source
    existing_source_id = get_source_id_from_url(client, url)
    if existing_source_id is not None:
        return existing_source_id

    try:
        # Calculate total word count from all chunks for this URL
        total_word_count = sum(len(content.split()) for content in contents)

        # Prepare source data (no 'name' column in sources table)
        source_data = {
            "url": url,
            "total_word_count": total_word_count,
        }

        # Insert new source (use upsert to handle race conditions)
        response = (
            client.table("sources").upsert(source_data, on_conflict="url").execute()
        )

        if response.data and len(response.data) > 0:
            source_id = response.data[0]["source_id"]
            print(f"✅ Created new source: {url} (ID: {source_id})")
            return source_id
        else:
            print(f"⚠️ Failed to create source for URL {url}: No data returned")
            return None

    except Exception as e:
        print(f"Error creating source for URL {url}: {e}")
        # Try to get it again in case another process created it
        return get_source_id_from_url(client, url)


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
            "Authorization": f"Bearer {supabase_key}",
        }

        response = requests.post(url, json=params, headers=headers, timeout=30)

        if response.status_code == 200:
            rows = response.json()
        else:
            print(
                f"[search_documents] Edge function failed with status {response.status_code}: {response.text}"
            )
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
