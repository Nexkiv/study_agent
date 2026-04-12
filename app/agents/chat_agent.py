"""
RAG Chat Agent for StudyAgent.

Art history expert with tool use:
- search_class_materials: Semantic search via ChromaDB
- execute_python: Sandboxed code execution for calculations
- correct_spelling: Fuzzy matching for misspelled terms
- search_web: Web search via Tavily for external knowledge

Uses OpenAI GPT-4o-mini with responses API.
"""
from openai import AsyncOpenAI
import io
import contextlib
import re
from typing import Optional
from rapidfuzz import process, fuzz

from app.config import OPENAI_API_KEY, EMBEDDING_MODEL, TAVILY_API_KEY
from app.extensions import get_or_create_collection, db
from app.models import Class


# Singleton async client
_openai_async_client = None


def get_async_openai_client() -> AsyncOpenAI:
    """Get or create async OpenAI client for agent."""
    global _openai_async_client
    if _openai_async_client is None:
        _openai_async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_async_client


def correct_spelling(query: str, class_name: str) -> str:
    """
    Correct misspelled terms in user query using fuzzy matching against class materials.

    Extracts capitalized terms (likely proper nouns) from class materials and corrects
    similar words in the query. For example: "Alderfini" -> "Arnolfini"

    Args:
        query: User's search query (may contain misspellings)
        class_name: Name of class to extract terms from

    Returns:
        Corrected query with proper spellings from materials
    """
    try:
        # Get class from database
        class_obj = Class.query.filter_by(name=class_name).first()
        if not class_obj:
            return query  # Return original if class not found

        # Get ChromaDB collection
        collection = get_or_create_collection(class_obj.id)

        # Check if collection has documents
        if collection.count() == 0:
            return query  # Return original if no materials

        # Get all documents from collection
        results = collection.get()
        if not results['documents']:
            return query

        # Extract capitalized words (proper nouns) from all documents
        # These are likely artist names, artwork titles, periods, etc.
        proper_nouns = set()
        for doc in results['documents']:
            # Find capitalized words (but skip common words like "The", "A", etc.)
            words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', doc)
            proper_nouns.update(words)

        # Remove common words that aren't art history terms
        common_words = {'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'And',
                       'But', 'Or', 'As', 'By', 'From', 'With', 'This', 'That'}
        proper_nouns = proper_nouns - common_words

        # Convert to list for rapidfuzz
        term_list = list(proper_nouns)

        if not term_list:
            return query  # No terms to match against

        # Find and replace misspelled words in query
        corrected_query = query
        words = query.split()

        for i, word in enumerate(words):
            # Skip very short words and common words
            if len(word) < 4 or word.lower() in {'what', 'when', 'where', 'who', 'how', 'why'}:
                continue

            # Clean punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            if not clean_word:
                continue

            # Find best match using fuzzy matching
            # Only correct if similarity is high (>= 80) but not exact (< 100)
            match_result = process.extractOne(
                clean_word,
                term_list,
                scorer=fuzz.ratio
            )

            if match_result:
                match_term, score, _ = match_result
                # Correct if it's a close match but not exact
                # Threshold lowered to 65% to catch "Alderfini" -> "Arnolfini" (66.67%)
                if 65 <= score < 100:
                    # Preserve original punctuation
                    replacement = word.replace(clean_word, match_term)
                    corrected_query = corrected_query.replace(word, replacement, 1)

        return corrected_query

    except Exception as e:
        # If anything fails, return original query
        print(f"Spelling correction error: {e}")
        return query


# Art history expert system prompt
ART_HISTORY_SYSTEM_PROMPT = """You are an expert art history study assistant with deep knowledge of Western art from antiquity through contemporary periods.

When answering questions:
1. ALWAYS ground your response in the student's uploaded course materials
2. Use the search_class_materials tool to find relevant information before answering
3. Use proper art historical terminology (chiaroscuro, sfumato, tenebrism, impasto, etc.)
4. When discussing artworks, reference: artist, title, date, medium, period if known
5. Cite which lecture/reading the information comes from (use source metadata)
6. If asked to calculate or analyze data, use the execute_python tool
7. For historical connections or context not in materials, use the search_web tool

Tool usage guidelines:
- search_class_materials: Primary source - always check course materials first
- search_web: Use for broader context, artist biographies, historical connections (e.g., family relationships, patronage networks)
- execute_python: For date math, timeline visualizations, statistical analysis

Examples:
- "How are the Portinari and Medici families related?" → search_class_materials first, then search_web for historical context
- "How many years between X and Y?" → execute_python

CRITICAL: Do NOT make up information. If the uploaded materials don't contain the answer, use search_web or say so clearly.
"""


def create_search_tool(class_name: str, default_n_results: int = 5):
    """
    Factory function for search tool with class context.

    Args:
        class_name: Name of the class to search
        default_n_results: Default number of results (5 for chat, 20 for flashcards)

    Returns a tool function bound to specific class.
    """
    async def search_class_materials(query: str, n_results: int = None) -> str:
        """
        Search uploaded class materials using semantic similarity.

        Args:
            query: Search query (natural language)
            n_results: Number of results to return (uses default if not specified)

        Returns:
            Formatted search results with source attribution
        """
        # Use default if not specified
        if n_results is None:
            n_results = default_n_results
        try:
            # Get class from database
            class_obj = Class.query.filter_by(name=class_name).first()
            if not class_obj:
                return f"Error: Class '{class_name}' not found"

            # Get ChromaDB collection
            collection = get_or_create_collection(class_obj.id)

            # Check if collection has documents
            if collection.count() == 0:
                return "No materials uploaded yet. Please upload lecture notes or readings first."

            # Generate query embedding (must use same model as ingestion)
            client = get_async_openai_client()
            embedding_response = await client.embeddings.create(
                input=[query],
                model=EMBEDDING_MODEL  # 'text-embedding-3-small'
            )
            query_embedding = embedding_response.data[0].embedding

            # Query ChromaDB with embedding
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection.count())
            )

            # Format results with source attribution
            if not results['documents'] or not results['documents'][0]:
                return f"No relevant information found for: {query}"

            formatted_results = []
            for i, (doc, metadata) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0]
            )):
                source = metadata.get('source', 'Unknown')
                formatted_results.append(
                    f"[Source: {source}]\n{doc}\n"
                )

            return "\n---\n".join(formatted_results)

        except Exception as e:
            return f"Search error: {str(e)}"

    return search_class_materials


# Allowed modules for sandbox
ALLOWED_MODULES = {
    'math', 'statistics', 'datetime',
    'numpy', 'sympy'  # Optional: only if installed
}


async def execute_python(code: str) -> str:
    """
    Execute Python code in a sandboxed environment.

    Allowed modules: math, statistics, datetime, numpy, sympy
    No file I/O, network, or subprocess operations.

    Args:
        code: Python code to execute

    Returns:
        stdout output or error message
    """
    # Capture stdout
    stdout = io.StringIO()

    # Restricted globals (no builtins)
    restricted_globals = {"__builtins__": {}}

    # Import allowed modules into local namespace
    local_vars = {}
    for module_name in ALLOWED_MODULES:
        try:
            local_vars[module_name] = __import__(module_name)
        except ImportError:
            # Module not installed, skip silently
            pass

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, restricted_globals, local_vars)

        output = stdout.getvalue()
        return output if output else "Code executed successfully (no output)"

    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"


async def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web for art history information using Tavily API.

    Use this tool to find historical context, artist biographies, or connections
    between artworks/families that aren't in the uploaded materials.

    Args:
        query: Search query (e.g., "Portinari family connection to Medici")
        max_results: Number of results to return (default 3)

    Returns:
        Formatted search results with URLs and snippets
    """
    if not TAVILY_API_KEY:
        return ("⚠️ Web search is not configured. TAVILY_API_KEY is missing from .env file.\n"
                "To enable web search:\n"
                "1. Sign up at https://tavily.com\n"
                "2. Add TAVILY_API_KEY=your_key_here to .env\n"
                "3. Restart the application")

    try:
        from tavily import TavilyClient

        # Initialize Tavily client
        client = TavilyClient(api_key=TAVILY_API_KEY)

        # Search with art history context
        search_context = f"art history {query}"
        response = client.search(
            query=search_context,
            max_results=max_results,
            search_depth="advanced",  # More comprehensive search
            include_domains=["wikipedia.org", "metmuseum.org", "arthistory.net"],  # Prefer reliable sources
        )

        # Format results
        if not response.get('results'):
            return f"No web results found for: {query}"

        formatted_results = []
        for i, result in enumerate(response['results'][:max_results], 1):
            title = result.get('title', 'Unknown')
            url = result.get('url', '')
            content = result.get('content', '')

            formatted_results.append(
                f"{i}. **{title}**\n"
                f"   Source: {url}\n"
                f"   {content}\n"
            )

        return "\n".join(formatted_results)

    except ImportError:
        return "Error: tavily-python library not installed. Run: pip install tavily-python"
    except Exception as e:
        return f"Web search error: {type(e).__name__}: {str(e)}"


def create_rag_agent_config(class_name: str) -> dict:
    """
    Create agent configuration for RAG chat.

    Args:
        class_name: Name of class for context binding

    Returns:
        Agent config dict compatible with run_agent()
    """
    return {
        "name": "rag_chat_agent",
        "description": "Art history study assistant with RAG, web search, and code execution",
        "model": "gpt-4o-mini",  # OpenAI model
        "prompt": ART_HISTORY_SYSTEM_PROMPT,
        "tools": ["search_class_materials", "search_web", "execute_python"],
        "kwargs": {
            "temperature": 0.7  # Balanced creativity/consistency
        }
    }
