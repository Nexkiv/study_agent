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

from app.config import OPENAI_API_KEY, EMBEDDING_MODEL, TAVILY_API_KEY, CHAT_DEFAULT_N_RESULTS
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


# Common English words that should never be "corrected" to art terms.
# Prevents "table" → "Marble", "format" → "Reformation", etc.
COMMON_ENGLISH_WORDS = {
    'about', 'above', 'across', 'after', 'again', 'against', 'along', 'also',
    'always', 'among', 'another', 'answer', 'around', 'asked', 'away', 'back',
    'based', 'because', 'been', 'before', 'began', 'being', 'below', 'best',
    'better', 'between', 'body', 'book', 'both', 'bring', 'build', 'built',
    'called', 'came', 'case', 'cause', 'change', 'city', 'close', 'come',
    'compare', 'complete', 'consider', 'could', 'course', 'cover', 'create',
    'current', 'define', 'describe', 'detail', 'develop', 'discuss', 'does',
    'done', 'down', 'during', 'each', 'early', 'else', 'enough', 'even',
    'every', 'example', 'explain', 'face', 'fact', 'family', 'feel', 'field',
    'figure', 'file', 'find', 'first', 'following', 'form', 'format', 'found',
    'from', 'full', 'gave', 'general', 'give', 'given', 'goes', 'going',
    'good', 'great', 'group', 'grow', 'guide', 'half', 'hand', 'hard', 'have',
    'head', 'help', 'here', 'high', 'hold', 'home', 'house', 'idea', 'image',
    'important', 'include', 'into', 'just', 'keep', 'kind', 'know', 'known',
    'land', 'large', 'last', 'late', 'later', 'lead', 'left', 'less', 'life',
    'light', 'like', 'line', 'list', 'little', 'live', 'long', 'look', 'made',
    'main', 'major', 'make', 'many', 'mark', 'material', 'matter', 'mean',
    'might', 'mind', 'more', 'most', 'move', 'much', 'must', 'name', 'need',
    'never', 'next', 'note', 'nothing', 'number', 'often', 'once', 'only',
    'open', 'order', 'other', 'over', 'page', 'part', 'pass', 'past', 'period',
    'person', 'place', 'plan', 'play', 'point', 'possible', 'power', 'present',
    'probably', 'problem', 'provide', 'public', 'pull', 'push', 'quite', 'rather',
    'read', 'real', 'right', 'room', 'round', 'rule', 'said', 'same', 'school',
    'seem', 'sense', 'several', 'shall', 'show', 'side', 'since', 'small',
    'some', 'something', 'soon', 'sort', 'space', 'stand', 'start', 'state',
    'still', 'story', 'study', 'such', 'sure', 'system', 'table', 'take',
    'talk', 'tell', 'term', 'terms', 'test', 'than', 'that', 'their', 'them',
    'then', 'there', 'these', 'they', 'thing', 'think', 'those', 'though',
    'thought', 'three', 'through', 'time', 'together', 'told', 'took', 'total',
    'toward', 'true', 'turn', 'type', 'under', 'understand', 'unit', 'until',
    'upon', 'used', 'using', 'very', 'view', 'want', 'water', 'well', 'were',
    'while', 'whole', 'will', 'with', 'within', 'without', 'word', 'work',
    'world', 'would', 'write', 'year', 'your',
}


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

            # Skip common English words — only correct likely art term misspellings
            if clean_word.lower() in COMMON_ENGLISH_WORDS:
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
ART_HISTORY_SYSTEM_PROMPT = """You are an expert art history study assistant helping students review their uploaded course materials.

**CRITICAL CONSTRAINT — NO HALLUCINATIONS:**
- ONLY include information found in search results from the student's materials
- Do NOT add terms, definitions, artworks, or concepts from your general knowledge
- Do NOT embellish or supplement search results with external art history knowledge
- If the search results don't contain something, do NOT include it — even if you know it's relevant
- Every fact in your response must be traceable to a specific search result

When answering questions:
1. ALWAYS use search_class_materials to find relevant information BEFORE answering
2. When discussing artworks, reference: artist, title, date, medium, period — but ONLY if these details appear in the search results
3. If asked to calculate or analyze data, use the execute_python tool
4. For historical connections or context NOT in the materials, use search_web — and clearly label this as external

Tool usage guidelines:
- search_class_materials: Primary source — ALWAYS search course materials first. Supports three modes:
  * Semantic search: provide a query to find related content by meaning
  * Section filter: provide a section name to get content from a specific section
  * Keyword filter: provide a keyword to find chunks containing that exact term
  * Combine modes for targeted searches (e.g., query + section filter)
  * Pass "" for any parameter you don't need
- list_sections: Call this to see all available sections in the materials. Use it when you need comprehensive coverage.
- search_web: ONLY for context not found in materials. Always label web results separately.
- execute_python: For date math, timeline visualizations, statistical analysis

Search strategy for comprehensive questions (e.g., "all terms", "list everything", "who do I need to know"):
1. Call list_sections to see all available sections
2. Search each relevant section individually using the section filter
3. Synthesize results from ALL searches before answering
This ensures complete coverage — do NOT rely on a single broad search for comprehensive listings.

Source attribution (REQUIRED):
- At the END of every response, include a "Sources:" section
- Group cited facts by their source document and section
- Format each line as: "- {Source Name} > {Section}: {key terms or facts cited from that source}"
- If information came from search_web, list separately as: "- Web Search: {what was looked up}"
- If you could not find information in materials or web search, say so clearly

Example source attribution:
Sources:
- Study Guide > Early Northern Renaissance: Diptych, Triptych, Grisaille, Disguised symbolism
- Study Guide > Early Southern Renaissance: Linear perspective, Contrapposto, Fresco
- Web Search: Medici family patronage history
"""


def _format_results(results: dict) -> str:
    """Format ChromaDB query results with source attribution."""
    if not results['documents'] or not results['documents'][0]:
        return ""

    formatted = []
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        source = metadata.get('source', 'Unknown')
        section = metadata.get('section', '')
        if section:
            source_label = f"[Source: {source} | Section: {section}]"
        else:
            source_label = f"[Source: {source}]"
        formatted.append(f"{source_label}\n{doc}\n")

    return "\n---\n".join(formatted)


def create_search_tool(class_name: str, default_n_results: int = CHAT_DEFAULT_N_RESULTS):
    """
    Factory function for search tool with class context.

    Args:
        class_name: Name of the class to search
        default_n_results: Default number of results (from config, overridable for flashcards)

    Returns a tool function bound to specific class.
    """
    async def search_class_materials(query: str, section: str, keyword: str) -> str:
        """
        Search uploaded class materials. Supports semantic search, section filtering, and keyword matching.

        Args:
            query: Search query (natural language). Use "" to skip semantic search and use only section/keyword filters.
            section: Filter results to a specific section name (e.g., "Early Northern Renaissance"). Use "" for no filter. Supports partial matching.
            keyword: Filter results to chunks containing this keyword (case-insensitive substring match). Use "" for no filter.

        Returns:
            Formatted search results with source and section attribution
        """
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

            # Resolve section filter — find matching section names via substring
            section_where = None
            if section:
                all_meta = collection.get(include=["metadatas"])
                all_sections = {m.get('section', '') for m in all_meta['metadatas']} - {''}
                section_lower = section.lower()
                matching = [s for s in all_sections if section_lower in s.lower()]
                if not matching:
                    return f"No sections matching '{section}' found. Use list_sections to see available sections."
                if len(matching) == 1:
                    section_where = {"section": matching[0]}
                else:
                    section_where = {"section": {"$in": matching}}

            # Build query kwargs
            query_kwargs = {
                "n_results": min(n_results, collection.count())
            }

            if section_where:
                query_kwargs["where"] = section_where

            # Keyword filter (works on document text)
            if keyword:
                query_kwargs["where_document"] = {"$contains": keyword}

            # Semantic search with embedding
            if query:
                client = get_async_openai_client()
                embedding_response = await client.embeddings.create(
                    input=[query],
                    model=EMBEDDING_MODEL
                )
                query_kwargs["query_embeddings"] = [embedding_response.data[0].embedding]
            else:
                # No query — use get() for section/keyword-only search
                get_kwargs = {}
                if section_where:
                    get_kwargs["where"] = section_where

                all_results = collection.get(**get_kwargs, include=["documents", "metadatas"])
                if not all_results['documents']:
                    return f"No results found for section='{section}', keyword='{keyword}'"

                # Manual keyword filter if needed
                if keyword:
                    keyword_lower = keyword.lower()
                    filtered_docs = []
                    filtered_metas = []
                    for doc, meta in zip(all_results['documents'], all_results['metadatas']):
                        if keyword_lower in doc.lower():
                            filtered_docs.append(doc)
                            filtered_metas.append(meta)
                    all_results = {'documents': [filtered_docs], 'metadatas': [filtered_metas]}
                else:
                    all_results = {
                        'documents': [all_results['documents']],
                        'metadatas': [all_results['metadatas']]
                    }

                result_text = _format_results(all_results)
                return result_text if result_text else f"No results found for section='{section}', keyword='{keyword}'"

            results = collection.query(**query_kwargs)
            result_text = _format_results(results)
            return result_text if result_text else f"No relevant information found for: {query}"

        except Exception as e:
            return f"Search error: {str(e)}"

    return search_class_materials


def create_list_sections_tool(class_name: str):
    """
    Factory function for list_sections tool with class context.
    """
    async def list_sections() -> str:
        """
        List all document sections available in the class materials.
        Returns section names that can be used with search_class_materials' section parameter.
        Call this first when you need comprehensive coverage across all sections.
        """
        try:
            class_obj = Class.query.filter_by(name=class_name).first()
            if not class_obj:
                return f"Error: Class '{class_name}' not found"

            collection = get_or_create_collection(class_obj.id)
            if collection.count() == 0:
                return "No materials uploaded yet."

            # Get all metadata to extract unique sections
            all_data = collection.get(include=["metadatas"])
            sections = set()
            for meta in all_data['metadatas']:
                section = meta.get('section', '')
                if section:
                    sections.add(section)

            if not sections:
                return "No sections detected in uploaded materials."

            sorted_sections = sorted(sections)
            return "Available sections:\n" + "\n".join(f"- {s}" for s in sorted_sections)

        except Exception as e:
            return f"Error listing sections: {str(e)}"

    return list_sections


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
        "tools": ["search_class_materials", "list_sections", "search_web", "execute_python"],
        "kwargs": {
            "temperature": 0.7  # Balanced creativity/consistency
        }
    }
