"""
Study agent for flashcard generation using OpenAI Structured Outputs.

This module implements flashcard generation with:
- LLM-based intent parsing (domain-agnostic)
- Deterministic multi-query search (parallel, no agent loop)
- Direct structured output generation (no agent overhead)
- Post-processing: exact dedup, fuzzy dedup, category filter
"""

import asyncio
import json
import logging
from typing import List, Dict, Tuple

from rapidfuzz import fuzz

from app.agents.chat_agent import get_async_openai_client, create_search_tool
from app.agents.run_agent import _cancel_flag, AgentCancelled
from app.config import DEFAULT_CHAT_MODEL, FLASHCARD_DEFAULT_N_RESULTS

logger = logging.getLogger(__name__)

# Structured output schema for flashcard generation
FLASHCARD_SCHEMA = {
    "type": "object",
    "properties": {
        "flashcards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "Term, concept, or key item name"
                    },
                    "definition": {
                        "type": "string",
                        "description": "Clear definition with context"
                    }
                },
                "required": ["term", "definition"],
                "additionalProperties": False
            }
        }
    },
    "required": ["flashcards"],
    "additionalProperties": False
}


CATEGORY_FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "matches": {"type": "boolean"}
                },
                "required": ["index", "matches"],
                "additionalProperties": False
            }
        }
    },
    "required": ["results"],
    "additionalProperties": False
}


INTENT_PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "user_category": {
            "type": "string",
            "description": "What the user wants flashcards about (e.g., 'artists and people', 'places and locations', 'key terms and vocabulary'). Use 'all topics' if the user wants everything."
        },
        "is_specific_category": {
            "type": "boolean",
            "description": "True if the user asked for a specific subset of content, false if they want everything."
        },
        "search_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-4 semantic search queries to find this category of content in course materials."
        }
    },
    "required": ["user_category", "is_specific_category", "search_queries"],
    "additionalProperties": False
}


def _check_cancelled():
    """Check if generation has been cancelled and raise if so."""
    if _cancel_flag.is_set():
        logger.info("Flashcard generation cancelled by user")
        raise AgentCancelled("Generation cancelled by user")


async def parse_user_intent(client, topic: str) -> dict:
    """
    Use LLM to parse the user's topic into a structured intent.

    Returns dict with user_category, is_specific_category, and search_queries.
    Domain-agnostic — works for any subject.
    """
    response = await client.responses.create(
        model=DEFAULT_CHAT_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You parse a student's flashcard request into structured intent.\n"
                    "Determine what category of content they want and generate search queries to find it.\n\n"
                    "Examples:\n"
                    '- "Artists and People" → category: "people (artists, historical figures, individuals)", specific: true, queries: ["artists", "people", "historical figures", "individuals"]\n'
                    '- "Places to Know" → category: "places and locations (cities, regions, buildings, sites)", specific: true, queries: ["places", "locations", "cities", "buildings"]\n'
                    '- "Key Terms" → category: "terms, concepts, and definitions", specific: true, queries: ["terms", "definitions", "concepts", "vocabulary"]\n'
                    '- "Early Northern Renaissance" → category: "all topics", specific: false, queries: ["Early Northern Renaissance"]\n'
                    '- "Everything about Chapter 5" → category: "all topics", specific: false, queries: ["Chapter 5"]\n'
                    '- "Artworks and paintings" → category: "specific artworks (paintings, sculptures, architectural works)", specific: true, queries: ["artworks", "paintings", "sculptures", "works"]\n'
                )
            },
            {
                "role": "user",
                "content": f"Parse this flashcard request: \"{topic}\""
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "intent_parse",
                "schema": INTENT_PARSE_SCHEMA,
                "strict": True
            }
        }
    )

    result = json.loads(response.output[0].content[0].text)
    logger.info(f"Parsed intent: category='{result['user_category']}', specific={result['is_specific_category']}, queries={result['search_queries']}")
    return result


async def gather_section_content(class_name: str, search_queries: List[str]) -> str:
    """
    Gather relevant content via parallel broad searches (no per-section explosion).

    Fires one search per query (3-4 embedding calls total) without section filters.
    ChromaDB semantic search already returns the most relevant chunks across all sections.

    Returns concatenated context string.
    """
    search_fn = create_search_tool(class_name, default_n_results=FLASHCARD_DEFAULT_N_RESULTS)

    # Fire all queries in parallel — broad search (no section filter)
    all_results = await asyncio.gather(*[
        search_fn(query=q, section="", keyword="")
        for q in search_queries
    ])

    # Deduplicate chunks by content
    seen_chunks = set()
    unique_chunks = []
    for result_text in all_results:
        if result_text.startswith("No ") or result_text.startswith("Error") or result_text.startswith("Search error"):
            continue
        for chunk in result_text.split("\n---\n"):
            chunk = chunk.strip()
            if not chunk:
                continue
            # Use first 200 chars as dedup key
            dedup_key = chunk[:200].strip().lower()
            if dedup_key not in seen_chunks:
                seen_chunks.add(dedup_key)
                unique_chunks.append(chunk)

    logger.info(f"Gathered {len(unique_chunks)} unique chunks from {len(search_queries)} searches")
    return "\n\n---\n\n".join(unique_chunks)


def fuzzy_deduplicate(flashcards: List[Dict[str, str]], threshold: int = 80) -> List[Dict[str, str]]:
    """
    Merge near-duplicate flashcards using rapidfuzz.

    Catches cases like "Jan van Eyck" / "Van Eyck" or "Limbourg Brothers" / "The Limbourg Brothers".
    When duplicates are found, keeps the longer/more complete term and its definition.
    """
    if not flashcards:
        return flashcards

    kept = []  # List of (term_lower, card)

    for card in flashcards:
        term = card['term'].strip()
        term_lower = term.lower()
        is_duplicate = False

        for i, (existing_lower, existing_card) in enumerate(kept):
            # Check substring containment
            if term_lower in existing_lower or existing_lower in term_lower:
                is_duplicate = True
            # Check fuzzy similarity
            elif fuzz.ratio(term_lower, existing_lower) >= threshold:
                is_duplicate = True

            if is_duplicate:
                # Keep the longer/more complete term
                if len(term) > len(existing_card['term'].strip()):
                    kept[i] = (term_lower, card)
                    logger.debug(f"Fuzzy dedup: replaced '{existing_card['term']}' with '{term}'")
                else:
                    logger.debug(f"Fuzzy dedup: dropped '{term}' (keeping '{existing_card['term']}')")
                break

        if not is_duplicate:
            kept.append((term_lower, card))

    result = [card for _, card in kept]
    removed = len(flashcards) - len(result)
    if removed > 0:
        logger.info(f"Fuzzy dedup removed {removed} near-duplicate flashcards")
    return result


async def filter_flashcards_by_category(
    client,
    flashcards: List[Dict[str, str]],
    categories: List[str],
) -> List[Dict[str, str]]:
    """
    Post-process flashcards to remove items that don't match requested categories.

    Uses a structured output call to classify each flashcard term as matching
    or not matching the requested categories.
    """
    if not flashcards or not categories:
        return flashcards

    category_str = ", ".join(categories)
    terms_list = "\n".join(f"{i}: {card['term']}" for i, card in enumerate(flashcards))

    response = await client.responses.create(
        model=DEFAULT_CHAT_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a strict classifier. You MUST return a result for EVERY item in the list.\n"
                    f"For each item, determine if it is a {category_str}.\n"
                    "Return matches=true ONLY for items that clearly belong to the requested category.\n"
                    "Return matches=false for items that do not clearly fit the category.\n"
                    "When in doubt, return matches=false — it is better to exclude a borderline item "
                    "than to include something irrelevant."
                )
            },
            {
                "role": "user",
                "content": f"Classify EVERY item below — does it refer to a {category_str}? You must return one result per item.\n\n{terms_list}"
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "category_filter",
                "schema": CATEGORY_FILTER_SCHEMA,
                "strict": True
            }
        }
    )

    result = json.loads(response.output[0].content[0].text)
    remove_indices = {r["index"] for r in result["results"] if not r["matches"]}

    filtered = [card for i, card in enumerate(flashcards) if i not in remove_indices]
    removed_count = len(flashcards) - len(filtered)
    if removed_count > 0:
        logger.info(f"Category filter removed {removed_count} flashcards not matching: {category_str}")

    return filtered


async def generate_flashcards_for_topic(
    class_name: str,
    topic: str,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Main entry point for flashcard generation.

    Pipeline:
    1. LLM intent parse — understand what category of content the user wants
    2. Parallel broad searches — gather relevant content (3-4 embedding calls)
    3. Direct structured output — generate flashcards (no agent loop overhead)
    4. Post-processing — exact dedup, fuzzy dedup, category filter

    Args:
        class_name: Name of the class to generate flashcards for
        topic: User's topic/request (e.g., "Artists and People", "Places to Know")

    Returns:
        Tuple of (flashcards_list, status_message)
    """
    logger.info(f"Generating flashcards for class '{class_name}' on topic: {topic}")

    client = get_async_openai_client()

    # Step 1: Parse user intent
    intent = await parse_user_intent(client, topic)
    user_category = intent["user_category"]
    is_specific_category = intent["is_specific_category"]
    search_queries = intent["search_queries"]

    _check_cancelled()

    # Step 2: Gather content via parallel broad searches
    context = await gather_section_content(class_name, search_queries)

    if not context.strip():
        return [], "No relevant materials found. Please upload course materials first."

    _check_cancelled()

    # Step 3: Direct structured output call (no agent loop)
    logger.info(f"Calling Structured Outputs API for flashcards on '{topic}'")

    system_content = "You are generating study flashcards from course materials."
    if is_specific_category:
        system_content += f" The user specifically requested: {user_category}. Only include items that match this category."
    system_content += (
        " Extract key information from the context and create clear, testable flashcards."
        "\n\nCRITICAL: Only create flashcards from the provided context. Do NOT add information from your general knowledge."
        " Be exhaustive — generate ONE flashcard for EVERY distinct matching item. Do not skip items or cap the number."
    )

    user_content = (
        f"Context from course materials:\n\n{context}\n\n"
        f"Generate one flashcard for EVERY distinct item in the context above on the topic: {topic}\n\n"
        f"Do not skip any items. Be exhaustive — if there are 60 items, generate 60 flashcards."
    )
    if is_specific_category:
        user_content += f"\n\nFocus specifically on items matching: {user_category}. Do NOT include items from other categories."
    else:
        user_content += "\n\nInclude all relevant items: terms, concepts, people, works, and key details as they appear in the context."

    try:
        response = await client.responses.create(
            model=DEFAULT_CHAT_MODEL,
            input=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "flashcard_set",
                    "schema": FLASHCARD_SCHEMA,
                    "strict": True
                }
            }
        )

        message_content = response.output[0].content
        if not message_content or len(message_content) == 0:
            return [], "No flashcards generated — empty response from API."

        flashcard_json = message_content[0].text
        if not flashcard_json:
            return [], "No flashcards generated — empty content block."

        result = json.loads(flashcard_json)
        if 'flashcards' not in result:
            return [], "No flashcards generated — invalid response format."

        generated_flashcards = result['flashcards']
        logger.info(f"Generated {len(generated_flashcards)} raw flashcards")

    except Exception as e:
        error_msg = f"Flashcard generation failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

    _check_cancelled()

    # Step 4a: Exact deduplication (case-insensitive)
    seen_terms = set()
    deduplicated = []
    for card in generated_flashcards:
        term_lower = card['term'].lower().strip()
        if term_lower not in seen_terms:
            seen_terms.add(term_lower)
            deduplicated.append(card)

    exact_dupes = len(generated_flashcards) - len(deduplicated)
    if exact_dupes > 0:
        logger.info(f"Exact dedup removed {exact_dupes} duplicates")

    # Step 4b: Fuzzy deduplication (near-duplicates)
    deduplicated = fuzzy_deduplicate(deduplicated)

    # Step 4c: Category filter (only when user asked for a specific category)
    if is_specific_category:
        deduplicated = await filter_flashcards_by_category(
            client, deduplicated, [user_category]
        )

    status = f"Generated {len(deduplicated)} flashcards."
    logger.info(status)
    return deduplicated, status
