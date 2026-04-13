"""
Study agent for flashcard generation using OpenAI Structured Outputs.

This module implements intelligent, RAG-based flashcard generation where
the agent uses search_class_materials to find relevant content and then
generates flashcards with guaranteed JSON structure.
"""

import json
import logging
from typing import List, Dict, Tuple

from app.agents.chat_agent import get_async_openai_client, create_search_tool, create_list_sections_tool
from app.agents.run_agent import run_agent
from app.agents.tools import ToolBox
from app.config import DEFAULT_CHAT_MODEL, FLASHCARD_DEFAULT_N_RESULTS

logger = logging.getLogger(__name__)

# System prompt for art history flashcard generation
ART_HISTORY_FLASHCARD_PROMPT = """You are an expert art history flashcard generator.

Your goal: Create high-quality study flashcards from the student's uploaded course materials.

Process:
1. Call list_sections to see all available sections in the materials
2. Search each relevant section using search_class_materials with the section filter
3. Extract key information from ALL search results: artworks, artists, movements, terms, dates
4. Generate flashcards using generate_flashcards_structured tool with ALL gathered content

**CRITICAL CONSTRAINTS - Read Carefully**:
1. **Content Source - NO HALLUCINATIONS**:
   - ONLY create flashcards from content found in the search results above
   - Do NOT add information from your general knowledge base about art history
   - Do NOT include terms, artists, artworks, or movements unless they appear in the search results
   - If you think of something relevant but don't see it in search results, DO NOT include it
   - Stick strictly to the uploaded study materials - no external knowledge

2. **Respect User Intent**:
   - The user will specify what type of flashcards they want (terms, people, artworks, etc.)
   - Generate flashcards matching their request ONLY
   - If user asks for "terms and people", do NOT include artworks
   - If user asks for "artworks", do NOT include general terminology
   - Stay focused on what the user specifically requested

Example of what NOT to do:
❌ Adding "Baroque" because you know it's important (not in search results - HALLUCINATION)
❌ Adding "Cubism" because it follows in art history (not in search results - HALLUCINATION)
❌ Adding "Last Supper" when user asked for "terms and people" (wrong scope - not respecting user intent)
❌ Adding general definitions you know but didn't find in search results

Example of what TO do:
✅ When user asks for "terms and people": Include "Fresco" (term) and "Jan van Eyck" (person), exclude "Last Supper" (artwork)
✅ When user asks for "artworks": Include "Last Supper", "Pietà", "Birth of Venus"
✅ When user asks for "terms": Include only terminology like "Contrapposto", "Memento Mori"
✅ Always verify each flashcard exists in search results (no hallucinations)

Art history flashcard format:
- Term: Artist name, artwork title, period/style, or terminology
- Definition: Clear, concise explanation including:
  * For artworks: Artist, date, medium, style, key characteristics
  * For movements: Time period, key characteristics, major artists
  * For terms: Definition + example of usage

Quality guidelines:
- Generate flashcards for ALL content found **in the search results**
- Count should match the search results content (not your knowledge base)
- **Verify each term exists in search results before adding**
- **Avoid duplicates**: If you see the same term/artwork multiple times, create only ONE flashcard
- Focus on testable facts (who, what, when, where) **from the materials**
- Use proper art historical terminology **as it appears in the materials**
- Include specific dates when provided in search results
- Connect artworks to broader movements **only if shown in search results**

Example good flashcard:
Term: "Jan van Eyck, Arnolfini Portrait, 1434"
Definition: "Northern Renaissance oil painting depicting Giovanni Arnolfini and his wife. Showcases van Eyck's mastery of oil technique with incredible detail in fabrics, mirror reflection, and symbolic objects. Exemplifies Northern Renaissance realism and symbolism."

IMPORTANT: After searching for relevant information, you MUST call the generate_flashcards_structured tool to create the final flashcards."""

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
                        "description": "Artist/artwork/term/movement name"
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


def create_study_agent_config(class_name: str) -> dict:
    """
    Create agent configuration for flashcard generation.

    Args:
        class_name: Name of the class to generate flashcards for

    Returns:
        Agent configuration dict with system prompt, model, tools, kwargs
    """
    return {
        "name": "study_agent",
        "description": f"Art history flashcard generator for {class_name}",
        "model": DEFAULT_CHAT_MODEL,  # gpt-4o-mini supports both tool use and structured outputs
        "prompt": ART_HISTORY_FLASHCARD_PROMPT,
        "tools": ["search_class_materials", "list_sections", "generate_flashcards_structured"],
        "kwargs": {
            "temperature": 0.5  # Lower temperature for more consistent flashcard quality
        }
    }


async def generate_flashcards_for_topic(
    class_name: str,
    topic: str,
    count: int = 60
) -> Tuple[List[Dict[str, str]], str]:
    """
    Main entry point for flashcard generation.

    Uses an agentic approach where the model:
    1. Searches for relevant materials using search_class_materials tool
    2. Calls generate_flashcards_structured tool with retrieved context
    3. Returns structured flashcard data

    Args:
        class_name: Name of the class to generate flashcards for
        topic: User's topic/request (can be vague like "Baroque period")
        count: Target number of flashcards to generate (default 15)

    Returns:
        Tuple of (flashcards_list, status_message)
        - flashcards_list: List of dicts with 'term' and 'definition' keys
        - status_message: Status info about agent execution

    Raises:
        Exception: If agent fails or structured output is invalid
    """
    logger.info(f"Generating flashcards for class '{class_name}' on topic: {topic}")

    # Create toolbox for this session
    toolbox = ToolBox()
    client = get_async_openai_client()

    # Store flashcards in closure variable that tool can access
    generated_flashcards = []

    search_tool = create_search_tool(class_name, default_n_results=FLASHCARD_DEFAULT_N_RESULTS)
    list_sections_tool = create_list_sections_tool(class_name)
    toolbox.tool(search_tool)
    toolbox.tool(list_sections_tool)

    # Register generate_flashcards_structured tool
    @toolbox.tool
    async def generate_flashcards_structured(context: str, topic: str, count: int = 60) -> str:
        """
        Generate flashcards using OpenAI Structured Outputs.

        This tool uses the Structured Outputs API to generate flashcards
        with guaranteed JSON schema validation.

        Args:
            context: Retrieved course material text from search
            topic: User's requested topic
            count: Number of flashcards to generate (default 15)

        Returns:
            JSON string with flashcard array for agent consumption
        """
        nonlocal generated_flashcards  # Access closure variable

        logger.info(f"Calling Structured Outputs API for {count} flashcards on '{topic}'")

        try:
            # Call OpenAI with Structured Outputs
            response = await client.responses.create(
                model=DEFAULT_CHAT_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": "You are generating art history flashcards. Extract key information from the context and create clear, testable flashcards."
                    },
                    {
                        "role": "user",
                        "content": f"Context from course materials:\n\n{context}\n\nGenerate {count} high-quality flashcards on the topic: {topic}\n\nFocus on artworks, artists, movements, and key terminology."
                    }
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

            # Extract the structured output
            # response.output[0].content is a LIST of content blocks
            message_content = response.output[0].content
            if not message_content or len(message_content) == 0:
                logger.error("No content blocks in response")
                return "Error: Response contained no content blocks"

            # Get text from first content block
            flashcard_json = message_content[0].text
            if not flashcard_json:
                logger.error("Empty text in content block")
                return "Error: Content block was empty"

            # Parse JSON
            result = json.loads(flashcard_json)

            # Validate structure
            if 'flashcards' not in result:
                logger.error("Response missing 'flashcards' key")
                return "Error: Response missing 'flashcards' key"

            # Store flashcards in closure variable
            generated_flashcards = result['flashcards']
            logger.info(f"Generated {len(generated_flashcards)} flashcards")

            # Return success message for agent
            return f"Successfully generated {len(generated_flashcards)} flashcards. The flashcards have been created and are ready to use."

        except Exception as e:
            logger.error(f"Structured output generation failed: {e}")
            return f"Error generating flashcards: {str(e)}"

    # Create agent config
    agent_config = create_study_agent_config(class_name)

    # Parse user's request to understand what they want
    topic_lower = topic.lower()
    wants_terms = 'term' in topic_lower or 'definition' in topic_lower or 'concept' in topic_lower
    wants_people = 'people' in topic_lower or 'individual' in topic_lower or 'artist' in topic_lower or 'person' in topic_lower
    wants_artworks = 'artwork' in topic_lower or 'painting' in topic_lower or 'sculpture' in topic_lower or 'piece' in topic_lower

    # Build a single comprehensive search query
    query_parts = []
    if wants_terms:
        query_parts.append("terms definitions concepts")
    if wants_people:
        query_parts.append("artists individuals people")
    if wants_artworks:
        query_parts.append("artworks paintings sculptures")

    # Fallback if no specific intent detected
    if not query_parts:
        query_parts = ["terms definitions concepts artists artworks"]

    # Add topic to query
    query_parts.append(topic)

    comprehensive_query = " ".join(query_parts)
    search_instructions = [
        f"1. Call list_sections to see all available sections",
        f"2. Search each relevant section using search_class_materials with section filter and query '{comprehensive_query}'",
    ]

    # Build scope guidance
    scope_guidance = []
    if wants_terms:
        scope_guidance.append("✅ INCLUDE: Terms and definitions (e.g., 'Fresco', 'Contrapposto')")
    if wants_people:
        scope_guidance.append("✅ INCLUDE: People and individuals (e.g., 'Jan van Eyck', 'Martin Luther')")
    if wants_artworks:
        scope_guidance.append("✅ INCLUDE: Specific artworks (e.g., 'Last Supper', 'Pietà')")

    # Exclusions based on what user DIDN'T ask for
    exclusions = []
    if not wants_artworks and (wants_terms or wants_people):
        exclusions.append("❌ EXCLUDE: Specific artwork titles (user didn't ask for artworks)")
    if not wants_terms and (wants_people or wants_artworks):
        exclusions.append("❌ EXCLUDE: General terminology (user didn't ask for terms)")
    if not wants_people and (wants_terms or wants_artworks):
        exclusions.append("❌ EXCLUDE: Artist/people names (user didn't ask for people)")

    # Prepare user message with intent-based instructions
    user_message = (
        f"Generate flashcards based on: {topic}\n\n"
        f"USER'S REQUEST ANALYSIS:\n"
        f"{'- User wants: TERMS (definitions, concepts)\n' if wants_terms else ''}"
        f"{'- User wants: PEOPLE (artists, individuals)\n' if wants_people else ''}"
        f"{'- User wants: ARTWORKS (specific pieces)\n' if wants_artworks else ''}"
        f"\n"
        f"Search strategy:\n"
        f"{chr(10).join(search_instructions)}\n\n"
        f"CRITICAL CONSTRAINTS:\n"
        f"- Only create flashcards from content found in the search results above (NO HALLUCINATIONS)\n"
        f"- Do NOT add anything from your general knowledge base\n"
        f"- Respect user's request - generate ONLY what they asked for:\n"
        f"  {chr(10).join(scope_guidance)}\n"
        f"{'  ' + chr(10).join(exclusions) + chr(10) if exclusions else ''}"
        f"- Every flashcard must be traceable to the search results\n\n"
        f"Generate flashcards for ALL relevant content found IN THE SEARCH RESULTS matching user's request.\n\n"
        f"Before finalizing, verify:\n"
        f"1. Each flashcard appears in search results (no hallucinations from your knowledge)\n"
        f"2. Flashcards match user's request scope (terms/people/artworks as specified)\n\n"
        f"Then use generate_flashcards_structured to create the flashcards."
    )

    try:
        # Run agent loop
        response_text = await run_agent(
            client=client,
            toolbox=toolbox,
            agent=agent_config,
            user_message=user_message,
            history=[],
            usage=None
        )

        # Check if flashcards were generated
        if generated_flashcards:
            # Deduplicate flashcards (case-insensitive, keep first occurrence)
            seen_terms = set()
            deduplicated = []
            original_count = len(generated_flashcards)

            for card in generated_flashcards:
                term_lower = card['term'].lower().strip()
                if term_lower not in seen_terms:
                    seen_terms.add(term_lower)
                    deduplicated.append(card)

            duplicate_count = original_count - len(deduplicated)
            status = f"Generated {len(deduplicated)} flashcards."
            logger.info(status)
            return deduplicated, status
        else:
            # No flashcards generated - agent didn't call the tool or it failed
            error_msg = "Agent completed but did not generate flashcards. Try a more specific topic or check that materials are uploaded."
            logger.warning(error_msg)
            return [], error_msg

    except Exception as e:
        error_msg = f"Flashcard generation failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
