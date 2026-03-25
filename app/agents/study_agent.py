"""
Study agent for flashcard generation using OpenAI Structured Outputs.

This module implements intelligent, RAG-based flashcard generation where
the agent uses search_class_materials to find relevant content and then
generates flashcards with guaranteed JSON structure.
"""

import json
import logging
from typing import List, Dict, Tuple

from app.agents.chat_agent import get_async_openai_client, create_search_tool
from app.agents.run_agent import run_agent
from app.agents.tools import ToolBox
from app.config import DEFAULT_CHAT_MODEL

logger = logging.getLogger(__name__)

# System prompt for art history flashcard generation
ART_HISTORY_FLASHCARD_PROMPT = """You are an expert art history flashcard generator.

Your goal: Create high-quality study flashcards from the student's uploaded course materials.

Process:
1. Use search_class_materials to find relevant content for the requested topic
2. Extract key information: artworks, artists, movements, terms, dates
3. Generate flashcards using generate_flashcards_structured tool

Art history flashcard format:
- Term: Artist name, artwork title, period/style, or terminology
- Definition: Clear, concise explanation including:
  * For artworks: Artist, date, medium, style, key characteristics
  * For movements: Time period, key characteristics, major artists
  * For terms: Definition + example of usage

Quality guidelines:
- 10-20 flashcards per generation (not too many)
- Focus on testable facts (who, what, when, where)
- Use proper art historical terminology
- Include specific dates when known
- Connect artworks to broader movements/contexts

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
        "tools": ["search_class_materials", "generate_flashcards_structured"],
        "kwargs": {
            "temperature": 0.5  # Lower temperature for more consistent flashcard quality
        }
    }


async def generate_flashcards_for_topic(
    class_name: str,
    topic: str,
    count: int = 15
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

    # Register search_class_materials tool (reuse from chat_agent)
    search_tool = create_search_tool(class_name)
    toolbox.tool(search_tool)

    # Register generate_flashcards_structured tool
    @toolbox.tool
    async def generate_flashcards_structured(context: str, topic: str, count: int = 15) -> str:
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

    # Prepare user message with topic and count
    user_message = (
        f"Generate {count} flashcards on the following topic: {topic}\n\n"
        f"First, search for relevant materials using search_class_materials. "
        f"Then, use the generate_flashcards_structured tool to create the flashcards."
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
            status = f"Agent completed successfully. Search and generation tools used."
            logger.info(status)
            return generated_flashcards, status
        else:
            # No flashcards generated - agent didn't call the tool or it failed
            error_msg = "Agent completed but did not generate flashcards. Try a more specific topic or check that materials are uploaded."
            logger.warning(error_msg)
            return [], error_msg

    except Exception as e:
        error_msg = f"Flashcard generation failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
