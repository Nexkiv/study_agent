"""
Deterministic export pipelines for flashcards.

This module provides pure data transformation functions for exporting
flashcards to various formats (Quizlet TSV, Anki CSV). No LLM reasoning,
just text formatting.
"""

import csv
import io
from typing import List, Dict


def export_to_quizlet(flashcards: List[Dict[str, str]]) -> str:
    """
    Export flashcards to Quizlet TSV format.

    Quizlet uses tab-separated values with one flashcard per line:
    term \t definition

    Args:
        flashcards: List of dicts with 'term' and 'definition' keys

    Returns:
        TSV string ready for Quizlet import

    Example:
        >>> cards = [{"term": "Baroque", "definition": "17th century style..."}]
        >>> tsv = export_to_quizlet(cards)
        >>> print(tsv)
        Baroque\t17th century style...
    """
    if not flashcards:
        return ""

    lines = []
    for card in flashcards:
        # Get term and definition, strip tabs and newlines (breaks Quizlet import)
        term = card.get('term', '').replace('\t', ' ').replace('\n', ' ').strip()
        definition = card.get('definition', '').replace('\t', ' ').replace('\n', ' ').strip()

        # Skip empty cards
        if not term or not definition:
            continue

        # Quizlet format: term \t definition
        lines.append(f"{term}\t{definition}")

    return '\n'.join(lines)


def export_to_anki(flashcards: List[Dict[str, str]]) -> str:
    """
    Export flashcards to Anki CSV format.

    Anki uses CSV format with proper escaping for quotes and commas:
    term, definition

    Args:
        flashcards: List of dicts with 'term' and 'definition' keys

    Returns:
        CSV string ready for Anki import

    Example:
        >>> cards = [{"term": "Baroque", "definition": "Style with quotes"}]
        >>> csv_output = export_to_anki(cards)
        >>> # CSV module handles proper escaping automatically
    """
    if not flashcards:
        return ""

    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    for card in flashcards:
        term = card.get('term', '').strip()
        definition = card.get('definition', '').strip()

        # Skip empty cards
        if not term or not definition:
            continue

        # Write as CSV row (handles escaping automatically)
        writer.writerow([term, definition])

    return output.getvalue()
