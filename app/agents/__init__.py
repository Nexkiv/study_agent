"""
Agentic layer for StudyAgent.

Contains:
- run_agent.py: Agent execution loop from CS 301R course
- tools.py: ToolBox class for function tool definitions
- chat_agent.py: RAG chat agent with tool use (Phase 3)
- study_agent.py: Flashcard/quiz generation (Phase 4)
"""
from app.agents.run_agent import run_agent, as_tool, Agent, current_agent, conclude
from app.agents.tools import ToolBox

__all__ = [
    'run_agent',
    'as_tool',
    'Agent',
    'current_agent',
    'conclude',
    'ToolBox'
]
