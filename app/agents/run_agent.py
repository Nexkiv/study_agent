import asyncio
import json
import logging
import time
from contextvars import ContextVar
from typing import TypedDict

current_agent = ContextVar('current_agent')
logger = logging.getLogger(__name__)


class Agent(TypedDict):
    name: str
    description: str
    model: str
    prompt: str
    tools: list[str]
    kwargs: dict


def conclude():
    """
    Conclude the conversation.
    """


async def run_agent(
        client,
        toolbox,
        agent: Agent,
        user_message: str = None,
        history=None,
        usage=None
) -> str | None:
    current_agent.set(agent)

    if history is None:
        history = []
    if usage is None:
        usage = []

    if user_message:
        history.append({'role': 'user', 'content': user_message})

    while True:
        history_for_response = history
        if prompt := agent.get('prompt'):
            history_for_response = history_for_response + [{'role': 'system', 'content': prompt}]

        start = time.time()
        logger.debug('AGENT %s', agent['name'])
        response = await client.responses.create(
            input=history_for_response,
            model=agent.get('model', 'gpt-5-mini'),
            tools=toolbox.get_tools(agent.get('tools', [])),
            **agent.get('kwargs', {})
        )
        logger.debug(
            'RESPONSE from %s in %.2f seconds',
            agent['name'],
            time.time() - start,
        )

        usage.append((agent.get('model', response.model), response.usage))
        history.extend(
            response.output
        )

        # output -> we're done
        if outputs := [
            item
            for item in response.output
            if item.type == 'message'
        ]:
            return '\n'.join(
                chunk.text
                for item in outputs
                for chunk in item.content
            )

        # tool calls
        tool_calls = {
            item.call_id: toolbox.run_tool(item.name, **json.loads(item.arguments))
            for item in response.output
            if item.type == 'function_call'
        }

        results = await asyncio.gather(*(
            asyncio.create_task(tool_call)
            for tool_call in tool_calls.values()
        ))

        for call_id, result in zip(tool_calls.keys(), results):
            history.append({
                'type': 'function_call_output',
                'call_id': call_id,
                'output': str(result)
            })

        if any(
                item.type == 'function_call'
                and item.name == conclude.__name__
                for item in response.output
        ):
            return None


def as_tool(
        client, toolbox, agent,
        history=None,
        usage=None
):
    async def function(input: str) -> str:
        return await run_agent(
            client, toolbox, agent,
            user_message=input, history=history, usage=usage
        )

    function.__name__ = agent['name']
    function.__doc__ = agent.get('description', '')

    return function
