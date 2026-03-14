import inspect
import logging
from types import UnionType
from typing import Any, Callable, get_type_hints, Literal, get_origin, get_args, Union

from openai.types.responses import FunctionToolParam

_tools: dict[str, Callable] = {}
logger = logging.getLogger(__name__)


def _get_schema_type(_type: str):
    type_map = {
        'str': "string",
        'int': "integer",
        'float': "number",
        'bool': "boolean",
    }

    if result := type_map.get(_type):
        return {"type": result}

    return None


def _is_optional(annotation) -> bool:
    origin = get_origin(annotation)
    args = get_args(annotation)
    return (origin is UnionType or origin is Union) and type(None) in args


def _get_strict_json_schema_type(annotation) -> dict:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if _is_optional(annotation):
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _get_strict_json_schema_type(non_none_args[0])
        raise TypeError(f"Unsupported Union with multiple non-None values: {annotation}")

    if result := _get_schema_type(str(annotation.__name__)):
        return result

    if result := _get_schema_type(str(origin.__name__)):
        return result

    if origin is Literal:
        values = args
        if all(isinstance(v, (str, int, bool)) for v in values):
            return {"type": "string" if all(isinstance(v, str) for v in values) else "number", "enum": list(values)}
        raise TypeError("Unsupported Literal values in annotation")

    raise TypeError(f"Unsupported parameter type: {annotation}")


def _inspect_signature(func):
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    params = {}
    required = []

    for name, param in sig.parameters.items():
        if name in {"self", "ctx"}:
            continue

        ann = type_hints.get(name, param.annotation)
        if ann is inspect._empty:
            raise TypeError(f"Missing type annotation for parameter: {name}")

        schema_entry = _get_strict_json_schema_type(ann)

        required.append(name)
        params[name] = schema_entry

    return params, required


def _parse_signature(signature: str):
    # foo: int
    # bar: tuple[str, str]
    params = {}
    required = []
    for line in signature.splitlines():
        name, ptype = line.split(':')
        ptype = ptype.strip()
        params[name] = _get_schema_type(ptype)
        required.append(name)
    return params, required


def generate_function_schema(func: Callable[..., Any]) -> FunctionToolParam:
    params, required = _inspect_signature(func)

    return {
        "type": "function",
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {
            "type": "object",
            "properties": params,
            "required": required,
            "additionalProperties": False
        },
        "strict": True
    }


class ToolBox:
    _tools: list[FunctionToolParam]

    def __init__(self):
        self._funcs = {}
        self._tools = []

    def tool(self, func):
        self._funcs[func.__name__] = func
        self._tools.append(generate_function_schema(func))
        return func

    def get_tools(self, tool_names: list[str]) -> list[FunctionToolParam]:
        tls = [
            tool
            for tool in self._tools
            if tool['name'] in tool_names
        ]
        if 'web_search' in tool_names:
            # noinspection PyTypeChecker
            tls.append({'type': 'web_search'})
        return tls

    async def run_tool(self, tool_name: str, **kwargs):
        logger.debug('TOOL %s(%s)', tool_name, kwargs)
        tool = self._funcs.get(tool_name)
        result = tool(**kwargs)
        if inspect.iscoroutine(result):
            result = await result

        logger.debug('TOOL %s(%s) -> %s', tool_name, kwargs, result)
        return result
