# Pricing per 1M tokens (USD) for recent OpenAI models, fetched March 6, 2026 from https://developers.openai.com/api/docs/pricing.
import logging
import sys

from openai.types.responses import ResponseUsage

logger = logging.getLogger(__name__)

PRICING = {
    'gpt-5.4': {'input': 2.25, 'cached': 0.225, 'output': 18.00},
    'gpt-5.4-pro': {'input': 27.00, 'cached': 0.0, 'output': 216.00},
    'gpt-5.3-chat-latest': {'input': 1.75, 'cached': 0.175, 'output': 14.00},
    'gpt-5.3-codex': {'input': 1.75, 'cached': 0.175, 'output': 14.00},
    'gpt-5.2': {'input': 1.75, 'cached': 0.175, 'output': 14.00},
    'gpt-5.2-chat-latest': {'input': 1.75, 'cached': 0.175, 'output': 14.00},
    'gpt-5.2-codex': {'input': 1.75, 'cached': 0.175, 'output': 14.00},
    'gpt-5.2-pro': {'input': 21.00, 'cached': 0.0, 'output': 168.00},
    'gpt-5.1': {'input': 1.25, 'cached': 0.125, 'output': 10.00},
    'gpt-5.1-chat-latest': {'input': 1.25, 'cached': 0.125, 'output': 10.00},
    'gpt-5.1-codex': {'input': 1.25, 'cached': 0.125, 'output': 10.00},
    'gpt-5.1-codex-max': {'input': 1.25, 'cached': 0.125, 'output': 10.00},
    'gpt-5': {'input': 1.25, 'cached': 0.125, 'output': 10.00},
    'gpt-5-chat-latest': {'input': 1.25, 'cached': 0.125, 'output': 10.00},
    'gpt-5-codex': {'input': 1.25, 'cached': 0.125, 'output': 10.00},
    'gpt-5-pro': {'input': 15.00, 'cached': 0.0, 'output': 120.00},
    'gpt-5-mini': {'input': 0.25, 'cached': 0.025, 'output': 2.00},
    'gpt-5-nano': {'input': 0.05, 'cached': 0.005, 'output': 0.40},
    'gpt-4.1': {'input': 2.00, 'cached': 0.50, 'output': 8.00},
    'gpt-4.1-mini': {'input': 0.40, 'cached': 0.10, 'output': 1.60},
    'gpt-4.1-nano': {'input': 0.10, 'cached': 0.025, 'output': 0.40},
}


def _calculate_cost_usd(totals: dict[str, dict]) -> float:
    total = 0
    for model, usage in totals.items():
        rates = PRICING.get(model)
        if not rates:
            logger.warning('No pricing rates configured for model %s', model)
            continue
        total += usage['input'] * rates['input']
        total += usage['cached'] * rates.get('cached', rates['input'])
        total += usage['output'] * rates['output']
    # Prices are per 1M tokens.
    return total / 1_000_000


def _aggregate_usage(usages: list[tuple[str, ResponseUsage]]):
    total = {}
    for model, usage in usages:
        if model not in total:
            total[model] = {'input': 0, 'cached': 0, 'output': 0, 'reasoning': 0}
        total[model]['input'] += usage.input_tokens
        total[model]['cached'] += usage.input_tokens_details.cached_tokens
        total[model]['output'] += usage.output_tokens
        total[model]['reasoning'] += usage.output_tokens_details.reasoning_tokens
    return total


def print_usage(usages: list[tuple[str, ResponseUsage]], file=sys.stderr):
    print(' Usage '.center(30, '-'), file=file)
    totals = _aggregate_usage(usages)
    for model, total in totals.items():
        print(model.center(30, '~'), file=file)
        for key, value in total.items():
            print(f'{key.title()} (tokens):', value, file=file)
        cost = _calculate_cost_usd({model: total})
        print(f'{model} cost (USD): ${cost:.6f}', file=file)

    cost = _calculate_cost_usd(totals)
    print('~'*30, file=file)
    print(f'Total cost (USD): ${cost:.6f}', file=file)
