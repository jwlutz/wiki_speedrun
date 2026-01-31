#!/usr/bin/env python3
"""Analyze OpenRouter models for Wikipedia speedrun."""

import json
from pathlib import Path

project_root = Path(__file__).parent.parent
models_file = project_root / "data" / "openrouter_models.json"

with open(models_file) as f:
    models = json.load(f)

# Filter for text models
text_models = [m for m in models if m.get('architecture') in ('text->text', 'text+image->text')]

# Calculate cost per 1M tokens
for m in text_models:
    prompt = m['pricing'].get('prompt')
    completion = m['pricing'].get('completion')
    if prompt:
        m['prompt_per_1m'] = float(prompt) * 1_000_000
    if completion:
        m['completion_per_1m'] = float(completion) * 1_000_000

# Sort by prompt price
text_models.sort(key=lambda x: x.get('prompt_per_1m', 999999))

print('=' * 85)
print('CHEAPEST TEXT MODELS (by input price)')
print('=' * 85)
print(f"{'Model ID':<55} {'In/1M':>10} {'Out/1M':>10} {'Context':>8}")
print('-' * 85)

for m in text_models[:25]:
    model_id = m['id'][:53]
    input_price = f"${m.get('prompt_per_1m', 0):.3f}"
    output_price = f"${m.get('completion_per_1m', 0):.3f}"
    context = str(m.get('context_length', 'N/A'))[:8]
    print(f"{model_id:<55} {input_price:>10} {output_price:>10} {context:>8}")

print()
print('=' * 85)
print('FRONTIER/FAST MODELS FOR SPEEDRUN')
print('=' * 85)

# Find specific models we care about
target_keywords = [
    'claude-3-haiku',
    'claude-3.5-haiku',
    'claude-3-5-haiku',
    'claude-haiku-4',
    'gemini-flash',
    'gemini-2.0-flash',
    'gemini-3',
    'gpt-4o-mini',
    'gpt-5',
    'deepseek-chat',
    'deepseek-v3',
    'mistral-small',
    'llama-3',
]

model_lookup = {m['id']: m for m in models}
found_ids = set()

for keyword in target_keywords:
    for mid, m in model_lookup.items():
        if keyword.lower() in mid.lower() and mid not in found_ids:
            found_ids.add(mid)
            input_price = float(m['pricing'].get('prompt', 0)) * 1_000_000
            output_price = float(m['pricing'].get('completion', 0)) * 1_000_000
            ctx = m.get('context_length', 'N/A')
            print(f"{mid:<55} ${input_price:>7.3f} in, ${output_price:>7.3f} out, {ctx} ctx")

print()
print('=' * 85)
print('FREE MODELS')
print('=' * 85)

free_models = [m for m in models if m['pricing'].get('prompt') == '0' and m['pricing'].get('completion') == '0']
for m in free_models[:15]:
    ctx = m.get('context_length', 'N/A')
    print(f"{m['id']:<55} FREE ({ctx} ctx)")
