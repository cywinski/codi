# ABOUTME: Generate prompts as JSON with both addition and subtraction variants.
# ABOUTME: Limits to 50 prompts per template with X > Y constraint for subtraction.

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.templates import ADDITION_FIRST_TEMPLATES, SUBTRACTION_FIRST_TEMPLATES


# %%
def get_answer(x: int, y: int, z: int, operation: str = "addition") -> int:
    """Compute ground truth answer based on operation type.

    Addition: (X+Y)*Z + (X+Y)
    Subtraction: (X-Y)*Z + (X-Y)
    """
    if operation == "addition":
        step_1 = x + y
    elif operation == "subtraction":
        step_1 = x - y
    else:
        raise ValueError(f"Unknown operation: {operation}")
    step_2 = step_1 * z
    step_3 = step_1 + step_2
    return int(step_3)


def get_template(template_idx: int, operation: str = "addition") -> str:
    if operation == "addition":
        templates = ADDITION_FIRST_TEMPLATES
    elif operation == "subtraction":
        templates = SUBTRACTION_FIRST_TEMPLATES
    else:
        raise ValueError(f"Unknown operation: {operation}")
    return templates[template_idx]


# %%
# Generate all combinations where X > Y (1-8 range)
combinations = [
    (x, y, z) for x in range(1, 9) for y in range(1, 9) for z in range(1, 9) if x > y
]

# Seed for reproducibility
np.random.seed(42)
np.random.shuffle(combinations)

# Limit to 50 per template
combinations_limited = combinations[:50]

print(f"Using {len(combinations_limited)} combinations per template")

# %%
# Create output directory
output_dir = Path("prompts")
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Generate combined prompts
prompts_data = []

for template_idx in range(len(ADDITION_FIRST_TEMPLATES)):
    addition_template = get_template(template_idx, operation="addition")
    subtraction_template = get_template(template_idx, operation="subtraction")

    for prompt_idx, (x, y, z) in enumerate(combinations_limited):
        addition_prompt = addition_template.format(X=x, Y=y, Z=z)
        subtraction_prompt = subtraction_template.format(X=x, Y=y, Z=z)

        addition_answer = get_answer(x, y, z, operation="addition")
        subtraction_answer = get_answer(x, y, z, operation="subtraction")

        entry = {
            "id": len(prompts_data),
            "template_idx": template_idx,
            "prompt_idx_in_template": prompt_idx,
            "X": int(x),
            "Y": int(y),
            "Z": int(z),
            "addition": {
                "prompt": addition_prompt,
                "ground_truth": addition_answer,
            },
            "subtraction": {
                "prompt": subtraction_prompt,
                "ground_truth": subtraction_answer,
            },
        }
        prompts_data.append(entry)

# %%
# Save to JSON
output_file = output_dir / "prompts.json"
with open(output_file, "w") as f:
    json.dump(
        {
            "config": {
                "num_templates": len(ADDITION_FIRST_TEMPLATES),
                "prompts_per_template": len(combinations_limited),
                "total_prompts": len(prompts_data),
                "constraint": "X > Y",
                "seed": 42,
            },
            "prompts": prompts_data,
        },
        f,
        indent=2,
    )

print(f"✓ Prompts saved to {output_file}")
print(f"  Total: {len(prompts_data)} prompts")
print(f"  Templates: {len(ADDITION_FIRST_TEMPLATES)}")
print(f"  Prompts per template: {len(combinations_limited)}")
print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
