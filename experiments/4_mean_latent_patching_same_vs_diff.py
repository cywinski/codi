# %%
# ABOUTME: Mean latent patching experiment comparing same vs different intermediate values.
# ABOUTME: Tests whether latent vectors encode intermediate computational results.
"""
Mean Latent Patching Experiment: Same vs Different Intermediate Values

This experiment tests whether patching latent embeddings affects accuracy differently
when using mean latents from prompts with the SAME vs DIFFERENT intermediate values.

Key hypothesis: If latent vectors encode intermediate computational results,
patching with mean latents from prompts with the same intermediate values
should preserve accuracy better than patching with different intermediate values.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import extract_answer_number
from src.model import CODI
from src.templates import ADDITION_FIRST_TEMPLATES, SUBTRACTION_FIRST_TEMPLATES


# ============================================================================
# Utility Functions
# ============================================================================


def prepare_inputs(model, tokenizer, prompt):
    """Construct input sequence: [Prompt Tokens] + [BOCOT]"""
    device = model.codi.device

    inputs = tokenizer(
        prompt, return_tensors="pt", padding=False, add_special_tokens=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Add BOCOT token
    bot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
    input_ids_bot = torch.cat(
        [input_ids, torch.tensor([[bot_id]], device=device)], dim=1
    )
    attention_mask_bot = torch.cat(
        [attention_mask, torch.ones((1, 1), device=device)], dim=1
    )

    return input_ids_bot, attention_mask_bot


def get_answer(x: int, y: int, z: int, operation: str) -> tuple[int, int, int]:
    """
    Compute ground truth answer and intermediate values.

    Returns: (answer, step_1, step_2)
    """
    if operation == "addition":
        step_1 = x + y
    elif operation == "subtraction":
        step_1 = x - y
    else:
        raise ValueError(f"Unknown operation: {operation}")
    step_2 = step_1 * z
    answer = step_1 + step_2
    return answer, step_1, step_2


# ============================================================================
# Latent Embedding Collection
# ============================================================================


def collect_latent_embeddings(model, tokenizer, prompt, num_latent_iterations=6):
    """
    Collect all latent embeddings including initial and loop outputs.

    Returns a list of num_latent_iterations + 1 tensors:
    - Index 0: Initial latent embedding from BOCOT position (before loop)
    - Index 1 to num_latent_iterations: Output embeddings from each latent iteration

    Each tensor has shape [1, 1, hidden_size].
    """
    device = model.codi.device
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)

    latent_outputs = []

    with torch.no_grad():
        # Prefill with normal prompt
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        # Get initial latent embedding from BOCOT position
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Store the initial latent embedding (index 0)
        latent_outputs.append(latent_embd.cpu().clone())

        # Latent loop - collect outputs at each position (indices 1 to num_latent_iterations)
        for _ in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values

            # Get output latent embedding
            output_latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                output_latent_embd = model.prj(output_latent_embd).to(
                    dtype=model.codi.dtype
                )

            # Store the output embedding for this position
            latent_outputs.append(output_latent_embd.cpu().clone())

            # Use output as next input
            latent_embd = output_latent_embd

    return latent_outputs


def compute_mean_latents(latent_list, num_positions):
    """
    Compute mean latent embeddings at each position from a list of prompts.

    Args:
        latent_list: List of lists, where latent_list[prompt_idx][position]
                     is the latent embedding for that prompt at that position.
        num_positions: Total number of latent positions (including initial).

    Returns:
        List of mean embeddings, one for each position.
    """
    mean_latents = []

    for pos in range(num_positions):
        embeddings = [prompt_latents[pos] for prompt_latents in latent_list]
        stacked = torch.cat(embeddings, dim=0)
        mean_emb = stacked.mean(dim=0, keepdim=True)
        mean_latents.append(mean_emb)

    return mean_latents


# ============================================================================
# Generation with Mean Latent Patching
# ============================================================================


def generate_with_mean_latent_patching(
    model,
    tokenizer,
    prompt,
    patch_position,
    mean_latent,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Generate with the latent embedding patched at a specific position.

    Args:
        patch_position: Which latent vector to patch (0 = initial, 1-6 = loop outputs).
        mean_latent: The mean latent embedding to use for patching.
    """
    device = model.codi.device
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    batch_size = input_ids.size(0)

    with torch.no_grad():
        # Prefill with normal prompt
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        # Get initial latent embedding from BOCOT position
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Patch initial latent if position is 0
        if patch_position == 0:
            latent_embd = mean_latent.to(device).to(dtype=model.codi.dtype)

        # Latent loop with mean patching
        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values

            # Get output latent embedding
            output_latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                output_latent_embd = model.prj(output_latent_embd).to(
                    dtype=model.codi.dtype
                )

            # Patch output if this is the target position (positions 1-6 map to loop indices 0-5)
            if patch_position == i + 1:
                latent_embd = mean_latent.to(device).to(dtype=model.codi.dtype)
            else:
                latent_embd = output_latent_embd

        # Generate from eocot token
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)
        output = eot_emb

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()

        for _ in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]

            if greedy:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)

            sequences = torch.cat(
                [sequences, next_token_ids.expand(batch_size, -1)], dim=1
            )

            for batch_idx in range(batch_size):
                if not finished[batch_idx]:
                    token_id = next_token_ids[batch_idx, 0].item()
                    if token_id == tokenizer.eos_token_id:
                        finished[batch_idx] = True

            if finished.all():
                break

            next_token_emb = model.get_embd(model.codi, model.model_name)(
                next_token_ids
            ).to(device)
            output = next_token_emb

        return {"sequences": sequences}


# ============================================================================
# Prompt Generation
# ============================================================================


def generate_prompts(limit: int | None = None):
    """
    Generate prompts from both addition and subtraction templates.

    Returns list of dicts with prompt info including intermediate values.
    """
    # Generate all valid combinations where X > Y (for subtraction to be positive)
    combinations = [
        (x, y, z)
        for x in range(1, 9)
        for y in range(1, 9)
        for z in range(1, 9)
        if x > y
    ]

    np.random.seed(42)
    np.random.shuffle(combinations)

    # Apply limit per template if specified
    if limit is not None:
        combinations = combinations[:limit]

    prompts = []
    prompt_id = 0

    # Generate prompts for both operations
    for operation, templates in [
        ("addition", ADDITION_FIRST_TEMPLATES),
        ("subtraction", SUBTRACTION_FIRST_TEMPLATES),
    ]:
        for template_idx, template in enumerate(templates):
            for x, y, z in combinations:
                answer, step_1, step_2 = get_answer(x, y, z, operation)

                prompts.append(
                    {
                        "id": prompt_id,
                        "operation": operation,
                        "template_idx": template_idx,
                        "X": x,
                        "Y": y,
                        "Z": z,
                        "prompt": template.format(X=x, Y=y, Z=z),
                        "ground_truth": answer,
                        "step_1": step_1,
                        "step_2": step_2,
                        "intermediate_key": (step_1, z),  # For grouping
                    }
                )
                prompt_id += 1

    return prompts


def group_prompts_by_intermediate(prompts):
    """
    Group prompts by their intermediate values (step_1, Z).

    Two prompts have the same intermediate values if:
    - They have the same step_1 (X+Y or X-Y)
    - They have the same Z (multiplier)

    This means they will have the same step_2 = step_1 * Z and same final answer.
    """
    groups = defaultdict(list)
    for idx, p in enumerate(prompts):
        key = p["intermediate_key"]
        groups[key].append(idx)
    return groups


# ============================================================================
# Main Experiment
# ============================================================================


def main(
    limit: int | None = None,
    num_samples: int = 3,
    greedy: bool = False,
    temperature: float = 1.0,
    seed: int = 42,
):
    """
    Run the mean latent patching experiment comparing same vs different intermediates.

    Args:
        limit: Limit number of (X, Y, Z) combinations per template (for testing)
        num_samples: Number of samples per prompt per condition
        greedy: Use greedy decoding
        temperature: Temperature for sampling
        seed: Random seed
    """
    load_dotenv()
    np.random.seed(seed)

    # Create results directory
    results_dir = (
        Path(__file__).parent.parent / "results" / "mean_latent_patching_same_vs_diff"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Model setup
    print("Loading model...")
    model = CODI.from_pretrained(
        checkpoint_path="bcywinski/codi_llama1b-answer_only",
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        lora_r=128,
        lora_alpha=32,
        num_latent=6,
        use_prj=True,
        device="cuda",
        dtype="bfloat16",
        strict=False,
        checkpoint_save_path="./checkpoints/bcywinski/codi_llama1b-answer_only",
        remove_eos=False,
        full_precision=True,
    )
    tokenizer = model.tokenizer
    num_latent_iterations = 6
    num_positions = num_latent_iterations + 1  # Initial + 6 loop outputs = 7 positions

    # Generate prompts
    print("\nGenerating prompts...")
    prompts = generate_prompts(limit=limit)
    print(f"Generated {len(prompts)} prompts")

    # Group prompts by intermediate values
    groups = group_prompts_by_intermediate(prompts)
    print(f"Found {len(groups)} unique intermediate value groups")

    # Filter to groups with at least 2 prompts (need at least 1 for mean)
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"Groups with >= 2 prompts: {len(valid_groups)}")

    # Get all valid prompt indices
    valid_prompt_indices = set()
    for indices in valid_groups.values():
        valid_prompt_indices.update(indices)
    valid_prompt_indices = sorted(valid_prompt_indices)
    print(f"Total valid prompts: {len(valid_prompt_indices)}")

    # Phase 1: Collect latent embeddings from all valid prompts
    print("\nPhase 1: Collecting latent embeddings...")
    prompt_latents = {}  # prompt_idx -> latent_outputs
    for idx in tqdm(valid_prompt_indices, desc="Collecting latents"):
        prompt = prompts[idx]["prompt"]
        latent_outputs = collect_latent_embeddings(
            model, tokenizer, prompt, num_latent_iterations=num_latent_iterations
        )
        prompt_latents[idx] = latent_outputs

    # Phase 2: Pre-compute mean latents for each group
    print("\nPhase 2: Computing group mean latents...")
    group_mean_latents = {}  # (step_1, z) -> mean_latents (excluding computation)
    for key, indices in tqdm(valid_groups.items(), desc="Computing means"):
        # Get all latents for this group
        group_latent_list = [prompt_latents[idx] for idx in indices]
        group_mean_latents[key] = compute_mean_latents(
            group_latent_list, num_positions=num_positions
        )

    # Positions to test: baseline (None) + each latent position (0 to 6)
    positions_to_test = [None] + list(range(num_positions))

    # Results storage
    results_same = {pos: [] for pos in positions_to_test}  # Same intermediate
    results_diff = {pos: [] for pos in positions_to_test}  # Different intermediate
    detailed_results = []

    # Phase 3: Run patching experiment
    print("\nPhase 3: Running patching experiment...")

    # Select a subset of prompts for testing if there are too many
    test_indices = valid_prompt_indices
    if len(test_indices) > 200:
        np.random.shuffle(test_indices)
        test_indices = test_indices[:200]
        print(f"Sampling {len(test_indices)} prompts for testing")

    # Get all unique intermediate keys for "different" patching
    all_keys = list(valid_groups.keys())

    for prompt_idx in tqdm(test_indices, desc="Testing prompts"):
        prompt_info = prompts[prompt_idx]
        prompt = prompt_info["prompt"]
        ground_truth = prompt_info["ground_truth"]
        own_key = prompt_info["intermediate_key"]

        # Get group indices for same intermediate (excluding self)
        same_group_indices = [i for i in valid_groups[own_key] if i != prompt_idx]
        if len(same_group_indices) == 0:
            continue  # Skip if no other prompts with same intermediate

        # Compute mean latents for "same" (excluding self)
        same_latent_list = [prompt_latents[idx] for idx in same_group_indices]
        same_mean_latents = compute_mean_latents(same_latent_list, num_positions)

        # Select a different intermediate key for "different" patching
        different_keys = [k for k in all_keys if k != own_key]
        if len(different_keys) == 0:
            continue
        diff_key = different_keys[np.random.randint(len(different_keys))]
        diff_mean_latents = group_mean_latents[diff_key]

        prompt_results = {
            "prompt_idx": prompt_idx,
            "prompt_info": {
                "X": prompt_info["X"],
                "Y": prompt_info["Y"],
                "Z": prompt_info["Z"],
                "operation": prompt_info["operation"],
                "ground_truth": ground_truth,
                "step_1": prompt_info["step_1"],
                "step_2": prompt_info["step_2"],
            },
            "same_key": own_key,
            "diff_key": diff_key,
            "positions": {},
        }

        for position in positions_to_test:
            pos_key = "baseline" if position is None else str(position)
            prompt_results["positions"][pos_key] = {"same": [], "diff": []}

            for sample_idx in range(num_samples):
                # Baseline or Same intermediate patching
                if position is None:
                    # Baseline: no patching
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                        model.codi.device
                    )
                    attention_mask = tokenizer(
                        prompt, return_tensors="pt"
                    ).attention_mask.to(model.codi.device)
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=128,
                        num_latent_iterations=num_latent_iterations,
                        temperature=temperature,
                        greedy=greedy,
                        return_latent_vectors=False,
                        remove_eos=False,
                        output_attentions=False,
                        skip_thinking=False,
                        output_hidden_states=True,
                        sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
                        eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
                        verbalize_cot=False,
                    )
                    generated_text = tokenizer.decode(
                        output["sequences"][0], skip_special_tokens=False
                    )
                    answer = extract_answer_number(generated_text)
                    correct = (
                        answer is not None
                        and answer != float("inf")
                        and int(answer) == ground_truth
                    )

                    # Baseline is the same for both conditions
                    results_same[position].append(correct)
                    results_diff[position].append(correct)
                    prompt_results["positions"][pos_key]["same"].append(correct)
                    prompt_results["positions"][pos_key]["diff"].append(correct)
                else:
                    # Same intermediate patching
                    output_same = generate_with_mean_latent_patching(
                        model,
                        tokenizer,
                        prompt,
                        patch_position=position,
                        mean_latent=same_mean_latents[position],
                        max_new_tokens=128,
                        num_latent_iterations=num_latent_iterations,
                        temperature=temperature,
                        greedy=greedy,
                    )
                    text_same = tokenizer.decode(
                        output_same["sequences"][0], skip_special_tokens=False
                    )
                    answer_same = extract_answer_number(text_same)
                    correct_same = (
                        answer_same is not None
                        and answer_same != float("inf")
                        and int(answer_same) == ground_truth
                    )
                    results_same[position].append(correct_same)
                    prompt_results["positions"][pos_key]["same"].append(correct_same)

                    # Different intermediate patching
                    output_diff = generate_with_mean_latent_patching(
                        model,
                        tokenizer,
                        prompt,
                        patch_position=position,
                        mean_latent=diff_mean_latents[position],
                        max_new_tokens=128,
                        num_latent_iterations=num_latent_iterations,
                        temperature=temperature,
                        greedy=greedy,
                    )
                    text_diff = tokenizer.decode(
                        output_diff["sequences"][0], skip_special_tokens=False
                    )
                    answer_diff = extract_answer_number(text_diff)
                    correct_diff = (
                        answer_diff is not None
                        and answer_diff != float("inf")
                        and int(answer_diff) == ground_truth
                    )
                    results_diff[position].append(correct_diff)
                    prompt_results["positions"][pos_key]["diff"].append(correct_diff)

        detailed_results.append(prompt_results)

    # Calculate aggregate statistics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Calculate accuracies
    accuracies_same = {}
    accuracies_diff = {}
    std_same = {}
    std_diff = {}

    for pos in positions_to_test:
        pos_key = "baseline" if pos is None else str(pos)
        if len(results_same[pos]) > 0:
            accuracies_same[pos_key] = np.mean(results_same[pos])
            std_same[pos_key] = np.std(results_same[pos]) / np.sqrt(
                len(results_same[pos])
            )
        else:
            accuracies_same[pos_key] = 0.0
            std_same[pos_key] = 0.0

        if len(results_diff[pos]) > 0:
            accuracies_diff[pos_key] = np.mean(results_diff[pos])
            std_diff[pos_key] = np.std(results_diff[pos]) / np.sqrt(
                len(results_diff[pos])
            )
        else:
            accuracies_diff[pos_key] = 0.0
            std_diff[pos_key] = 0.0

        print(
            f"Position {pos_key:10s}: Same={accuracies_same[pos_key]:.4f}±{std_same[pos_key]:.4f}  Diff={accuracies_diff[pos_key]:.4f}±{std_diff[pos_key]:.4f}"
        )

    # Calculate deltas from baseline
    baseline_acc = accuracies_same.get("baseline", 0.0)
    baseline_std = std_same.get("baseline", 0.0)
    print(f"\nBaseline accuracy: {baseline_acc:.4f} ± {baseline_std:.4f}")

    deltas_same = {}
    deltas_diff = {}
    delta_std_same = {}
    delta_std_diff = {}

    for pos in range(num_positions):
        pos_key = str(pos)
        deltas_same[pos_key] = accuracies_same[pos_key] - baseline_acc
        deltas_diff[pos_key] = accuracies_diff[pos_key] - baseline_acc
        delta_std_same[pos_key] = np.sqrt(std_same[pos_key] ** 2 + baseline_std**2)
        delta_std_diff[pos_key] = np.sqrt(std_diff[pos_key] ** 2 + baseline_std**2)

    print("\nDeltas from baseline:")
    for pos in range(num_positions):
        pos_key = str(pos)
        print(
            f"  Position {pos_key}: Same={deltas_same[pos_key]:+.4f}±{delta_std_same[pos_key]:.4f}  Diff={deltas_diff[pos_key]:+.4f}±{delta_std_diff[pos_key]:.4f}"
        )

    # Save results
    output_data = {
        "config": {
            "limit": limit,
            "num_samples": num_samples,
            "greedy": greedy,
            "temperature": temperature,
            "seed": seed,
            "num_prompts_tested": len(test_indices),
            "num_groups": len(valid_groups),
        },
        "accuracies_same": accuracies_same,
        "accuracies_diff": accuracies_diff,
        "std_same": std_same,
        "std_diff": std_diff,
        "deltas_same": deltas_same,
        "deltas_diff": deltas_diff,
        "delta_std_same": delta_std_same,
        "delta_std_diff": delta_std_diff,
        "baseline_accuracy": baseline_acc,
        "baseline_std": baseline_std,
        "detailed_results": detailed_results,
    }

    output_file = results_dir / "results.json"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    # Create visualization
    print("\nCreating visualization...")
    create_visualizations(
        accuracies_same,
        accuracies_diff,
        std_same,
        std_diff,
        deltas_same,
        deltas_diff,
        delta_std_same,
        delta_std_diff,
        baseline_acc,
        baseline_std,
        num_positions,
        results_dir,
    )

    print("\nExperiment complete!")


def create_visualizations(
    accuracies_same,
    accuracies_diff,
    std_same,
    std_diff,
    deltas_same,
    deltas_diff,
    delta_std_same,
    delta_std_diff,
    baseline_acc,
    baseline_std,
    num_positions,
    results_dir,
):
    """Create both accuracy and delta plots."""
    # Font sizes matching experiment 1 style
    fontsize_title = 20
    fontsize_label = 18
    fontsize_tick = 16
    fontsize_legend = 14

    positions = list(range(num_positions))
    x = np.arange(len(positions))
    width = 0.35

    # =========================================================================
    # Plot 1: Absolute accuracy
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    same_accs = [accuracies_same[str(i)] for i in positions]
    diff_accs = [accuracies_diff[str(i)] for i in positions]
    same_errors = [std_same[str(i)] for i in positions]
    diff_errors = [std_diff[str(i)] for i in positions]

    ax.bar(
        x - width / 2,
        same_accs,
        width,
        yerr=same_errors,
        label="Same Intermediate",
        color="#4CAF50",
        alpha=0.8,
        capsize=8,
        edgecolor="black",
        linewidth=2,
    )
    ax.bar(
        x + width / 2,
        diff_accs,
        width,
        yerr=diff_errors,
        label="Different Intermediate",
        color="#e74c3c",
        alpha=0.8,
        capsize=8,
        edgecolor="black",
        linewidth=2,
    )

    # Baseline horizontal dotted line
    ax.axhline(
        y=baseline_acc,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Baseline ({baseline_acc:.3f})",
    )

    ax.set_xlabel("Latent Vector Index", fontsize=fontsize_label, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=fontsize_label, fontweight="bold")
    ax.set_title(
        "Patching latent vectors with vectors averaged from different prompts",
        fontsize=fontsize_title,
        fontweight="bold",
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in positions], fontsize=fontsize_tick)
    ax.tick_params(axis="y", labelsize=fontsize_tick)
    ax.legend(loc="best", fontsize=fontsize_legend)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    output_path = results_dir / "accuracy_by_position.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)

    # =========================================================================
    # Plot 2: Delta from baseline
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    same_deltas = [deltas_same[str(i)] for i in positions]
    diff_deltas = [deltas_diff[str(i)] for i in positions]
    same_delta_errors = [delta_std_same[str(i)] for i in positions]
    diff_delta_errors = [delta_std_diff[str(i)] for i in positions]

    ax.bar(
        x - width / 2,
        same_deltas,
        width,
        yerr=same_delta_errors,
        label="Same Intermediate",
        color="#4CAF50",
        alpha=0.8,
        capsize=8,
        edgecolor="black",
        linewidth=2,
    )
    ax.bar(
        x + width / 2,
        diff_deltas,
        width,
        yerr=diff_delta_errors,
        label="Different Intermediate",
        color="#e74c3c",
        alpha=0.8,
        capsize=8,
        edgecolor="black",
        linewidth=2,
    )

    # Reference line at 0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5)

    ax.set_xlabel("Latent Vector Index", fontsize=fontsize_label, fontweight="bold")
    ax.set_ylabel(
        "Accuracy Delta from Baseline", fontsize=fontsize_label, fontweight="bold"
    )
    ax.set_title(
        "Patching latent vectors with vectors averaged from different prompts",
        fontsize=fontsize_title,
        fontweight="bold",
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in positions], fontsize=fontsize_tick)
    ax.tick_params(axis="y", labelsize=fontsize_tick)
    ax.legend(loc="best", fontsize=fontsize_legend)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)

    # Adjust y-axis
    all_vals = same_deltas + diff_deltas
    all_errs = same_delta_errors + diff_delta_errors
    y_min = min(all_vals) - max(all_errs) - 0.1
    y_max = max(all_vals) + max(all_errs) + 0.15
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    output_path = results_dir / "delta_by_position.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close(fig)


# %%
if __name__ == "__main__":
    fire.Fire(main)
