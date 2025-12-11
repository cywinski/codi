# %%
"""
Random Latent Patching Experiment: Testing the effect of patching specific latent
positions with random vectors on model performance.

This script compares baseline generation with all possible combinations of patching
latent vectors with random noise, evaluating which latent positions are most critical
for correct reasoning.
"""

import json
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from src.datasets import extract_answer_number
from src.model import CODI

# ============================================================================
# Utility Functions
# ============================================================================


def get_transformer_layers(model):
    """Robustly retrieve the ModuleList of decoder layers, handling PEFT wrappers."""
    obj = model.codi

    if hasattr(obj, "get_base_model"):
        obj = obj.get_base_model()

    if hasattr(obj, "model"):
        obj = obj.model

    if hasattr(obj, "model"):
        obj = obj.model

    if hasattr(obj, "layers"):
        return obj.layers

    # Fallback search
    print("Warning: Standard layer path not found. Searching modules...")
    for name, module in model.codi.named_modules():
        if "layers" in name and isinstance(module, torch.nn.ModuleList):
            return module

    raise AttributeError(f"Could not find transformer layers in {type(model.codi)}")


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


# ============================================================================
# Latent Capture and Patching Functions
# ============================================================================


def capture_latent_embeddings(model, tokenizer, prompt, num_latents=6):
    """Capture the latent embeddings (input tokens) used during latent thinking."""
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    device = model.codi.device

    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )

        past_key_values = outputs.past_key_values
        initial_latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            initial_latent_embd = model.prj(initial_latent_embd).to(
                dtype=model.codi.dtype
            )

        latent_embeddings = [initial_latent_embd.clone()]
        latent_embd = initial_latent_embd

        for i in range(num_latents):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

            if i < num_latents - 1:
                latent_embeddings.append(latent_embd.clone())

    return {"latent_embeddings": latent_embeddings}


def generate_random_latent_embeddings(model, latent_embeddings, seed=None):
    """Generate random latent embeddings with the same shape as the original embeddings."""
    if seed is not None:
        torch.manual_seed(seed)

    random_embeddings = []
    for emb in latent_embeddings:
        # Generate random embeddings with same shape and dtype
        random_emb = torch.randn_like(emb)
        # Normalize to have similar magnitude as original embeddings
        original_norm = torch.norm(emb, dim=-1, keepdim=True)
        random_norm = torch.norm(random_emb, dim=-1, keepdim=True)
        random_emb = random_emb * (original_norm / (random_norm + 1e-8))
        random_embeddings.append(random_emb)

    return random_embeddings


def generate_with_patched_latent_embeddings(
    model,
    tokenizer,
    prompt,
    latent_embeddings_dict,
    patch_positions,
    random_embeddings,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Generate with specific latent positions patched with random embeddings.

    Args:
        patch_positions: List of indices indicating which latent positions to patch with random vectors
    """
    device = model.codi.device
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    latent_embeddings = latent_embeddings_dict["latent_embeddings"]

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with normal prompt
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        # Latent loop - patch specified positions with random embeddings
        latent_embd = latent_embeddings[0].to(device)

        for i in range(num_latent_iterations):
            if i < len(latent_embeddings):
                # Use random embedding if this position should be patched
                if i in patch_positions:
                    latent_embd_input = random_embeddings[i].to(device)
                else:
                    latent_embd_input = latent_embeddings[i].to(device)
            else:
                latent_embd_input = latent_embd

            outputs = model.codi(
                inputs_embeds=latent_embd_input,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Generate from eocot token
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)
        output = eot_emb

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()

        for step in range(max_new_tokens):
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
# Template Functions
# ============================================================================


def _get_templates():
    """Get the list of all available templates."""
    return [
        "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
        "A company starts with {X} employees. They hire {Y} more employees. Then each current employee brings in {Z} additional people. How many people are there now in the company? Give the answer only and nothing else.",
        "A school starts with {X} students. They enroll {Y} new students. Then each current student brings {Z} additional students. How many students are there now in the school? Give the answer only and nothing else.",
        "A club starts with {X} members. They add {Y} new members. Then each current member invites {Z} additional people. How many people are there now in the club? Give the answer only and nothing else.",
        "A restaurant starts with {X} customers. They welcome {Y} more customers. Then each current customer brings {Z} additional customers. How many customers are there now in the restaurant? Give the answer only and nothing else.",
        "A gym starts with {X} members. They sign up {Y} new members. Then each current member refers {Z} additional people. How many people are there now in the gym? Give the answer only and nothing else.",
        "A band starts with {X} musicians. They add {Y} more musicians. Then each current musician brings {Z} additional musicians. How many musicians are there now in the band? Give the answer only and nothing else.",
        "A community starts with {X} residents. They welcome {Y} new residents. Then each current resident invites {Z} additional people. How many people are there now in the community? Give the answer only and nothing else.",
        "A group starts with {X} participants. They add {Y} new participants. Then each current participant brings {Z} additional people. How many people are there now in the group? Give the answer only and nothing else.",
        "A workshop starts with {X} attendees. They register {Y} more attendees. Then each current attendee brings {Z} additional people. How many people are there now in the workshop? Give the answer only and nothing else.",
    ]


def get_num_templates():
    """Get the number of available templates."""
    return len(_get_templates())


def get_template_and_param_mapping(template_idx):
    """Get template string and parameter mapping function for a given template index."""
    templates = _get_templates()

    def get_params(x, y, z, idx):
        """Get mapped parameters for template index."""
        # All templates use the same calculation: (X+Y) * (1+Z)
        param_mappings = [
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
            (x, y, z),
        ]
        return param_mappings[idx]

    return templates[template_idx], lambda x, y, z: get_params(x, y, z, template_idx)


def get_all_patch_combinations(num_latents):
    """
    Generate all possible combinations of latent positions to patch.
    Returns a list of tuples, where each tuple contains the indices to patch.
    """
    all_combinations = [()]  # Empty tuple means no patching (baseline)

    # Generate all combinations of different sizes
    for size in range(1, num_latents + 1):
        for combo in combinations(range(num_latents), size):
            all_combinations.append(combo)

    return all_combinations


def combination_to_string(combo, num_latents=6):
    """Convert a combination tuple to a readable string representation."""
    if len(combo) == 0:
        return "None"
    elif len(combo) == num_latents:
        return "All"
    else:
        return ",".join(str(i) for i in combo)


# ============================================================================
# Main Experiment
# ============================================================================


def main():
    """Run the random latent patching experiment."""
    load_dotenv()

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
        remove_eos=True,
        full_precision=True,
    )
    tokenizer = model.tokenizer

    def get_answer(X, Y, Z):
        """Compute ground truth answer: (X+Y) * (1+Z)"""
        step_1 = X + Y
        step_2 = step_1 * Z
        step_3 = step_1 + step_2
        return step_3

    # Experiment parameters
    num_samples_per_prompt = 3
    temperature = 1.0
    greedy = False
    num_test_cases = 50
    num_latents = 6
    random_seed = 42

    # Generate all possible combinations of base X, Y, Z in [1, 10]
    print("\nGenerating all possible combinations...")
    all_combinations = []
    for X in range(1, 11):
        for Y in range(1, 11):
            for Z in range(1, 11):
                all_combinations.append((X, Y, Z))

    print(f"Total combinations: {len(all_combinations)}")

    # Shuffle and select test cases
    np.random.seed(42)
    shuffled_combinations = np.array(all_combinations)
    np.random.shuffle(shuffled_combinations)
    test_case_combinations = shuffled_combinations[:num_test_cases].tolist()

    print(f"Test case samples: {len(test_case_combinations)}")

    # Get all patching combinations
    patch_combinations = get_all_patch_combinations(num_latents)
    print(f"\nTotal patching combinations to test: {len(patch_combinations)}")
    print(
        f"Combinations: {[combination_to_string(c) for c in patch_combinations[:10]]}..."
    )

    # Use first template for this experiment
    template_idx = 0
    template_str, param_mapping = get_template_and_param_mapping(template_idx)
    print(f"\nUsing template: {template_str[:80]}...")

    # Generate test cases
    print("\nGenerating test cases...")
    test_cases = []
    for i, (X, Y, Z) in enumerate(test_case_combinations):
        mapped_X, mapped_Y, mapped_Z = param_mapping(X, Y, Z)
        prompt = template_str.format(X=mapped_X, Y=mapped_Y, Z=mapped_Z)
        ground_truth = get_answer(X, Y, Z)
        test_cases.append(
            {
                "id": i,
                "X": X,
                "Y": Y,
                "Z": Z,
                "prompt": prompt,
                "ground_truth": ground_truth,
            }
        )

    # Storage for results
    results = []
    # Store accuracies per prompt per sample for error bar calculation
    combination_accuracies_per_prompt = {combo: [] for combo in patch_combinations}
    combination_accuracies_all = {combo: [] for combo in patch_combinations}

    # Run experiment
    print("\nRunning experiment...")
    for test_case in tqdm(test_cases, desc="Test cases"):
        prompt = test_case["prompt"]
        ground_truth = test_case["ground_truth"]

        try:
            # Capture latent embeddings once
            latent_embeddings_dict = capture_latent_embeddings(
                model, tokenizer, prompt, num_latents=num_latents
            )

            # Generate random embeddings once
            random_embeddings = generate_random_latent_embeddings(
                model, latent_embeddings_dict["latent_embeddings"], seed=random_seed
            )

            # Test each patching combination
            test_case_results = {
                "id": test_case["id"],
                "X": test_case["X"],
                "Y": test_case["Y"],
                "Z": test_case["Z"],
                "ground_truth": ground_truth,
                "combinations": {},
            }

            for combo in patch_combinations:
                combo_str = combination_to_string(combo, num_latents)
                combo_samples = []
                prompt_correct_list = []

                for sample_idx in range(num_samples_per_prompt):
                    if len(combo) == 0:
                        # Baseline: no patching, use regular generation
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
                            num_latent_iterations=num_latents,
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
                    else:
                        # Patched generation
                        output = generate_with_patched_latent_embeddings(
                            model,
                            tokenizer,
                            prompt,
                            latent_embeddings_dict=latent_embeddings_dict,
                            patch_positions=list(combo),
                            random_embeddings=random_embeddings,
                            max_new_tokens=128,
                            num_latent_iterations=num_latents,
                            temperature=temperature,
                            greedy=greedy,
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

                    combo_samples.append(
                        {
                            "answer": answer,
                            "correct": correct,
                            "text": generated_text,
                        }
                    )
                    prompt_correct_list.append(correct)
                    combination_accuracies_all[combo].append(correct)

                # Store per-prompt accuracy (mean across samples for this prompt)
                prompt_accuracy = np.mean(prompt_correct_list)
                combination_accuracies_per_prompt[combo].append(prompt_accuracy)

                test_case_results["combinations"][combo_str] = combo_samples

            results.append(test_case_results)

        except Exception as e:
            print(f"\nError processing test case {test_case['id']}: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "id": test_case["id"],
                    "X": test_case["X"],
                    "Y": test_case["Y"],
                    "Z": test_case["Z"],
                    "ground_truth": ground_truth,
                    "error": str(e),
                }
            )

    # Calculate accuracies and standard errors for each combination
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)

    combination_mean_accuracies = {}
    combination_std_errors = {}

    for combo in patch_combinations:
        combo_str = combination_to_string(combo, num_latents)
        if len(combination_accuracies_per_prompt[combo]) > 0:
            # Mean across prompts
            mean_acc = np.mean(combination_accuracies_per_prompt[combo])
            # Standard error across prompts (for error bars)
            std_err = np.std(combination_accuracies_per_prompt[combo]) / np.sqrt(
                len(combination_accuracies_per_prompt[combo])
            )
            combination_mean_accuracies[combo_str] = mean_acc
            combination_std_errors[combo_str] = std_err
            print(f"{combo_str:20s}: {mean_acc:.4f} ± {std_err:.4f}")
        else:
            combination_mean_accuracies[combo_str] = 0.0
            combination_std_errors[combo_str] = 0.0

    # Get baseline accuracy for delta calculation
    baseline_acc = combination_mean_accuracies.get("None", 0.0)
    baseline_std_err = combination_std_errors.get("None", 0.0)
    print(f"\nBaseline accuracy: {baseline_acc:.4f} ± {baseline_std_err:.4f}")

    # Calculate deltas and propagate errors
    combination_deltas = {}
    combination_delta_errors = {}
    for combo_str, acc in combination_mean_accuracies.items():
        if combo_str != "None":
            combination_deltas[combo_str] = acc - baseline_acc
            # Error propagation for difference: sqrt(err1^2 + err2^2)
            combination_delta_errors[combo_str] = np.sqrt(
                combination_std_errors[combo_str] ** 2 + baseline_std_err**2
            )

    # Save results
    output_file = "nov_29_random_latent_patching_results.json"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(
            {
                "results": results,
                "combination_accuracies": {
                    k: v for k, v in combination_mean_accuracies.items()
                },
                "combination_std_errors": {
                    k: v for k, v in combination_std_errors.items()
                },
                "combination_deltas": combination_deltas,
                "combination_delta_errors": combination_delta_errors,
                "baseline_accuracy": baseline_acc,
                "baseline_std_error": baseline_std_err,
                "num_samples_per_prompt": num_samples_per_prompt,
                "num_test_cases": num_test_cases,
            },
            f,
            indent=2,
        )

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(
        combination_mean_accuracies,
        combination_std_errors,
        combination_deltas,
        combination_delta_errors,
        baseline_acc,
        baseline_std_err,
        num_latents,
    )

    print("\nExperiment complete!")


def create_visualizations(
    combination_mean_accuracies,
    combination_std_errors,
    combination_deltas,
    combination_delta_errors,
    baseline_acc,
    baseline_std_err,
    num_latents,
):
    """Create separate visualizations with error bars for the random latent patching experiment."""

    # Sort combinations by number of patches and then by delta
    sorted_combos = sorted(
        combination_deltas.items(),
        key=lambda x: (
            len(x[0].split(","))
            if x[0] not in ["None", "All"]
            else (0 if x[0] == "None" else num_latents),
            -x[1],
        ),
    )

    # ========== Plot 1: Accuracy by combination ==========
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    combo_labels = ["None"] + [c[0] for c in sorted_combos]
    accuracies = [baseline_acc] + [
        combination_mean_accuracies[c[0]] for c in sorted_combos
    ]
    errors = [baseline_std_err] + [combination_std_errors[c[0]] for c in sorted_combos]

    # Color by number of patches
    colors = []
    for label in combo_labels:
        if label == "None":
            colors.append("green")
        elif label == "All":
            colors.append("red")
        else:
            num_patches = len(label.split(","))
            colors.append(plt.cm.YlOrRd(num_patches / num_latents))

    x_pos = np.arange(len(combo_labels))
    ax1.bar(
        x_pos,
        accuracies,
        yerr=errors,
        color=colors,
        alpha=0.7,
        capsize=3,
        error_kw={"linewidth": 1},
    )
    ax1.axhline(
        y=baseline_acc, color="green", linestyle="--", label="Baseline", linewidth=2
    )
    ax1.set_xlabel("Patched Positions", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(
        "Accuracy by Patching Combination (with Standard Error)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(combo_labels, rotation=90, ha="right", fontsize=6)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "nov_29_random_latent_patching_accuracy_by_combination.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Saved: nov_29_random_latent_patching_accuracy_by_combination.png")
    plt.close()

    # ========== Plot 2: Delta from baseline ==========
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    delta_labels = [c[0] for c in sorted_combos]
    deltas = [c[1] for c in sorted_combos]
    delta_errors = [combination_delta_errors[c[0]] for c in sorted_combos]

    colors_delta = []
    for label in delta_labels:
        if label == "All":
            colors_delta.append("red")
        else:
            num_patches = len(label.split(","))
            colors_delta.append(plt.cm.YlOrRd(num_patches / num_latents))

    x_pos_delta = np.arange(len(delta_labels))
    ax2.bar(
        x_pos_delta,
        deltas,
        yerr=delta_errors,
        color=colors_delta,
        alpha=0.7,
        capsize=3,
        error_kw={"linewidth": 1},
    )
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Patched Positions", fontsize=12)
    ax2.set_ylabel("Accuracy Delta from Baseline", fontsize=12)
    ax2.set_title(
        "Impact of Patching - Delta from Baseline (with Standard Error)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x_pos_delta)
    ax2.set_xticklabels(delta_labels, rotation=90, ha="right", fontsize=6)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "nov_29_random_latent_patching_delta_from_baseline.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Saved: nov_29_random_latent_patching_delta_from_baseline.png")
    plt.close()

    # ========== Plot 3: Accuracy by number of patches ==========
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    acc_by_num_patches = {i: [] for i in range(num_latents + 1)}

    acc_by_num_patches[0].append(baseline_acc)
    for combo_str, acc in combination_mean_accuracies.items():
        if combo_str == "None":
            continue
        elif combo_str == "All":
            acc_by_num_patches[num_latents].append(acc)
        else:
            num_patches = len(combo_str.split(","))
            acc_by_num_patches[num_patches].append(acc)

    # Calculate mean and std for each number of patches
    num_patches_keys = sorted(acc_by_num_patches.keys())
    mean_accs = [
        np.mean(acc_by_num_patches[k]) if acc_by_num_patches[k] else 0
        for k in num_patches_keys
    ]
    std_accs = [
        np.std(acc_by_num_patches[k]) if len(acc_by_num_patches[k]) > 1 else 0
        for k in num_patches_keys
    ]

    ax3.errorbar(
        num_patches_keys,
        mean_accs,
        yerr=std_accs,
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
        color="blue",
        ecolor="red",
    )
    ax3.axhline(
        y=baseline_acc, color="green", linestyle="--", label="Baseline", linewidth=2
    )
    ax3.fill_between(
        num_patches_keys,
        [baseline_acc - baseline_std_err] * len(num_patches_keys),
        [baseline_acc + baseline_std_err] * len(num_patches_keys),
        alpha=0.2,
        color="green",
        label="Baseline ± SE",
    )
    ax3.set_xlabel("Number of Patched Positions", fontsize=12)
    ax3.set_ylabel("Mean Accuracy", fontsize=12)
    ax3.set_title(
        "Accuracy vs Number of Patched Positions (with Standard Deviation)",
        fontsize=14,
        fontweight="bold",
    )
    ax3.set_xticks(num_patches_keys)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "nov_29_random_latent_patching_by_num_patches.png", dpi=300, bbox_inches="tight"
    )
    print("Saved: nov_29_random_latent_patching_by_num_patches.png")
    plt.close()

    # ========== Plot 4: Single position patching analysis ==========
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    single_patch_combos = [
        (str(i), combination_mean_accuracies[str(i)], combination_std_errors[str(i)])
        for i in range(num_latents)
    ]
    single_labels = [c[0] for c in single_patch_combos]
    single_accs = [c[1] for c in single_patch_combos]
    single_errors = [c[2] for c in single_patch_combos]
    single_deltas = [acc - baseline_acc for acc in single_accs]
    single_delta_errors = [
        np.sqrt(err**2 + baseline_std_err**2) for err in single_errors
    ]

    x_pos_single = np.arange(len(single_labels))
    bars = ax4.bar(
        x_pos_single,
        single_deltas,
        yerr=single_delta_errors,
        color=plt.cm.YlOrRd(0.5),
        alpha=0.7,
        capsize=5,
        error_kw={"linewidth": 2},
    )
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax4.set_xlabel("Single Latent Position Patched", fontsize=12)
    ax4.set_ylabel("Accuracy Delta from Baseline", fontsize=12)
    ax4.set_title(
        "Impact of Patching Individual Latent Positions (with Standard Error)",
        fontsize=14,
        fontweight="bold",
    )
    ax4.set_xticks(x_pos_single)
    ax4.set_xticklabels(single_labels, fontsize=12)
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, delta, err) in enumerate(
        zip(bars, single_deltas, single_delta_errors)
    ):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{delta:.3f}\n±{err:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        "nov_29_random_latent_patching_single_positions.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Saved: nov_29_random_latent_patching_single_positions.png")
    plt.close()

    # ========== Plot 5: Heatmap for pairwise combinations ==========
    fig5, ax5 = plt.subplots(figsize=(10, 8))

    # Create matrix for pairwise combinations
    pairwise_matrix = np.zeros((num_latents, num_latents))
    pairwise_matrix[:] = np.nan

    for combo_str, acc in combination_mean_accuracies.items():
        if combo_str in ["None", "All"]:
            continue
        positions = [int(x) for x in combo_str.split(",")]
        if len(positions) == 2:
            i, j = positions
            delta = acc - baseline_acc
            pairwise_matrix[i, j] = delta
            pairwise_matrix[j, i] = delta

    # Plot heatmap
    im = ax5.imshow(pairwise_matrix, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.1)
    ax5.set_xticks(np.arange(num_latents))
    ax5.set_yticks(np.arange(num_latents))
    ax5.set_xticklabels([f"L{i}" for i in range(num_latents)], fontsize=12)
    ax5.set_yticklabels([f"L{i}" for i in range(num_latents)], fontsize=12)
    ax5.set_xlabel("Latent Position", fontsize=14)
    ax5.set_ylabel("Latent Position", fontsize=14)
    ax5.set_title(
        "Pairwise Latent Patching Impact (Delta from Baseline)",
        fontsize=14,
        fontweight="bold",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label("Accuracy Delta", fontsize=12)

    # Add text annotations
    for i in range(num_latents):
        for j in range(num_latents):
            if not np.isnan(pairwise_matrix[i, j]):
                text = ax5.text(
                    j,
                    i,
                    f"{pairwise_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

    plt.tight_layout()
    plt.savefig(
        "nov_29_random_latent_patching_pairwise_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Saved: nov_29_random_latent_patching_pairwise_heatmap.png")
    plt.close()


if __name__ == "__main__":
    main()
