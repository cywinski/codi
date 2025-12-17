# %%
# ABOUTME: Tests latent vector steering by patching with averaged vectors from specific intermediate values.
# ABOUTME: Evaluates if patching a latent position can steer the model to produce a target intermediate computation.
"""
Latent Steering by Intermediate Value Experiment

This script tests whether patching a specific latent vector position with an averaged
latent vector (computed from prompts with a specific first intermediate value) can
steer the model's output to match that intermediate value's expected answer.

Approach:
1. For each template, mean-ablate the prompt activations
2. Compute averaged latent vectors grouped by intermediate value (X+Y) in range 2-14
3. For each test case:
   - Baseline: Mean-Abl Prompt + Patched Original Latents
   - Intervention: Patch one latent position with averaged vector for target intermediate value
   - Check if output matches expected answer for the patched intermediate value
"""

import json
import sys
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
# Configuration
# ============================================================================

RESULTS_DIR = Path(__file__).parent.parent / "results" / "latent_steering_intermediate"

# Range of intermediate values to test (X+Y ranges from 2 to 14 with X,Y in 1-7)
MIN_INTERMEDIATE = 2
MAX_INTERMEDIATE = 4


# ============================================================================
# Template Utilities
# ============================================================================


def get_all_templates():
    """Get all templates as a single list (addition first, then subtraction)."""
    return ADDITION_FIRST_TEMPLATES + SUBTRACTION_FIRST_TEMPLATES


def get_num_templates():
    """Get the total number of templates."""
    return len(get_all_templates())


def is_subtraction_template(template_idx):
    """Check if a template index corresponds to a subtraction template."""
    return template_idx >= len(ADDITION_FIRST_TEMPLATES)


def get_mapped_params(X, Y, Z, template_idx):
    """
    Get the mapped parameters for a given template.

    For addition templates: use (X, Y, Z) directly
    For subtraction templates: use (X + 2Y, Y, Z) to get the same intermediate values
    """
    if is_subtraction_template(template_idx):
        return (X + 2 * Y, Y, Z)
    return (X, Y, Z)


def get_answer(X, Y, Z):
    """
    Compute ground truth answer: (X+Y) * (1+Z)
    """
    step_1 = X + Y
    step_2 = step_1 * Z
    step_3 = step_1 + step_2
    return step_3


def get_answer_from_intermediate(intermediate_value, Z):
    """
    Compute answer given intermediate value (X+Y) and Z.
    Answer = intermediate * (1 + Z)
    """
    return intermediate_value * (1 + Z)


# ============================================================================
# Model Utilities
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

    bot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
    input_ids_bot = torch.cat(
        [input_ids, torch.tensor([[bot_id]], device=device)], dim=1
    )
    attention_mask_bot = torch.cat(
        [attention_mask, torch.ones((1, 1), device=device)], dim=1
    )

    return input_ids_bot, attention_mask_bot


# ============================================================================
# Activation Capture Functions
# ============================================================================


def capture_prompt_activations(model, tokenizer, prompt):
    """Capture residual stream activations for all layers and all prompt tokens."""
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    device = model.codi.device
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    captured_prefill = [None] * num_layers
    handles = []

    def get_prefill_hook(layer_idx):
        def hook(module, args, output):
            act = output[0] if isinstance(output, tuple) else output
            captured_prefill[layer_idx] = act.detach().cpu()

        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(get_prefill_hook(i)))

    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )

    for h in handles:
        h.remove()

    seq_len = input_ids.shape[1] - 1
    prefill_tensor = torch.stack([x[0, :seq_len, :] for x in captured_prefill])

    return {"prefill": prefill_tensor, "input_ids": input_ids[:, :seq_len]}


def compute_mean_prompt_activations(model, tokenizer, template_str, combinations):
    """Compute mean activations across multiple template prompts using specific combinations."""
    prompts = []
    for X, Y, Z in combinations:
        prompt = template_str.format(X=X, Y=Y, Z=Z)
        prompts.append(prompt)

    all_activations = []
    all_seq_lens = []

    for prompt in tqdm(prompts, desc="Capturing activations", leave=False):
        result = capture_prompt_activations(model, tokenizer, prompt)
        activations = result["prefill"]
        seq_len = activations.shape[1]
        all_activations.append(activations)
        all_seq_lens.append(seq_len)

    if len(set(all_seq_lens)) > 1:
        raise ValueError(
            f"All prompts must tokenize to the same number of tokens, but got lengths: {all_seq_lens}"
        )

    max_seq_len = max(all_seq_lens)
    num_layers = all_activations[0].shape[0]
    hidden_dim = all_activations[0].shape[2]

    mean_activations_list = []
    for pos in range(max_seq_len):
        pos_activations = []
        for act in all_activations:
            seq_len = act.shape[1]
            if pos < seq_len:
                pos_activations.append(act[:, pos, :])

        if len(pos_activations) > 0:
            stacked_pos = torch.stack(pos_activations, dim=0)
            mean_pos = torch.mean(stacked_pos, dim=0)
            mean_activations_list.append(mean_pos)
        else:
            mean_activations_list.append(torch.zeros(num_layers, hidden_dim))

    mean_activations = torch.stack(mean_activations_list, dim=1)

    return {
        "mean_prefill": mean_activations,
        "token_counts": torch.tensor(all_seq_lens),
        "max_seq_len": max_seq_len,
    }


def capture_latent_embeddings(model, tokenizer, prompt, num_latents=6):
    """Capture the latent embeddings using model.generate with return_latent_vectors=True."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.codi.device)
    attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask.to(
        model.codi.device
    )

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        num_latent_iterations=num_latents,
        greedy=True,
        return_latent_vectors=True,
        sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
        eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
    )

    latent_vectors = output["latent_vectors"]
    return {"latent_embeddings": latent_vectors}


def compute_mean_latent_embeddings_by_intermediate_value(
    model,
    tokenizer,
    current_template_idx,
    all_combinations,
    num_latents=6,
):
    """
    Compute mean latent embeddings grouped by intermediate value (X+Y).

    For each intermediate value in range MIN_INTERMEDIATE to MAX_INTERMEDIATE,
    computes mean latent vectors across all other templates with prompts
    that yield that intermediate value.

    Returns:
        Dict mapping intermediate_value -> list of mean latent embeddings per step
    """
    num_templates = get_num_templates()
    all_templates = get_all_templates()

    # Group combinations by intermediate value
    combos_by_intermediate = {}
    for X, Y, Z in all_combinations:
        intermediate = X + Y
        if intermediate not in combos_by_intermediate:
            combos_by_intermediate[intermediate] = []
        combos_by_intermediate[intermediate].append((X, Y, Z))

    mean_embeddings_by_intermediate = {}

    for intermediate_value in range(MIN_INTERMEDIATE, MAX_INTERMEDIATE + 1):
        if intermediate_value not in combos_by_intermediate:
            continue

        combos_for_intermediate = combos_by_intermediate[intermediate_value][:]
        np.random.shuffle(combos_for_intermediate)
        all_latent_embeddings = []

        # Sample from combinations for this intermediate value
        sampled_combos = combos_for_intermediate[:5]  # Limit samples per intermediate

        for template_idx in range(num_templates):
            if template_idx == current_template_idx:
                continue

            template_str = all_templates[template_idx]

            for X, Y, Z in sampled_combos:
                mapped_X, mapped_Y, mapped_Z = get_mapped_params(X, Y, Z, template_idx)
                prompt = template_str.format(X=mapped_X, Y=mapped_Y, Z=mapped_Z)
                print("Prompt", prompt)
                print("Intermediate", intermediate_value)

                try:
                    latent_embeddings_dict = capture_latent_embeddings(
                        model, tokenizer, prompt, num_latents=num_latents
                    )
                    all_latent_embeddings.append(
                        latent_embeddings_dict["latent_embeddings"]
                    )
                except Exception:
                    continue

        if len(all_latent_embeddings) == 0:
            continue

        # Average across all collected embeddings
        num_latent_steps = len(all_latent_embeddings[0])
        mean_latent_embeddings = []

        for step_idx in range(num_latent_steps):
            step_embeddings = [emb[step_idx] for emb in all_latent_embeddings]
            stacked_step = torch.stack(step_embeddings, dim=0)
            mean_step = torch.mean(stacked_step, dim=0)
            mean_latent_embeddings.append(mean_step)

        mean_embeddings_by_intermediate[intermediate_value] = mean_latent_embeddings

    return mean_embeddings_by_intermediate


# ============================================================================
# Generation Functions with Interventions
# ============================================================================


def _create_mean_ablation_hook(
    layer_idx, mean_prefill, prompt_seq_len, max_seq_len, device
):
    """Create a hook function that replaces prompt activations with mean activations."""

    def hook(module, args, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            new_hidden = hidden_states.clone()
            seq_len = min(hidden_states.shape[1], prompt_seq_len, max_seq_len)
            for token_idx in range(seq_len):
                new_hidden[:, token_idx, :] = (
                    mean_prefill[layer_idx, token_idx, :].unsqueeze(0).to(device)
                )
            return (new_hidden,) + output[1:]
        else:
            new_hidden = output.clone()
            seq_len = min(output.shape[1], prompt_seq_len, max_seq_len)
            for token_idx in range(seq_len):
                new_hidden[:, token_idx, :] = (
                    mean_prefill[layer_idx, token_idx, :].unsqueeze(0).to(device)
                )
            return new_hidden

    return hook


def generate_with_mean_ablated_prompt_patched_latent_embeddings(
    model,
    tokenizer,
    prompt,
    mean_activations_dict,
    latent_embeddings,
    max_new_tokens=128,
    num_latent_iterations=6,
    greedy=True,
):
    """
    Mean-ablated prompt + patched latent embeddings.
    Mean-ablates prompt activations, patches in provided latent embeddings.
    """
    device = model.codi.device
    layers = get_transformer_layers(model)

    mean_prefill = mean_activations_dict["mean_prefill"].to(device)
    max_seq_len = mean_prefill.shape[1]
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    prompt_seq_len = input_ids.shape[1] - 1

    patch_handles = []
    for layer_idx, layer in enumerate(layers):
        hook = _create_mean_ablation_hook(
            layer_idx, mean_prefill, prompt_seq_len, max_seq_len, device
        )
        patch_handles.append(layer.register_forward_hook(hook))

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        for h in patch_handles:
            h.remove()

        latent_embd = latent_embeddings[0].to(device)

        for i in range(num_latent_iterations):
            if i < len(latent_embeddings):
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

            next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)

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


def generate_with_patched_latent_at_position(
    model,
    tokenizer,
    prompt,
    mean_activations_dict,
    original_latent_embeddings,
    patched_latent_embedding,
    patch_position,
    max_new_tokens=128,
    num_latent_iterations=6,
):
    """
    Mean-ablated prompt + original latents with ONE position patched.

    Args:
        original_latent_embeddings: List of original latent embeddings
        patched_latent_embedding: Single latent embedding to patch in
        patch_position: Which latent position to patch (0 to num_latent_iterations-1)
    """
    # Create modified latent embeddings list
    modified_latents = []
    for i in range(len(original_latent_embeddings)):
        if i == patch_position:
            modified_latents.append(patched_latent_embedding)
        else:
            modified_latents.append(original_latent_embeddings[i])

    return generate_with_mean_ablated_prompt_patched_latent_embeddings(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        mean_activations_dict=mean_activations_dict,
        latent_embeddings=modified_latents,
        max_new_tokens=max_new_tokens,
        num_latent_iterations=num_latent_iterations,
        greedy=True,
    )


# ============================================================================
# Main Experiment
# ============================================================================


def main(
    num_test_cases: int = 20,
    num_mean_activation_samples: int = 50,
    seed: int = 42,
    max_templates: int = None,
    patch_position: int = 0,
):
    """
    Run the latent steering experiment.

    Args:
        num_test_cases: Number of test cases per template
        num_mean_activation_samples: Number of samples for computing mean activations
        seed: Random seed for reproducibility
        max_templates: Maximum number of templates to process (None = all 20)
        patch_position: Which latent position to patch (0 to num_latents-1)
    """
    load_dotenv()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Testing patch position: {patch_position}")

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

    total_templates = get_num_templates()
    all_templates = get_all_templates()
    num_templates = (
        min(max_templates, total_templates) if max_templates else total_templates
    )
    print(f"Processing {num_templates}/{total_templates} templates")

    print("\nGenerating all possible combinations...")
    all_combinations = []
    for X in range(1, 8):
        for Y in range(1, 8):
            for Z in range(1, 8):
                all_combinations.append((X, Y, Z))

    print(f"Total combinations: {len(all_combinations)}")

    np.random.seed(seed)
    shuffled_combinations = np.array(all_combinations)
    np.random.shuffle(shuffled_combinations)

    mean_activation_combinations = shuffled_combinations[
        :num_mean_activation_samples
    ].tolist()
    test_case_combinations = shuffled_combinations[
        num_mean_activation_samples : num_mean_activation_samples + num_test_cases
    ].tolist()

    print(f"Mean activation samples: {len(mean_activation_combinations)}")
    print(f"Test case samples: {len(test_case_combinations)}")

    # Results storage
    all_template_results = []

    for template_idx in range(num_templates):
        print(f"\n{'=' * 80}")
        print(f"Processing Template {template_idx + 1}/{num_templates}")
        template_type = (
            "subtraction" if is_subtraction_template(template_idx) else "addition"
        )
        print(f"Type: {template_type}")
        print(f"{'=' * 80}")

        template_str = all_templates[template_idx]
        print(f"Template: {template_str[:80]}...")

        # Compute mean prompt activations
        print("Computing mean prompt activations...")
        mean_act_mapped_combos = [
            get_mapped_params(X, Y, Z, template_idx)
            for X, Y, Z in mean_activation_combinations
        ]
        mean_activations_dict = compute_mean_prompt_activations(
            model,
            tokenizer,
            template_str,
            mean_act_mapped_combos,
        )

        # Compute mean latent embeddings by intermediate value
        print("Computing mean latent embeddings by intermediate value...")
        mean_embeddings_by_intermediate = (
            compute_mean_latent_embeddings_by_intermediate_value(
                model,
                tokenizer,
                template_idx,
                all_combinations,
                num_latents=6,
            )
        )
        print(
            f"  Computed for intermediate values: {sorted(mean_embeddings_by_intermediate.keys())}"
        )

        # Generate test cases
        template_test_cases = []
        for i, (X, Y, Z) in enumerate(test_case_combinations):
            mapped_X, mapped_Y, mapped_Z = get_mapped_params(X, Y, Z, template_idx)
            prompt = template_str.format(X=mapped_X, Y=mapped_Y, Z=mapped_Z)
            ground_truth = get_answer(X, Y, Z)
            intermediate = X + Y
            template_test_cases.append(
                {
                    "id": i,
                    "base_X": X,
                    "base_Y": Y,
                    "base_Z": Z,
                    "mapped_X": mapped_X,
                    "mapped_Y": mapped_Y,
                    "mapped_Z": mapped_Z,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "intermediate": intermediate,
                }
            )

        # Run experiments
        print(f"Running experiments for template {template_idx + 1}...")
        template_results = []

        # Track accuracies
        baseline_correct_count = 0
        steering_results = {
            iv: {"correct": 0, "total": 0}
            for iv in range(MIN_INTERMEDIATE, MAX_INTERMEDIATE + 1)
        }

        for test_case in tqdm(template_test_cases, desc=f"Template {template_idx + 1}"):
            prompt = test_case["prompt"]
            ground_truth = test_case["ground_truth"]
            original_intermediate = test_case["intermediate"]
            Z = test_case["base_Z"]
            print("Prompt", prompt)

            try:
                # Capture original latent embeddings
                original_latent_dict = capture_latent_embeddings(
                    model, tokenizer, prompt, num_latents=6
                )
                original_latents = original_latent_dict["latent_embeddings"]

                # Baseline: Mean-Abl Prompt + Patched Original Latents
                output_baseline = (
                    generate_with_mean_ablated_prompt_patched_latent_embeddings(
                        model,
                        tokenizer,
                        prompt,
                        mean_activations_dict,
                        original_latents,
                        max_new_tokens=128,
                        num_latent_iterations=6,
                        greedy=True,
                    )
                )
                generated_text_baseline = tokenizer.decode(
                    output_baseline["sequences"][0], skip_special_tokens=False
                )
                baseline_answer = extract_answer_number(generated_text_baseline)
                baseline_correct = (
                    baseline_answer is not None
                    and baseline_answer != float("inf")
                    and int(baseline_answer) == ground_truth
                )
                if baseline_correct:
                    baseline_correct_count += 1

                case_result = {
                    "id": test_case["id"],
                    "base_X": test_case["base_X"],
                    "base_Y": test_case["base_Y"],
                    "base_Z": test_case["base_Z"],
                    "ground_truth": ground_truth,
                    "original_intermediate": original_intermediate,
                    "baseline_answer": baseline_answer,
                    "baseline_correct": baseline_correct,
                    "steering_results": {},
                }

                # Test steering for each target intermediate value
                for target_intermediate in mean_embeddings_by_intermediate.keys():
                    if target_intermediate == original_intermediate:
                        # Skip if same as original
                        continue

                    target_latents = mean_embeddings_by_intermediate[
                        target_intermediate
                    ]
                    target_answer = get_answer_from_intermediate(target_intermediate, Z)

                    # Patch the latent at the specified position
                    output_steered = generate_with_patched_latent_at_position(
                        model,
                        tokenizer,
                        prompt,
                        mean_activations_dict,
                        original_latents,
                        target_latents[patch_position],
                        patch_position,
                        max_new_tokens=128,
                        num_latent_iterations=6,
                    )
                    generated_text_steered = tokenizer.decode(
                        output_steered["sequences"][0], skip_special_tokens=False
                    )

                    steered_answer = extract_answer_number(generated_text_steered)
                    steered_correct = (
                        steered_answer is not None
                        and steered_answer != float("inf")
                        and int(steered_answer) == target_answer
                    )
                    print(f"Steered answer: {steered_answer}")
                    print(f"Target answer: {target_answer}")
                    print(f"Correct: {steered_correct}")

                    case_result["steering_results"][target_intermediate] = {
                        "target_answer": target_answer,
                        "steered_answer": steered_answer,
                        "correct": steered_correct,
                    }

                    steering_results[target_intermediate]["total"] += 1
                    if steered_correct:
                        steering_results[target_intermediate]["correct"] += 1

                template_results.append(case_result)

            except Exception as e:
                print(f"\nError processing test case {test_case['id']}: {e}")
                import traceback

                traceback.print_exc()

        # Calculate accuracies
        baseline_accuracy = baseline_correct_count / len(template_test_cases)

        steering_accuracies = {}
        total_correct = 0
        total_count = 0
        for iv in range(MIN_INTERMEDIATE, MAX_INTERMEDIATE + 1):
            if steering_results[iv]["total"] > 0:
                acc = steering_results[iv]["correct"] / steering_results[iv]["total"]
                steering_accuracies[iv] = acc
                total_correct += steering_results[iv]["correct"]
                total_count += steering_results[iv]["total"]
        steering_overall = total_correct / total_count if total_count > 0 else 0

        print(f"\nTemplate {template_idx + 1} Results ({template_type}):")
        print(f"  Baseline Accuracy: {baseline_accuracy:.2%}")
        print(f"  Steering Overall: {steering_overall:.2%}")

        all_template_results.append(
            {
                "template_idx": template_idx,
                "template_str": template_str,
                "template_type": template_type,
                "baseline_accuracy": baseline_accuracy,
                "steering_overall": steering_overall,
                "steering_accuracies_by_iv": steering_accuracies,
                "detailed_results": template_results,
            }
        )

    # Calculate overall statistics
    overall_baseline = np.mean([tr["baseline_accuracy"] for tr in all_template_results])
    overall_baseline_std = np.std(
        [tr["baseline_accuracy"] for tr in all_template_results]
    )

    steering_accs = [tr["steering_overall"] for tr in all_template_results]
    overall_steering_mean = np.mean(steering_accs)
    overall_steering_std = np.std(steering_accs)

    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"Baseline Accuracy: {overall_baseline:.2%} +/- {overall_baseline_std:.2%}")
    print(
        f"Steering (Position {patch_position}): {overall_steering_mean:.2%} +/- {overall_steering_std:.2%}"
    )

    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Overall comparison
    ax1 = axes[0]
    methods = ["Baseline", f"Steering\n(Pos {patch_position})"]
    means = [overall_baseline, overall_steering_mean]
    stds = [overall_baseline_std, overall_steering_std]
    colors = ["#2ecc71", "#3498db"]

    ax1.bar(
        range(len(methods)),
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Latent Steering: Overall Accuracy", fontsize=14, fontweight="bold", pad=10
    )
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Steering accuracy by intermediate value
    ax2 = axes[1]
    iv_values = []
    iv_accs = []
    for iv in range(MIN_INTERMEDIATE, MAX_INTERMEDIATE + 1):
        acc_list = []
        for tr in all_template_results:
            if iv in tr["steering_accuracies_by_iv"]:
                acc_list.append(tr["steering_accuracies_by_iv"][iv])
        if acc_list:
            iv_accs.append(np.mean(acc_list))
            iv_values.append(iv)

    ax2.plot(iv_values, iv_accs, "o-", color="#3498db", linewidth=2, markersize=6)
    ax2.set_xlabel("Target Intermediate Value", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Steering Accuracy", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Steering Accuracy by Target Value (Pos {patch_position})",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    chart_filename = RESULTS_DIR / "latent_steering_results.png"
    fig.savefig(chart_filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Chart saved to {chart_filename}")

    # Save results
    results_file = RESULTS_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "config": {
                    "num_templates": num_templates,
                    "num_test_cases_per_template": num_test_cases,
                    "num_mean_activation_samples": num_mean_activation_samples,
                    "seed": seed,
                    "patch_position": patch_position,
                    "intermediate_range": [MIN_INTERMEDIATE, MAX_INTERMEDIATE],
                },
                "overall_results": {
                    "baseline_accuracy": float(overall_baseline),
                    "baseline_std": float(overall_baseline_std),
                    "steering_accuracy": float(overall_steering_mean),
                    "steering_std": float(overall_steering_std),
                },
                "per_template_results": [
                    {
                        "template_idx": tr["template_idx"],
                        "template_type": tr["template_type"],
                        "baseline_accuracy": float(tr["baseline_accuracy"]),
                        "steering_overall": float(tr["steering_overall"]),
                        "steering_accuracies_by_iv": {
                            str(k): float(v)
                            for k, v in tr["steering_accuracies_by_iv"].items()
                        },
                    }
                    for tr in all_template_results
                ],
                "detailed_results": all_template_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    fire.Fire(main)

# %%
