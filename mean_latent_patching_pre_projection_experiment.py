# %%
"""
Mean Latent Patching (Pre-Projection) Experiment: Testing the effect of patching
output latent embeddings BEFORE the projection layer with the mean latent vector
from OTHER prompts at the same position.

This script collects latent embeddings before the projection layer is applied,
then patches specific positions with the mean embedding and applies projection after.
"""

import json

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
# Latent Embedding Collection (Pre-Projection)
# ============================================================================


def collect_latent_embeddings_pre_projection(model, tokenizer, prompt, num_latents=6):
    """
    Collect the output latent embeddings BEFORE projection for each latent iteration.

    Returns a list of tensors, one for each latent position (0 to num_latents-1).
    Each tensor has shape [1, 1, hidden_size] and is the raw hidden state before projection.
    """
    device = model.codi.device
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)

    latent_outputs_pre_proj = []

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
        latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        # Apply projection for input to next iteration
        if model.use_prj:
            latent_embd = model.prj(latent_embd_pre_proj).to(dtype=model.codi.dtype)
        else:
            latent_embd = latent_embd_pre_proj

        # Latent loop - collect pre-projection outputs at each position
        for i in range(num_latents):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values

            # Get output latent embedding BEFORE projection
            latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Store the pre-projection embedding for this position
            latent_outputs_pre_proj.append(latent_embd_pre_proj.cpu().clone())

            # Apply projection for input to next iteration
            if model.use_prj:
                latent_embd = model.prj(latent_embd_pre_proj).to(dtype=model.codi.dtype)
            else:
                latent_embd = latent_embd_pre_proj

    return latent_outputs_pre_proj


def compute_mean_latents_excluding(all_latents, exclude_idx, num_latents=6):
    """
    Compute mean latent embeddings at each position, excluding a specific prompt.

    Args:
        all_latents: List of lists, where all_latents[prompt_idx][position] is the
                     latent embedding for that prompt at that position.
        exclude_idx: Index of the prompt to exclude from the mean calculation.
        num_latents: Number of latent positions.

    Returns:
        List of mean embeddings, one for each position.
    """
    mean_latents = []

    for pos in range(num_latents):
        # Collect embeddings from all prompts except the excluded one
        embeddings = []
        for prompt_idx, prompt_latents in enumerate(all_latents):
            if prompt_idx != exclude_idx:
                embeddings.append(prompt_latents[pos])

        # Stack and compute mean
        stacked = torch.cat(embeddings, dim=0)  # [num_prompts-1, 1, hidden_size]
        mean_emb = stacked.mean(dim=0, keepdim=True)  # [1, 1, hidden_size]
        mean_latents.append(mean_emb)

    return mean_latents


# ============================================================================
# Generation with Mean Latent Patching (Pre-Projection)
# ============================================================================


def generate_with_mean_latent_patching_pre_projection(
    model,
    tokenizer,
    prompt,
    patch_position,
    mean_latent_pre_proj,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Generate with the output latent embedding patched BEFORE projection at a specific
    position using the mean latent from other prompts.

    Args:
        patch_position: Which latent iteration's output to patch (before projection).
        mean_latent_pre_proj: The mean pre-projection latent embedding to use for patching.
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

        # Get initial latent embedding from BOCOT position (pre-projection)
        latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        # Apply projection for input
        if model.use_prj:
            latent_embd = model.prj(latent_embd_pre_proj).to(dtype=model.codi.dtype)
        else:
            latent_embd = latent_embd_pre_proj

        # Latent loop with pre-projection mean patching
        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values

            # Get output latent embedding BEFORE projection
            latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Patch pre-projection output if this is the target position
            if i == patch_position:
                latent_embd_pre_proj = mean_latent_pre_proj.to(device).to(
                    dtype=model.codi.dtype
                )

            # Apply projection after (possibly patched) pre-projection embedding
            if model.use_prj:
                latent_embd = model.prj(latent_embd_pre_proj).to(dtype=model.codi.dtype)
            else:
                latent_embd = latent_embd_pre_proj

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


def get_template_and_param_mapping(template_idx):
    """Get template string and parameter mapping function for a given template index."""
    templates = _get_templates()
    return templates[template_idx], lambda x, y, z: (x, y, z)


# ============================================================================
# Main Experiment
# ============================================================================


def main():
    """Run the mean latent patching (pre-projection) experiment."""
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

    # Phase 1: Collect pre-projection latent embeddings from all prompts
    print("\nPhase 1: Collecting pre-projection latent embeddings from all prompts...")
    all_latents = []
    for test_case in tqdm(test_cases, desc="Collecting latents"):
        prompt = test_case["prompt"]
        latent_outputs = collect_latent_embeddings_pre_projection(
            model, tokenizer, prompt, num_latents=num_latents
        )
        all_latents.append(latent_outputs)

    print(f"Collected pre-projection latents from {len(all_latents)} prompts")
    print(f"Each prompt has {len(all_latents[0])} latent positions")
    print(f"Latent shape (pre-projection): {all_latents[0][0].shape}")

    # Positions to test: baseline (no patching) + each individual position (0-5)
    positions_to_test = [None] + list(range(num_latents))  # None = baseline

    # Storage for results
    results = []
    position_accuracies_per_prompt = {pos: [] for pos in positions_to_test}
    position_accuracies_all = {pos: [] for pos in positions_to_test}

    # Phase 2: Run patching experiment
    print("\nPhase 2: Running mean latent patching (pre-projection) experiment...")
    for prompt_idx, test_case in enumerate(tqdm(test_cases, desc="Test cases")):
        prompt = test_case["prompt"]
        ground_truth = test_case["ground_truth"]

        try:
            # Compute mean latents excluding current prompt
            mean_latents = compute_mean_latents_excluding(
                all_latents, prompt_idx, num_latents=num_latents
            )

            test_case_results = {
                "id": test_case["id"],
                "X": test_case["X"],
                "Y": test_case["Y"],
                "Z": test_case["Z"],
                "ground_truth": ground_truth,
                "positions": {},
            }

            for position in positions_to_test:
                position_samples = []
                prompt_correct_list = []

                for sample_idx in range(num_samples_per_prompt):
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
                        # Patched generation using mean latent at specific position
                        output = generate_with_mean_latent_patching_pre_projection(
                            model,
                            tokenizer,
                            prompt,
                            patch_position=position,
                            mean_latent_pre_proj=mean_latents[position],
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

                    position_samples.append(
                        {
                            "answer": answer,
                            "correct": correct,
                            "text": generated_text,
                        }
                    )
                    prompt_correct_list.append(correct)
                    position_accuracies_all[position].append(correct)

                # Store per-prompt accuracy
                prompt_accuracy = np.mean(prompt_correct_list)
                position_accuracies_per_prompt[position].append(prompt_accuracy)

                pos_key = "baseline" if position is None else str(position)
                test_case_results["positions"][pos_key] = position_samples

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

    # Calculate accuracies and standard errors
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)

    position_mean_accuracies = {}
    position_std_errors = {}

    for position in positions_to_test:
        pos_key = "baseline" if position is None else str(position)
        if len(position_accuracies_per_prompt[position]) > 0:
            mean_acc = np.mean(position_accuracies_per_prompt[position])
            std_err = np.std(position_accuracies_per_prompt[position]) / np.sqrt(
                len(position_accuracies_per_prompt[position])
            )
            position_mean_accuracies[pos_key] = mean_acc
            position_std_errors[pos_key] = std_err
            print(f"Position {pos_key:10s}: {mean_acc:.4f} ± {std_err:.4f}")
        else:
            position_mean_accuracies[pos_key] = 0.0
            position_std_errors[pos_key] = 0.0

    # Get baseline for delta calculation
    baseline_acc = position_mean_accuracies.get("baseline", 0.0)
    baseline_std_err = position_std_errors.get("baseline", 0.0)
    print(f"\nBaseline accuracy: {baseline_acc:.4f} ± {baseline_std_err:.4f}")

    # Calculate deltas
    position_deltas = {}
    position_delta_errors = {}
    for pos_key, acc in position_mean_accuracies.items():
        if pos_key != "baseline":
            position_deltas[pos_key] = acc - baseline_acc
            position_delta_errors[pos_key] = np.sqrt(
                position_std_errors[pos_key] ** 2 + baseline_std_err**2
            )

    # Save results
    output_file = "mean_latent_patching_pre_projection_results.json"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(
            {
                "results": results,
                "position_accuracies": position_mean_accuracies,
                "position_std_errors": position_std_errors,
                "position_deltas": position_deltas,
                "position_delta_errors": position_delta_errors,
                "baseline_accuracy": baseline_acc,
                "baseline_std_error": baseline_std_err,
                "num_samples_per_prompt": num_samples_per_prompt,
                "num_test_cases": num_test_cases,
            },
            f,
            indent=2,
        )

    # Create visualization
    print("\nCreating visualization...")
    create_visualization(
        position_deltas,
        position_delta_errors,
        baseline_acc,
        baseline_std_err,
        num_latents,
    )

    print("\nExperiment complete!")


def create_visualization(
    position_deltas,
    position_delta_errors,
    baseline_acc,
    baseline_std_err,
    num_latents,
):
    """Create bar plot of delta accuracy for each latent position with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    positions = list(range(num_latents))
    deltas = [position_deltas[str(i)] for i in positions]
    errors = [position_delta_errors[str(i)] for i in positions]

    # Create bar plot
    x_pos = np.arange(len(positions))
    bars = ax.bar(
        x_pos,
        deltas,
        yerr=errors,
        color=plt.cm.RdYlBu(0.2),
        alpha=0.7,
        capsize=5,
        error_kw={"linewidth": 2, "capthick": 2},
        edgecolor="black",
        linewidth=1.5,
    )

    # Add reference line at 0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Formatting
    ax.set_xlabel(
        "Latent Position (Patched Pre-Projection with Mean from Other Prompts)",
        fontsize=11,
    )
    ax.set_ylabel("Accuracy Delta from Baseline", fontsize=14)
    ax.set_title(
        "Impact of Mean Latent Patching (Pre-Projection) at Each Position\n"
        f"(Baseline accuracy: {baseline_acc:.3f} ± {baseline_std_err:.3f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"L{i}" for i in positions], fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (bar, delta, err) in enumerate(zip(bars, deltas, errors)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.01 if height >= 0 else -0.01),
            f"{delta:.3f}\n±{err:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    # Adjust y-axis to show error bars properly
    y_min = min(deltas) - max(errors) - 0.1
    y_max = max(deltas) + max(errors) + 0.15
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(
        "mean_latent_patching_pre_projection_delta_by_position.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Saved: mean_latent_patching_pre_projection_delta_by_position.png")
    plt.close()


if __name__ == "__main__":
    main()
