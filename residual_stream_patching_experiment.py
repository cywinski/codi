# %%
"""
Residual Stream Patching Experiment: Testing the effect of patching the residual
stream at every layer for specific latent positions with random vectors.

This script patches the hidden states (residual stream) at all layers for specific
latent vector positions and measures the impact on model accuracy.
"""

import json
from contextlib import contextmanager

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
# Residual Stream Patching Classes
# ============================================================================


class ResidualStreamPatcher:
    """
    Manages patching of residual streams at transformer layers.

    This class registers hooks on transformer layers to intercept and modify
    hidden states during the forward pass.
    """

    def __init__(self, model, num_latents=6):
        self.model = model
        self.num_latents = num_latents
        self.layers = get_transformer_layers(model)
        self.num_layers = len(self.layers)
        self.hooks = []

        # State tracking
        self.current_latent_idx = -1  # -1 means prefill phase
        self.patch_positions = set()  # Which latent positions to patch
        self.random_vectors = {}  # Layer -> random vector for patching
        self.is_patching_enabled = False

    def generate_random_vectors(self, hidden_size, dtype, device, seed=None):
        """Generate random vectors for each layer with similar magnitude to typical hidden states."""
        if seed is not None:
            torch.manual_seed(seed)

        self.random_vectors = {}
        for layer_idx in range(self.num_layers):
            # Generate random vector with unit norm, then scale
            random_vec = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
            # Normalize to have a reasonable magnitude (typical residual stream norm)
            random_vec = random_vec / torch.norm(random_vec, dim=-1, keepdim=True)
            # Scale to typical hidden state magnitude (empirically ~10-50 for LLaMA)
            random_vec = random_vec * 30.0
            self.random_vectors[layer_idx] = random_vec

    def set_patch_positions(self, positions):
        """Set which latent positions should be patched."""
        self.patch_positions = set(positions)

    def reset_latent_counter(self):
        """Reset the latent position counter (call before each generation)."""
        self.current_latent_idx = -1

    def increment_latent_counter(self):
        """Increment the latent position counter (call after each latent forward pass)."""
        self.current_latent_idx += 1

    def _create_hook(self, layer_idx):
        """Create a forward hook for a specific layer."""

        def hook(module, input, output):
            if not self.is_patching_enabled:
                return output

            # Check if we're in a latent position that should be patched
            if self.current_latent_idx in self.patch_positions:
                print(
                    f"Patching layer {layer_idx} at latent position {self.current_latent_idx}"
                )
                hidden_states = output[0]

                # Only patch if this is a single-token forward (latent iteration)
                if hidden_states.shape[1] == 1:
                    # Replace with random vector
                    if layer_idx in self.random_vectors:
                        patched_hidden = self.random_vectors[layer_idx].to(
                            hidden_states.device
                        )
                        # Return modified output (preserve other outputs like attention)
                        if isinstance(output, tuple):
                            return (patched_hidden,) + output[1:]
                        return patched_hidden

            return output

        return hook

    def register_hooks(self):
        """Register forward hooks on all transformer layers."""
        self.remove_hooks()  # Clean up any existing hooks

        for layer_idx, layer in enumerate(self.layers):
            hook = layer.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @contextmanager
    def patching_context(self, positions, seed=None):
        """Context manager for patching specific latent positions."""
        # Setup
        hidden_size = self.model.codi.config.hidden_size
        dtype = self.model.codi.dtype
        device = self.model.codi.device

        self.generate_random_vectors(hidden_size, dtype, device, seed=seed)
        self.set_patch_positions(positions)
        self.reset_latent_counter()
        self.register_hooks()
        self.is_patching_enabled = True

        try:
            yield self
        finally:
            # Cleanup
            self.is_patching_enabled = False
            self.remove_hooks()


# ============================================================================
# Generation with Residual Stream Patching
# ============================================================================


def generate_with_residual_patching(
    model,
    tokenizer,
    prompt,
    patcher,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Generate with residual stream patching at specified latent positions.

    The patcher should already be configured with the positions to patch.
    """
    device = model.codi.device
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    batch_size = input_ids.size(0)

    with torch.no_grad():
        # Reset latent counter for new generation
        patcher.reset_latent_counter()

        # Prefill with normal prompt (no patching during prefill)
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

        # Latent loop with residual stream patching
        for i in range(num_latent_iterations):
            # Increment counter BEFORE forward pass so hook knows current position
            patcher.increment_latent_counter()
            print(f"Latent iteration {i}")

            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Disable patching for token generation
        patcher.is_patching_enabled = False

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
    """Run the residual stream patching experiment."""
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

    # Create patcher
    patcher = ResidualStreamPatcher(model, num_latents=num_latents)

    # Positions to test: baseline (no patching) + each individual position
    positions_to_test = [None] + list(range(num_latents))  # None = baseline

    # Storage for results
    results = []
    position_accuracies_per_prompt = {pos: [] for pos in positions_to_test}
    position_accuracies_all = {pos: [] for pos in positions_to_test}

    # Run experiment
    print("\nRunning experiment...")
    for test_case in tqdm(test_cases, desc="Test cases"):
        prompt = test_case["prompt"]
        ground_truth = test_case["ground_truth"]

        try:
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
                        # Patched generation at specific position
                        with patcher.patching_context(
                            positions=[position], seed=random_seed + sample_idx
                        ):
                            output = generate_with_residual_patching(
                                model,
                                tokenizer,
                                prompt,
                                patcher,
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
    output_file = "residual_stream_patching_results.json"
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
                "num_layers": patcher.num_layers,
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
    ax.set_xlabel("Latent Position", fontsize=14)
    ax.set_ylabel("Accuracy Delta from Baseline", fontsize=14)
    ax.set_title(
        "Impact of Residual Stream Patching at Each Latent Position\n"
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
        "residual_stream_patching_delta_by_position.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Saved: residual_stream_patching_delta_by_position.png")
    plt.close()


if __name__ == "__main__":
    main()
