# %%
# ABOUTME: Mean ablation experiment using 20 combined addition/subtraction templates.
# ABOUTME: Tests latent reasoning transfer across semantically different but computationally equivalent prompts.
"""
Mean Ablation Experiment with Combined Addition/Subtraction Templates

This script extends the mean ablation experiment to use all 20 templates from src/templates.py,
treating addition and subtraction templates as one unified set. To ensure all prompts yield
identical intermediate numbers and final answers:
- Addition templates: use (X, Y, Z) directly
- Subtraction templates: use (X + 2Y, Y, Z)

This makes: addition: X + Y = subtraction: (X + 2Y) - Y = X + Y

Conditions tested:
1. Baseline: Normal generation
2. Mean-ablated + Regenerated Latents: Mean-ablated prompt with latents regenerated normally
3. Mean-ablated + Patched Embeddings: Mean-ablated prompt with original latent embeddings patched
4. Mean-ablated + Cross-Template Mean: Mean-ablated prompt with cross-template mean embeddings
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

RESULTS_DIR = Path(__file__).parent.parent / "results" / "mean_ablation_latent_transfer"


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

    This ensures: addition(X, Y, Z) has step_1 = X + Y
                  subtraction(X + 2Y, Y, Z) has step_1 = (X + 2Y) - Y = X + Y
    """
    if is_subtraction_template(template_idx):
        return (X + 2 * Y, Y, Z)
    return (X, Y, Z)


def get_answer(X, Y, Z):
    """
    Compute ground truth answer: (X+Y) * (1+Z)

    Both addition and subtraction templates (with correct param mapping)
    yield this same computation.
    """
    step_1 = X + Y
    step_2 = step_1 * Z
    step_3 = step_1 + step_2
    return step_3


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
    print(f"Computing mean prompt activations across {len(combinations)} samples...")

    prompts = []
    for X, Y, Z in combinations:
        prompt = template_str.format(X=X, Y=Y, Z=Z)
        prompts.append(prompt)

    all_activations = []
    all_seq_lens = []

    for prompt in tqdm(prompts, desc="Capturing activations"):
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


def compute_mean_latent_embeddings_across_other_templates(
    model, tokenizer, current_template_idx, X, Y, Z, num_latents=6
):
    """
    Compute mean latent embeddings by averaging across all other templates
    (excluding current_template_idx) using the same base X, Y, Z values.
    """
    num_templates = get_num_templates()
    all_templates = get_all_templates()
    all_latent_embeddings = []

    for template_idx in range(num_templates):
        if template_idx == current_template_idx:
            continue

        template_str = all_templates[template_idx]
        mapped_X, mapped_Y, mapped_Z = get_mapped_params(X, Y, Z, template_idx)
        prompt = template_str.format(X=mapped_X, Y=mapped_Y, Z=mapped_Z)

        latent_embeddings_dict = capture_latent_embeddings(
            model, tokenizer, prompt, num_latents=num_latents
        )
        all_latent_embeddings.append(latent_embeddings_dict["latent_embeddings"])

    num_other_templates = len(all_latent_embeddings)
    if num_other_templates == 0:
        raise ValueError("No other templates to average over")

    num_latent_steps = len(all_latent_embeddings[0])
    mean_latent_embeddings = []

    for step_idx in range(num_latent_steps):
        step_embeddings = [emb[step_idx] for emb in all_latent_embeddings]
        stacked_step = torch.stack(step_embeddings, dim=0)
        mean_step = torch.mean(stacked_step, dim=0)
        mean_latent_embeddings.append(mean_step)

    return {"latent_embeddings": mean_latent_embeddings}


def compute_mean_latent_embeddings_with_different_values(
    model, tokenizer, current_template_idx, X, Y, Z, all_combinations, num_latents=6
):
    """
    Compute mean latent embeddings by averaging across all other templates
    using DIFFERENT X, Y, Z values (control condition).

    This serves as a control to test whether cross-template transfer works because
    latents encode computation-specific information vs just having any latent vectors.
    """
    num_templates = get_num_templates()
    all_templates = get_all_templates()
    all_latent_embeddings = []

    # Find combinations with different intermediate values
    current_intermediate = X + Y
    different_combos = [
        (ox, oy, oz)
        for ox, oy, oz in all_combinations
        if ox + oy != current_intermediate
    ]

    if len(different_combos) == 0:
        raise ValueError("No combinations with different intermediate values found")

    # Sample one different combination per template (excluding current template)
    np.random.shuffle(different_combos)
    combo_idx = 0

    for template_idx in range(num_templates):
        if template_idx == current_template_idx:
            continue

        if combo_idx >= len(different_combos):
            combo_idx = 0

        diff_X, diff_Y, diff_Z = different_combos[combo_idx]
        combo_idx += 1

        template_str = all_templates[template_idx]
        mapped_X, mapped_Y, mapped_Z = get_mapped_params(
            diff_X, diff_Y, diff_Z, template_idx
        )
        prompt = template_str.format(X=mapped_X, Y=mapped_Y, Z=mapped_Z)

        latent_embeddings_dict = capture_latent_embeddings(
            model, tokenizer, prompt, num_latents=num_latents
        )
        all_latent_embeddings.append(latent_embeddings_dict["latent_embeddings"])

    num_other_templates = len(all_latent_embeddings)
    if num_other_templates == 0:
        raise ValueError("No other templates to average over")

    num_latent_steps = len(all_latent_embeddings[0])
    mean_latent_embeddings = []

    for step_idx in range(num_latent_steps):
        step_embeddings = [emb[step_idx] for emb in all_latent_embeddings]
        stacked_step = torch.stack(step_embeddings, dim=0)
        mean_step = torch.mean(stacked_step, dim=0)
        mean_latent_embeddings.append(mean_step)

    return {"latent_embeddings": mean_latent_embeddings}


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


def generate_with_mean_ablated_prompt_regenerated_latents(
    model,
    tokenizer,
    prompt,
    mean_activations_dict,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Mean-ablated prompt + regenerated latents.
    Mean-ablates prompt activations, then lets latents regenerate normally.
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

        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if model.use_prj:
            latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        for i in range(num_latent_iterations):
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


def generate_with_mean_ablated_prompt_patched_latent_embeddings(
    model,
    tokenizer,
    prompt,
    mean_activations_dict,
    latent_embeddings_dict,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Mean-ablated prompt + patched latent embeddings.
    Mean-ablates prompt activations, patches in original latent embeddings,
    but lets layer activations compute normally.
    """
    device = model.codi.device
    layers = get_transformer_layers(model)

    mean_prefill = mean_activations_dict["mean_prefill"].to(device)
    max_seq_len = mean_prefill.shape[1]
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    prompt_seq_len = input_ids.shape[1] - 1

    latent_embeddings = latent_embeddings_dict["latent_embeddings"]

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


def generate_with_mean_ablated_prompt_cross_template_mean_embeddings(
    model,
    tokenizer,
    prompt,
    mean_activations_dict,
    cross_template_mean_embeddings_dict,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Mean-ablated prompt + cross-template mean latent embeddings.
    Mean-ablates prompt activations, patches in mean latent embeddings computed
    across other templates (with same base X, Y, Z), but lets layer activations compute normally.
    """
    device = model.codi.device
    layers = get_transformer_layers(model)

    mean_prefill = mean_activations_dict["mean_prefill"].to(device)
    max_seq_len = mean_prefill.shape[1]
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    prompt_seq_len = input_ids.shape[1] - 1

    latent_embeddings = cross_template_mean_embeddings_dict["latent_embeddings"]

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
# Main Experiment
# ============================================================================


def main(
    num_samples_per_prompt: int = 5,
    temperature: float = 1.0,
    greedy: bool = False,
    num_test_cases: int = 25,
    num_mean_activation_samples: int = 50,
    seed: int = 42,
    max_templates: int = None,
):
    """
    Run the mean ablation experiment with combined addition/subtraction templates.

    Args:
        num_samples_per_prompt: Number of generation samples per test case
        temperature: Sampling temperature (used when greedy=False)
        greedy: Use greedy decoding if True
        num_test_cases: Number of test cases per template
        num_mean_activation_samples: Number of samples for computing mean activations
        seed: Random seed for reproducibility
        max_templates: Maximum number of templates to process (None = all 20)
    """
    load_dotenv()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    print(
        f"Processing {num_templates}/{total_templates} templates (10 addition + 10 subtraction available)"
    )

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

    all_template_results = []
    all_template_baseline_accuracies = []

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

        print("Generating prompts for mean activation computation...")
        mean_act_mapped_combos = [
            get_mapped_params(X, Y, Z, template_idx)
            for X, Y, Z in mean_activation_combinations
        ]

        print(f"Computing mean prompt activations for template {template_idx + 1}...")
        mean_activations_dict = compute_mean_prompt_activations(
            model,
            tokenizer,
            template_str,
            mean_act_mapped_combos,
        )

        print(f"\nGenerating test cases for template {template_idx + 1}...")
        template_test_cases = []
        for i, (X, Y, Z) in enumerate(test_case_combinations):
            mapped_X, mapped_Y, mapped_Z = get_mapped_params(X, Y, Z, template_idx)
            prompt = template_str.format(X=mapped_X, Y=mapped_Y, Z=mapped_Z)
            ground_truth = get_answer(X, Y, Z)
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
                }
            )

        print(f"\nRunning experiment for template {template_idx + 1}...")
        template_results = []
        baseline_position_correct = [[] for _ in range(num_samples_per_prompt)]
        regenerated_position_correct = [[] for _ in range(num_samples_per_prompt)]
        patched_position_correct = [[] for _ in range(num_samples_per_prompt)]
        cross_template_mean_position_correct = [
            [] for _ in range(num_samples_per_prompt)
        ]
        different_values_position_correct = [[] for _ in range(num_samples_per_prompt)]

        for test_case in tqdm(template_test_cases, desc=f"Template {template_idx + 1}"):
            prompt = test_case["prompt"]
            ground_truth = test_case["ground_truth"]

            try:
                latent_embeddings_dict = capture_latent_embeddings(
                    model, tokenizer, prompt, num_latents=6
                )

                cross_template_mean_embeddings_dict = (
                    compute_mean_latent_embeddings_across_other_templates(
                        model,
                        tokenizer,
                        template_idx,
                        test_case["base_X"],
                        test_case["base_Y"],
                        test_case["base_Z"],
                        num_latents=6,
                    )
                )

                different_values_embeddings_dict = (
                    compute_mean_latent_embeddings_with_different_values(
                        model,
                        tokenizer,
                        template_idx,
                        test_case["base_X"],
                        test_case["base_Y"],
                        test_case["base_Z"],
                        all_combinations,
                        num_latents=6,
                    )
                )

                prompt_results = {
                    "id": test_case["id"],
                    "base_X": test_case["base_X"],
                    "base_Y": test_case["base_Y"],
                    "base_Z": test_case["base_Z"],
                    "mapped_X": test_case["mapped_X"],
                    "mapped_Y": test_case["mapped_Y"],
                    "mapped_Z": test_case["mapped_Z"],
                    "ground_truth": ground_truth,
                    "baseline_samples": [],
                    "regenerated_samples": [],
                    "patched_samples": [],
                    "cross_template_mean_samples": [],
                    "different_values_samples": [],
                }

                for sample_idx in range(num_samples_per_prompt):
                    # Baseline
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                        model.codi.device
                    )
                    attention_mask = tokenizer(
                        prompt, return_tensors="pt"
                    ).attention_mask.to(model.codi.device)
                    output_baseline = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=128,
                        num_latent_iterations=6,
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
                    generated_text_baseline = tokenizer.decode(
                        output_baseline["sequences"][0], skip_special_tokens=False
                    )
                    baseline_answer = extract_answer_number(generated_text_baseline)
                    baseline_correct = (
                        baseline_answer is not None
                        and baseline_answer != float("inf")
                        and int(baseline_answer) == ground_truth
                    )
                    prompt_results["baseline_samples"].append(
                        {
                            "answer": baseline_answer,
                            "correct": baseline_correct,
                            "text": generated_text_baseline,
                        }
                    )
                    baseline_position_correct[sample_idx].append(baseline_correct)

                    # Mean-ablated + regenerated latents
                    output_regen = (
                        generate_with_mean_ablated_prompt_regenerated_latents(
                            model,
                            tokenizer,
                            prompt,
                            mean_activations_dict=mean_activations_dict,
                            max_new_tokens=128,
                            num_latent_iterations=6,
                            temperature=temperature,
                            greedy=greedy,
                        )
                    )
                    generated_text_regen = tokenizer.decode(
                        output_regen["sequences"][0],
                        skip_special_tokens=False,
                    )
                    regen_answer = extract_answer_number(generated_text_regen)
                    regen_correct = (
                        regen_answer is not None
                        and regen_answer != float("inf")
                        and int(regen_answer) == ground_truth
                    )
                    prompt_results["regenerated_samples"].append(
                        {
                            "answer": regen_answer,
                            "correct": regen_correct,
                            "text": generated_text_regen,
                        }
                    )
                    regenerated_position_correct[sample_idx].append(regen_correct)

                    # Mean-ablated + patched embeddings
                    output_patched = (
                        generate_with_mean_ablated_prompt_patched_latent_embeddings(
                            model,
                            tokenizer,
                            prompt,
                            mean_activations_dict=mean_activations_dict,
                            latent_embeddings_dict=latent_embeddings_dict,
                            max_new_tokens=128,
                            num_latent_iterations=6,
                            temperature=temperature,
                            greedy=greedy,
                        )
                    )
                    generated_text_patched = tokenizer.decode(
                        output_patched["sequences"][0],
                        skip_special_tokens=False,
                    )
                    patched_answer = extract_answer_number(generated_text_patched)
                    patched_correct = (
                        patched_answer is not None
                        and patched_answer != float("inf")
                        and int(patched_answer) == ground_truth
                    )
                    prompt_results["patched_samples"].append(
                        {
                            "answer": patched_answer,
                            "correct": patched_correct,
                            "text": generated_text_patched,
                        }
                    )
                    patched_position_correct[sample_idx].append(patched_correct)

                    # Mean-ablated + cross-template mean embeddings
                    output_cross = generate_with_mean_ablated_prompt_cross_template_mean_embeddings(
                        model,
                        tokenizer,
                        prompt,
                        mean_activations_dict=mean_activations_dict,
                        cross_template_mean_embeddings_dict=cross_template_mean_embeddings_dict,
                        max_new_tokens=128,
                        num_latent_iterations=6,
                        temperature=temperature,
                        greedy=greedy,
                    )
                    generated_text_cross = tokenizer.decode(
                        output_cross["sequences"][0],
                        skip_special_tokens=False,
                    )
                    cross_answer = extract_answer_number(generated_text_cross)
                    cross_correct = (
                        cross_answer is not None
                        and cross_answer != float("inf")
                        and int(cross_answer) == ground_truth
                    )
                    prompt_results["cross_template_mean_samples"].append(
                        {
                            "answer": cross_answer,
                            "correct": cross_correct,
                            "text": generated_text_cross,
                        }
                    )
                    cross_template_mean_position_correct[sample_idx].append(
                        cross_correct
                    )

                    # Mean-ablated + different values embeddings (control)
                    output_diff = generate_with_mean_ablated_prompt_cross_template_mean_embeddings(
                        model,
                        tokenizer,
                        prompt,
                        mean_activations_dict=mean_activations_dict,
                        cross_template_mean_embeddings_dict=different_values_embeddings_dict,
                        max_new_tokens=128,
                        num_latent_iterations=6,
                        temperature=temperature,
                        greedy=greedy,
                    )
                    generated_text_diff = tokenizer.decode(
                        output_diff["sequences"][0],
                        skip_special_tokens=False,
                    )
                    diff_answer = extract_answer_number(generated_text_diff)
                    diff_correct = (
                        diff_answer is not None
                        and diff_answer != float("inf")
                        and int(diff_answer) == ground_truth
                    )
                    prompt_results["different_values_samples"].append(
                        {
                            "answer": diff_answer,
                            "correct": diff_correct,
                            "text": generated_text_diff,
                        }
                    )
                    different_values_position_correct[sample_idx].append(diff_correct)

                template_results.append(prompt_results)

            except Exception as e:
                print(f"\nError processing test case {test_case['id']}: {e}")
                import traceback

                traceback.print_exc()
                template_results.append(
                    {
                        "id": test_case["id"],
                        "base_X": test_case["base_X"],
                        "base_Y": test_case["base_Y"],
                        "base_Z": test_case["base_Z"],
                        "ground_truth": ground_truth,
                        "error": str(e),
                    }
                )

        def calc_accuracies(position_correct):
            """Calculate position-wise accuracies from position_correct list."""
            position_accuracies = []
            for position in range(num_samples_per_prompt):
                if len(position_correct[position]) > 0:
                    position_accuracies.append(np.mean(position_correct[position]))
            mean_acc = np.mean(position_accuracies) if position_accuracies else 0.0
            std_acc = np.std(position_accuracies) if position_accuracies else 0.0
            return position_accuracies, mean_acc, std_acc

        (
            baseline_position_accuracies,
            baseline_mean_acc,
            baseline_std_acc,
        ) = calc_accuracies(baseline_position_correct)
        (
            regenerated_position_accuracies,
            regenerated_mean_acc,
            regenerated_std_acc,
        ) = calc_accuracies(regenerated_position_correct)
        (
            patched_position_accuracies,
            patched_mean_acc,
            patched_std_acc,
        ) = calc_accuracies(patched_position_correct)
        (
            cross_template_mean_position_accuracies,
            cross_template_mean_mean_acc,
            cross_template_mean_std_acc,
        ) = calc_accuracies(cross_template_mean_position_correct)
        (
            different_values_position_accuracies,
            different_values_mean_acc,
            different_values_std_acc,
        ) = calc_accuracies(different_values_position_correct)

        print(f"\nTemplate {template_idx + 1} Results ({template_type}):")
        print("  Mean Accuracies:")
        print(
            f"    Baseline:            {baseline_mean_acc:.2%} +/- {baseline_std_acc:.2%}"
        )
        print(
            f"    Regenerated:         {regenerated_mean_acc:.2%} +/- {regenerated_std_acc:.2%}"
        )
        print(
            f"    Patched:             {patched_mean_acc:.2%} +/- {patched_std_acc:.2%}"
        )
        print(
            f"    Cross-Template Mean: {cross_template_mean_mean_acc:.2%} +/- {cross_template_mean_std_acc:.2%}"
        )
        print(
            f"    Different Values:    {different_values_mean_acc:.2%} +/- {different_values_std_acc:.2%}"
        )

        all_template_results.append(
            {
                "template_idx": template_idx,
                "template_str": template_str,
                "template_type": template_type,
                "baseline_mean_acc": baseline_mean_acc,
                "baseline_std_acc": baseline_std_acc,
                "baseline_position_accuracies": baseline_position_accuracies,
                "regenerated_mean_acc": regenerated_mean_acc,
                "regenerated_std_acc": regenerated_std_acc,
                "regenerated_position_accuracies": regenerated_position_accuracies,
                "patched_mean_acc": patched_mean_acc,
                "patched_std_acc": patched_std_acc,
                "patched_position_accuracies": patched_position_accuracies,
                "cross_template_mean_mean_acc": cross_template_mean_mean_acc,
                "cross_template_mean_std_acc": cross_template_mean_std_acc,
                "cross_template_mean_position_accuracies": cross_template_mean_position_accuracies,
                "different_values_mean_acc": different_values_mean_acc,
                "different_values_std_acc": different_values_std_acc,
                "different_values_position_accuracies": different_values_position_accuracies,
                "results": template_results,
            }
        )
        all_template_baseline_accuracies.append(baseline_mean_acc)

    # Calculate overall averages
    overall_baseline_mean = np.mean(all_template_baseline_accuracies)
    overall_baseline_std = np.std(all_template_baseline_accuracies)

    all_regenerated_accuracies = [
        tr["regenerated_mean_acc"] for tr in all_template_results
    ]
    all_patched_accuracies = [tr["patched_mean_acc"] for tr in all_template_results]
    all_cross_template_mean_accuracies = [
        tr["cross_template_mean_mean_acc"] for tr in all_template_results
    ]
    all_different_values_accuracies = [
        tr["different_values_mean_acc"] for tr in all_template_results
    ]

    overall_regenerated_mean = np.mean(all_regenerated_accuracies)
    overall_regenerated_std = np.std(all_regenerated_accuracies)
    overall_patched_mean = np.mean(all_patched_accuracies)
    overall_patched_std = np.std(all_patched_accuracies)
    overall_cross_template_mean_mean = np.mean(all_cross_template_mean_accuracies)
    overall_cross_template_mean_std = np.std(all_cross_template_mean_accuracies)
    overall_different_values_mean = np.mean(all_different_values_accuracies)
    overall_different_values_std = np.std(all_different_values_accuracies)

    # Calculate by template type
    addition_results = [
        tr for tr in all_template_results if tr["template_type"] == "addition"
    ]
    subtraction_results = [
        tr for tr in all_template_results if tr["template_type"] == "subtraction"
    ]

    addition_baseline_mean = np.mean(
        [tr["baseline_mean_acc"] for tr in addition_results]
    )
    addition_baseline_std = np.std([tr["baseline_mean_acc"] for tr in addition_results])
    subtraction_baseline_mean = np.mean(
        [tr["baseline_mean_acc"] for tr in subtraction_results]
    )
    subtraction_baseline_std = np.std(
        [tr["baseline_mean_acc"] for tr in subtraction_results]
    )

    print("\nCreating visualization...")

    fontsize_title = 20
    fontsize_label = 18
    fontsize_tick = 14

    fig, ax = plt.subplots(figsize=(14, 6))
    methods = [
        "Standard",
        "Mean-Abl Prompt",
        "Mean-Abl Prompt\n+ Patched Latents",
        "Mean-Abl Prompt\n+ Cross Latents\n(same values)",
        "Mean-Abl Prompt\n+ Cross Latents\n(diff values)",
    ]
    mean_accs = [
        overall_baseline_mean,
        overall_regenerated_mean,
        overall_patched_mean,
        overall_cross_template_mean_mean,
        overall_different_values_mean,
    ]
    std_accs = [
        overall_baseline_std,
        overall_regenerated_std,
        overall_patched_std,
        overall_cross_template_mean_std,
        overall_different_values_std,
    ]
    colors = ["#2ecc71", "#e74c3c", "#9b59b6", "#3498db", "#95a5a6"]

    for i, (method, acc, std, color) in enumerate(
        zip(methods, mean_accs, std_accs, colors)
    ):
        ax.bar(
            i,
            acc,
            yerr=std,
            capsize=8,
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
            width=0.6,
        )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=fontsize_tick)
    ax.set_ylabel("Accuracy", fontsize=fontsize_label, fontweight="bold")
    ax.set_title(
        "How much performance can the latent vectors recover?",
        fontsize=fontsize_title,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
    ax.tick_params(axis="y", labelsize=fontsize_tick)

    plt.tight_layout()
    chart_filename = RESULTS_DIR / "mean_ablation_results.png"
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
                    "total_templates_available": total_templates,
                    "max_templates": max_templates,
                    "num_test_cases_per_template": num_test_cases,
                    "num_samples_per_prompt": num_samples_per_prompt,
                    "temperature": temperature,
                    "greedy": greedy,
                    "seed": seed,
                    "num_mean_activation_samples": num_mean_activation_samples,
                },
                "overall_results": {
                    "baseline_mean": float(overall_baseline_mean),
                    "baseline_std": float(overall_baseline_std),
                    "regenerated_mean": float(overall_regenerated_mean),
                    "regenerated_std": float(overall_regenerated_std),
                    "patched_mean": float(overall_patched_mean),
                    "patched_std": float(overall_patched_std),
                    "cross_template_mean_mean": float(overall_cross_template_mean_mean),
                    "cross_template_mean_std": float(overall_cross_template_mean_std),
                    "different_values_mean": float(overall_different_values_mean),
                    "different_values_std": float(overall_different_values_std),
                },
                "by_template_type": {
                    "addition": {
                        "baseline_mean": float(addition_baseline_mean),
                        "baseline_std": float(addition_baseline_std),
                    },
                    "subtraction": {
                        "baseline_mean": float(subtraction_baseline_mean),
                        "baseline_std": float(subtraction_baseline_std),
                    },
                },
                "per_template_results": [
                    {
                        "template_idx": tr["template_idx"],
                        "template_type": tr["template_type"],
                        "baseline_mean": float(tr["baseline_mean_acc"]),
                        "baseline_std": float(tr["baseline_std_acc"]),
                        "regenerated_mean": float(tr["regenerated_mean_acc"]),
                        "regenerated_std": float(tr["regenerated_std_acc"]),
                        "patched_mean": float(tr["patched_mean_acc"]),
                        "patched_std": float(tr["patched_std_acc"]),
                        "cross_template_mean_mean": float(
                            tr["cross_template_mean_mean_acc"]
                        ),
                        "cross_template_mean_std": float(
                            tr["cross_template_mean_std_acc"]
                        ),
                        "different_values_mean": float(tr["different_values_mean_acc"]),
                        "different_values_std": float(tr["different_values_std_acc"]),
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
