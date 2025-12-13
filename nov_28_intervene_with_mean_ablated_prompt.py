# %%
"""
Mean Ablation Experiment: Testing the effect of mean-ablating prompt activations
on model performance with different latent reasoning strategies.

This script compares four conditions:
1. Baseline: Normal generation
2. Mean-ablated + Frozen Latents: Mean-ablated prompt with all latent activations frozen
3. Mean-ablated + Regenerated Latents: Mean-ablated prompt with latents regenerated normally
4. Mean-ablated + Patched Embeddings: Mean-ablated prompt with only latent embeddings patched
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


def load_latent_vectors(
    metadata_file="nov_27_all_test_cases_metadata.json",
    latent_vectors_file="nov_27_all_latent_vectors.npy",
    valid_mask_file="nov_27_valid_mask.npy",
):
    """Load metadata and latent vectors from files."""
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    latent_vectors = np.load(latent_vectors_file)
    valid_mask = np.load(valid_mask_file)
    return metadata, latent_vectors, valid_mask


def get_average_latent_vector(
    metadata, latent_vectors, valid_mask, latent_idx, filter_dict=None
):
    """Get average latent vector for a specific latent index, optionally filtered."""
    indices = np.where(valid_mask)[0]

    if filter_dict:
        filtered_indices = []
        for idx in indices:
            match = True
            for key, value in filter_dict.items():
                if metadata[idx].get(key) != value:
                    match = False
                    break
            if match:
                filtered_indices.append(idx)
        indices = np.array(filtered_indices)

    if len(indices) == 0:
        print(f"No cases match the filter criteria: {filter_dict}")
        return None

    selected_latents = latent_vectors[indices, latent_idx, :]
    avg_vector = np.mean(selected_latents, axis=0)

    print(f"Computed average for latent {latent_idx} across {len(indices)} cases")
    if filter_dict:
        print(f"Filter: {filter_dict}")

    return avg_vector


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

    # Stack activations: (NumLayers, PromptSeqLen, HiddenDim)
    seq_len = input_ids.shape[1] - 1  # Exclude BOCOT token
    prefill_tensor = torch.stack([x[0, :seq_len, :] for x in captured_prefill])

    return {"prefill": prefill_tensor, "input_ids": input_ids[:, :seq_len]}


def compute_mean_prompt_activations(model, tokenizer, template, combinations):
    """Compute mean activations across multiple template prompts using specific combinations."""
    print(f"Computing mean prompt activations across {len(combinations)} samples...")

    # Generate prompts from combinations
    prompts = []
    for X, Y, Z in combinations:
        prompt = template.format(X=X, Y=Y, Z=Z)
        prompts.append(prompt)

    all_activations = []
    all_seq_lens = []

    for prompt in tqdm(prompts, desc="Capturing activations"):
        result = capture_prompt_activations(model, tokenizer, prompt)
        activations = result["prefill"]
        seq_len = activations.shape[1]
        all_activations.append(activations)
        all_seq_lens.append(seq_len)

    max_seq_len = max(all_seq_lens)
    num_layers = all_activations[0].shape[0]
    hidden_dim = all_activations[0].shape[2]

    # Compute mean activations position by position
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


def capture_target_frozen_latents(model, tokenizer, prompt, num_latents=6):
    """Capture latent activations from a prompt run for freezing."""
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    device = model.codi.device
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    captured_latents = [[] for _ in range(num_layers)]
    handles = []

    def get_latent_hook(layer_idx):
        def hook(module, args, output):
            act = output[0] if isinstance(output, tuple) else output
            captured_latents[layer_idx].append(act.detach().cpu())

        return hook

    with torch.no_grad():
        # Prefill phase
        outputs = model.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )

        past_key_values_after_prefill = outputs.past_key_values
        initial_latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if model.use_prj:
            initial_latent_embd = model.prj(initial_latent_embd).to(
                dtype=model.codi.dtype
            )

        # Latent loop - capture all activations
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(get_latent_hook(i)))

        past_key_values_list = [past_key_values_after_prefill]
        latent_embd = initial_latent_embd

        for i in range(num_latents):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                past_key_values=past_key_values_list[-1],
                output_hidden_states=True,
            )
            past_key_values_list.append(outputs.past_key_values)
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        for h in handles:
            h.remove()

    # Consolidate: (Layers, NumLatents, Dim)
    latent_stacked_layers = []
    for layer_list in captured_latents:
        if len(layer_list) > 0:
            layer_steps = torch.cat(layer_list, dim=0)
            latent_stacked_layers.append(layer_steps)
        else:
            hidden_dim = initial_latent_embd.shape[-1]
            latent_stacked_layers.append(torch.zeros(num_latents, hidden_dim))
    latent_tensor = torch.stack(latent_stacked_layers)

    return {
        "past_key_values_after_prefill": past_key_values_after_prefill,
        "initial_latent_embd": initial_latent_embd,
        "latent_activations": latent_tensor,
        "past_key_values_list": past_key_values_list,
    }


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


def generate_with_mean_ablated_prompt_frozen_latents(
    model,
    tokenizer,
    prompt,
    mean_activations_dict,
    frozen_latents_dict,
    max_new_tokens=128,
    temperature=0.1,
    greedy=True,
):
    """
    Condition 2: Mean-ablated prompt + frozen latents.
    Mean-ablates prompt activations, then freezes all latent layer activations.
    """
    device = model.codi.device
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    mean_prefill = mean_activations_dict["mean_prefill"].to(device)
    max_seq_len = mean_prefill.shape[1]
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    prompt_seq_len = input_ids.shape[1] - 1

    # Set up mean ablation hooks for prefill
    patch_handles = []
    for layer_idx, layer in enumerate(layers):
        hook = _create_mean_ablation_hook(
            layer_idx, mean_prefill, prompt_seq_len, max_seq_len, device
        )
        patch_handles.append(layer.register_forward_hook(hook))

    frozen_latents = frozen_latents_dict["latent_activations"].to(device)
    num_latents = frozen_latents.shape[1]
    frozen_latent_handles = []

    def get_frozen_latent_hook(latent_step, layer_idx):
        def hook(module, args, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                new_hidden = hidden_states.clone()
                new_hidden[:, 0, :] = (
                    frozen_latents[layer_idx, latent_step, :].unsqueeze(0).to(device)
                )
                return (new_hidden,) + output[1:]
            else:
                new_hidden = output.clone()
                new_hidden[:, 0, :] = (
                    frozen_latents[layer_idx, latent_step, :].unsqueeze(0).to(device)
                )
                return new_hidden

        return hook

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with mean ablation
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

        # Latent loop with frozen activations
        latent_embd = frozen_latents_dict["initial_latent_embd"].to(device)

        for i in range(num_latents):
            for layer_idx, layer in enumerate(layers):
                hook = get_frozen_latent_hook(i, layer_idx)
                frozen_latent_handles.append(layer.register_forward_hook(hook))

            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            for h in frozen_latent_handles:
                h.remove()
            frozen_latent_handles = []

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Generate from eocot token
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)

        outputs = model.codi(
            inputs_embeds=eot_emb,
            output_hidden_states=False,
            attention_mask=None,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        logits_at_eocot = outputs.logits[:, -1, : model.codi.config.vocab_size - 1]

        # Token generation
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()
        output = eot_emb

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

        return {"sequences": sequences, "logits_at_eocot": logits_at_eocot}


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
    Condition 3: Mean-ablated prompt + regenerated latents.
    Mean-ablates prompt activations, then lets latents regenerate normally.
    """
    device = model.codi.device
    layers = get_transformer_layers(model)

    mean_prefill = mean_activations_dict["mean_prefill"].to(device)
    max_seq_len = mean_prefill.shape[1]
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)
    prompt_seq_len = input_ids.shape[1] - 1

    # Set up mean ablation hooks
    patch_handles = []
    for layer_idx, layer in enumerate(layers):
        hook = _create_mean_ablation_hook(
            layer_idx, mean_prefill, prompt_seq_len, max_seq_len, device
        )
        patch_handles.append(layer.register_forward_hook(hook))

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with mean ablation
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

        # Latent loop - regenerate normally
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
    Condition 4: Mean-ablated prompt + patched latent embeddings.
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

    # Set up mean ablation hooks
    patch_handles = []
    for layer_idx, layer in enumerate(layers):
        hook = _create_mean_ablation_hook(
            layer_idx, mean_prefill, prompt_seq_len, max_seq_len, device
        )
        patch_handles.append(layer.register_forward_hook(hook))

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with mean ablation
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

        # Latent loop - patch embeddings but compute activations normally
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


def generate_with_template_prompt_patched_embeddings(
    model,
    tokenizer,
    template_prompt,
    latent_embeddings_dict,
    max_new_tokens=128,
    num_latent_iterations=6,
    temperature=0.1,
    greedy=True,
):
    """
    Template baseline with patched embeddings.
    Uses template prompt with X, Y, Z placeholders, patches in latent embeddings from concrete prompt.
    """
    device = model.codi.device
    input_ids, attention_mask = prepare_inputs(model, tokenizer, template_prompt)
    latent_embeddings = latent_embeddings_dict["latent_embeddings"]

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with template prompt
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        # Latent loop - patch embeddings from concrete prompt
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


def generate_with_template_prompt_frozen_latents(
    model,
    tokenizer,
    template_prompt,
    frozen_latents_dict,
    max_new_tokens=128,
    temperature=0.1,
    greedy=True,
):
    """
    Template baseline with frozen latents.
    Uses template prompt with X, Y, Z placeholders, freezes latent activations from concrete prompt.
    """
    device = model.codi.device
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    input_ids, attention_mask = prepare_inputs(model, tokenizer, template_prompt)
    frozen_latents = frozen_latents_dict["latent_activations"].to(device)
    num_latents = frozen_latents.shape[1]
    frozen_latent_handles = []

    def get_frozen_latent_hook(latent_step, layer_idx):
        def hook(module, args, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                new_hidden = hidden_states.clone()
                new_hidden[:, 0, :] = (
                    frozen_latents[layer_idx, latent_step, :].unsqueeze(0).to(device)
                )
                return (new_hidden,) + output[1:]
            else:
                new_hidden = output.clone()
                new_hidden[:, 0, :] = (
                    frozen_latents[layer_idx, latent_step, :].unsqueeze(0).to(device)
                )
                return new_hidden

        return hook

    batch_size = input_ids.size(0)
    past_key_values = None

    with torch.no_grad():
        # Prefill with template prompt
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        # Latent loop with frozen activations from concrete prompt
        latent_embd = frozen_latents_dict["initial_latent_embd"].to(device)

        for i in range(num_latents):
            for layer_idx, layer in enumerate(layers):
                hook = get_frozen_latent_hook(i, layer_idx)
                frozen_latent_handles.append(layer.register_forward_hook(hook))

            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )

            for h in frozen_latent_handles:
                h.remove()
            frozen_latent_handles = []

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd).to(dtype=model.codi.dtype)

        # Generate from eocot token
        eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
        eot_ids = torch.tensor([[eot_id]], dtype=torch.long, device=device)
        eot_emb = model.get_embd(model.codi, model.model_name)(eot_ids).to(device)

        outputs = model.codi(
            inputs_embeds=eot_emb,
            output_hidden_states=False,
            attention_mask=None,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        logits_at_eocot = outputs.logits[:, -1, : model.codi.config.vocab_size - 1]

        # Token generation
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        sequences = input_ids.clone()
        output = eot_emb

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

        return {"sequences": sequences, "logits_at_eocot": logits_at_eocot}


# ============================================================================
# Main Experiment
# ============================================================================


def generate_test_cases(template, get_answer, combinations):
    """Generate test cases from specific combinations."""
    test_cases = []
    for i, (X, Y, Z) in enumerate(combinations):
        prompt = template.format(X=X, Y=Y, Z=Z)
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
    return test_cases


def main():
    """Run the main experiment comparing all four conditions."""
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

    # Task setup
    template = """A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else."""

    # Template with placeholders (no concrete numbers)
    template_abstract = """A team starts with  X members. They recruit  Y new members. Then each current member recruits  Z additional people. How many people are there now on the team? Give the answer only and nothing else."""

    def get_answer(X, Y, Z):
        step_1 = X + Y
        step_2 = step_1 * Z
        step_3 = step_1 + step_2
        return step_3

    # Second template with different first operation but same computational structure
    template2 = """A warehouse stores {A} crates. {B} crates are shipped out. Each remaining crate contains {C} items. How many items are stored in total? Give the answer only and nothing else."""

    def map_to_template2(X, Y, Z):
        """Map X, Y, Z to A, B, C such that both templates yield same answer.
        Template 1: step1 = X + Y
        Template 2: step1 = A - B
        For same results: A - B = X + Y, C = Z
        Using: A = X + 2Y, B = Y, C = Z
        """
        A = X + 2 * Y
        B = Y
        C = Z
        return A, B, C

    # Experiment configuration - enable/disable specific conditions
    ENABLED_CONDITIONS = {
        "baseline": True,
        "mean_ablated_frozen": True,
        "mean_ablated_regenerated": True,
        "mean_ablated_patched": True,
        "template_plain": True,
        "template_patched": True,
        "template_frozen": True,
        "template2_baseline": True,
        "cross_template_frozen": True,
        "cross_template_patched": True,
    }

    # Experiment parameters
    num_samples_per_prompt = 1
    temperature = 1.0
    greedy = False
    num_test_cases = 50
    num_mean_activation_samples = 50

    # Generate all possible combinations of X, Y, Z in [1, 10]
    print("\nGenerating all possible combinations...")
    all_combinations = []
    for X in range(1, 11):
        for Y in range(1, 11):
            for Z in range(1, 11):
                all_combinations.append((X, Y, Z))

    print(f"Total combinations: {len(all_combinations)}")

    # Shuffle and split into disjoint sets
    np.random.seed(42)
    shuffled_combinations = np.array(all_combinations)
    np.random.shuffle(shuffled_combinations)

    # Split: first N for mean activations, next M for test cases
    mean_activation_combinations = shuffled_combinations[
        :num_mean_activation_samples
    ].tolist()
    test_case_combinations = shuffled_combinations[
        num_mean_activation_samples : num_mean_activation_samples + num_test_cases
    ].tolist()

    print(f"Mean activation samples: {len(mean_activation_combinations)}")
    print(f"Test case samples: {len(test_case_combinations)}")
    print(
        f"Verifying no overlap: {len(set(map(tuple, mean_activation_combinations)) & set(map(tuple, test_case_combinations)))} overlapping combinations"
    )

    # Generate test cases
    print(f"\nGenerating {len(test_case_combinations)} test cases...")
    test_cases = generate_test_cases(template, get_answer, test_case_combinations)
    print(f"Generated {len(test_cases)} test cases")

    # Compute mean activations
    print("\nComputing mean prompt activations...")
    mean_activations_dict = compute_mean_prompt_activations(
        model, tokenizer, template, mean_activation_combinations
    )

    # Run experiment
    print(
        f"\nRunning experiment on {num_test_cases} prompts with {num_samples_per_prompt} samples each..."
    )
    results = []

    # Track position-wise correctness for each condition
    baseline_position_correct = [[] for _ in range(num_samples_per_prompt)]
    frozen_position_correct = [[] for _ in range(num_samples_per_prompt)]
    regenerated_position_correct = [[] for _ in range(num_samples_per_prompt)]
    patched_position_correct = [[] for _ in range(num_samples_per_prompt)]
    template_plain_position_correct = [[] for _ in range(num_samples_per_prompt)]
    template_patched_position_correct = [[] for _ in range(num_samples_per_prompt)]
    template_frozen_position_correct = [[] for _ in range(num_samples_per_prompt)]
    # Cross-template transfer conditions
    template2_baseline_position_correct = [[] for _ in range(num_samples_per_prompt)]
    cross_template_frozen_position_correct = [[] for _ in range(num_samples_per_prompt)]
    cross_template_patched_position_correct = [
        [] for _ in range(num_samples_per_prompt)
    ]

    for test_case in tqdm(test_cases, desc="Processing prompts"):
        prompt = test_case["prompt"]
        ground_truth = test_case["ground_truth"]
        print(f"\nProcessing prompt {test_case['id']}: {prompt}")

        try:
            # Capture data for interventions (only once per prompt)
            frozen_latents_dict = capture_target_frozen_latents(
                model, tokenizer, prompt, num_latents=6
            )
            latent_embeddings_dict = capture_latent_embeddings(
                model, tokenizer, prompt, num_latents=6
            )

            # Compute template2 values and capture latents from template2
            X, Y, Z = test_case["X"], test_case["Y"], test_case["Z"]
            A, B, C = map_to_template2(X, Y, Z)
            prompt2 = template2.format(A=A, B=B, C=C)

            # Capture latents from template2 for cross-template transfer
            frozen_latents_dict_t2 = capture_target_frozen_latents(
                model, tokenizer, prompt2, num_latents=6
            )
            latent_embeddings_dict_t2 = capture_latent_embeddings(
                model, tokenizer, prompt2, num_latents=6
            )

            # Storage for this prompt's samples
            prompt_results = {
                "id": test_case["id"],
                "X": test_case["X"],
                "Y": test_case["Y"],
                "Z": test_case["Z"],
                "A": A,
                "B": B,
                "C": C,
                "ground_truth": ground_truth,
                "baseline_samples": [],
                "frozen_samples": [],
                "regenerated_samples": [],
                "patched_samples": [],
                "template_plain_samples": [],
                "template_patched_samples": [],
                "template_frozen_samples": [],
                "template2_baseline_samples": [],
                "cross_template_frozen_samples": [],
                "cross_template_patched_samples": [],
            }

            # Generate multiple samples for each condition
            for sample_idx in range(num_samples_per_prompt):
                print(f"  Sample {sample_idx + 1}/{num_samples_per_prompt}")

                # Condition 1: Baseline
                if ENABLED_CONDITIONS["baseline"]:
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

                # Condition 2: Mean-ablated + frozen latents
                if ENABLED_CONDITIONS["mean_ablated_frozen"]:
                    output_mean_ablated_frozen = (
                        generate_with_mean_ablated_prompt_frozen_latents(
                            model,
                            tokenizer,
                            prompt,
                            mean_activations_dict=mean_activations_dict,
                            frozen_latents_dict=frozen_latents_dict,
                            max_new_tokens=128,
                            temperature=temperature,
                            greedy=greedy,
                        )
                    )
                    generated_text_mean_ablated_frozen = tokenizer.decode(
                        output_mean_ablated_frozen["sequences"][0],
                        skip_special_tokens=False,
                    )
                    mean_ablated_frozen_answer = extract_answer_number(
                        generated_text_mean_ablated_frozen
                    )
                    frozen_correct = (
                        mean_ablated_frozen_answer is not None
                        and mean_ablated_frozen_answer != float("inf")
                        and int(mean_ablated_frozen_answer) == ground_truth
                    )
                    prompt_results["frozen_samples"].append(
                        {
                            "answer": mean_ablated_frozen_answer,
                            "correct": frozen_correct,
                            "text": generated_text_mean_ablated_frozen,
                        }
                    )
                    frozen_position_correct[sample_idx].append(frozen_correct)

                # Condition 3: Mean-ablated + regenerated latents
                if ENABLED_CONDITIONS["mean_ablated_regenerated"]:
                    output_mean_ablated_regenerated = (
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
                    generated_text_mean_ablated_regenerated = tokenizer.decode(
                        output_mean_ablated_regenerated["sequences"][0],
                        skip_special_tokens=False,
                    )
                    regenerated_answer = extract_answer_number(
                        generated_text_mean_ablated_regenerated
                    )
                    regenerated_correct = (
                        regenerated_answer is not None
                        and regenerated_answer != float("inf")
                        and int(regenerated_answer) == ground_truth
                    )
                    prompt_results["regenerated_samples"].append(
                        {
                            "answer": regenerated_answer,
                            "correct": regenerated_correct,
                            "text": generated_text_mean_ablated_regenerated,
                        }
                    )
                    regenerated_position_correct[sample_idx].append(regenerated_correct)

                # Condition 4: Mean-ablated + patched embeddings
                if ENABLED_CONDITIONS["mean_ablated_patched"]:
                    output_mean_ablated_patched_embeddings = (
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
                    generated_text_mean_ablated_patched_embeddings = tokenizer.decode(
                        output_mean_ablated_patched_embeddings["sequences"][0],
                        skip_special_tokens=False,
                    )
                    patched_answer = extract_answer_number(
                        generated_text_mean_ablated_patched_embeddings
                    )
                    patched_correct = (
                        patched_answer is not None
                        and patched_answer != float("inf")
                        and int(patched_answer) == ground_truth
                    )
                    prompt_results["patched_samples"].append(
                        {
                            "answer": patched_answer,
                            "correct": patched_correct,
                            "text": generated_text_mean_ablated_patched_embeddings,
                        }
                    )
                    patched_position_correct[sample_idx].append(patched_correct)

                # Condition 5: Template prompt (plain)
                if ENABLED_CONDITIONS["template_plain"]:
                    input_ids_template = tokenizer(
                        template_abstract, return_tensors="pt"
                    ).input_ids.to(model.codi.device)
                    attention_mask_template = tokenizer(
                        template_abstract, return_tensors="pt"
                    ).attention_mask.to(model.codi.device)
                    output_template_plain = model.generate(
                        input_ids=input_ids_template,
                        attention_mask=attention_mask_template,
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
                    generated_text_template_plain = tokenizer.decode(
                        output_template_plain["sequences"][0], skip_special_tokens=False
                    )
                    template_plain_answer = extract_answer_number(
                        generated_text_template_plain
                    )
                    template_plain_correct = (
                        template_plain_answer is not None
                        and template_plain_answer != float("inf")
                        and int(template_plain_answer) == ground_truth
                    )
                    prompt_results["template_plain_samples"].append(
                        {
                            "answer": template_plain_answer,
                            "correct": template_plain_correct,
                            "text": generated_text_template_plain,
                        }
                    )
                    template_plain_position_correct[sample_idx].append(
                        template_plain_correct
                    )

                # Condition 6: Template prompt + patched embeddings
                if ENABLED_CONDITIONS["template_patched"]:
                    output_template_patched = (
                        generate_with_template_prompt_patched_embeddings(
                            model,
                            tokenizer,
                            template_abstract,
                            latent_embeddings_dict=latent_embeddings_dict,
                            max_new_tokens=128,
                            num_latent_iterations=6,
                            temperature=temperature,
                            greedy=greedy,
                        )
                    )
                    generated_text_template_patched = tokenizer.decode(
                        output_template_patched["sequences"][0],
                        skip_special_tokens=False,
                    )
                    template_patched_answer = extract_answer_number(
                        generated_text_template_patched
                    )
                    template_patched_correct = (
                        template_patched_answer is not None
                        and template_patched_answer != float("inf")
                        and int(template_patched_answer) == ground_truth
                    )
                    prompt_results["template_patched_samples"].append(
                        {
                            "answer": template_patched_answer,
                            "correct": template_patched_correct,
                            "text": generated_text_template_patched,
                        }
                    )
                    template_patched_position_correct[sample_idx].append(
                        template_patched_correct
                    )

                # Condition 7: Template prompt + frozen latents
                if ENABLED_CONDITIONS["template_frozen"]:
                    output_template_frozen = (
                        generate_with_template_prompt_frozen_latents(
                            model,
                            tokenizer,
                            template_abstract,
                            frozen_latents_dict=frozen_latents_dict,
                            max_new_tokens=128,
                            temperature=temperature,
                            greedy=greedy,
                        )
                    )
                    generated_text_template_frozen = tokenizer.decode(
                        output_template_frozen["sequences"][0],
                        skip_special_tokens=False,
                    )
                    template_frozen_answer = extract_answer_number(
                        generated_text_template_frozen
                    )
                    template_frozen_correct = (
                        template_frozen_answer is not None
                        and template_frozen_answer != float("inf")
                        and int(template_frozen_answer) == ground_truth
                    )
                    prompt_results["template_frozen_samples"].append(
                        {
                            "answer": template_frozen_answer,
                            "correct": template_frozen_correct,
                            "text": generated_text_template_frozen,
                        }
                    )
                    template_frozen_position_correct[sample_idx].append(
                        template_frozen_correct
                    )

                # Condition 8: Template2 baseline
                if ENABLED_CONDITIONS["template2_baseline"]:
                    input_ids_t2 = tokenizer(prompt2, return_tensors="pt").input_ids.to(
                        model.codi.device
                    )
                    attention_mask_t2 = tokenizer(
                        prompt2, return_tensors="pt"
                    ).attention_mask.to(model.codi.device)
                    output_template2_baseline = model.generate(
                        input_ids=input_ids_t2,
                        attention_mask=attention_mask_t2,
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
                    generated_text_t2_baseline = tokenizer.decode(
                        output_template2_baseline["sequences"][0],
                        skip_special_tokens=False,
                    )
                    t2_baseline_answer = extract_answer_number(
                        generated_text_t2_baseline
                    )
                    t2_baseline_correct = (
                        t2_baseline_answer is not None
                        and t2_baseline_answer != float("inf")
                        and int(t2_baseline_answer) == ground_truth
                    )
                    prompt_results["template2_baseline_samples"].append(
                        {
                            "answer": t2_baseline_answer,
                            "correct": t2_baseline_correct,
                            "text": generated_text_t2_baseline,
                        }
                    )
                    template2_baseline_position_correct[sample_idx].append(
                        t2_baseline_correct
                    )

                # Condition 9: Cross-template frozen (mean-ablate template1, freeze latents from template2)
                if ENABLED_CONDITIONS["cross_template_frozen"]:
                    output_cross_frozen = generate_with_mean_ablated_prompt_frozen_latents(
                        model,
                        tokenizer,
                        prompt,  # Template 1 prompt
                        mean_activations_dict=mean_activations_dict,
                        frozen_latents_dict=frozen_latents_dict_t2,  # Latents from template 2
                        max_new_tokens=128,
                        temperature=temperature,
                        greedy=greedy,
                    )
                    generated_text_cross_frozen = tokenizer.decode(
                        output_cross_frozen["sequences"][0],
                        skip_special_tokens=False,
                    )
                    cross_frozen_answer = extract_answer_number(
                        generated_text_cross_frozen
                    )
                    cross_frozen_correct = (
                        cross_frozen_answer is not None
                        and cross_frozen_answer != float("inf")
                        and int(cross_frozen_answer) == ground_truth
                    )
                    prompt_results["cross_template_frozen_samples"].append(
                        {
                            "answer": cross_frozen_answer,
                            "correct": cross_frozen_correct,
                            "text": generated_text_cross_frozen,
                        }
                    )
                    cross_template_frozen_position_correct[sample_idx].append(
                        cross_frozen_correct
                    )

                # Condition 10: Cross-template patched (mean-ablate template1, patch embeddings from template2)
                if ENABLED_CONDITIONS["cross_template_patched"]:
                    output_cross_patched = generate_with_mean_ablated_prompt_patched_latent_embeddings(
                        model,
                        tokenizer,
                        prompt,  # Template 1 prompt
                        mean_activations_dict=mean_activations_dict,
                        latent_embeddings_dict=latent_embeddings_dict_t2,  # Latent embeddings from template 2
                        max_new_tokens=128,
                        num_latent_iterations=6,
                        temperature=temperature,
                        greedy=greedy,
                    )
                    generated_text_cross_patched = tokenizer.decode(
                        output_cross_patched["sequences"][0],
                        skip_special_tokens=False,
                    )
                    cross_patched_answer = extract_answer_number(
                        generated_text_cross_patched
                    )
                    cross_patched_correct = (
                        cross_patched_answer is not None
                        and cross_patched_answer != float("inf")
                        and int(cross_patched_answer) == ground_truth
                    )
                    prompt_results["cross_template_patched_samples"].append(
                        {
                            "answer": cross_patched_answer,
                            "correct": cross_patched_correct,
                            "text": generated_text_cross_patched,
                        }
                    )
                    cross_template_patched_position_correct[sample_idx].append(
                        cross_patched_correct
                    )

            results.append(prompt_results)

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

    # Calculate position-wise accuracies
    total_valid = len([r for r in results if "error" not in r])

    baseline_position_accuracies = []
    frozen_position_accuracies = []
    regenerated_position_accuracies = []
    patched_position_accuracies = []
    template_plain_position_accuracies = []
    template_patched_position_accuracies = []
    template_frozen_position_accuracies = []
    template2_baseline_position_accuracies = []
    cross_template_frozen_position_accuracies = []
    cross_template_patched_position_accuracies = []

    for position in range(num_samples_per_prompt):
        if len(baseline_position_correct[position]) > 0:
            baseline_position_accuracies.append(
                np.mean(baseline_position_correct[position])
            )
            frozen_position_accuracies.append(
                np.mean(frozen_position_correct[position])
            )
            regenerated_position_accuracies.append(
                np.mean(regenerated_position_correct[position])
            )
            patched_position_accuracies.append(
                np.mean(patched_position_correct[position])
            )
            template_plain_position_accuracies.append(
                np.mean(template_plain_position_correct[position])
            )
            template_patched_position_accuracies.append(
                np.mean(template_patched_position_correct[position])
            )
            template_frozen_position_accuracies.append(
                np.mean(template_frozen_position_correct[position])
            )
            template2_baseline_position_accuracies.append(
                np.mean(template2_baseline_position_correct[position])
            )
            cross_template_frozen_position_accuracies.append(
                np.mean(cross_template_frozen_position_correct[position])
            )
            cross_template_patched_position_accuracies.append(
                np.mean(cross_template_patched_position_correct[position])
            )

    # Calculate mean and std dev across positions
    baseline_mean_acc = np.mean(baseline_position_accuracies)
    baseline_std_acc = np.std(baseline_position_accuracies)
    frozen_mean_acc = np.mean(frozen_position_accuracies)
    frozen_std_acc = np.std(frozen_position_accuracies)
    regenerated_mean_acc = np.mean(regenerated_position_accuracies)
    regenerated_std_acc = np.std(regenerated_position_accuracies)
    patched_mean_acc = np.mean(patched_position_accuracies)
    patched_std_acc = np.std(patched_position_accuracies)
    template_plain_mean_acc = np.mean(template_plain_position_accuracies)
    template_plain_std_acc = np.std(template_plain_position_accuracies)
    template_patched_mean_acc = np.mean(template_patched_position_accuracies)
    template_patched_std_acc = np.std(template_patched_position_accuracies)
    template_frozen_mean_acc = np.mean(template_frozen_position_accuracies)
    template_frozen_std_acc = np.std(template_frozen_position_accuracies)
    template2_baseline_mean_acc = np.mean(template2_baseline_position_accuracies)
    template2_baseline_std_acc = np.std(template2_baseline_position_accuracies)
    cross_template_frozen_mean_acc = np.mean(cross_template_frozen_position_accuracies)
    cross_template_frozen_std_acc = np.std(cross_template_frozen_position_accuracies)
    cross_template_patched_mean_acc = np.mean(
        cross_template_patched_position_accuracies
    )
    cross_template_patched_std_acc = np.std(cross_template_patched_position_accuracies)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Total test cases: {len(results)}")
    print(f"Valid test cases: {total_valid}")
    print(f"Samples per prompt: {num_samples_per_prompt}")
    print(f"Temperature: {temperature}")

    print("\n" + "-" * 80)
    print("POSITION-WISE ACCURACIES:")
    print("-" * 80)
    for pos in range(num_samples_per_prompt):
        print(f"Position {pos + 1}:")
        print(f"  Baseline:          {baseline_position_accuracies[pos]:.2%}")
        print(f"  Frozen:            {frozen_position_accuracies[pos]:.2%}")
        print(f"  Regenerated:       {regenerated_position_accuracies[pos]:.2%}")
        print(f"  Patched:           {patched_position_accuracies[pos]:.2%}")
        print(f"  Template Plain:    {template_plain_position_accuracies[pos]:.2%}")
        print(f"  Template Patched:  {template_patched_position_accuracies[pos]:.2%}")
        print(f"  Template Frozen:   {template_frozen_position_accuracies[pos]:.2%}")
        print(f"  Template2 Base:    {template2_baseline_position_accuracies[pos]:.2%}")
        print(
            f"  Cross-T Frozen:    {cross_template_frozen_position_accuracies[pos]:.2%}"
        )
        print(
            f"  Cross-T Patched:   {cross_template_patched_position_accuracies[pos]:.2%}"
        )

    print("\n" + "-" * 80)
    print("MEAN AND STD DEV ACROSS POSITIONS:")
    print("-" * 80)
    print(f"Baseline:          {baseline_mean_acc:.2%} ± {baseline_std_acc:.2%}")
    print(f"Frozen:            {frozen_mean_acc:.2%} ± {frozen_std_acc:.2%}")
    print(f"Regenerated:       {regenerated_mean_acc:.2%} ± {regenerated_std_acc:.2%}")
    print(f"Patched:           {patched_mean_acc:.2%} ± {patched_std_acc:.2%}")
    print(
        f"Template Plain:    {template_plain_mean_acc:.2%} ± {template_plain_std_acc:.2%}"
    )
    print(
        f"Template Patched:  {template_patched_mean_acc:.2%} ± {template_patched_std_acc:.2%}"
    )
    print(
        f"Template Frozen:   {template_frozen_mean_acc:.2%} ± {template_frozen_std_acc:.2%}"
    )
    print(
        f"Template2 Base:    {template2_baseline_mean_acc:.2%} ± {template2_baseline_std_acc:.2%}"
    )
    print(
        f"Cross-T Frozen:    {cross_template_frozen_mean_acc:.2%} ± {cross_template_frozen_std_acc:.2%}"
    )
    print(
        f"Cross-T Patched:   {cross_template_patched_mean_acc:.2%} ± {cross_template_patched_std_acc:.2%}"
    )

    print("\n" + "-" * 80)
    print("DIFFERENCES VS BASELINE (Mean):")
    print("-" * 80)
    print(f"  Frozen:            {frozen_mean_acc - baseline_mean_acc:.2%}")
    print(f"  Regenerated:       {regenerated_mean_acc - baseline_mean_acc:.2%}")
    print(f"  Patched:           {patched_mean_acc - baseline_mean_acc:.2%}")
    print(f"  Template Plain:    {template_plain_mean_acc - baseline_mean_acc:.2%}")
    print(f"  Template Patched:  {template_patched_mean_acc - baseline_mean_acc:.2%}")
    print(f"  Template Frozen:   {template_frozen_mean_acc - baseline_mean_acc:.2%}")
    print(f"  Template2 Base:    {template2_baseline_mean_acc - baseline_mean_acc:.2%}")
    print(
        f"  Cross-T Frozen:    {cross_template_frozen_mean_acc - baseline_mean_acc:.2%}"
    )
    print(
        f"  Cross-T Patched:   {cross_template_patched_mean_acc - baseline_mean_acc:.2%}"
    )

    # Save results
    results_file = "nov_28_mean_ablated_frozen_latents_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "num_test_cases": len(results),
                "num_samples_per_prompt": num_samples_per_prompt,
                "temperature": temperature,
                "total_valid": total_valid,
                "baseline_position_accuracies": [
                    float(x) for x in baseline_position_accuracies
                ],
                "frozen_position_accuracies": [
                    float(x) for x in frozen_position_accuracies
                ],
                "regenerated_position_accuracies": [
                    float(x) for x in regenerated_position_accuracies
                ],
                "patched_position_accuracies": [
                    float(x) for x in patched_position_accuracies
                ],
                "template_plain_position_accuracies": [
                    float(x) for x in template_plain_position_accuracies
                ],
                "template_patched_position_accuracies": [
                    float(x) for x in template_patched_position_accuracies
                ],
                "template_frozen_position_accuracies": [
                    float(x) for x in template_frozen_position_accuracies
                ],
                "template2_baseline_position_accuracies": [
                    float(x) for x in template2_baseline_position_accuracies
                ],
                "cross_template_frozen_position_accuracies": [
                    float(x) for x in cross_template_frozen_position_accuracies
                ],
                "cross_template_patched_position_accuracies": [
                    float(x) for x in cross_template_patched_position_accuracies
                ],
                "baseline_mean_accuracy": float(baseline_mean_acc),
                "baseline_std_accuracy": float(baseline_std_acc),
                "frozen_mean_accuracy": float(frozen_mean_acc),
                "frozen_std_accuracy": float(frozen_std_acc),
                "regenerated_mean_accuracy": float(regenerated_mean_acc),
                "regenerated_std_accuracy": float(regenerated_std_acc),
                "patched_mean_accuracy": float(patched_mean_acc),
                "patched_std_accuracy": float(patched_std_acc),
                "template_plain_mean_accuracy": float(template_plain_mean_acc),
                "template_plain_std_accuracy": float(template_plain_std_acc),
                "template_patched_mean_accuracy": float(template_patched_mean_acc),
                "template_patched_std_accuracy": float(template_patched_std_acc),
                "template_frozen_mean_accuracy": float(template_frozen_mean_acc),
                "template_frozen_std_accuracy": float(template_frozen_std_acc),
                "template2_baseline_mean_accuracy": float(template2_baseline_mean_acc),
                "template2_baseline_std_accuracy": float(template2_baseline_std_acc),
                "cross_template_frozen_mean_accuracy": float(
                    cross_template_frozen_mean_acc
                ),
                "cross_template_frozen_std_accuracy": float(
                    cross_template_frozen_std_acc
                ),
                "cross_template_patched_mean_accuracy": float(
                    cross_template_patched_mean_acc
                ),
                "cross_template_patched_std_accuracy": float(
                    cross_template_patched_std_acc
                ),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_file}")

    # Create visualizations
    print("\nCreating visualizations...")

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))

    # Mean accuracy with error bars (std dev)
    methods = [
        "Baseline",
        "Mean-Abl\n+ Frozen",
        "Mean-Abl\n+ Regen",
        "Mean-Abl\n+ Patched",
        "Template\nPlain",
        "Template\n+ Patched",
        "Template\n+ Frozen",
        "Template2\nBaseline",
        "Cross-T\nFrozen",
        "Cross-T\nPatched",
    ]
    mean_accs = [
        baseline_mean_acc * 100,
        frozen_mean_acc * 100,
        regenerated_mean_acc * 100,
        patched_mean_acc * 100,
        template_plain_mean_acc * 100,
        template_patched_mean_acc * 100,
        template_frozen_mean_acc * 100,
        template2_baseline_mean_acc * 100,
        cross_template_frozen_mean_acc * 100,
        cross_template_patched_mean_acc * 100,
    ]
    std_accs = [
        baseline_std_acc * 100,
        frozen_std_acc * 100,
        regenerated_std_acc * 100,
        patched_std_acc * 100,
        template_plain_std_acc * 100,
        template_patched_std_acc * 100,
        template_frozen_std_acc * 100,
        template2_baseline_std_acc * 100,
        cross_template_frozen_std_acc * 100,
        cross_template_patched_std_acc * 100,
    ]
    colors = [
        "#2ecc71",
        "#3498db",
        "#e74c3c",
        "#9b59b6",
        "#f39c12",
        "#16a085",
        "#e67e22",
        "#95a5a6",
        "#d35400",
        "#c0392b",
    ]

    bars = ax.bar(
        methods,
        mean_accs,
        yerr=std_accs,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
        capsize=5,
        error_kw={"linewidth": 2},
    )
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Mean Accuracy Across {num_samples_per_prompt} Samples (Temp={temperature})\n({num_test_cases} Test Cases, Error bars = std dev)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim([0, 100])
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, acc, std in zip(bars, mean_accs, std_accs):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 2,
            f"{acc:.1f}%\n±{std:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    chart_filename = "nov_28_mean_ablated_frozen_latents_charts.png"
    fig.savefig(chart_filename, dpi=300, bbox_inches="tight")
    print(f"Chart saved to {chart_filename}")

    plt.show()


if __name__ == "__main__":
    main()

# %%
