# ABOUTME: Logit lens visualization for CODI latent vectors.
# ABOUTME: Shows most likely tokens at each layer for each latent reasoning position.

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import CODI

# %%
# Parameters
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = "cuda"
DTYPE = "bfloat16"

PROMPT = "A team starts with 3 members. They recruit 5 new members. Then each current member recruits 2 additional people. How many people are there now on the team? Give the answer only and nothing else."
NUM_LATENT_ITERATIONS = 6
TOP_K_TOKENS = 10  # Number of top tokens to display per layer


# %%
def get_lm_head(model):
    """Get the language model head (unembedding matrix) from the model."""
    codi = model.codi
    if hasattr(codi, "get_base_model"):
        return codi.get_base_model().lm_head
    return codi.lm_head


def get_layer_norm(model):
    """Get the final layer norm before the lm_head."""
    codi = model.codi
    if hasattr(codi, "get_base_model"):
        base = codi.get_base_model()
    else:
        base = codi

    # LLaMA/Mistral architecture
    if hasattr(base, "model") and hasattr(base.model, "norm"):
        return base.model.norm
    # GPT-2 architecture
    if hasattr(base, "transformer") and hasattr(base.transformer, "ln_f"):
        return base.transformer.ln_f
    return None


def logit_lens(hidden_states, lm_head, layer_norm=None, top_k=5):
    """
    Apply logit lens to hidden states.

    Args:
        hidden_states: Tuple of hidden states from each layer, each of shape (batch, seq, hidden)
        lm_head: The unembedding matrix (linear layer)
        layer_norm: Optional final layer norm to apply before unembedding
        top_k: Number of top tokens to return

    Returns:
        List of (layer_idx, top_tokens, top_probs) for the last position
    """
    results = []

    for layer_idx, h in enumerate(hidden_states):
        # Take the last position
        h_last = h[:, -1, :]  # (batch, hidden)

        # Optionally apply layer norm (for fair comparison with final output)
        if layer_norm is not None:
            h_last = layer_norm(h_last)

        # Project through unembedding
        logits = lm_head(h_last)  # (batch, vocab)
        probs = torch.softmax(logits, dim=-1)

        # Get top-k
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        results.append(
            {
                "layer": layer_idx,
                "top_indices": top_indices[0].cpu().tolist(),
                "top_probs": top_probs[0].cpu().tolist(),
            }
        )

    return results


def run_inference_with_logit_lens(
    model, tokenizer, prompt, num_latent_iterations, top_k=5
):
    """
    Run CODI inference and capture logit lens results for each latent position.

    Returns:
        Dict with prompt_logit_lens and latent_logit_lens for each iteration
    """
    device = next(model.parameters()).device

    # Get model components
    lm_head = get_lm_head(model)
    layer_norm = get_layer_norm(model)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Special tokens
    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")

    # Add BOT tokens
    bot_tensor = torch.tensor(
        [tokenizer.eos_token_id, sot_token], dtype=torch.long, device=device
    ).unsqueeze(0)

    input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
    attention_mask_bot = torch.cat((attention_mask, torch.ones_like(bot_tensor)), dim=1)

    results = {
        "prompt": prompt,
        "num_latent_iterations": num_latent_iterations,
        "prompt_hidden_states_lens": None,
        "latent_positions": [],
    }

    with torch.no_grad():
        # Encode prompt
        outputs = model.codi(
            input_ids=input_ids_bot,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask_bot,
        )
        past_key_values = outputs.past_key_values

        # Logit lens on prompt (last position = after <|bocot|>)
        prompt_lens = logit_lens(outputs.hidden_states, lm_head, layer_norm, top_k)
        results["prompt_hidden_states_lens"] = prompt_lens

        # Get initial latent embedding
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        # Optionally project
        if model.use_prj:
            latent_embd = model.prj(latent_embd)
            latent_embd = latent_embd.to(dtype=model.codi.dtype)

        # Latent iterations
        for i in range(num_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            # Logit lens on this latent position
            latent_lens = logit_lens(outputs.hidden_states, lm_head, layer_norm, top_k)
            results["latent_positions"].append(
                {"iteration": i, "logit_lens": latent_lens}
            )

            # Get next latent embedding
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd)
                latent_embd = latent_embd.to(dtype=model.codi.dtype)

    return results


# %%
def visualize_logit_lens(results, tokenizer, figsize=(14, 10)):
    """
    Visualize logit lens results as a heatmap with token annotations.
    """
    num_latent = len(results["latent_positions"])
    if num_latent == 0:
        print("No latent positions to visualize")
        return

    num_layers = len(results["latent_positions"][0]["logit_lens"])

    # Include prompt position (one position to the left)
    include_prompt = results["prompt_hidden_states_lens"] is not None
    num_positions = num_latent + (1 if include_prompt else 0)

    # Create matrix of top-1 probabilities
    prob_matrix = np.zeros((num_layers, num_positions))
    token_matrix = [[None] * num_positions for _ in range(num_layers)]

    # Add prompt position (column 0)
    if include_prompt:
        for layer_data in results["prompt_hidden_states_lens"]:
            layer_idx = layer_data["layer"]
            top_token_id = layer_data["top_indices"][0]
            top_prob = layer_data["top_probs"][0]

            prob_matrix[layer_idx, 0] = top_prob
            token_matrix[layer_idx][0] = tokenizer.decode([top_token_id])

    # Add latent positions (columns 1 onwards)
    for pos_idx, pos_data in enumerate(results["latent_positions"]):
        col_idx = pos_idx + (1 if include_prompt else 0)
        for layer_data in pos_data["logit_lens"]:
            layer_idx = layer_data["layer"]
            top_token_id = layer_data["top_indices"][0]
            top_prob = layer_data["top_probs"][0]

            prob_matrix[layer_idx, col_idx] = top_prob
            token_matrix[layer_idx][col_idx] = tokenizer.decode([top_token_id])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Top-1 probability", rotation=270, labelpad=15, fontsize=14)

    # Annotate cells with tokens
    for i in range(num_layers):
        for j in range(num_positions):
            token = token_matrix[i][j]
            prob = prob_matrix[i, j]
            # Escape special characters for display
            token_display = repr(token)[1:-1] if token else ""
            text_color = "white" if prob > 0.5 else "black"
            ax.text(
                j,
                i,
                token_display,
                ha="center",
                va="center",
                color=text_color,
                fontsize=16,
            )

    # Labels
    ax.set_xlabel("Latent vector index", fontsize=14, fontweight="bold")
    ax.set_ylabel("Layer", fontsize=14, fontweight="bold")
    ax.set_xticks(range(num_positions))
    ax.set_xticklabels([str(i) for i in range(num_positions)], fontsize=12)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"{i}" for i in range(num_layers)], fontsize=12)

    plt.tight_layout()
    return fig


def print_logit_lens_table(results, tokenizer, top_k=5):
    """
    Print a detailed table of logit lens results.
    """
    print("=" * 80)
    print(f"Prompt: {results['prompt']}")
    print(f"Number of latent iterations: {results['num_latent_iterations']}")
    print("=" * 80)

    for pos_idx, pos_data in enumerate(results["latent_positions"]):
        print(f"\n{'=' * 40}")
        print(f"LATENT POSITION {pos_idx}")
        print(f"{'=' * 40}")

        for layer_data in pos_data["logit_lens"]:
            layer = layer_data["layer"]
            tokens = [
                tokenizer.decode([tid]) for tid in layer_data["top_indices"][:top_k]
            ]
            probs = layer_data["top_probs"][:top_k]

            token_str = " | ".join(
                [f"{repr(t):>10s} ({p:.3f})" for t, p in zip(tokens, probs)]
            )
            print(f"Layer {layer:2d}: {token_str}")


# %%
def ensure_tokenizer_special_tokens(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]}
    )


# %%
load_dotenv()

print("Loading model...")
model = CODI.from_pretrained(
    checkpoint_path=CHECKPOINT_PATH,
    model_name_or_path=MODEL_NAME_OR_PATH,
    lora_r=128,
    lora_alpha=32,
    num_latent=6,
    use_prj=True,
    device=DEVICE,
    dtype=DTYPE,
    strict=False,
    checkpoint_save_path=f"./checkpoints/{CHECKPOINT_PATH}",
    remove_eos=False,
    full_precision=True,
)
tokenizer = model.tokenizer
ensure_tokenizer_special_tokens(tokenizer, model)
# %%
print("Running inference with logit lens...")
results = run_inference_with_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompt=PROMPT,
    num_latent_iterations=NUM_LATENT_ITERATIONS,
    top_k=TOP_K_TOKENS,
)
# %%
# Print detailed table
print_logit_lens_table(results, tokenizer, top_k=TOP_K_TOKENS)

# Visualize
fig = visualize_logit_lens(results, tokenizer)
if fig is not None:
    results_dir = "results/logit_lens_latents"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "logit_lens_latents.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path}")
    plt.show()


# %%
