# ABOUTME: Trains difference-in-means linear probes on CODI residual stream at each layer.
# ABOUTME: Evaluates probe accuracy for each layer and latent position, producing a heatmap.

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CODI

load_dotenv()

# %%
# Configuration
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
NUM_LATENT = 6
USE_PRJ = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Train/test split
TRAIN_RATIO = 0.7
RANDOM_SEED = 42

# Paths
PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "prompts.json"
OUTPUT_DIR = (
    Path(__file__).parent.parent
    / "results"
    / "operation_probe_residual_stream_accuracy"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
def load_prompts(prompts_path: Path) -> list[dict]:
    """Load prompts from JSON file."""
    with open(prompts_path, "r") as f:
        data = json.load(f)
    return data["prompts"]


def train_test_split(
    prompts: list[dict], train_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split prompts into train and test sets."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(prompts))
    n_train = int(len(prompts) * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    train_prompts = [prompts[i] for i in train_indices]
    test_prompts = [prompts[i] for i in test_indices]
    return train_prompts, test_prompts


# %%
def prepare_inputs(model, tokenizer, prompt):
    """Prepare input tensors with BOT token appended."""
    device = model.codi.device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")
    bot_tensor = torch.tensor(
        [tokenizer.eos_token_id, sot_token], dtype=torch.long, device=device
    ).unsqueeze(0)

    input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
    attention_mask_bot = torch.cat((attention_mask, torch.ones_like(bot_tensor)), dim=1)

    return input_ids_bot, attention_mask_bot


def extract_residual_stream_at_latent_positions(
    model: CODI,
    prompt: str,
    num_latent_iterations: int,
) -> dict[int, list[torch.Tensor]]:
    """
    Extract residual stream (hidden states) at each layer for each latent position.

    Returns:
        Dict mapping latent_position -> list of hidden states per layer.
        Each hidden state has shape [hidden_dim] on CPU.
        latent_position 0 is the initial BOCOT position, 1..num_latent are iterations.
    """
    tokenizer = model.tokenizer
    input_ids, attention_mask = prepare_inputs(model, tokenizer, prompt)

    num_latent_positions = num_latent_iterations + 1
    # result[latent_pos][layer_idx] = tensor of shape [hidden_dim]
    result: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_latent_positions)}

    with torch.no_grad():
        # Initial forward pass to get hidden states at BOCOT position
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask,
        )
        past_key_values = outputs.past_key_values

        # Hidden states for latent position 0 (BOCOT position)
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor has shape (batch, seq_len, hidden_dim)
        # We want the last position (-1) for each layer
        for layer_idx, h in enumerate(outputs.hidden_states):
            result[0].append(h[0, -1, :].cpu())

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

            # Hidden states for latent position i+1
            latent_pos = i + 1
            for layer_idx, h in enumerate(outputs.hidden_states):
                result[latent_pos].append(h[0, -1, :].cpu())

            # Get next latent embedding
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd)
                latent_embd = latent_embd.to(dtype=model.codi.dtype)

    return result


# %%
def _l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    """L2 normalize a vector."""
    denom = vec.norm()
    if denom.item() == 0:
        return vec
    return vec / denom


def train_probes_by_layer_and_position(
    pos_hidden_states: dict[int, dict[int, list[torch.Tensor]]],
    neg_hidden_states: dict[int, dict[int, list[torch.Tensor]]],
    num_latent_positions: int,
    num_layers: int,
) -> dict[int, dict[int, dict[str, torch.Tensor]]]:
    """
    Train difference-in-means probes for each layer and latent position.

    Args:
        pos_hidden_states: {latent_pos: {layer_idx: [tensors]}}
        neg_hidden_states: {latent_pos: {layer_idx: [tensors]}}
        num_latent_positions: Number of latent positions (including initial)
        num_layers: Number of layers

    Returns:
        {latent_pos: {layer_idx: {"direction": tensor, "threshold": tensor}}}
    """
    probes: dict[int, dict[int, dict[str, torch.Tensor]]] = {}

    for latent_pos in range(num_latent_positions):
        probes[latent_pos] = {}
        for layer_idx in range(num_layers):
            pos_stack = torch.stack(
                pos_hidden_states[latent_pos][layer_idx], dim=0
            ).float()
            neg_stack = torch.stack(
                neg_hidden_states[latent_pos][layer_idx], dim=0
            ).float()

            pos_mean = pos_stack.mean(dim=0)
            neg_mean = neg_stack.mean(dim=0)
            direction = _l2_normalize(pos_mean - neg_mean)

            # Compute threshold as midpoint between pos and neg projections
            pos_proj = torch.dot(direction, pos_mean).item()
            neg_proj = torch.dot(direction, neg_mean).item()
            threshold = (pos_proj + neg_proj) / 2.0

            probes[latent_pos][layer_idx] = {
                "direction": direction,
                "threshold": torch.tensor(threshold, dtype=torch.float32),
            }

    return probes


def evaluate_probe_accuracy(
    probes: dict[int, dict[int, dict[str, torch.Tensor]]],
    pos_hidden_states: dict[int, dict[int, list[torch.Tensor]]],
    neg_hidden_states: dict[int, dict[int, list[torch.Tensor]]],
    num_latent_positions: int,
    num_layers: int,
) -> dict[int, dict[int, dict[str, float]]]:
    """
    Evaluate probe accuracy for each layer and latent position.

    Returns:
        {latent_pos: {layer_idx: {"accuracy": float, "std": float}}}
    """
    results: dict[int, dict[int, dict[str, float]]] = {}

    for latent_pos in range(num_latent_positions):
        results[latent_pos] = {}
        for layer_idx in range(num_layers):
            direction = probes[latent_pos][layer_idx]["direction"].float()
            threshold = probes[latent_pos][layer_idx]["threshold"].item()

            correct_list: list[int] = []

            for vec in pos_hidden_states[latent_pos][layer_idx]:
                score = torch.dot(direction, vec.float()).item()
                pred_pos = score > threshold
                correct_list.append(int(pred_pos))

            for vec in neg_hidden_states[latent_pos][layer_idx]:
                score = torch.dot(direction, vec.float()).item()
                pred_pos = score > threshold
                correct_list.append(int(not pred_pos))

            correct_arr = np.array(correct_list)
            accuracy = correct_arr.mean()
            std = correct_arr.std()

            results[latent_pos][layer_idx] = {
                "accuracy": float(accuracy),
                "std": float(std),
            }

    return results


# %%
print("Loading CODI model...")
model = CODI.from_pretrained(
    checkpoint_path=CHECKPOINT_PATH,
    model_name_or_path=BASE_MODEL,
    lora_r=128,
    lora_alpha=32,
    num_latent=NUM_LATENT,
    use_prj=USE_PRJ,
    device=DEVICE,
    dtype="bfloat16",
    strict=False,
    remove_eos=False,
    full_precision=True,
)
model.eval()
print(f"Model loaded on {DEVICE}")

# Configure tokenizer with special tokens
tokenizer = model.tokenizer
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = model.pad_token_id
tokenizer.add_special_tokens({"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]})
tokenizer.bocot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")
tokenizer.eocot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")

NUM_LATENT_POSITIONS = NUM_LATENT + 1

# Determine number of layers
# hidden_states includes embedding layer (layer 0) plus all transformer layers
# For Llama-3.2-1B, this is typically 17 (1 embedding + 16 transformer layers)
NUM_LAYERS = model.codi.config.num_hidden_layers + 1
print(f"Number of layers (including embedding): {NUM_LAYERS}")

# %%
print("\nLoading prompts...")
prompts = load_prompts(PROMPTS_PATH)
print(f"Loaded {len(prompts)} prompts")

train_prompts, test_prompts = train_test_split(prompts, TRAIN_RATIO, RANDOM_SEED)
print(f"Train: {len(train_prompts)}, Test: {len(test_prompts)}")

# %%
print("\nExtracting TRAINING hidden states (addition vs subtraction)...")
# train_pos_hidden_states[latent_pos][layer_idx] = list of tensors
train_pos_hidden_states: dict[int, dict[int, list[torch.Tensor]]] = {
    i: {j: [] for j in range(NUM_LAYERS)} for i in range(NUM_LATENT_POSITIONS)
}
train_neg_hidden_states: dict[int, dict[int, list[torch.Tensor]]] = {
    i: {j: [] for j in range(NUM_LAYERS)} for i in range(NUM_LATENT_POSITIONS)
}

for prompt_data in tqdm(train_prompts, desc="Training prompts"):
    add_prompt = prompt_data["addition"]["prompt"]
    sub_prompt = prompt_data["subtraction"]["prompt"]

    add_hidden = extract_residual_stream_at_latent_positions(
        model, add_prompt, NUM_LATENT
    )
    sub_hidden = extract_residual_stream_at_latent_positions(
        model, sub_prompt, NUM_LATENT
    )

    for latent_pos in range(NUM_LATENT_POSITIONS):
        for layer_idx in range(NUM_LAYERS):
            train_pos_hidden_states[latent_pos][layer_idx].append(
                add_hidden[latent_pos][layer_idx]
            )
            train_neg_hidden_states[latent_pos][layer_idx].append(
                sub_hidden[latent_pos][layer_idx]
            )

print("Training probes (difference-in-means)...")
probes = train_probes_by_layer_and_position(
    pos_hidden_states=train_pos_hidden_states,
    neg_hidden_states=train_neg_hidden_states,
    num_latent_positions=NUM_LATENT_POSITIONS,
    num_layers=NUM_LAYERS,
)

# %%
print("\nExtracting TEST hidden states (addition vs subtraction)...")
test_pos_hidden_states: dict[int, dict[int, list[torch.Tensor]]] = {
    i: {j: [] for j in range(NUM_LAYERS)} for i in range(NUM_LATENT_POSITIONS)
}
test_neg_hidden_states: dict[int, dict[int, list[torch.Tensor]]] = {
    i: {j: [] for j in range(NUM_LAYERS)} for i in range(NUM_LATENT_POSITIONS)
}

for prompt_data in tqdm(test_prompts, desc="Test prompts"):
    add_prompt = prompt_data["addition"]["prompt"]
    sub_prompt = prompt_data["subtraction"]["prompt"]

    add_hidden = extract_residual_stream_at_latent_positions(
        model, add_prompt, NUM_LATENT
    )
    sub_hidden = extract_residual_stream_at_latent_positions(
        model, sub_prompt, NUM_LATENT
    )

    for latent_pos in range(NUM_LATENT_POSITIONS):
        for layer_idx in range(NUM_LAYERS):
            test_pos_hidden_states[latent_pos][layer_idx].append(
                add_hidden[latent_pos][layer_idx]
            )
            test_neg_hidden_states[latent_pos][layer_idx].append(
                sub_hidden[latent_pos][layer_idx]
            )

# %%
print("\nEvaluating probe accuracy on test set...")
accuracy_results = evaluate_probe_accuracy(
    probes=probes,
    pos_hidden_states=test_pos_hidden_states,
    neg_hidden_states=test_neg_hidden_states,
    num_latent_positions=NUM_LATENT_POSITIONS,
    num_layers=NUM_LAYERS,
)

print("\nProbe accuracy summary (averaged across layers):")
for latent_pos in range(NUM_LATENT_POSITIONS):
    layer_accs = [
        accuracy_results[latent_pos][l]["accuracy"] for l in range(NUM_LAYERS)
    ]
    avg_acc = np.mean(layer_accs)
    print(f"  Latent Position {latent_pos}: avg_accuracy={avg_acc:.4f}")

# %%
# Save JSON results
results_data = {
    "config": {
        "checkpoint_path": CHECKPOINT_PATH,
        "base_model": BASE_MODEL,
        "num_latent": NUM_LATENT,
        "use_prj": USE_PRJ,
        "train_ratio": TRAIN_RATIO,
        "random_seed": RANDOM_SEED,
        "num_train_prompts": len(train_prompts),
        "num_test_prompts": len(test_prompts),
        "num_layers": NUM_LAYERS,
        "num_latent_positions": NUM_LATENT_POSITIONS,
    },
    "accuracy": {
        str(latent_pos): {
            str(layer_idx): accuracy_results[latent_pos][layer_idx]
            for layer_idx in range(NUM_LAYERS)
        }
        for latent_pos in range(NUM_LATENT_POSITIONS)
    },
}

results_file = OUTPUT_DIR / "probe_results.json"
with open(results_file, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"\nSaved results to {results_file}")

# %%
# Create accuracy heatmap (layer x latent position)
accuracy_matrix = np.zeros((NUM_LAYERS, NUM_LATENT_POSITIONS))
for latent_pos in range(NUM_LATENT_POSITIONS):
    for layer_idx in range(NUM_LAYERS):
        accuracy_matrix[layer_idx, latent_pos] = accuracy_results[latent_pos][
            layer_idx
        ]["accuracy"]

fontsize_title = 20
fontsize_label = 18
fontsize_tick = 14
fontsize_cbar = 14

fig, ax = plt.subplots(figsize=(10, 12))

# Plot heatmap
im = ax.imshow(accuracy_matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label(
    "Accuracy", rotation=270, labelpad=20, fontsize=fontsize_cbar, fontweight="bold"
)
cbar.ax.tick_params(labelsize=fontsize_tick)

# Annotate cells with accuracy values
for i in range(NUM_LAYERS):
    for j in range(NUM_LATENT_POSITIONS):
        acc = accuracy_matrix[i, j]
        text_color = "white" if acc < 0.7 else "black"
        ax.text(
            j,
            i,
            f"{acc:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=10,
            fontweight="bold",
        )

# Labels
ax.set_xlabel("Latent Position", fontsize=fontsize_label, fontweight="bold")
ax.set_ylabel("Layer", fontsize=fontsize_label, fontweight="bold")
ax.set_xticks(range(NUM_LATENT_POSITIONS))
ax.set_xticklabels(
    [str(i) for i in range(NUM_LATENT_POSITIONS)], fontsize=fontsize_tick
)
ax.set_yticks(range(NUM_LAYERS))
ax.set_yticklabels([str(i) for i in range(NUM_LAYERS)], fontsize=fontsize_tick)
ax.set_title(
    "Operation Probe Accuracy by Layer and Latent Position",
    fontsize=fontsize_title,
    fontweight="bold",
    pad=16,
)

plt.tight_layout()

heatmap_path = OUTPUT_DIR / "accuracy_heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved accuracy heatmap to {heatmap_path}")

# %%
# Plot accuracy by layer for each latent position (small multiples)
n_cols = 4
n_rows = (NUM_LATENT_POSITIONS + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharey=True)
axes = axes.flatten()

colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#34495e"]

for latent_pos in range(NUM_LATENT_POSITIONS):
    ax = axes[latent_pos]
    layer_accs = [
        accuracy_results[latent_pos][l]["accuracy"] for l in range(NUM_LAYERS)
    ]

    ax.plot(
        range(NUM_LAYERS),
        layer_accs,
        color=colors[latent_pos % len(colors)],
        linewidth=2.5,
        marker="o",
        markersize=5,
    )
    ax.fill_between(
        range(NUM_LAYERS),
        [0.5] * NUM_LAYERS,
        layer_accs,
        alpha=0.15,
        color=colors[latent_pos % len(colors)],
    )

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_title(f"Latent Position {latent_pos}", fontsize=14, fontweight="bold")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    ax.grid(axis="both", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=11)

# Hide unused subplots
for i in range(NUM_LATENT_POSITIONS, len(axes)):
    axes[i].set_visible(False)

# Common labels
fig.supxlabel("Layer", fontsize=fontsize_label, fontweight="bold", y=0.02)
fig.supylabel("Accuracy", fontsize=fontsize_label, fontweight="bold", x=0.02)
fig.suptitle(
    "Operation Probe Accuracy by Layer (per Latent Position)",
    fontsize=fontsize_title,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()

line_plot_path = OUTPUT_DIR / "accuracy_by_layer.png"
plt.savefig(line_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved accuracy by layer plot to {line_plot_path}")

print("\nExperiment complete!")
