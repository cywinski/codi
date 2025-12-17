# ABOUTME: Trains a difference-in-means linear probe on CODI latent vectors to classify operation type.
# ABOUTME: Uses prompts.json dataset with train/test split and evaluates probe accuracy per latent position.

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
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "operation_probe_latent_vectors"
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
def extract_latent_vectors(
    model: CODI,
    prompt: str,
    num_latent_iterations: int,
    sot_token: int,
    eot_token: int,
) -> list[torch.Tensor]:
    """
    Extract CODI latent vectors using the generate method.

    Returns:
        List of length (num_latent_iterations + 1) where index i corresponds to
        latent position i, each tensor has shape [hidden_dim] on CPU.
    """
    tokenizer = model.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=32,
            num_latent_iterations=num_latent_iterations,
            greedy=True,
            return_latent_vectors=True,
            sot_token=sot_token,
            eot_token=eot_token,
        )

    # latent_vectors is a list of tensors with shape (batch_size, 1, hidden_dim)
    latent_vectors = outputs["latent_vectors"]

    # Extract and move to CPU
    extracted = []
    for lv in latent_vectors:
        # Shape: (1, 1, hidden_dim) -> (hidden_dim,)
        extracted.append(lv.squeeze(0).squeeze(0).cpu())

    return extracted


# %%
def _l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    """L2 normalize a vector."""
    denom = vec.norm()
    if denom.item() == 0:
        return vec
    return vec / denom


def train_probe_by_latent_position(
    pos_latents: dict[int, list[torch.Tensor]],
    neg_latents: dict[int, list[torch.Tensor]],
    num_latent_positions: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """
    Train a difference-in-means linear direction probe for each latent position.

    Probe direction = normalize(mean(addition) - mean(subtraction))
    Threshold = midpoint between class means along that direction
    """
    probes: dict[int, dict[str, torch.Tensor]] = {}
    for latent_pos in range(num_latent_positions):
        pos_stack = torch.stack(pos_latents[latent_pos], dim=0).float()
        neg_stack = torch.stack(neg_latents[latent_pos], dim=0).float()

        pos_mean = pos_stack.mean(dim=0)
        neg_mean = neg_stack.mean(dim=0)
        direction = _l2_normalize(pos_mean - neg_mean)

        pos_proj_mean = torch.dot(direction, pos_mean).item()
        neg_proj_mean = torch.dot(direction, neg_mean).item()
        threshold = 0.5 * (pos_proj_mean + neg_proj_mean)

        probes[latent_pos] = {
            "direction": direction,
            "threshold": torch.tensor(threshold, dtype=torch.float32),
        }
    return probes


def evaluate_probe_accuracy_by_latent_position(
    probes: dict[int, dict[str, torch.Tensor]],
    pos_latents: dict[int, list[torch.Tensor]],
    neg_latents: dict[int, list[torch.Tensor]],
    num_latent_positions: int,
) -> dict[int, dict[str, float]]:
    """
    Evaluate probe accuracy for each latent position.

    Returns accuracy and standard deviation for each position.
    """
    results: dict[int, dict[str, float]] = {}
    for latent_pos in range(num_latent_positions):
        direction = probes[latent_pos]["direction"].float()
        threshold = probes[latent_pos]["threshold"].item()

        correct_list: list[int] = []

        for vec in pos_latents[latent_pos]:
            score = torch.dot(direction, vec.float()).item()
            pred_pos = score > threshold
            correct_list.append(int(pred_pos))

        for vec in neg_latents[latent_pos]:
            score = torch.dot(direction, vec.float()).item()
            pred_pos = score > threshold
            correct_list.append(int(not pred_pos))

        correct_arr = np.array(correct_list)
        accuracy = correct_arr.mean()
        std = correct_arr.std()

        results[latent_pos] = {"accuracy": float(accuracy), "std": float(std)}

    return results


def evaluate_probe_dot_products_by_latent_position(
    probes: dict[int, dict[str, torch.Tensor]],
    pos_latents: dict[int, list[torch.Tensor]],
    neg_latents: dict[int, list[torch.Tensor]],
    num_latent_positions: int,
) -> dict[int, dict[str, object]]:
    """
    Evaluate dot products (cosine similarity) with each latent-position probe direction.

    Returns per latent position:
      - addition: {mean, std, dot_products}
      - subtraction: {mean, std, dot_products}
    """
    results: dict[int, dict[str, object]] = {}
    for latent_pos in range(num_latent_positions):
        direction = probes[latent_pos]["direction"].float()

        add_dots: list[float] = []
        sub_dots: list[float] = []

        for vec in pos_latents[latent_pos]:
            vec_n = _l2_normalize(vec.float())
            add_dots.append(torch.dot(direction, vec_n).item())

        for vec in neg_latents[latent_pos]:
            vec_n = _l2_normalize(vec.float())
            sub_dots.append(torch.dot(direction, vec_n).item())

        results[latent_pos] = {
            "addition": {
                "mean": float(np.mean(add_dots)),
                "std": float(np.std(add_dots)),
                "dot_products": add_dots,
            },
            "subtraction": {
                "mean": float(np.mean(sub_dots)),
                "std": float(np.std(sub_dots)),
                "dot_products": sub_dots,
            },
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
SOT_TOKEN = tokenizer.bocot_id
EOT_TOKEN = tokenizer.eocot_id

NUM_LATENT_POSITIONS = NUM_LATENT + 1

# %%
print("\nLoading prompts...")
prompts = load_prompts(PROMPTS_PATH)
print(f"Loaded {len(prompts)} prompts")

train_prompts, test_prompts = train_test_split(prompts, TRAIN_RATIO, RANDOM_SEED)
print(f"Train: {len(train_prompts)}, Test: {len(test_prompts)}")

# %%
print("\nExtracting TRAINING latent vectors (addition vs subtraction)...")
train_pos_latents: dict[int, list[torch.Tensor]] = {
    i: [] for i in range(NUM_LATENT_POSITIONS)
}
train_neg_latents: dict[int, list[torch.Tensor]] = {
    i: [] for i in range(NUM_LATENT_POSITIONS)
}

for prompt_data in tqdm(train_prompts, desc="Training prompts"):
    add_prompt = prompt_data["addition"]["prompt"]
    sub_prompt = prompt_data["subtraction"]["prompt"]

    add_latents = extract_latent_vectors(
        model, add_prompt, NUM_LATENT, SOT_TOKEN, EOT_TOKEN
    )
    sub_latents = extract_latent_vectors(
        model, sub_prompt, NUM_LATENT, SOT_TOKEN, EOT_TOKEN
    )

    for latent_pos in range(NUM_LATENT_POSITIONS):
        train_pos_latents[latent_pos].append(add_latents[latent_pos])
        train_neg_latents[latent_pos].append(sub_latents[latent_pos])

print("Training probes (difference-in-means)...")
probes = train_probe_by_latent_position(
    pos_latents=train_pos_latents,
    neg_latents=train_neg_latents,
    num_latent_positions=NUM_LATENT_POSITIONS,
)

# %%
print("\nExtracting TEST latent vectors (addition vs subtraction)...")
test_pos_latents: dict[int, list[torch.Tensor]] = {
    i: [] for i in range(NUM_LATENT_POSITIONS)
}
test_neg_latents: dict[int, list[torch.Tensor]] = {
    i: [] for i in range(NUM_LATENT_POSITIONS)
}

for prompt_data in tqdm(test_prompts, desc="Test prompts"):
    add_prompt = prompt_data["addition"]["prompt"]
    sub_prompt = prompt_data["subtraction"]["prompt"]

    add_latents = extract_latent_vectors(
        model, add_prompt, NUM_LATENT, SOT_TOKEN, EOT_TOKEN
    )
    sub_latents = extract_latent_vectors(
        model, sub_prompt, NUM_LATENT, SOT_TOKEN, EOT_TOKEN
    )

    for latent_pos in range(NUM_LATENT_POSITIONS):
        test_pos_latents[latent_pos].append(add_latents[latent_pos])
        test_neg_latents[latent_pos].append(sub_latents[latent_pos])

# %%
print("\nEvaluating probe accuracy on test set...")
accuracy_results = evaluate_probe_accuracy_by_latent_position(
    probes=probes,
    pos_latents=test_pos_latents,
    neg_latents=test_neg_latents,
    num_latent_positions=NUM_LATENT_POSITIONS,
)

print("\nProbe accuracy by latent position:")
for latent_pos in range(NUM_LATENT_POSITIONS):
    acc = accuracy_results[latent_pos]["accuracy"]
    std = accuracy_results[latent_pos]["std"]
    print(f"  Latent Position {latent_pos}: accuracy={acc:.4f} ± {std:.4f}")

# %%
print("\nEvaluating dot products on test set...")
dot_product_results = evaluate_probe_dot_products_by_latent_position(
    probes=probes,
    pos_latents=test_pos_latents,
    neg_latents=test_neg_latents,
    num_latent_positions=NUM_LATENT_POSITIONS,
)

print("\nDot products (cosine similarity) by latent position:")
for latent_pos in range(NUM_LATENT_POSITIONS):
    add_mean = dot_product_results[latent_pos]["addition"]["mean"]
    sub_mean = dot_product_results[latent_pos]["subtraction"]["mean"]
    print(
        f"  Latent Position {latent_pos}: addition={add_mean:.4f} subtraction={sub_mean:.4f}"
    )

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
    },
    "accuracy": {str(k): v for k, v in accuracy_results.items()},
    "dot_products": {str(k): v for k, v in dot_product_results.items()},
}

results_file = OUTPUT_DIR / "probe_results.json"
with open(results_file, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"\nSaved results to {results_file}")

# %%
# Plot 1: Probe accuracy by latent position (bar plot)
xs = list(range(NUM_LATENT_POSITIONS))
accuracies = [accuracy_results[i]["accuracy"] for i in xs]
stds = [accuracy_results[i]["std"] for i in xs]

fontsize_title = 20
fontsize_label = 18
fontsize_tick = 16

fig, ax = plt.subplots(figsize=(10, 6))

# Create gradient colors for bars (light blue to dark blue)
colors = []
for i in range(len(xs)):
    normalized = i / (len(xs) - 1) if len(xs) > 1 else 0
    r = int(52 + (26 - 52) * normalized)
    g = int(152 + (80 - 152) * normalized)
    b = int(219 + (184 - 219) * normalized)
    colors.append(f"#{r:02x}{g:02x}{b:02x}")

# Bar plot
for i in range(len(xs)):
    ax.bar(
        i,
        accuracies[i],
        color=colors[i],
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
        width=0.6,
    )

ax.set_xticks(xs)
ax.set_xticklabels([str(x) for x in xs], fontsize=fontsize_tick)
ax.set_xlabel("Latent Vector Index", fontsize=fontsize_label, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=fontsize_label, fontweight="bold")
ax.set_title(
    "Math Operation Probe trained on Latent Vectors",
    fontsize=fontsize_title,
    fontweight="bold",
    pad=16,
)
ax.set_ylim(0.0, 1.05)
ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
ax.tick_params(axis="y", labelsize=fontsize_tick)
plt.tight_layout()

accuracy_plot_path = OUTPUT_DIR / "probe_accuracy_by_latent_position.png"
plt.savefig(accuracy_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved accuracy plot to {accuracy_plot_path}")
