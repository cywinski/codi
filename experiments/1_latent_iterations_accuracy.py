# ABOUTME: Evaluate CODI accuracy on latent-iteration count for both addition and subtraction.
# ABOUTME: Evaluates the same 50 prompts (seeded) for both operations and plots results side-by-side.

# %%
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import extract_answer_number
from src.model import CODI

# %%
# Parameters (edit these at the top for quick experimentation)
CHECKPOINT_PATH = "bcywinski/codi_llama1b-answer_only"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = "cuda"  # set to "cpu" if needed
DTYPE = "bfloat16"

NUM_SAMPLES_PER_PROMPT = 5
TEMPERATURE = 1.0
GREEDY = False
MAX_NEW_TOKENS = 128

PROMPTS_JSON_PATH = (
    Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    / "prompts"
    / "prompts.json"
)
RESULTS_DIR = Path("results/latent_iterations_accuracy")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# %%
def load_prompts_from_json():
    """Load all prompts from prompts.json file."""
    with open(PROMPTS_JSON_PATH, "r") as f:
        data = json.load(f)
    return data["prompts"]


def build_test_cases(prompts: list, template_idx: int, operation: str = "addition"):
    """Build test cases for a given template and operation from loaded prompts."""
    test_cases = []
    for prompt_data in prompts:
        if prompt_data["template_idx"] == template_idx:
            operation_key = operation
            test_cases.append(
                {
                    "id": prompt_data["id"],
                    "X": prompt_data["X"],
                    "Y": prompt_data["Y"],
                    "Z": prompt_data["Z"],
                    "prompt": prompt_data[operation_key]["prompt"],
                    "ground_truth": prompt_data[operation_key]["ground_truth"],
                }
            )
    return test_cases


# %%
def ensure_tokenizer_special_tokens(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]}
    )


def evaluate_accuracy_by_latent_iterations(
    model,
    tokenizer,
    test_cases,
    num_samples_per_prompt: int,
    temperature: float,
    greedy: bool,
    max_new_tokens: int,
    seed: int,
):
    """Evaluate accuracy across latent iteration counts."""
    iteration_values = list(range(0, 7))
    mean_accs = []
    std_errs = []
    per_prompt_accs_by_iter = {}

    for num_latent_iterations in iteration_values:
        skip_thinking = num_latent_iterations == 0
        prompt_accuracies = []

        for tc in tqdm(
            test_cases,
            desc=f"iters={num_latent_iterations} (skip={skip_thinking})",
            leave=False,
        ):
            prompt = tc["prompt"]
            ground_truth = tc["ground_truth"]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(model.codi.device)
            attention_mask = inputs["attention_mask"].to(model.codi.device)

            sample_correct = []
            for sample_idx in range(num_samples_per_prompt):
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    num_latent_iterations=num_latent_iterations,
                    temperature=temperature,
                    greedy=greedy,
                    return_latent_vectors=False,
                    remove_eos=False,
                    output_hidden_states=True,
                    output_attentions=False,
                    skip_thinking=skip_thinking,
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
                    and int(answer) == int(ground_truth)
                )
                sample_correct.append(bool(correct))

            prompt_accuracies.append(float(np.mean(sample_correct)))

        prompt_accuracies = np.asarray(prompt_accuracies, dtype=np.float64)
        per_prompt_accs_by_iter[str(num_latent_iterations)] = prompt_accuracies.tolist()

        mean_acc = float(np.mean(prompt_accuracies))
        std_err = float(np.std(prompt_accuracies) / np.sqrt(len(prompt_accuracies)))
        mean_accs.append(mean_acc)
        std_errs.append(std_err)

    return {
        "iteration_values": iteration_values,
        "mean_accuracies": mean_accs,
        "std_errors": std_errs,
        "per_prompt_accuracies": per_prompt_accs_by_iter,
    }


# %%
def plot_aggregated_accuracy(
    iteration_values,
    addition_means,
    addition_stds,
    subtraction_means,
    subtraction_stds,
    out_path,
):
    """Plot average accuracy across all templates and operations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(iteration_values))
    labels = [str(v) for v in iteration_values]

    fontsize_title = 20
    fontsize_label = 18
    fontsize_tick = 16

    # Compute average and combined error
    avg_means = (np.array(addition_means) + np.array(subtraction_means)) / 2
    avg_stds = np.sqrt(
        (np.array(addition_stds) ** 2 + np.array(subtraction_stds) ** 2) / 2
    )

    # Create gradient colors for bars (gray for iteration 0, gradient for others)
    colors = []
    for i in range(len(iteration_values)):
        if iteration_values[i] == 0:
            colors.append("#95a5a6")  # Gray for iteration 0
        else:
            # Gradient from light blue to dark blue
            normalized = (iteration_values[i] - 1) / (max(iteration_values) - 1)
            r = int(52 + (26 - 52) * normalized)  # 52 to 26
            g = int(152 + (80 - 152) * normalized)  # 152 to 80
            b = int(219 + (184 - 219) * normalized)  # 219 to 184
            colors.append(f"#{r:02x}{g:02x}{b:02x}")

    # Single bars with gradient
    for i in range(len(iteration_values)):
        ax.bar(
            i,
            avg_means[i],
            yerr=avg_stds[i],
            capsize=8,
            color=colors[i],
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
            width=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize_tick)
    ax.set_xlabel(
        "Latent Reasoning Iterations", fontsize=fontsize_label, fontweight="bold"
    )
    ax.set_ylabel("Accuracy", fontsize=fontsize_label, fontweight="bold")
    ax.set_title(
        "Accuracy vs Latent Reasoning Iterations",
        fontsize=fontsize_title,
        fontweight="bold",
        pad=16,
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
    ax.tick_params(axis="y", labelsize=fontsize_tick)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def aggregate_results_across_templates(all_results_by_template):
    """Aggregate results across all templates by averaging."""
    iteration_values = all_results_by_template[0]["addition"]["iteration_values"]

    all_addition_means = []
    all_subtraction_means = []

    for template_results in all_results_by_template:
        all_addition_means.append(template_results["addition"]["mean_accuracies"])
        all_subtraction_means.append(template_results["subtraction"]["mean_accuracies"])

    all_addition_means = np.array(all_addition_means)
    all_subtraction_means = np.array(all_subtraction_means)

    aggregated = {
        "addition": {
            "mean_accuracies": np.mean(all_addition_means, axis=0).tolist(),
            "std_errors": np.std(all_addition_means, axis=0).tolist(),
        },
        "subtraction": {
            "mean_accuracies": np.mean(all_subtraction_means, axis=0).tolist(),
            "std_errors": np.std(all_subtraction_means, axis=0).tolist(),
        },
        "iteration_values": iteration_values,
    }
    return aggregated


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

print("\n" + "=" * 80)
print("LOADING PROMPTS FROM JSON")
print("=" * 80)

# Load prompts from JSON file
all_prompts = load_prompts_from_json()
print(f"Loaded {len(all_prompts)} prompts from {PROMPTS_JSON_PATH}")

# Determine number of templates from prompts
num_templates = max(p["template_idx"] for p in all_prompts) + 1
print(f"Found {num_templates} templates")

# %%
# Evaluate all prompts for each latent iteration
iteration_values = list(range(0, 7))
for num_latent_iterations in iteration_values:
    skip_thinking = num_latent_iterations == 0
    print("\n" + "=" * 80)
    print(f"LATENT ITERATIONS: {num_latent_iterations} (skip_thinking={skip_thinking})")
    print("=" * 80)

    for sample_idx in range(NUM_SAMPLES_PER_PROMPT):
        print(f"Sample {sample_idx + 1} / {NUM_SAMPLES_PER_PROMPT}")

        for prompt_idx, prompt_data in enumerate(all_prompts):
            if prompt_idx % 50 == 0:
                print(f"  Prompt {prompt_idx} / {len(all_prompts)}")

            for operation in ["addition", "subtraction"]:
                prompt = prompt_data[operation]["prompt"]
                ground_truth = prompt_data[operation]["ground_truth"]

                inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(model.codi.device)
                attention_mask = inputs["attention_mask"].to(model.codi.device)

                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=tokenizer,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_latent_iterations=num_latent_iterations,
                    temperature=TEMPERATURE,
                    greedy=GREEDY,
                    return_latent_vectors=False,
                    remove_eos=False,
                    output_hidden_states=True,
                    output_attentions=False,
                    skip_thinking=skip_thinking,
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
                    and int(answer) == int(ground_truth)
                )

                # Store all samples (not just one)
                if f"preds_{num_latent_iterations}_{operation}" not in prompt_data:
                    prompt_data[f"preds_{num_latent_iterations}_{operation}"] = []
                    prompt_data[f"correct_{num_latent_iterations}_{operation}"] = []

                prompt_data[f"preds_{num_latent_iterations}_{operation}"].append(answer)
                prompt_data[f"correct_{num_latent_iterations}_{operation}"].append(
                    correct
                )

# Aggregate results by template and operation
all_results_by_template = []
for template_idx in range(num_templates):
    template_prompts = [p for p in all_prompts if p["template_idx"] == template_idx]

    addition_results = {
        "iteration_values": iteration_values,
        "mean_accuracies": [],
        "std_errors": [],
    }
    subtraction_results = {
        "iteration_values": iteration_values,
        "mean_accuracies": [],
        "std_errors": [],
    }

    for num_latent_iterations in iteration_values:
        # Collect all sample accuracies across all prompts
        addition_all_accs = []
        subtraction_all_accs = []

        for p in template_prompts:
            for correct in p[f"correct_{num_latent_iterations}_addition"]:
                addition_all_accs.append(1.0 if correct else 0.0)
            for correct in p[f"correct_{num_latent_iterations}_subtraction"]:
                subtraction_all_accs.append(1.0 if correct else 0.0)

        addition_all_accs = np.array(addition_all_accs)
        subtraction_all_accs = np.array(subtraction_all_accs)

        addition_results["mean_accuracies"].append(float(np.mean(addition_all_accs)))
        addition_results["std_errors"].append(
            float(np.std(addition_all_accs) / np.sqrt(len(addition_all_accs)))
        )
        subtraction_results["mean_accuracies"].append(
            float(np.mean(subtraction_all_accs))
        )
        subtraction_results["std_errors"].append(
            float(np.std(subtraction_all_accs) / np.sqrt(len(subtraction_all_accs)))
        )

    all_results_by_template.append(
        {
            "template_idx": template_idx,
            "addition": addition_results,
            "subtraction": subtraction_results,
        }
    )

# Aggregate results
aggregated_results = aggregate_results_across_templates(all_results_by_template)
# %%
# Save detailed results per template
detailed_json_path = RESULTS_DIR / "latent_iterations_accuracy_by_template.json"
with open(detailed_json_path, "w") as f:
    json.dump(
        {
            "config": {
                "checkpoint_path": CHECKPOINT_PATH,
                "model_name_or_path": MODEL_NAME_OR_PATH,
                "device": DEVICE,
                "dtype": DTYPE,
                "prompts_source": str(PROMPTS_JSON_PATH),
                "num_templates": num_templates,
                "num_samples_per_prompt": NUM_SAMPLES_PER_PROMPT,
                "temperature": TEMPERATURE,
                "greedy": GREEDY,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            "results_by_template": all_results_by_template,
        },
        f,
        indent=2,
    )

# Save aggregated results
aggregated_json_path = RESULTS_DIR / "latent_iterations_accuracy_aggregated.json"
with open(aggregated_json_path, "w") as f:
    json.dump(
        {
            "config": {
                "checkpoint_path": CHECKPOINT_PATH,
                "model_name_or_path": MODEL_NAME_OR_PATH,
                "device": DEVICE,
                "dtype": DTYPE,
                "prompts_source": str(PROMPTS_JSON_PATH),
                "num_templates": num_templates,
                "num_samples_per_prompt": NUM_SAMPLES_PER_PROMPT,
                "temperature": TEMPERATURE,
                "greedy": GREEDY,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            "aggregated_results": aggregated_results,
        },
        f,
        indent=2,
    )
# %%
# Create aggregated plot
plot_path = RESULTS_DIR / "latent_iterations_accuracy_aggregated.png"
plot_aggregated_accuracy(
    iteration_values=aggregated_results["iteration_values"],
    addition_means=aggregated_results["addition"]["mean_accuracies"],
    addition_stds=aggregated_results["addition"]["std_errors"],
    subtraction_means=aggregated_results["subtraction"]["mean_accuracies"],
    subtraction_stds=aggregated_results["subtraction"]["std_errors"],
    out_path=plot_path,
)

# %%
