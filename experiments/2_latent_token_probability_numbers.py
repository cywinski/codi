# ABOUTME: Compute latent-vector token probabilities for arithmetic intermediates/answer.
# ABOUTME: Aggregates across prompts.json (addition+subtraction) and plots mean±std vs latent index.

# %%
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import extract_answer_number
from src.model import CODI

# %%
PROMPTS_JSON_PATH = (
    Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    / "prompts"
    / "prompts.json"
)


def ensure_tokenizer_special_tokens(tokenizer, model) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|bocot|>", "<|eocot|>"]}
    )


def load_prompts_from_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["prompts"]


def compute_steps(x: int, y: int, z: int, operation: str) -> tuple[int, int, int]:
    if operation == "addition":
        step_1 = x + y
    elif operation == "subtraction":
        step_1 = x - y
    else:
        raise ValueError(f"Unknown operation: {operation}")
    step_2 = step_1 * z
    answer = step_1 + step_2
    return int(step_1), int(step_2), int(answer)


def get_single_token_id_for_number(tokenizer, n: int) -> Optional[int]:
    text = str(int(n))
    candidates = [text, f" {text}", f"\n{text}"]

    for s in candidates:
        tid = tokenizer.convert_tokens_to_ids(s)
        if isinstance(tid, int) and tid >= 0:
            return tid

    for s in candidates:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])

    return None


@dataclass(frozen=True)
class Config:
    checkpoint_path: str = "bcywinski/codi_llama1b-answer_only"
    model_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    num_latent: int = 6
    num_latent_iterations: int = 6
    max_new_tokens: int = 128
    temperature: float = 0.1
    greedy: bool = True
    limit_prompts: Optional[int] = None
    seed: int = 0
    prompts_json_path: str = str(PROMPTS_JSON_PATH)
    results_dir: str = "results/latent_token_probability_numbers"


def plot_row_mean_std(
    x_values: list[int],
    series_by_panel: list[tuple[str, dict[str, tuple[np.ndarray, np.ndarray]]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(18, 5.8), sharey=True, constrained_layout=True
    )

    fontsize_title = 16
    fontsize_label = 15
    fontsize_tick = 13

    colors = {
        "step1": "#1f77b4",
        "step2": "#ff7f0e",
        "answer": "#2ca02c",
    }
    labels = {
        "step1": "step1",
        "step2": "step2",
        "answer": "answer",
    }

    for ax, (panel_title, series) in zip(axes, series_by_panel):
        for key, (mean, std) in series.items():
            ax.plot(
                x_values,
                mean,
                label=labels.get(key, key),
                color=colors.get(key, None),
                linewidth=3,
                markersize=10,
                marker="o",
            )
            ax.fill_between(
                x_values,
                mean - std,
                mean + std,
                color=colors.get(key, None),
                alpha=0.2,
                linewidth=0,
            )
        ax.set_title(panel_title, fontsize=fontsize_title, fontweight="bold", pad=8)
        ax.set_xticks(x_values)
        ax.tick_params(axis="both", labelsize=fontsize_tick)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25, axis="y")
        ax.set_xlabel("Latent vector index", fontsize=fontsize_label, fontweight="bold")
        ax.legend(
            loc="upper right",
            fontsize=fontsize_tick,
            frameon=True,
        )

    axes[0].set_ylabel("Token probability", fontsize=fontsize_label, fontweight="bold")

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _aggregate_mean_std(arrs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.stack(arrs, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def main(
    checkpoint_path: str = Config.checkpoint_path,
    model_name_or_path: str = Config.model_name_or_path,
    device: str = Config.device,
    dtype: str = Config.dtype,
    num_latent: int = Config.num_latent,
    num_latent_iterations: int = Config.num_latent_iterations,
    max_new_tokens: int = Config.max_new_tokens,
    temperature: float = Config.temperature,
    greedy: bool = Config.greedy,
    limit_prompts: Optional[int] = Config.limit_prompts,
    seed: int = Config.seed,
    prompts_json_path: str = Config.prompts_json_path,
    results_dir: str = Config.results_dir,
) -> None:
    load_dotenv()
    torch.manual_seed(seed)
    np.random.seed(seed)

    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        checkpoint_path=checkpoint_path,
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        num_latent=num_latent,
        num_latent_iterations=num_latent_iterations,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        greedy=greedy,
        limit_prompts=limit_prompts,
        seed=seed,
        prompts_json_path=prompts_json_path,
        results_dir=results_dir,
    )

    print("Loading model...")
    model = CODI.from_pretrained(
        checkpoint_path=checkpoint_path,
        model_name_or_path=model_name_or_path,
        lora_r=128,
        lora_alpha=32,
        num_latent=num_latent,
        use_prj=True,
        device=device,
        dtype=dtype,
        strict=False,
        checkpoint_save_path=f"./checkpoints/{checkpoint_path}",
        remove_eos=False,
        full_precision=True,
    )
    tokenizer = model.tokenizer
    ensure_tokenizer_special_tokens(tokenizer, model)

    embed_matrix = model.codi.model.model.embed_tokens.weight

    prompts = load_prompts_from_json(Path(prompts_json_path))
    if limit_prompts is not None:
        prompts = prompts[: int(limit_prompts)]

    cases: list[dict[str, Any]] = []
    for p in prompts:
        for operation in ["addition", "subtraction"]:
            x, y, z = int(p["X"]), int(p["Y"]), int(p["Z"])
            step1, step2, answer = compute_steps(x, y, z, operation)
            cases.append(
                {
                    "id": int(p["id"]),
                    "template_idx": int(p["template_idx"]),
                    "operation": operation,
                    "prompt": p[operation]["prompt"],
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "step1": step1,
                    "step2": step2,
                    "answer": answer,
                }
            )

    print(
        f"Loaded {len(prompts)} prompts -> {len(cases)} cases (addition+subtraction)."
    )

    missing_token_count = 0
    records: list[dict[str, Any]] = []

    sot_token = tokenizer.convert_tokens_to_ids("<|bocot|>")
    eot_token = tokenizer.convert_tokens_to_ids("<|eocot|>")

    for tc in tqdm(cases, desc="prompts", leave=True):
        tid_step1 = get_single_token_id_for_number(tokenizer, tc["step1"])
        tid_step2 = get_single_token_id_for_number(tokenizer, tc["step2"])
        tid_answer = get_single_token_id_for_number(tokenizer, tc["answer"])

        if tid_step1 is None or tid_step2 is None or tid_answer is None:
            missing_token_count += 1
            continue

        inputs = tokenizer(tc["prompt"], return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(model.codi.device)
        attention_mask = inputs["attention_mask"].to(model.codi.device)

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            num_latent_iterations=num_latent_iterations,
            temperature=temperature,
            greedy=greedy,
            return_latent_vectors=True,
            remove_eos=False,
            output_attentions=False,
            skip_thinking=False,
            verbalize_cot=False,
            output_hidden_states=True,
            sot_token=sot_token,
            eot_token=eot_token,
        )

        generated_text = tokenizer.decode(
            output["sequences"][0], skip_special_tokens=False
        )
        pred = extract_answer_number(generated_text)
        is_correct = (
            pred is not None and pred != float("inf") and int(pred) == int(tc["answer"])
        )

        latent_vectors = (
            torch.stack(output["latent_vectors_pre_prj"]).squeeze(1).squeeze(1)
        )  # [num_latents+1, hidden]
        logits = latent_vectors @ embed_matrix.T
        probs = logits.float().softmax(dim=-1)

        records.append(
            {
                "correct": bool(is_correct),
                "p_step1": probs[:, tid_step1].detach().cpu().numpy(),
                "p_step2": probs[:, tid_step2].detach().cpu().numpy(),
                "p_answer": probs[:, tid_answer].detach().cpu().numpy(),
            }
        )

    if not records:
        raise RuntimeError(
            "No prompts produced valid single-token IDs for all targets. "
            "Try a different tokenizer/model or reduce the numeric range."
        )

    records_all = records
    records_correct = [r for r in records if r["correct"]]
    records_incorrect = [r for r in records if not r["correct"]]

    def agg(rec_list: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if not rec_list:
            nan = np.full_like(records_all[0]["p_step1"], np.nan, dtype=np.float64)
            return {"step1": (nan, nan), "step2": (nan, nan), "answer": (nan, nan)}
        m1, s1 = _aggregate_mean_std([r["p_step1"] for r in rec_list])
        m2, s2 = _aggregate_mean_std([r["p_step2"] for r in rec_list])
        ma, sa = _aggregate_mean_std([r["p_answer"] for r in rec_list])
        return {"step1": (m1, s1), "step2": (m2, s2), "answer": (ma, sa)}

    series_all = agg(records_all)
    series_correct = agg(records_correct)
    series_incorrect = agg(records_incorrect)

    latent_indices = list(range(0, int(records_all[0]["p_step1"].shape[0])))

    out_png = results_dir_path / "latent_token_probability_numbers.png"
    plot_row_mean_std(
        x_values=latent_indices,
        series_by_panel=[
            ("All prompts", series_all),
            (f"Correct answers (n={len(records_correct)})", series_correct),
            (f"Incorrect answers (n={len(records_incorrect)})", series_incorrect),
        ],
        out_path=out_png,
    )

    out_json = results_dir_path / "latent_token_probability_numbers.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "num_cases_total": len(cases),
                "num_cases_used": int(len(records_all)),
                "num_cases_correct": int(len(records_correct)),
                "num_cases_incorrect": int(len(records_incorrect)),
                "num_cases_skipped_missing_single_token": int(missing_token_count),
                "latent_indices": latent_indices,
                "panels": {
                    "all": {
                        "mean": {
                            "step1": series_all["step1"][0].tolist(),
                            "step2": series_all["step2"][0].tolist(),
                            "answer": series_all["answer"][0].tolist(),
                        },
                        "std": {
                            "step1": series_all["step1"][1].tolist(),
                            "step2": series_all["step2"][1].tolist(),
                            "answer": series_all["answer"][1].tolist(),
                        },
                    },
                    "correct": {
                        "mean": {
                            "step1": series_correct["step1"][0].tolist(),
                            "step2": series_correct["step2"][0].tolist(),
                            "answer": series_correct["answer"][0].tolist(),
                        },
                        "std": {
                            "step1": series_correct["step1"][1].tolist(),
                            "step2": series_correct["step2"][1].tolist(),
                            "answer": series_correct["answer"][1].tolist(),
                        },
                    },
                    "incorrect": {
                        "mean": {
                            "step1": series_incorrect["step1"][0].tolist(),
                            "step2": series_incorrect["step2"][0].tolist(),
                            "answer": series_incorrect["answer"][0].tolist(),
                        },
                        "std": {
                            "step1": series_incorrect["step1"][1].tolist(),
                            "step2": series_incorrect["step2"][1].tolist(),
                            "answer": series_incorrect["answer"][1].tolist(),
                        },
                    },
                },
            },
            f,
            indent=2,
        )

    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")
    if missing_token_count:
        print(
            f"Skipped {missing_token_count}/{len(cases)} cases due to missing single-token IDs."
        )


if __name__ == "__main__":
    fire.Fire(main)
