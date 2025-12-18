# CODI interpretability

* Our AI Alignment blog post
* Original paper [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074)


## Setup

```
uv venv
uv pip install -r requirements.txt
```

Set required environment variables in a `.env` file.

## Train a CODI model

```bash
python train.py configs/llama1b_gsm8k-aug-nl.yaml
```

This repository supports also a distributed training with torchrun.
```bash
torchrun --nproc_per_node=4 train.py configs/llama1b_gsm8k-aug-nl.yaml
```

## Evaluate a CODI model

```bash
bash scripts/test_llama1b.sh
```

## Reproduce the experiments in the post

First prepare the prompts as JSON.
```bash
python prepare_prompts_json.py
```

Then execute script experiments. Experiments are sorted according to the order in the blog post.

```bash
python experiments/1_latent_iterations_accuracy.py
python experiments/2_latent_token_probability_numbers.py
python experiments/3_logit_lens_latents.py
python experiments/4_mean_latent_patching_same_vs_diff.py --limit=10
python experiments/5_mean_ablation_combined_templates.py
python experiments/6_operation_probe_latent_vectors.py
```
