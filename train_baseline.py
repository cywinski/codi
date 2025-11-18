# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

import wandb
from datasets import Dataset
from src.datasets import make_supervised_data_module
from src.model import DataArguments, ModelArguments, TrainingArguments

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Also try loading from current directory
    load_dotenv()


def is_main_process():
    """Check if this is the main process (rank 0)"""
    # Check if we're in distributed mode
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    # Check environment variables set by torchrun
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) == 0

    # Single process mode
    return True


def create_sft_dataset(train_dataset, model):
    """Create a HuggingFace Dataset for SFT training."""

    def _map_tokens_to_sft(item):
        encoder_input_ids = item["encoder_input_ids"]
        decoder_input_ids = item["decoder_input_ids"]
        encoder_len = len(encoder_input_ids)
        decoder_len = len(decoder_input_ids)
        input_ids = (
            encoder_input_ids.tolist()
            + item["ref_input_ids"][encoder_len - 1 : -decoder_len + 1].tolist()
            + decoder_input_ids.tolist()
        )
        assistant_masks = [0] * encoder_len + [1] * (len(input_ids) - encoder_len)
        assert len(assistant_masks) == len(input_ids)
        return {
            "input_ids": input_ids,
            "assistant_masks": assistant_masks,
        }

    # Transform all items
    transformed_data = [_map_tokens_to_sft(item) for item in train_dataset]

    # Create HuggingFace Dataset from list of dictionaries
    return Dataset.from_list(transformed_data)


def train():
    parser = argparse.ArgumentParser(description="Train CODI model")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = (
        "false"  # Disable tokenizers parallelism warning
    )
    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create dataclass instances from config
    model_config = config.get("model_args", {})
    # Use environment variable for token if not provided in config
    if model_config.get("token") is None:
        model_config["token"] = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    model_args = ModelArguments(**model_config)
    data_args = DataArguments(**config.get("data_args", {}))

    # TrainingArguments needs special handling since it inherits from transformers.TrainingArguments
    training_config = config.get("training_args", {})
    # Convert learning_rate to float if it's a string (YAML may parse scientific notation as string)
    if "learning_rate" in training_config and isinstance(
        training_config["learning_rate"], str
    ):
        training_config["learning_rate"] = float(training_config["learning_rate"])

    # Add hub-related parameters if hub config exists
    hub_config = config.get("hub", {})
    if hub_config:
        wandb_config = config.get("wandb", {})
        env_vars = {
            "hf_token": os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or model_config.get("token")
        }

        training_config["push_to_hub"] = hub_config.get("push_to_hub", False)
        if training_config["push_to_hub"]:
            training_config["hub_model_id"] = (
                f"{hub_config.get('account', '')}/{wandb_config.get('run_name', training_config.get('expt_name', 'default'))}"
            )
            training_config["hub_token"] = env_vars["hf_token"]
            training_config["hub_private_repo"] = hub_config.get("private_repo", False)

    training_args = TrainingArguments(**training_config)

    ##########################
    #       Peft Model       #
    ##########################
    lora_config = None
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(
            name in model_args.model_name_or_path.lower()
            for name in ["llama", "mistral", "falcon", "qwen"]
        ):
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            raise ValueError(
                f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}."
            )

        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16,
    )
    ori_vocab_size = model.config.vocab_size

    # special tokens to enclose the latent embeddings
    model.pad_token_id = ori_vocab_size
    model.bot_id = ori_vocab_size + 1
    model.eot_id = ori_vocab_size + 2

    # Resize embeddings (only for full precision models, quantized models are already on device)
    if model_args.full_precision:
        model.resize_token_embeddings(
            ori_vocab_size + 3, mean_resizing=False
        )  # dummy values for mem tokens
        # Move to GPU after resizing embeddings
        if torch.cuda.is_available():
            model = model.to("cuda")
    else:
        # For quantized models, resize is handled differently
        model.resize_token_embeddings(ori_vocab_size + 3, mean_resizing=False)

    model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:  # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split("/")[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    # Initialize wandb if configured
    wandb_config = config.get("wandb", {})
    if wandb_config and training_args.report_to and "wandb" in training_args.report_to:
        if is_main_process():
            wandb.init(
                project=wandb_config.get("project", "codi"),
                name=wandb_config.get("run_name", training_args.expt_name),
                config={
                    "model_args": model_config,
                    "data_args": config.get("data_args", {}),
                    "training_args": training_config,
                },
                settings=wandb.Settings(
                    start_method="thread"
                ),  # Use thread-based initialization
            )

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model=model,
        training_args=training_args,
    )
    train_dataset = data_module["train_dataset"]
    sft_train_dataset = create_sft_dataset(train_dataset, model)
    sft_config = SFTConfig(
        output_dir=training_args.output_dir,
        logging_dir=training_args.logging_dir,
        logging_steps=training_args.logging_steps,
        seed=training_args.seed,
        max_length=training_args.model_max_length,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        bf16=training_args.bf16,
        save_strategy=training_args.save_strategy,
        save_safetensors=training_args.save_safetensors,
        weight_decay=training_args.weight_decay,
        warmup_ratio=training_args.warmup_ratio,
        lr_scheduler_type=training_args.lr_scheduler_type,
        report_to=training_args.report_to,
        max_steps=training_args.max_steps,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        max_grad_norm=training_args.max_grad_norm,
        ddp_find_unused_parameters=training_args.ddp_find_unused_parameters,
        dataloader_num_workers=training_args.dataloader_num_workers,
        push_to_hub=hub_config.get("push_to_hub", False),
        hub_model_id=f"{hub_config.get('account', '')}/{wandb_config.get('run_name', training_args.expt_name)}",
        hub_token=env_vars["hf_token"],
        hub_private_repo=hub_config.get("private_repo", False),
        run_name=wandb_config.get("run_name", training_args.expt_name),
        completion_only_loss=True,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_train_dataset,
        args=sft_config,
    )
    trainer.train()


if __name__ == "__main__":
    train()
