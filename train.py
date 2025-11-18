# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import argparse
import os
from math import ceil
from pathlib import Path

import torch.distributed as dist
import transformers
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from transformers import Trainer

import wandb
from src.datasets import make_supervised_data_module
from src.model import CODI, DataArguments, ModelArguments, TrainingArguments

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


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = (
            batch_size * self.args.world_size * gradient_accumulation_steps
        )
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        # Call the model's forward method
        outputs = model(**inputs)
        loss = outputs["loss"]
        # "ce_loss": ce_loss_total, "mse_loss": mse_loss_total, "ref_ce_loss": ref_ce_loss
        if step % self.args.logging_steps == 0:
            self.log(
                {
                    "loss": loss,
                    "ce_loss": outputs["ce_loss"],
                    "distill_loss": outputs["distill_loss"],
                    "ref_ce_loss": outputs["ref_ce_loss"],
                }
            )
        return loss

    # def log(self, logs, start_time=None):
    #     if self.state.global_step is not None:
    #         for k, v in logs.items():
    #             super().log({k: v})


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

    model = CODI(model_args, training_args, lora_config)
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
    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()

    # to avoid the error of saving the model
    # if "llama" in model_args.model_name_or_path:
    #    trainer.model.codi.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.model.lm_head.weight.clone())
    # if "gpt2" in model_args.model_name_or_path:
    #    trainer.model.codi.transformer.wte.weight = torch.nn.Parameter(model.codi.lm_head.weight.clone())
    # if "qwen" in model_args.model_name_or_path.lower():
    #    trainer.model.codi.base_model.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.base_model.model.lm_head.weight.clone())

    # trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
