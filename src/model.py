import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    full_precision: bool = field(
        default=True, metadata={"help": "whether use int4 for the base model"}
    )
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={
            "help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."
        },
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."
        },
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    ckpt_dir: Optional[str] = field(
        default=None, metadata={"help": "checkpoint dir for inference."}
    )


@dataclass
class DataArguments:
    data_names: list[str] = field(
        default_factory=list,
        metadata={"help": "List of dataset names to concatenate for training."},
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )
    batch_size: int = field(default=1, metadata={"help": "batch size during inference"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    per_device_train_batch_size: int = field(
        default=1,
    )
    per_device_eval_batch_size: int = field(
        default=1,
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    icot_train_path: str = field(
        default="/users/k24020023/efficient_cot/icae/code/coconut/icot_gsm8k/train.txt",
        metadata={"help": "The training data path"},
    )
    num_latent: int = field(
        default=5, metadata={"help": "The number of latent for training or inference."}
    )
    use_lora: bool = field(default=True, metadata={"help": "Use lora or not."})
    greedy: bool = field(
        default=False, metadata={"help": "Greedy decoding during inference."}
    )
    exp_mode: bool = field(
        default=False, metadata={"help": "Use partial number of data. for debugging."}
    )
    exp_data_num: int = field(
        default=10000, metadata={"help": "The number of data used in exp mode"}
    )
    use_prj: bool = field(
        default=False,
        metadata={"help": "Use a prj module after the llm for latent generation."},
    )
    prj_dim: int = field(
        default=2048, metadata={"help": "The hidden dim of the projection module."}
    )
    prj_dropout: float = field(
        default=0.0, metadata={"help": "Dropout ratio of the projection module."}
    )
    prj_no_ln: bool = field(
        default=False,
        metadata={"help": "Remove the Layer Norm layer for the projection module."},
    )
    distill_loss_div_std: bool = field(
        default=False,
        metadata={"help": "Divide the distillation loss by a std for normallisation."},
    )
    distill_loss_type: str = field(
        default="smooth_l1",
        metadata={"help": "Specify the distillation loss. Use smoothL1 by default."},
    )
    distill_loss_factor: float = field(
        default=1.0, metadata={"help": "A multiplier of the distillation loss."}
    )
    ref_loss_factor: float = field(
        default=1.0, metadata={"help": "A multiplier of the distillation loss."}
    )
    inf_latent_iterations: int = field(default=1, metadata={"help": ""})
    inf_num_iterations: int = field(
        default=5, metadata={"help": "Run multiple times during inference"}
    )
    remove_eos: bool = field(
        default=False, metadata={"help": "Do not add <eos> as a delimiter to split QA."}
    )
    print_ref_model_stats: bool = field(
        default=False, metadata={"help": "Print some stats for the teacher task."}
    )
    include_last_cot: bool = field(
        default=False,
        metadata={"help": "Include the last CoT step in the training data."},
    )
    fix_attn_mask: bool = field(
        default=False, metadata={"help": "Correct a bug about attention mask."}
    )
    log_full: bool = field(default=False, metadata={"help": "Log all losses."})
    print_loss: bool = field(default=True)
    max_token_num: int = field(
        default=1000, metadata={"help": "Limit the longest data to avoid OOM."}
    )


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class CODI(torch.nn.Module):
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
        lora_r: int = 128,
        lora_alpha: int = 16,
        num_latent: int = 5,
        use_prj: bool = False,
        prj_dim: int = 2048,
        prj_dropout: float = 0.0,
        prj_no_ln: bool = False,
        remove_eos: bool = False,
        model_max_length: int = 28000,
        attn_implementation: str = "flash_attention_2",
        full_precision: bool = True,
        device: str = "cuda",
        dtype: str = "bfloat16",
        token: Optional[str] = None,
        strict: bool = False,
        checkpoint_save_path: Optional[str] = None,
    ):
        """
        Load a pretrained CODI model from a checkpoint.

        Args:
            checkpoint_path: Fine-tuned checkpoint. Can be either:
                - Local path to directory containing model.safetensors or pytorch_model.bin
                - HuggingFace model ID where the checkpoint is stored
            model_name_or_path: Base model (e.g., "mistralai/Mistral-7B-Instruct-v0.2" or "meta-llama/Llama-2-7b").
                Can be either:
                - A HuggingFace model ID (uses default HF cache ~/.cache/huggingface/)
                - A local path to a saved base model
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            num_latent: Number of latent reasoning iterations during training
            use_prj: Whether to use projection module
            prj_dim: Projection module hidden dimension
            prj_dropout: Projection module dropout
            prj_no_ln: Remove LayerNorm from projection module
            remove_eos: Do not add <eos> as delimiter
            model_max_length: Maximum sequence length
            attn_implementation: Attention implementation ("flash_attention_2", "eager", etc.)
            full_precision: Use full precision (bf16/fp16) vs quantized (4-bit)
            device: Device to load model on ("cuda" or "cpu")
            dtype: Data type ("bfloat16" or "float16")
            token: HuggingFace token for private models
            strict: Whether to strictly enforce state_dict loading
            checkpoint_save_path: Optional path to save the checkpoint if downloading from HF.
                If None and checkpoint_path is a HF ID, will save to "./checkpoints/{checkpoint_name}"

        Returns:
            CODI model with loaded weights
        """
        # Step 1: Handle base model (model_name_or_path)
        # Base model uses standard HuggingFace caching - no custom handling needed
        # It will be downloaded to ~/.cache/huggingface/ automatically if it's a HF ID
        print(f"Base model: {model_name_or_path}")
        if not os.path.exists(model_name_or_path):
            print(
                "  → Will be downloaded from HuggingFace (cached in ~/.cache/huggingface/)"
            )
        else:
            print("  → Loading from local path")

        # Create model arguments
        model_args = ModelArguments(
            model_name_or_path=model_name_or_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_init=True,
            full_precision=full_precision,
            # attn_implementation=attn_implementation,
            train=False,  # Inference mode
            token=token,
        )

        # Create training arguments
        bf16 = dtype == "bfloat16"
        training_args = TrainingArguments(
            output_dir="./output",  # Dummy, not used for inference
            model_max_length=model_max_length,
            num_latent=num_latent,
            use_lora=True,
            use_prj=use_prj,
            prj_dim=prj_dim,
            prj_dropout=prj_dropout,
            prj_no_ln=prj_no_ln,
            remove_eos=remove_eos,
            bf16=bf16,
        )

        # Determine target modules based on model architecture
        if any(
            name in model_name_or_path.lower()
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
        elif any(name in model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            raise ValueError(
                f"Unsupported model architecture: {model_name_or_path}. "
                f"Supported: LLAMA, Mistral, Falcon, Qwen, Phi, GPT-2"
            )

        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )

        # Initialize model
        model = cls(model_args, training_args, lora_config)

        # Step 2: Handle checkpoint (checkpoint_path)
        # Check if checkpoint_path is a HuggingFace ID or local path
        is_hf_checkpoint = not os.path.exists(checkpoint_path)

        checkpoint_file = None
        if is_hf_checkpoint:
            # Download checkpoint from HuggingFace to a specific local path
            print(f"Checkpoint: {checkpoint_path} (HuggingFace ID)")

            if checkpoint_save_path is None:
                # Create default save path
                checkpoint_name_clean = checkpoint_path.replace("/", "_")
                checkpoint_save_path = os.path.join(
                    "./checkpoints", checkpoint_name_clean
                )

            print(f"  → Downloading to: {checkpoint_save_path}")

            from huggingface_hub import snapshot_download

            # Download the entire checkpoint directory to the specified path
            local_checkpoint_path = snapshot_download(
                repo_id=checkpoint_path,
                token=token,
                allow_patterns=["*.safetensors", "*.bin", "*.json"],
                local_dir=checkpoint_save_path,
                local_dir_use_symlinks=False,  # Download actual files, not symlinks
            )
            print(f"  → Checkpoint saved to: {local_checkpoint_path}")
            checkpoint_path = local_checkpoint_path
        else:
            print(f"Checkpoint: {checkpoint_path} (local path)")

        # Load checkpoint from local path
        if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
            checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
            state_dict = load_file(checkpoint_file)
        elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
            checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
            state_dict = torch.load(checkpoint_file, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_path}. "
                f"Looking for 'model.safetensors' or 'pytorch_model.bin'"
            )

        print(f"Loading checkpoint from {checkpoint_file}")
        model.load_state_dict(state_dict, strict=strict)
        model.codi.tie_weights()

        # Move to device and set dtype
        if device == "cuda" and torch.cuda.is_available():
            model = model.to(device)
            if dtype == "bfloat16":
                model = model.to(torch.bfloat16)
            else:
                model = model.to(torch.float16)
        elif device == "cpu":
            model = model.to("cpu")

        model.eval()
        print(f"Model loaded successfully from {checkpoint_path}")

        return model

    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        model_wrapper_class = AutoModelForCausalLM
        if model_args.full_precision:
            self.codi = model_wrapper_class.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if training_args.bf16 is False else torch.bfloat16
                ),
                resume_download=True,
            )
        else:
            self.codi = model_wrapper_class.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if training_args.bf16 is False else torch.bfloat16
                ),
                resume_download=True,
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                ),
            )

        ori_vocab_size = self.codi.config.vocab_size
        self.training = self.model_args.train

        # special tokens to enclose the latent embeddings
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1
        self.eot_id = ori_vocab_size + 2

        self.codi.resize_token_embeddings(
            ori_vocab_size + 3
        )  # dummy values for mem tokens

        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # LoRA
        if training_args.use_lora:
            self.codi = get_peft_model(self.codi, lora_config)

        # Projection Layer
        self.use_prj = training_args.use_prj
        self.prj_no_ln = training_args.prj_no_ln
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not self.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))

        # Losses
        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor

        # Cross Entropy Loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # Distillation Loss
        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError

        # general
        self.fix_attn_mask = training_args.fix_attn_mask

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token_id = self.pad_token_id

        if self.training:
            self.init()

    @property
    def config(self):
        """Expose the underlying model's config for compatibility with transformers Trainer."""
        # Handle PEFT-wrapped models
        if hasattr(self.codi, "get_base_model"):
            return self.codi.get_base_model().config
        return self.codi.config

    def get_embd(self, model, model_name):
        try:
            if "pythia" in model_name:
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" in model_name:
                try:
                    return model.get_base_model().transformer.wte
                except Exception:  # no lora
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception:  # no lora
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            raise NotImplementedError

    def init(self):
        print_trainable_parameters(self)
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")

    def forward(
        self,
        encoder_input_ids: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        ref_input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        model_answer_position: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        ref_labels: torch.LongTensor = None,
        step: int = None,
        step_ratio: float = None,
    ):
        if not self.fix_attn_mask:
            ref_attention_mask = None

        # Encode the question
        past_key_values = None
        outputs = self.codi(
            input_ids=encoder_input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=encoder_attention_mask,
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(
            1
        )  # as the next input
        if self.use_prj:
            latent_embd = self.prj(latent_embd)

        len_pred_loss = 0
        dynamic_mask = None
        if self.fix_attn_mask:
            dynamic_mask = torch.ones(
                (encoder_attention_mask.size(0), self.num_latent),
                device=ref_labels.device,
            )

        # Iterate over the latent embeddings
        distill_loss_total = 0
        ce_loss_total = 0

        with torch.no_grad():
            ref_outputs = self.codi(
                input_ids=ref_input_ids,
                output_hidden_states=True,
                attention_mask=ref_attention_mask,
            )
        ref_outputs_with_grad = self.codi(
            input_ids=ref_input_ids,
            output_hidden_states=True,
            attention_mask=ref_attention_mask,
        )

        # Formatting for deprecated exps
        ref_outputs_list = [ref_outputs]
        ref_input_ids = [ref_input_ids]

        # Process the position tensor
        # Normalise the position definition
        if (
            "llama" in self.model_name.lower() or "qwen" in self.model_name.lower()
        ):  # there is one more token standing for " "
            model_answer_position = model_answer_position + 1
            ref_answer_position = ref_answer_position + 1

        # For DEBUG: Print the probability of the teacher task to predict the correct answer
        if self.training_args.print_ref_model_stats:
            for i, (ref_inputs, ref_outputs) in enumerate(
                zip(ref_input_ids, ref_outputs_list)
            ):
                # evalutae the reference model
                if len(ref_outputs_list) > 1:
                    pos = ref_answer_position[i]
                else:
                    pos = ref_answer_position
                ref_probs = torch.nn.functional.softmax(ref_outputs.logits, dim=-1)
                input_positions = (
                    (pos - 1)
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .expand(-1, -1, ref_probs.size(2))
                )
                ref_probs_at_positions = ref_probs.gather(1, input_positions)
                probe_positions_positions = pos.unsqueeze(1)
                probe_positions = ref_inputs.gather(
                    1, probe_positions_positions
                ).unsqueeze(1)
                ref_probs_of_target = ref_probs_at_positions.gather(2, probe_positions)
                print(
                    f"stage{i}: mean of the prob of the target token: {ref_probs_of_target.mean()}"
                )

        # the model answer position is the position of the eot token to predict the first token of the response
        model_answer_position = model_answer_position - 1
        ref_answer_position = ref_answer_position - 1

        num_latent = self.num_latent
        if self.num_latent != 0:
            for i in range(num_latent):
                # Implicit CoT generation
                outputs = self.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

                # Calculate the distillation loss
                if i == num_latent - 1:  # the last latent embedding
                    # Decode the final answer in natural language
                    embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)

                    if dynamic_mask is not None:  # Prevent attending the paddings
                        decoder_mask = torch.ones(
                            (embds.size(0), embds.size(1)), dtype=torch.bool
                        ).to(dynamic_mask)
                        dynamic_mask = torch.cat(
                            (encoder_attention_mask, dynamic_mask, decoder_mask), dim=1
                        )
                        dynamic_mask = dynamic_mask.bool()
                    # Student task's output
                    outputs = self.codi(
                        inputs_embeds=embds,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values,
                        attention_mask=dynamic_mask,
                    )
                    # Teacher task's output
                    ref_outputs = ref_outputs_list[0]

                    distill_loss = 0
                    # Calculate distillation loss between the teacher's logits and the student's logits for every layer
                    for j, (out, ref_out) in enumerate(
                        zip(outputs.hidden_states, ref_outputs.hidden_states)
                    ):
                        ref_selected = ref_out.gather(
                            1,
                            ref_answer_position.unsqueeze(-1)
                            .unsqueeze(-1)
                            .expand(-1, -1, ref_out.size(-1)),
                        )
                        out_selected = out.gather(
                            1,
                            model_answer_position.unsqueeze(-1)
                            .unsqueeze(-1)
                            .expand(-1, -1, out.size(-1)),
                        )

                        distill_loss_tmp = self.distill_loss_fct(
                            out_selected, ref_selected.detach()
                        )

                        if self.distill_loss_div_std:
                            if self.distill_loss_type == "l2":
                                distill_loss_tmp /= ref_selected.std()
                            distill_loss_tmp /= ref_selected.std()
                        distill_loss += distill_loss_tmp

                    distill_loss /= len(outputs.hidden_states)

                    if self.print_loss:
                        print(f"latent{i}: distill_loss={distill_loss}")

                    distill_loss_total += distill_loss

                    # Calculate the CE loss for the student task
                    if i == num_latent - 1:
                        logits = outputs.logits
                        effective_logits = logits[:, :-1, :]
                        effective_logits = effective_logits.reshape(-1, logits.size(-1))
                        target_ids = labels[:, 1:].reshape(-1)
                        ce_loss = self.loss_fct(effective_logits, target_ids)
                        ce_loss_total += ce_loss

        # Calculate the CE loss for the teacher task
        ref_ce_loss = 0
        ref_logits = ref_outputs_with_grad.logits
        effective_ref_logits = ref_logits[:, :-1, :]
        effective_ref_logits = effective_ref_logits.reshape(-1, ref_logits.size(-1))
        ref_target_ids = ref_labels[:, 1:].reshape(-1)
        ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids)
        ref_ce_loss *= self.ref_loss_factor

        # Weigh the distillation loss
        distill_loss *= self.distill_loss_factor
        distill_loss_total *= self.distill_loss_factor

        if self.print_loss:
            print(
                f"loss={ce_loss + distill_loss}, ce_loss={ce_loss}, distill_loss={distill_loss}, ce_loss_total={ce_loss_total}, distill_loss_total={distill_loss_total}, ref_ce_loss={ref_ce_loss}"
            )

        loss = ce_loss_total + distill_loss_total + ref_ce_loss

        if ce_loss_total != 0:
            ce_loss_total = ce_loss_total.detach().item()
        if distill_loss_total != 0:
            distill_loss_total = distill_loss_total.detach().item()
        if ref_ce_loss != 0:
            ref_ce_loss = ref_ce_loss.detach().item()

        return {
            "loss": loss,
            "logits": logits,
            "ce_loss": ce_loss_total,
            "distill_loss": distill_loss_total,
            "ref_ce_loss": ref_ce_loss,
        }

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        max_new_tokens: int = 256,
        num_latent_iterations: int = 1,
        temperature: float = 0.1,
        top_k: int = 40,
        top_p: float = 0.95,
        greedy: bool = False,
        return_latent_vectors: bool = True,
        remove_eos: bool = False,
    ):
        """
        Generate text with latent reasoning steps.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            tokenizer: Tokenizer for decoding. If None, uses self.tokenizer
            max_new_tokens: Maximum number of tokens to generate
            num_latent_iterations: Number of latent reasoning iterations
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            greedy: Whether to use greedy decoding
            return_latent_vectors: Whether to return latent reasoning vectors
            remove_eos: Whether to remove EOS token when adding BOT/EOT tokens

        Returns:
            dict with keys:
                - 'sequences': Generated token IDs of shape (batch_size, generated_length)
                - 'latent_vectors': List of latent reasoning vectors if return_latent_vectors=True
                  Each element is a tensor of shape (batch_size, 1, hidden_dim)
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        device = input_ids.device
        batch_size = input_ids.size(0)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Add BOT token to input
        if remove_eos:
            bot_tensor = torch.tensor(
                [self.bot_id], dtype=torch.long, device=device
            ).expand(batch_size, 1)
        else:
            bot_tensor = torch.tensor(
                [tokenizer.eos_token_id, self.bot_id], dtype=torch.long, device=device
            ).expand(batch_size, 2)

        input_ids = torch.cat((input_ids, bot_tensor), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(bot_tensor, device=device)), dim=1
        )
        print(tokenizer.convert_ids_to_tokens(input_ids[0]))

        latent_vectors = []

        with torch.no_grad():
            # Encode the input
            past_key_values = None
            outputs = self.codi(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.use_prj:
                latent_embd = self.prj(latent_embd)

            if return_latent_vectors:
                latent_vectors.append(latent_embd.clone())

            # Latent reasoning iterations
            for i in range(num_latent_iterations):
                outputs = self.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

                if return_latent_vectors:
                    latent_vectors.append(latent_embd.clone())

            # Add EOT token embeddings
            if remove_eos:
                eot_emb = (
                    self.get_embd(self.codi, self.model_name)(
                        torch.tensor([self.eot_id], dtype=torch.long, device=device)
                    )
                    .unsqueeze(0)
                    .to(device)
                )
            else:
                eot_emb = (
                    self.get_embd(self.codi, self.model_name)(
                        torch.tensor(
                            [self.eot_id, tokenizer.eos_token_id],
                            dtype=torch.long,
                            device=device,
                        )
                    )
                    .unsqueeze(0)
                    .to(device)
                )

            eot_emb = eot_emb.expand(batch_size, -1, -1)
            output_emb = eot_emb

            # Generate tokens
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            generated_tokens = [[] for _ in range(batch_size)]

            for step in range(max_new_tokens):
                out = self.codi(
                    inputs_embeds=output_emb,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, : self.codi.config.vocab_size - 1]

                # Sampling
                if greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= temperature

                    # Top-k filtering
                    if top_k > 1:
                        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    # Top-p filtering
                    if top_p < 1.0:
                        sorted_logit, sorted_indices = torch.sort(
                            logits, descending=True, dim=-1
                        )
                        cumulative_probs = torch.cumsum(
                            F.softmax(sorted_logit, dim=-1), dim=-1
                        )

                        sorted_indices_to_remove = cumulative_probs > top_p
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(
                                1, dims=-1
                            )
                            sorted_indices_to_remove[:, 0] = False

                        for b in range(logits.size(0)):
                            logits[
                                b, sorted_indices[b, sorted_indices_to_remove[b]]
                            ] = -float("inf")

                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # Ensure next_token_ids has at least 1 dimension for indexing
                if next_token_ids.dim() == 0:
                    next_token_ids = next_token_ids.unsqueeze(0)

                # Track generated tokens
                for b in range(batch_size):
                    if not finished[b]:
                        generated_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                # Break if all sequences finished
                if finished.all():
                    break

                # Get embeddings for next iteration
                output_emb = (
                    self.get_embd(self.codi, self.model_name)(next_token_ids)
                    .unsqueeze(1)
                    .to(device)
                )

        # Convert generated tokens to tensor
        max_len = max(len(seq) for seq in generated_tokens)
        sequences = torch.full(
            (batch_size, max_len),
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
            dtype=torch.long,
            device=device,
        )

        for b, tokens in enumerate(generated_tokens):
            sequences[b, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        result = {"sequences": sequences}

        if return_latent_vectors:
            result["latent_vectors"] = latent_vectors

        return result
