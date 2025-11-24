import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


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
    max_samples: int = field(
        default=None, metadata={"help": "maximum number of samples to use"}
    )


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
    sft_loss_factor: float = field(
        default=1.0, metadata={"help": "A multiplier of the SFT loss."}
    )
    ce_loss_factor: float = field(
        default=1.0, metadata={"help": "A multiplier of the CE loss."}
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
    answer_only: bool = field(
        default=False, metadata={"help": "Only use the answer token for training."}
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

        # Create model arguments
        model_args = ModelArguments(
            model_name_or_path=model_name_or_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_init=True,
            full_precision=full_precision,
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

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
        model = cls(model_args, training_args, lora_config, tokenizer)

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

            # Check if checkpoint already exists locally
            model_file_exists = os.path.exists(
                os.path.join(checkpoint_save_path, "model.safetensors")
            ) or os.path.exists(os.path.join(checkpoint_save_path, "pytorch_model.bin"))

            if model_file_exists:
                print(f"  → Checkpoint already exists at: {checkpoint_save_path}")
                print("  → Skipping download")
                checkpoint_path = checkpoint_save_path
            else:
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

    def __init__(self, model_args, training_args, lora_config, tokenizer):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        target_dtype = torch.float16 if training_args.bf16 is False else torch.bfloat16
        self.model_name = model_args.model_name_or_path
        model_wrapper_class = AutoModelForCausalLM
        if model_args.full_precision:
            self.codi = model_wrapper_class.from_pretrained(
                self.model_name,
                torch_dtype=target_dtype,
            )
        else:
            self.codi = model_wrapper_class.from_pretrained(
                self.model_name,
                torch_dtype=target_dtype,
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                ),
            )

        # ori_vocab_size = self.codi.config.vocab_size
        self.training = self.model_args.train
        self.tokenizer = tokenizer

        # special tokens to enclose the latent embeddings
        self.codi.resize_token_embeddings(len(tokenizer))
        # self.pad_token_id = ori_vocab_size
        # self.bot_id = ori_vocab_size + 1
        # self.eot_id = ori_vocab_size + 2

        # self.codi.resize_token_embeddings(
        #     ori_vocab_size + 3
        # )  # dummy values for mem tokens

        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent

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
            self.prj = self.prj.to(dtype=target_dtype)

        # Losses
        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor
        self.sft_loss_factor = training_args.sft_loss_factor
        self.ce_loss_factor = training_args.ce_loss_factor
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

        # if self.tokenizer.pad_token_id is None:
        #     self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        #     self.tokenizer.pad_token_id = self.pad_token_id

        self.to(dtype=target_dtype)
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
        encoder_input_ids_ans: torch.LongTensor = None,
        encoder_input_ids_lcot: torch.LongTensor = None,
        # encoder_input_ids_vcot: torch.LongTensor = None,
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
        encoder_input_ids_labels: Optional[torch.LongTensor] = None,
    ):
        if not self.fix_attn_mask:
            ref_attention_mask = None

        direct_ans_attention_mask = (
            encoder_input_ids_ans != self.tokenizer.pad_token_id
        ).to(encoder_input_ids_ans.dtype)
        ans_outputs = self.codi(
            input_ids=encoder_input_ids_ans,
            attention_mask=direct_ans_attention_mask,
            labels=encoder_input_ids_labels.long(),
        )
        ans_ce_loss = ans_outputs.loss
        ans_ce_loss *= self.sft_loss_factor

        # Encode the question for the student
        past_key_values = None
        outputs = self.codi(
            input_ids=encoder_input_ids_lcot,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=encoder_attention_mask,
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if self.use_prj:
            latent_embd = self.prj(latent_embd)
            latent_embd = latent_embd.to(
                dtype=self.codi.dtype
            )  # FIX: layer norm casts to fp32

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

        ref_outputs_list = [ref_outputs]
        ref_input_ids = [ref_input_ids]

        if self.training_args.print_ref_model_stats:
            for i, (ref_inputs, ref_outputs) in enumerate(
                zip(ref_input_ids, ref_outputs_list)
            ):
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
                    latent_embd = latent_embd.to(dtype=self.codi.dtype)

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

        ce_loss_total *= self.ce_loss_factor
        if self.print_loss:
            print(
                f"loss={ce_loss + distill_loss}, ce_loss={ce_loss}, distill_loss={distill_loss}, ce_loss_total={ce_loss_total}, distill_loss_total={distill_loss_total}, ref_ce_loss={ref_ce_loss}, sft_loss={ans_ce_loss}"
            )
        # Add the new ans ce loss to the overall loss
        loss = ce_loss_total + distill_loss_total + ref_ce_loss + ans_ce_loss

        # if ce_loss_total != 0:
        ce_loss_total = ce_loss_total.detach().item()
        distill_loss_total = distill_loss_total.detach().item()
        ref_ce_loss = ref_ce_loss.detach().item()
        ans_ce_loss = ans_ce_loss.detach().item()

        return {
            "loss": loss,
            "logits": logits,
            "ce_loss": ce_loss_total,
            "distill_loss": distill_loss_total,
            "ref_ce_loss": ref_ce_loss,
            "ans_ce_loss": ans_ce_loss,
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
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        skip_thinking: bool = False,
        sot_token: int = None,
        verbalize_cot: bool = False,
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
            output_attentions: Whether to return attention weights for each generation step
            output_hidden_states: Whether to return hidden states computed on the prompt until thinking
            skip_thinking: Whether to skip the thinking/latent reasoning phase
            sot_token: Whether to use the sot token for the generation
        Returns:
            dict with keys:
                - 'sequences': Generated token IDs of shape (batch_size, generated_length)
                - 'latent_vectors': List of latent reasoning vectors if return_latent_vectors=True
                  Each element is a tensor of shape (batch_size, 1, hidden_dim)
                - 'attentions': List of attention tensors for each generation step if output_attentions=True
                  Each element is a tuple of tensors, one per layer
                - 'hidden_states': Hidden states computed on the prompt until thinking if output_hidden_states=True
                  A tuple of tensors, one per layer, each of shape (batch_size, seq_len, hidden_dim)
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        device = input_ids.device
        batch_size = input_ids.size(0)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        past_key_values = None
        prompt_hidden_states = None

        if verbalize_cot:
            # just use generate from transformers
            print(tokenizer.convert_ids_to_tokens(input_ids[0]))
            outputs = self.codi.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Calculate the number of prompt tokens for each item in the batch
            prompt_lengths = [input_ids[i].size(0) for i in range(input_ids.size(0))]
            # Remove the prompt tokens from each generated output
            sequences = []
            for i, prompt_len in enumerate(prompt_lengths):
                sequences.append(outputs[i, prompt_len:])
            # Pad to the maximum generation length in batch if needed
            max_gen_len = max(seq.size(0) for seq in sequences)
            padded_sequences = torch.full(
                (len(sequences), max_gen_len),
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                dtype=outputs.dtype,
                device=outputs.device,
            )
            for i, seq in enumerate(sequences):
                padded_sequences[i, : seq.size(0)] = seq
            return {"sequences": padded_sequences}

        if remove_eos:
            bot_tensor = torch.tensor(
                [sot_token], dtype=torch.long, device=device
            ).expand(batch_size, 1)
        else:
            bot_tensor = torch.tensor(
                [tokenizer.eos_token_id, sot_token],
                dtype=torch.long,
                device=device,
            ).expand(batch_size, 2)

        input_ids_bot = torch.cat((input_ids, bot_tensor), dim=1)
        attention_mask_bot = torch.cat(
            (attention_mask, torch.ones_like(bot_tensor, device=device)), dim=1
        )

        if skip_thinking:
            # just use generate from transformers
            eot_tensor = torch.tensor(
                [tokenizer.encode("<|eocot|>", add_special_tokens=False)[0]],
                dtype=torch.long,
                device=device,
            ).expand(batch_size, 1)
            input_ids_bot = torch.cat((input_ids, bot_tensor, eot_tensor), dim=1)
            attention_mask_bot = torch.cat(
                (
                    attention_mask,
                    torch.ones_like(bot_tensor, device=device),
                    torch.ones_like(eot_tensor, device=device),
                ),
                dim=1,
            )
            print(tokenizer.convert_ids_to_tokens(input_ids_bot[0]))
            outputs = self.codi.generate(
                input_ids=input_ids_bot,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                attention_mask=attention_mask_bot,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Calculate the number of prompt tokens for each item in the batch
            prompt_lengths = [
                input_ids_bot[i].size(0) for i in range(input_ids_bot.size(0))
            ]
            # Remove the prompt tokens from each generated output
            sequences = []
            for i, prompt_len in enumerate(prompt_lengths):
                sequences.append(outputs[i, prompt_len:])
            # Pad to the maximum generation length in batch if needed
            max_gen_len = max(seq.size(0) for seq in sequences)
            padded_sequences = torch.full(
                (len(sequences), max_gen_len),
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                dtype=outputs.dtype,
                device=outputs.device,
            )
            for i, seq in enumerate(sequences):
                padded_sequences[i, : seq.size(0)] = seq
            return {"sequences": padded_sequences}

        with torch.no_grad():
            print(tokenizer.convert_ids_to_tokens(input_ids_bot[0]))
            outputs = self.codi(
                input_ids=input_ids_bot,
                use_cache=True,
                output_hidden_states=output_hidden_states,
                past_key_values=None,
                attention_mask=attention_mask_bot,
            )
            past_key_values = outputs.past_key_values
            prompt_hidden_states = outputs.hidden_states
            # Hidden state after <bot>
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            latent_vectors = []
            attentions_list = [] if output_attentions else None

            if return_latent_vectors:
                latent_vectors.append(latent_embd.clone())

            # Optionally project latent_embd if using prj
            if self.use_prj:
                latent_embd = self.prj(latent_embd)
                latent_embd = latent_embd.to(
                    dtype=self.codi.dtype
                )  # FIX: layer norm casts to fp32

            # Latent reasoning iterations
            for i in range(num_latent_iterations):
                print(f"Latent reasoning iteration {i + 1}")
                outputs = self.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if return_latent_vectors:
                    latent_vectors.append(latent_embd.clone())

                if self.use_prj:
                    latent_embd = self.prj(latent_embd)
                    latent_embd = latent_embd.to(
                        dtype=self.codi.dtype
                    )  # FIX: layer norm casts to fp32

                # Add EOT token embeddings
                if remove_eos:
                    eot_emb = (
                        self.get_embd(self.codi, self.model_name)(
                            torch.tensor(
                                [tokenizer.convert_tokens_to_ids("<|eocot|>")],
                                dtype=torch.long,
                                device=device,
                            )
                        )
                        .unsqueeze(0)
                        .to(device)
                    )
                else:
                    eot_emb = (
                        self.get_embd(self.codi, self.model_name)(
                            torch.tensor(
                                [
                                    tokenizer.convert_tokens_to_ids("<|eocot|>"),
                                    tokenizer.eos_token_id,
                                ],
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
                    output_attentions=output_attentions,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values

                # Collect attentions if requested
                if (
                    output_attentions
                    and hasattr(out, "attentions")
                    and out.attentions is not None
                ):
                    attentions_list.append(out.attentions)
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

        if return_latent_vectors and not skip_thinking:
            result["latent_vectors"] = latent_vectors

        if output_attentions:
            result["attentions"] = attentions_list

        if output_hidden_states and prompt_hidden_states is not None:
            result["hidden_states"] = prompt_hidden_states

        return result
