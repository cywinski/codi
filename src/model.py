## model.py
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
        print(f"Base model: {model_name_or_path}")
        if not os.path.exists(model_name_or_path):
            print(
                "  → Will be downloaded from HuggingFace (cached in ~/.cache/huggingface/)"
            )
        else:
            print("  → Loading from local path")

        model_args = ModelArguments(
            model_name_or_path=model_name_or_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_init=True,
            full_precision=full_precision,
            train=False,
            token=token,
        )

        bf16 = dtype == "bfloat16"
        training_args = TrainingArguments(
            output_dir="./output",
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
            raise ValueError(f"Unsupported model architecture: {model_name_or_path}.")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )

        model = cls(model_args, training_args, lora_config)

        is_hf_checkpoint = not os.path.exists(checkpoint_path)
        checkpoint_file = None
        if is_hf_checkpoint:
            print(f"Checkpoint: {checkpoint_path} (HuggingFace ID)")
            if checkpoint_save_path is None:
                checkpoint_name_clean = checkpoint_path.replace("/", "_")
                checkpoint_save_path = os.path.join(
                    "./checkpoints", checkpoint_name_clean
                )
            print(f"  → Downloading to: {checkpoint_save_path}")
            from huggingface_hub import snapshot_download

            local_checkpoint_path = snapshot_download(
                repo_id=checkpoint_path,
                token=token,
                allow_patterns=["*.safetensors", "*.bin", "*.json"],
                local_dir=checkpoint_save_path,
                local_dir_use_symlinks=False,
            )
            print(f"  → Checkpoint saved to: {local_checkpoint_path}")
            checkpoint_path = local_checkpoint_path
        else:
            print(f"Checkpoint: {checkpoint_path} (local path)")

        if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
            checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
            state_dict = load_file(checkpoint_file)
        elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
            checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
            state_dict = torch.load(checkpoint_file, map_location="cpu")
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}.")

        print(f"Loading checkpoint from {checkpoint_file}")
        model.load_state_dict(state_dict, strict=strict)
        model.codi.tie_weights()

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
        target_dtype = torch.float16 if training_args.bf16 is False else torch.bfloat16
        self.model_name = model_args.model_name_or_path
        model_wrapper_class = AutoModelForCausalLM
        if model_args.full_precision:
            self.codi = model_wrapper_class.from_pretrained(
                self.model_name,
                torch_dtype=target_dtype,
                resume_download=True,
            )
        else:
            self.codi = model_wrapper_class.from_pretrained(
                self.model_name,
                torch_dtype=target_dtype,
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
        self.ans_id = ori_vocab_size + 3  # Add <ans> token

        # Resize for [PAD], [BOT], [EOT], [ANS]
        self.codi.resize_token_embeddings(
            ori_vocab_size + 4
        )

        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # Register special tokens in tokenizer
        self.tokenizer.add_special_tokens({
            "pad_token": "[PAD]",
            "additional_special_tokens": ["<bot>", "<eot>", "<ans>"]
        })

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

        self.to(dtype=target_dtype)
        if self.training:
            self.init()

    @property
    def config(self):
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
                except Exception:
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception:
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

        # ------------------------------------------------------------------
        # 1. Run Teacher (Reference) Model
        #    Input: Question + CoT + <ans> + Answer
        # ------------------------------------------------------------------
        with torch.no_grad():
            ref_outputs = self.codi(
                input_ids=ref_input_ids,
                output_hidden_states=True,
                attention_mask=ref_attention_mask,
            )
            
            # CE Loss for Teacher (just for monitoring)
            ref_logits = ref_outputs.logits
            effective_ref_logits = ref_logits[:, :-1, :].reshape(-1, ref_logits.size(-1))
            ref_target_ids = ref_labels[:, 1:].reshape(-1)
            ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids)
            ref_ce_loss *= self.ref_loss_factor

        # ------------------------------------------------------------------
        # 2. Run Student Encoder
        #    Input: Question + <bot>
        # ------------------------------------------------------------------
        past_key_values = None
        outputs = self.codi(
            input_ids=encoder_input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=encoder_attention_mask,
        )
        past_key_values = outputs.past_key_values
        
        # Initial latent embedding from <bot> output
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if self.use_prj:
            latent_embd = self.prj(latent_embd).to(dtype=self.codi.dtype)

        # ------------------------------------------------------------------
        # 3. Latent Reasoning Loop
        #    Generates l1 ... l_k
        # ------------------------------------------------------------------
        num_latent = self.num_latent
        
        # Prepare dynamic mask
        dynamic_mask = None
        if self.fix_attn_mask:
            dynamic_mask = torch.ones(
                (encoder_attention_mask.size(0), num_latent),
                device=encoder_input_ids.device,
            )
            dynamic_mask = torch.cat((encoder_attention_mask, dynamic_mask), dim=1)

        for i in range(num_latent):
            outputs = self.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if self.use_prj:
                latent_embd = self.prj(latent_embd).to(dtype=self.codi.dtype)

        # ------------------------------------------------------------------
        # 4. Insert <ans> Token & Distillation
        #    Input 2 Path: ... -> Latents -> <ans>
        # ------------------------------------------------------------------
        
        # Create <ans> embeddings to feed into the student
        # decoder_input_ids[:, 0] should be the <ans> token id based on preprocess logic
        ans_token_ids = decoder_input_ids[:, 0:1] 
        ans_embds = self.get_embd(self.codi, self.model_name)(ans_token_ids)
        
        # Update mask for <ans> token
        if dynamic_mask is not None:
            ans_mask = torch.ones((ans_embds.size(0), 1), device=ans_embds.device)
            current_mask = torch.cat((dynamic_mask, ans_mask), dim=1).bool()
        else:
            current_mask = None

        # Pass <ans> through Student (using past_kv from latents)
        ans_outputs = self.codi(
            inputs_embeds=ans_embds,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=current_mask
        )
        
        # Update KV cache for generating the actual answer next
        past_key_values = ans_outputs.past_key_values
        
        # --- ALIGNMENT ---
        # Get Student State at <ans> output
        student_ans_state = ans_outputs.hidden_states[-1] # shape: (batch, 1, dim)

        # Get Teacher State at <ans> output
        # ref_answer_position points to the <ans> token index in ref_input_ids
        # We need the hidden state produced by the Teacher when it processed <ans>
        ref_ans_state_list = []
        for b in range(len(ref_input_ids)):
            pos = ref_answer_position[b]
            # Ensure index is within bounds
            if pos < ref_outputs.hidden_states[-1].size(1):
                ref_ans_state_list.append(ref_outputs.hidden_states[-1][b, pos, :])
            else:
                # Fallback for edge cases
                ref_ans_state_list.append(ref_outputs.hidden_states[-1][b, -1, :])
                
        ref_ans_state = torch.stack(ref_ans_state_list).unsqueeze(1) # (batch, 1, dim)

        # Calculate Distillation Loss
        distill_loss = self.distill_loss_fct(
            student_ans_state, ref_ans_state.detach()
        )
        
        if self.distill_loss_div_std and self.distill_loss_type == "l2":
             distill_loss /= ref_ans_state.std()

        # ------------------------------------------------------------------
        # 5. Answer Generation (Student)
        #    Input: Answer tokens (after <ans>)
        # ------------------------------------------------------------------
        
        # decoder_input_ids includes <ans> at index 0. We already processed it.
        # Now we process the rest: Answer tokens.
        answer_ids = decoder_input_ids[:, 1:]
        
        ce_loss = torch.tensor(0.0, device=distill_loss.device)
        
        if answer_ids.size(1) > 0:
            answer_embds = self.get_embd(self.codi, self.model_name)(answer_ids)
            
            # Extend mask
            if current_mask is not None:
                answer_mask = torch.ones((answer_embds.size(0), answer_embds.size(1)), device=answer_embds.device)
                full_mask = torch.cat((current_mask, answer_mask), dim=1).bool()
            else:
                full_mask = None

            # Pass Answer tokens (using past_kv from <ans>)
            outputs = self.codi(
                inputs_embeds=answer_embds,
                use_cache=True,
                output_hidden_states=False,
                past_key_values=past_key_values,
                attention_mask=full_mask
            )
            
            # ------------------------------------------------------------------
            # 6. Cross Entropy Loss on Answer
            # ------------------------------------------------------------------
            
            # Combine <ans> output logits and answer output logits
            # ans_outputs.logits predicts the first token of the answer
            # outputs.logits predicts the subsequent tokens
            full_logits = torch.cat([ans_outputs.logits, outputs.logits], dim=1)
            
            # Standard shifted CE loss:
            # Inputs: [<ans>, A1, A2, ...]
            # Targets: [A1, A2, ..., EOS] (labels[:, 1:])
            
            effective_logits = full_logits[:, :-1, :].reshape(-1, self.codi.config.vocab_size)
            effective_labels = labels[:, 1:].reshape(-1)
            
            ce_loss = self.loss_fct(effective_logits, effective_labels)

        # Weigh losses
        distill_loss_scaled = distill_loss * self.distill_loss_factor
        loss = ce_loss + distill_loss_scaled + ref_ce_loss

        return {
            "loss": loss,
            "logits": full_logits if answer_ids.size(1) > 0 else ans_outputs.logits,
            "ce_loss": ce_loss.detach().item() if isinstance(ce_loss, torch.Tensor) else 0,
            "distill_loss": distill_loss.detach().item(),
            "ref_ce_loss": ref_ce_loss.detach().item(),
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
        if tokenizer is None:
            tokenizer = self.tokenizer

        device = input_ids.device
        batch_size = input_ids.size(0)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Add BOT token to input
        bot_tensor = torch.tensor(
            [self.bot_id], dtype=torch.long, device=device
        ).expand(batch_size, 1)

        input_ids = torch.cat((input_ids, bot_tensor), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(bot_tensor, device=device)), dim=1
        )

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

            if return_latent_vectors:
                latent_vectors.append(latent_embd.clone())

            if self.use_prj:
                latent_embd = self.prj(latent_embd).to(dtype=self.codi.dtype)

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

                if return_latent_vectors:
                    latent_vectors.append(latent_embd.clone())

                if self.use_prj:
                    latent_embd = self.prj(latent_embd).to(dtype=self.codi.dtype)

            # Add <ans> token embeddings to signal answer generation start
            ans_emb = (
                self.get_embd(self.codi, self.model_name)(
                    torch.tensor([self.ans_id], dtype=torch.long, device=device)
                )
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            
            output_emb = ans_emb

            # Generate tokens
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            generated_tokens = [[] for _ in range(batch_size)]

            for step in range(max_new_tokens):
                out = self.codi(
                    inputs_embeds=output_emb,
                    use_cache=True,
                    output_hidden_states=False,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, : self.codi.config.vocab_size]

                # Sampling
                if greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= temperature
                    if top_k > 1:
                        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

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

                if next_token_ids.dim() == 0:
                    next_token_ids = next_token_ids.unsqueeze(0)

                for b in range(batch_size):
                    if not finished[b]:
                        generated_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                if finished.all():
                    break

                output_emb = (
                    self.get_embd(self.codi, self.model_name)(next_token_ids)
                    .unsqueeze(1)
                    .to(device)
                )

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