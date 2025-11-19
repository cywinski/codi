# %%
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# %%
# This is a LoRA fine-tuned model (not CODI with latent reasoning)
# It has resized token embeddings (+3 tokens for special tokens)
checkpoint_id = "bcywinski/cot_sft_llama1b_gsm8k-strategyqa-commonsense"
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"

# %%

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="cuda",
)
# Load tokenizer from the checkpoint (has custom tokens)
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_id, padding_side="left", use_fast=False
)


# %%
# Extend the token embeddings by 3
original_vocab_size = model.config.vocab_size
new_vocab_size = len(tokenizer.get_vocab()) + 3
model.resize_token_embeddings(new_vocab_size)
print(f"Resized embeddings from {original_vocab_size} to {new_vocab_size}")

# Add 3 special tokens if missing: pad, bot, eot
special_tokens_dict = {
    "pad_token": "[PAD]",
    "additional_special_tokens": ["<|bot|>", "<|eot|>"],
}
tokenizer.add_special_tokens(special_tokens_dict)

# Set IDs for the special tokens
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
tokenizer.bot_id = tokenizer.convert_tokens_to_ids("<|bot|>")
tokenizer.eot_id = tokenizer.convert_tokens_to_ids("<|eot|>")

# %%
# Load LoRA adapter from HuggingFace
# This downloads adapter_model.safetensors (or .bin) and adapter_config.json
print(f"Loading LoRA adapter from {checkpoint_id}...")
model = PeftModel.from_pretrained(
    model,
    checkpoint_id,
    is_trainable=False,
)
model.eval()
print("LoRA adapter loaded successfully")

# %%
# Prepare prompt
prompt = "Allison and Emma are building sandcastles. Emma's castle takes twice as long as Allison's to build. Maya, who is watching, spends three times as long just observing as it takes Allison to build her castle. If in total, they spend 36 hours, how many hours does Allison spend on building her sandcastle?"

# %%
# Tokenize
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = torch.cat(
    (
        inputs["input_ids"],
        torch.tensor([tokenizer.bot_id], dtype=torch.long).expand(
            inputs["input_ids"].size(0), 1
        ),
    ),
    dim=1,
).to(model.device)
attention_mask = torch.cat(
    (inputs["attention_mask"], torch.ones(1).expand(inputs["input_ids"].size(0), 1)),
    dim=1,
).to(model.device)
# %%
print(tokenizer.convert_ids_to_tokens(input_ids[0]))
# %%
# Generate
print(f"Prompt: {prompt}\n")
print("Generating...")

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        temperature=0.1,
        top_k=40,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

# %%
# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\nGenerated: {generated_text}")

# %%
