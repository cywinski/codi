# %%
import torch
import transformers
from dotenv import load_dotenv

from src.model import CODI

load_dotenv()

# llama3 3B id: bcywinski/codi_llama3b_gsm8k-strategyqa-commonsense
# llama3 1B id: bcywinski/llama1b_gsm8k-strategyqa-commonsense

# %%
model = CODI.from_pretrained(
    checkpoint_path="bcywinski/llama3b_gsm8k-strategyqa-commonsense",  # HF checkpoint ID
    model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",  # HF base model ID
    lora_r=128,
    lora_alpha=16,
    num_latent=6,  # Number of latent reasoning steps
    use_prj=False,  # Whether model uses projection layer
    device="cuda",
    dtype="bfloat16",
    strict=False,
    attn_implementation="eager",
    # Optional: specify where to save the checkpoint (default: ./checkpoints/{name})
    checkpoint_save_path="./models/codi_llama3b",
)
# %%
# Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model.model_name,
    padding_side="left",
    use_fast=False,
)

if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = model.pad_token_id

# %%
# Tokenize input
prompt = "Emily plans on having a picnic and the grocery store suggests 1/4 kg of fruit per person. If she's inviting 8 friends and the fruit costs $5.00 a kg, how much will this cost her?"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = torch.cat(
    (
        inputs["input_ids"],
        torch.tensor([model.bot_id], dtype=torch.long).expand(
            inputs["input_ids"].size(0), 1
        ),
    ),
    dim=1,
).to(model.codi.device)
attention_mask = torch.cat(
    (inputs["attention_mask"], torch.ones(1).expand(inputs["input_ids"].size(0), 1)),
    dim=1,
).to(model.codi.device)
# %%
print(tokenizer.convert_ids_to_tokens(input_ids[0]))
# %%
# Generate with latent reasoning
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    tokenizer=tokenizer,
    max_new_tokens=256,
    num_latent_iterations=6,  # How many latent reasoning steps
    temperature=0.1,
    greedy=False,
    return_latent_vectors=True,  # Get the latent reasoning vectors
)

# %%
# Decode and print output
generated_text = tokenizer.decode(output["sequences"][0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
print(f"\nNumber of latent reasoning vectors: {len(output['latent_vectors'])}")
print(f"Each latent vector shape: {output['latent_vectors'][0].shape}")

# %%
