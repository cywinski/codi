# %%
import torch
import transformers
from dotenv import load_dotenv

from src.model import CODI

load_dotenv()

# llama3 3B id: bcywinski/codi_llama3b_gsm8k-strategyqa-commonsense
# llama3 1B id: bcywinski/llama1b_gsm8k-strategyqa-commonsense
# original: zen-E/CODI-llama3.2-1b-Instruct

# %%
model = CODI.from_pretrained(
    checkpoint_path="zen-E/CODI-gpt2",  # HF checkpoint ID
    model_name_or_path="gpt2",  # HF base model ID
    lora_r=128,
    lora_alpha=32,
    num_latent=6,  # Number of latent reasoning steps
    use_prj=True,  # Whether model uses projection layer
    device="cuda",
    dtype="bfloat16",
    strict=False,
    attn_implementation="eager",
    # Optional: specify where to save the checkpoint (default: ./checkpoints/{name})
    checkpoint_save_path="./models/codi_gpt2",
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
tokenizer.add_special_tokens({"additional_special_tokens": ["<|bot|>", "<|eot|>"]})
tokenizer.bot_id = tokenizer.convert_tokens_to_ids("<|bot|>")
tokenizer.eot_id = tokenizer.convert_tokens_to_ids("<|eot|>")

# %%
# Tokenize input
prompt = "Mark's car breaks down and he needs to get a new radiator. The cost for a new radiator is $400 but he goes to get it at a junk shop and gets it for 80% off. He then hires a mechanic to install it and it takes 3 hours at $50 an hour. How much did he pay?"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(model.codi.device)
attention_mask = inputs["attention_mask"].to(model.codi.device)
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
    greedy=True,
    return_latent_vectors=True,  # Get the latent reasoning vectors
    remove_eos=True,
)

# %%
# Decode and print output
generated_text = tokenizer.decode(output["sequences"][0], skip_special_tokens=False)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
print(f"\nNumber of latent reasoning vectors: {len(output['latent_vectors'])}")
print(f"Each latent vector shape: {output['latent_vectors'][0].shape}")

# %%
latent_vectors = torch.stack(output["latent_vectors"]).squeeze(1).squeeze(1).to("cpu")
latent_vectors.shape
# %%
embed_matrix = model.codi.model.model.embed_tokens.weight.to("cpu")
# embed_matrix = model.codi.base_model.model.transformer.wte.weight.to("cpu")

# %%
sims = latent_vectors @ embed_matrix.T
sims.shape
# %%
sims = sims.softmax(dim=-1)
# %%
k = 20
for i in range(sims.shape[0]):
    topk = sims[i].topk(k, dim=-1)
    topk_indices = topk.indices.tolist()
    topk_values = topk.values.tolist()
    for j, value in zip(topk_indices, topk_values):
        print(tokenizer.convert_ids_to_tokens(torch.tensor(j).tolist()), value)
    print("-" * 100)
# %%
