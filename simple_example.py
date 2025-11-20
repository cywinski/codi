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
    checkpoint_path="bcywinski/codi_llama1b_gsm8k-strategyqa-commonsense",  # HF checkpoint ID
    model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",  # HF base model ID
    lora_r=128,
    lora_alpha=32,
    num_latent=6,  # Number of latent reasoning steps
    use_prj=True,  # Whether model uses projection layer
    device="cuda",
    dtype="bfloat16",
    strict=False,
    # Optional: specify where to save the checkpoint (default: ./checkpoints/{name})
    checkpoint_save_path="./checkpoints/bcywinski_codi_llama1b_gsm8k-strategyqa-commonsense",
    remove_eos=True,
    full_precision=True,
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
prompt = "Jenny buys 1 bag of cookies a week. The bag has 36 cookies and she puts 4 cookies in her son's lunch box 5 days a week. Her husband eats 1 cookie a day for 7 days. Jenny eats the rest of the cookies. How many cookies does Jenny eat?"
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
    print(f"\nLatent vector #{i + 1}:")
    print(f"{'Rank':<5} {'Token':<20} {'ID':<6} {'Similarity':<10}")
    print("-" * 50)
    for rank, (j, value) in enumerate(zip(topk_indices, topk_values), 1):
        token_str = tokenizer.convert_ids_to_tokens([j])[0]
        print(f"{rank:<5} {token_str:<20} {j:<6} {value:<10.6f}")
    print("-" * 50)
# %%
