# %%
import torch
import transformers
from dotenv import load_dotenv

from src.model import CODI

load_dotenv()
from datasets import load_dataset

# llama3 3B id: bcywinski/codi_llama3b_gsm8k-strategyqa-commonsense
# llama3 1B id: bcywinski/llama1b_gsm8k-strategyqa-commonsense
# original: zen-E/CODI-llama3.2-1b-Instruct

# %%
model = CODI.from_pretrained(
    checkpoint_path="bcywinski/codi_llama1b_gsm8k-strategyqa-commonsense2",  # HF checkpoint ID
    model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",  # HF base model ID
    lora_r=128,
    lora_alpha=32,
    num_latent=6,  # Number of latent reasoning steps
    use_prj=True,  # Whether model uses projection layer
    device="cuda",
    dtype="bfloat16",
    strict=False,
    # Optional: specify where to save the checkpoint (default: ./checkpoints/{name})
    checkpoint_save_path="./checkpoints/bcywinski_codi_llama1b_gsm8k-strategyqa-commonsense2",
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
gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="test")

# %%
from tqdm import tqdm

questions = []
intermediate_numbers = []
answers = []

import re

for i, example in enumerate(tqdm(gsm8k_dataset)):
    questions.append(example["question"])
    answer = example["answer"]

    cot, ans = answer.split("####")
    # Find all intermediate calculations of the format <<...=number>>
    intermediate_results = re.findall(r"<<[^=]*=([^>]*)>>", cot)
    # Strip leading/trailing whitespace from each result
    intermediate_results = [res.strip() for res in intermediate_results]
    intermediate_numbers.append(intermediate_results)
    answers.append(ans.strip())
    if i == 0:
        print(example)
        print(intermediate_numbers[-1])
# %%
import numpy as np

# Choose batch size for inference
batch_size = 32


# Prepare DataLoader-like batching over questions
def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


results = []  # Each element: dict with keys 'question', 'answer', 'latent_vectors'

for batch_idx, batch_questions in tqdm(
    enumerate(list(batchify(questions, batch_size))), desc="Batch inference"
):
    # Tokenize the batch
    inputs = tokenizer(
        batch_questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].to(model.codi.device)
    attention_mask = inputs["attention_mask"].to(model.codi.device)

    # Model inference
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        max_new_tokens=256,
        num_latent_iterations=6,  # adjust as desired
        temperature=0.1,
        greedy=True,
        return_latent_vectors=True,
        remove_eos=True,
    )

    # The batch size may be less than batch_size in the last batch
    for idx in range(len(batch_questions)):
        generated = output["sequences"][idx]
        pred_ans = tokenizer.decode(generated, skip_special_tokens=True)
        sample_latent_vectors = []
        for j in range(len(output["latent_vectors"])):
            sample_latent_vectors.append(output["latent_vectors"][j][idx][0])
        sample_latent_vectors = torch.stack(sample_latent_vectors).to("cpu")
        results.append(
            {
                "question": batch_questions[idx],
                "answer": pred_ans,
                "intermediate_numbers": intermediate_numbers[
                    batch_idx * batch_size + idx
                ],
                "latent_vectors": sample_latent_vectors,  # shape: (num_latents, dim)
            }
        )

# %%
import re


def find_intermediate_numbers(text):
    # Find all numbers in the text (integers or floats), negative/positive
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers


def find_token_for_number(tokenizer, number_str):
    """
    Tries to tokenize the number as a single token using two options:
    1. Direct string.
    2. String with leading space (for tokenizers that mark non-initial tokens with spaces).
    Returns the token id if it comprises a single token, else None.
    """
    for candidate in [str(number_str)]:
        tokenized = tokenizer(candidate, add_special_tokens=False)
        input_ids = tokenized["input_ids"]
        if len(input_ids) == 1:
            return input_ids[0]
    return None


similarity_results = []

embed_matrix = model.codi.model.model.embed_tokens.weight.to("cpu")  # (vocab_size, dim)

# Gather ALL ints appearing in *any* answer, and append their token numbers.
all_numbers_found = set()
all_numbers_tokens = set()
intermediate_ints = []
intermediate_tokens = []
for r in results:
    interm = r["intermediate_numbers"]
    tmp_ints = []
    tmp_tokens = []
    for num in interm:
        if num.isdigit():
            int_num = int(num)
            all_numbers_found.add(int_num)
            token_id = find_token_for_number(tokenizer, num)
            if token_id is not None:
                all_numbers_tokens.add(token_id)
                tmp_tokens.append(token_id)
            tmp_ints.append(int_num)
    intermediate_ints.append(tmp_ints)
    intermediate_tokens.append(tmp_tokens)
min_number = int(min(all_numbers_found))
max_number = int(min(max(all_numbers_found), 10000))

print(f"Min number: {min_number}, Max number: {max_number}")
all_numbers_tokens = list(all_numbers_tokens)
# %%
for i, result in tqdm(enumerate(results)):
    latent_vectors = result["latent_vectors"]
    if len(intermediate_tokens[i]) == 0:
        continue
    sims = latent_vectors @ embed_matrix.T
    sims = sims.softmax(dim=-1)
    intermediate_number_sims = sims[:, intermediate_tokens[i]].mean(dim=1)
    other_numbers_sims = sims[:, all_numbers_tokens].mean(dim=1)
    random_sims = sims.mean(dim=1)
    similarity_results.append(
        {
            "question": result["question"],
            "answer": result["answer"],
            "intermediate_numbers": intermediate_numbers,
            "intermediate_number_sims": intermediate_number_sims.tolist(),
            "other_numbers_sims": other_numbers_sims.tolist(),
            "random_sims": random_sims.tolist(),
        }
    )
# %%
import matplotlib.pyplot as plt

all_interm_means = []
all_other_means = []
all_random_means = []
for i in range(len(similarity_results)):
    all_interm_means.append(similarity_results[i]["intermediate_number_sims"])
    all_other_means.append(similarity_results[i]["other_numbers_sims"])
    all_random_means.append(similarity_results[i]["random_sims"])

# %%
plt.figure()
plt.plot(
    np.array(all_interm_means).mean(axis=0),
    label="Intermediate numbers",
    marker="o",
    linewidth=3,
    markersize=8,
)
plt.plot(
    np.array(all_other_means).mean(axis=0),
    label="All numbers",
    marker="o",
    linewidth=3,
    markersize=8,
)
plt.plot(
    np.array(all_random_means).mean(axis=0),
    label="Random",
    linestyle="--",
    marker="o",
    linewidth=3,
    markersize=8,
)
plt.legend()
plt.grid(True)
plt.xlabel("# Latent vector")
plt.ylabel("Average similarity")
plt.title("Average similarity between latent vectors and token embeddings")
plt.savefig("plots/token_numbers_similarity.png", dpi=300, bbox_inches="tight")
