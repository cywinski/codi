#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
import logging
import math
import os
import re

import torch
import transformers
from dotenv import load_dotenv

from datasets import load_dataset

# Load environment variables from .env file
load_dotenv()

do_print = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def evaluation(model_name_or_path, batch_size, token=None, greedy=True):
    """
    Evaluate a base LLaMA model on GSM8K dataset.

    Args:
        model_name_or_path: Path or HuggingFace ID of the base model
        batch_size: Batch size for evaluation
        token: HuggingFace token for private models
        greedy: Whether to use greedy decoding (True) or sampling (False)
    """
    # Load tokenizer
    logging.warning("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        token=token,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    logging.warning("Loading model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    dataset = load_dataset("gsm8k", "main")
    test_set = dataset["test"]

    logging.warning("Formatting inputs...")
    question_name = "question"
    answer_name = "answer"

    # Format prompts with instruction
    question = []
    question_original = []  # Store original questions for printing
    for example in test_set:
        q = example[question_name].strip().replace("  ", " ")
        question_original.append(q)
        formatted_prompt = (
            f'{q}\n\nAnswer in a format: "The answer is: {{}}" and add nothing else.'
        )
        question.append(formatted_prompt)

    answer = []

    # get numerical answer
    for example in test_set:
        example = example[answer_name]
        if "####" in example:
            ans = example.split("####")[-1]
        else:
            ans = example
        ans = ans.replace(",", "")  # handle numbers like 2,000
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question) / batch_size)
    logging.warning(
        f"Total example: {len(question)} | eval batch size: {batch_size} | eval steps: {eval_step}"
    )

    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch_questions = question[i * batch_size : (i + 1) * batch_size]
        else:
            batch_questions = question[i * batch_size :]

        # Apply chat template to each question
        tokenized_batch = []
        for q in batch_questions:
            # Format as a chat message
            messages = [{"role": "user", "content": q}]
            # Apply chat template
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Tokenize the formatted prompt
            tokenized = tokenizer(
                formatted,
                return_tensors="pt",
                padding=False,
            )
            tokenized_batch.append(tokenized)

        # Pad to same length (left padding since padding_side="left")
        max_length = max(t["input_ids"].size(1) for t in tokenized_batch)
        input_ids_list = []
        attention_mask_list = []
        for t in tokenized_batch:
            pad_length = max_length - t["input_ids"].size(1)
            if pad_length > 0:
                padded_ids = torch.cat(
                    [
                        torch.full(
                            (1, pad_length), tokenizer.pad_token_id, dtype=torch.long
                        ),
                        t["input_ids"],
                    ],
                    dim=1,
                )
                padded_mask = torch.cat(
                    [
                        torch.zeros((1, pad_length), dtype=torch.long),
                        t["attention_mask"],
                    ],
                    dim=1,
                )
            else:
                padded_ids = t["input_ids"]
                padded_mask = t["attention_mask"]
            input_ids_list.append(padded_ids)
            attention_mask_list.append(padded_mask)

        batch = {
            "input_ids": torch.cat(input_ids_list, dim=0).to(device),
            "attention_mask": torch.cat(attention_mask_list, dim=0).to(device),
        }
        question_data.append(batch)

    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1 if not greedy else 1.0,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": not greedy,
        "pad_token_id": tokenizer.pad_token_id,
    }

    ans_pred_list = []
    len_cot = []

    logging.warning("Generating answers...")
    for step, batch in enumerate(question_data):
        batch_size_actual = batch["input_ids"].size(0)
        with torch.no_grad():
            # Generate responses
            outputs = model.generate(
                **batch,
                **gen_kwargs,
            )

            # Decode generated tokens (excluding input tokens)
            input_length = batch["input_ids"].size(1)
            generated_tokens = outputs[:, input_length:]

            for mini_step in range(batch_size_actual):
                pred_token = generated_tokens[mini_step].cpu().tolist()
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)

                if do_print:
                    print(f"Question {step * batch_size + mini_step} Starts...")
                    print(f"Q: {question_original[step * batch_size + mini_step]}")
                    print(decoded_pred)
                    print(f"Question {step * batch_size + mini_step} Ends")
                    print(
                        f"Prediction={extract_answer_number(decoded_pred)}; Groundtruth={answer[step * batch_size + mini_step]}"
                    )
                    print("")

                ans_pred_list.append(extract_answer_number(decoded_pred))

    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"Model: {model_name_or_path} | GSM8K test accuracy: {100 * accuracy:.2f}%")
    print(f"Average length of COT: {sum(len_cot) / len(len_cot):.2f}")

    return 100 * accuracy


def extract_answer_number(sentence: str) -> float:
    """Extract the numerical answer from the generated text."""
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")

    # use the last number as the answer
    pred_answer = float(pred[-1])
    return pred_answer


def compute_accuracy(gold: list, pred: list):
    """Compute accuracy between gold and predicted answers."""
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1

    return acc / len(gold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test base LLaMA model on GSM8K")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Path or HuggingFace ID of the base model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for private models",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )

    args = parser.parse_args()

    # Use environment variable for token if not provided
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    accuracy = evaluation(
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        token=token,
        greedy=args.greedy,
    )

    print(f"\nFinal accuracy: {accuracy:.2f}%")
