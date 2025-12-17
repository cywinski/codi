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
import csv
import json
import logging
import math
import os
import statistics
import sys

import torch
import transformers
import yaml
from dotenv import load_dotenv

from datasets import concatenate_datasets, load_dataset
from src.model import (
    CODI,
    DataArguments,
    ModelArguments,
    TrainingArguments,
)

# Load environment variables from .env file
load_dotenv()

do_print = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def evaluation(
    model_args,
    data_args,
    training_args,
    iteration=None,
    output_prefix=None,
    skip_thinking=False,
    verbalize_cot=False,
    answer_only=True,
    temperature=0.1,
    ablate_latent=None,
):
    # Load model using from_pretrained
    model = CODI.from_pretrained(
        checkpoint_path=model_args.ckpt_dir,
        model_name_or_path=model_args.model_name_or_path,
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        num_latent=training_args.num_latent,
        use_prj=training_args.use_prj,
        remove_eos=training_args.remove_eos,
        model_max_length=training_args.model_max_length,
        full_precision=model_args.full_precision,
        device="cuda",
        dtype="bfloat16" if training_args.bf16 else "float16",
        token=model_args.token,
        strict=False,
    )

    # # Load tokenizer
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     model.model_name,
    #     token=model_args.token,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="left",
    #     use_fast=False,
    # )

    # if tokenizer.pad_token_id is None:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    #     tokenizer.pad_token_id = model.pad_token_id
    #     if tokenizer.pad_token_id is None:  # error handling
    #         tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    # # Add special tokens if not already present
    # tokenizer.add_special_tokens({"additional_special_tokens": ["<|bot|>", "<|eot|>"]})
    # tokenizer.bot_id = tokenizer.convert_tokens_to_ids("<|bot|>")
    # tokenizer.eot_id = tokenizer.convert_tokens_to_ids("<|eot|>")

    device = "cuda"

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    question_name = "question"
    answer_name = "answer"
    # Handle data_names as list (from command-line) or string (from config)
    data_name = (
        data_args.data_names[0]
        if isinstance(data_args.data_names, list) and len(data_args.data_names) > 0
        else data_args.data_names
    )

    if "gsm-hard" == data_name:
        dataset = load_dataset("juyoung-trl/gsm-hard")
        test_set = dataset["train"]
        question_name = "instruction"
        answer_name = "response"
    elif "multi-arith" == data_name:
        dataset = load_dataset("ChilleD/MultiArith")
        test_set = dataset["test"]
        answer_name = "final_ans"
    elif "svamp" == data_name:
        dataset = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        question_name = "question_concat"
        answer_name = "Answer"
    elif "commonsense" == data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        test_set = dataset["validation"]
    elif "gsm8k" == data_name:
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset["test"]
    elif "strategy" == data_name:
        dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")
        test_set = dataset["train"]
    else:
        raise NotImplementedError

    logging.warning("Formatting inputs...")
    print(f"skip_thinking: {skip_thinking}")
    print(f"answer_only: {answer_only}")
    if "strategy" == data_name:
        question = [
            f"{example[question_name].strip().replace('  ', ' ')}. Answer only 'true' or 'false' and nothing else."
            for example in test_set
        ]
    else:
        if answer_only:
            question = [
                f"{example[question_name].strip().replace('  ', ' ')} Output only the answer and nothing else."
                for example in test_set
            ]
        else:
            question = [
                example[question_name].strip().replace("  ", " ")
                for example in test_set
            ]
    answer = []

    # get numerical answer
    for example in test_set:
        example = example[answer_name]
        if isinstance(example, bool):
            answer.append(example)
            continue
        if example in ["True", "False"]:
            if example == "True":
                ans = True
            else:
                ans = False
            answer.append(ans)
            continue
        if example in "ABCDE":
            answer.append(example)
            continue
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
    eval_step = math.ceil(len(question) / data_args.batch_size)
    logging.warning(
        f"Total example: {len(question)} | eval batch size: {data_args.batch_size} | "
        f"eval steps: {eval_step}"
    )

    ans_pred_list = []
    len_cot = []
    detailed_results = []  # Store answer, ground truth, and correctness for each example
    model.eval()
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.tokenizer.pad_token_id = model.tokenizer.convert_tokens_to_ids("[PAD]")

    tokenizer = model.tokenizer
    tokenizer.bot_id = tokenizer.convert_tokens_to_ids("<|bocot|>")  # beginning of CoT
    tokenizer.eot_id = tokenizer.convert_tokens_to_ids("<|eocot|>")
    print(f"{len(tokenizer)=}")

    for step in range(eval_step):
        # Prepare batch
        if step < eval_step - 1:
            batch_questions = question[
                step * data_args.batch_size : (step + 1) * data_args.batch_size
            ]
        else:
            batch_questions = question[step * data_args.batch_size :]

        # Tokenize batch
        batch = tokenizer(
            batch_questions,
            return_tensors="pt",
            padding="longest",
            padding_side="left",
        )
        input_ids = batch["input_ids"].to(device)
        print(tokenizer.convert_ids_to_tokens(input_ids[0]))
        attention_mask = batch["attention_mask"].to(device)

        # Generate using the abstract generate method
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                max_new_tokens=256,
                output_hidden_states=True,
                num_latent_iterations=training_args.inf_latent_iterations,
                temperature=temperature,
                top_k=40,
                top_p=0.95,
                greedy=training_args.greedy,
                return_latent_vectors=False,
                remove_eos=training_args.remove_eos,
                skip_thinking=skip_thinking,
                verbalize_cot=verbalize_cot,
                sot_token=tokenizer.convert_tokens_to_ids("<|bocot|>"),
                eot_token=tokenizer.convert_tokens_to_ids("<|eocot|>"),
                ablate_latent=ablate_latent,
            )

        # Process generated sequences
        generated_sequences = output["sequences"]
        batch_size = generated_sequences.size(0)

        for mini_step in range(batch_size):
            # Extract generated tokens (sequences already contains only generated tokens)
            generated_tokens = generated_sequences[mini_step].tolist()

            # Remove padding tokens and EOS token if present
            if tokenizer.pad_token_id is not None:
                generated_tokens = [
                    t for t in generated_tokens if t != tokenizer.pad_token_id
                ]
            if tokenizer.eos_token_id in generated_tokens:
                eos_idx = generated_tokens.index(tokenizer.eos_token_id)
                generated_tokens = generated_tokens[:eos_idx]

            len_cot.append(len(generated_tokens))
            decoded_pred = tokenizer.decode(generated_tokens, skip_special_tokens=False)

            question_idx = step * data_args.batch_size + mini_step
            pred_answer = extract_answer_number(decoded_pred, data_args)

            if do_print:
                print(f"Question {question_idx} Starts...")
                print(f"Q: {question[question_idx]}")
                print(decoded_pred)
                print(f"Question {question_idx} Ends")
                print(f"Prediction={pred_answer}; Groundtruth={answer[question_idx]}")
                print("")

            ans_pred_list.append(pred_answer)

            # Determine correctness
            is_correct = False
            gt = answer[question_idx]
            if isinstance(pred_answer, list):
                if gt in pred_answer:
                    is_correct = True
            else:
                if pred_answer == gt:
                    is_correct = True

            # Store detailed result
            detailed_results.append(
                {
                    "question": question[question_idx],
                    "answer": pred_answer,
                    "ground_truth": gt,
                    "is_correct": is_correct,
                }
            )

    accuracy = compute_accuracy(answer, ans_pred_list)

    print(
        f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100 * accuracy:.2f}% | "
    )
    print(f"average length of COT: {sum(len_cot) / len(len_cot)}")

    # Return accuracy and detailed results (CSV saving moved to main loop)
    return 100 * accuracy, detailed_results


import re


def extract_answer_number(sentence: str, data_args) -> float:
    print(f"sentence: {sentence}")
    sentence = sentence.replace(",", "")
    if "commonsense" in data_args.data_names[0]:
        return sentence.strip()[-1]
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    pred = float(pred[-1])
    return pred
    # if not pred:
    #     # Handle data_names as list (from command-line) or string (from config)
    #     data_name = (
    #         data_args.data_names[0]
    #         if isinstance(data_args.data_names, list) and len(data_args.data_names) > 0
    #         else data_args.data_names
    #     )

    #     if "commonsense" in data_name:
    #         pred = sentence.split("The answer is:")[-1].strip()
    #         if pred and pred[0] not in "ABCDE":
    #             return "C"
    #         return pred[0] if pred else "C"
    #     elif "strategy" in data_name or "prontoqa" in data_name.lower():
    #         if "True" in sentence:
    #             return True
    #         elif "False" in sentence:
    #             return False
    #         else:
    #             raise ValueError
    #     return float("inf")

    # use the last number as the answer
    pred_answer = float(pred[-1])

    return pred_answer


def compute_accuracy(gold: list, pred: list):
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
    # Support config file as positional argument or --config_file flag
    # Check if first argument (after script name) looks like a config file
    config_path = None
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        # Check if it's a config file (not a flag and has config extension)
        if not first_arg.startswith("-") and (
            first_arg.endswith(".yaml")
            or first_arg.endswith(".yml")
            or first_arg.endswith(".json")
        ):
            config_path = first_arg
            # Remove it from sys.argv so HfArgumentParser doesn't see it
            sys.argv.pop(1)

    # Also check for --config_file flag and --output_prefix
    arg_parser = argparse.ArgumentParser(description="Test CODI model", add_help=False)
    arg_parser.add_argument(
        "--config_file", type=str, default=None, help="Path to JSON or YAML config file"
    )
    arg_parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Prefix to add to CSV output filename",
    )
    arg_parser.add_argument(
        "--skip_thinking",
        action="store_true",
        default=False,
        help="Skip thinking step",
    )
    arg_parser.add_argument(
        "--verbalize_cot",
        action="store_true",
        default=False,
        help="Verbalize chain of thought",
    )
    arg_parser.add_argument(
        "--csv_filename",
        type=str,
        default=None,
        help="Filename for CSV output (if not provided, auto-generated)",
    )
    arg_parser.add_argument(
        "--answer_only",
        type=lambda x: x.lower() == "true" if isinstance(x, str) else bool(x),
        default=None,
        nargs="?",
        const=True,
        help="Add 'Output only the answer and nothing else.' to each question. Use --answer_only True/False or just --answer_only (default: True, maintains current behavior)",
    )
    arg_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for generation (default: 0.1)",
    )
    arg_parser.add_argument(
        "--ablate_latent",
        type=str,
        default=None,
        help="Type of latent ablation: 'zero', 'random', or 'mean' (default: None, no ablation)",
    )
    config_args, remaining_args = arg_parser.parse_known_args()

    # Use --config_file flag if provided, otherwise use positional argument we found
    config_path = config_args.config_file or config_path

    # Extract output_prefix, csv_filename, answer_only, temperature, and ablate_latent before processing config
    output_prefix = config_args.output_prefix
    csv_filename = config_args.csv_filename
    answer_only = config_args.answer_only
    temperature = config_args.temperature
    ablate_latent = config_args.ablate_latent

    if config_path:
        # Remove --config_file and its value from sys.argv if present
        if "--config_file" in sys.argv:
            idx = sys.argv.index("--config_file")
            sys.argv.pop(idx)  # Remove --config_file
            if idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
                sys.argv.pop(idx)  # Remove config file path

        # Remove --output_prefix and its value from sys.argv if present
        if "--output_prefix" in sys.argv:
            idx = sys.argv.index("--output_prefix")
            sys.argv.pop(idx)  # Remove --output_prefix
            if idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
                sys.argv.pop(idx)  # Remove output_prefix value

        # Remove --csv_filename and its value from sys.argv if present
        if "--csv_filename" in sys.argv:
            idx = sys.argv.index("--csv_filename")
            sys.argv.pop(idx)  # Remove --csv_filename
            if idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
                sys.argv.pop(idx)  # Remove csv_filename value

        # Remove --skip_thinking from sys.argv if present
        if "--skip_thinking" in sys.argv:
            sys.argv.remove("--skip_thinking")

        # Remove --verbalize_cot from sys.argv if present
        if "--verbalize_cot" in sys.argv:
            sys.argv.remove("--verbalize_cot")

        # Remove --answer_only and its value from sys.argv if present
        if "--answer_only" in sys.argv:
            idx = sys.argv.index("--answer_only")
            sys.argv.pop(idx)  # Remove --answer_only
            if idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
                sys.argv.pop(idx)  # Remove answer_only value (e.g., "True" or "False")

        # Remove --temperature and its value from sys.argv if present
        if "--temperature" in sys.argv:
            idx = sys.argv.index("--temperature")
            sys.argv.pop(idx)  # Remove --temperature
            if idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
                sys.argv.pop(idx)  # Remove temperature value

        # Remove --ablate_latent and its value from sys.argv if present
        if "--ablate_latent" in sys.argv:
            idx = sys.argv.index("--ablate_latent")
            sys.argv.pop(idx)  # Remove --ablate_latent
            if idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
                sys.argv.pop(idx)  # Remove ablate_latent value

        # Load config file
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        elif config_path.endswith(".json"):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            raise ValueError(
                f"Config file must be .json, .yaml, or .yml, got {config_path}"
            )

        # Create dataclass instances from config
        model_config = config.get("model_args", {})
        # Use environment variable for token if not provided in config
        if model_config.get("token") is None:
            model_config["token"] = os.getenv("HF_TOKEN") or os.getenv(
                "HUGGINGFACE_TOKEN"
            )
        model_args = ModelArguments(**model_config)
        data_args = DataArguments(**config.get("data_args", {}))

        # TrainingArguments needs special handling
        training_config = config.get("training_args", {})
        training_args = TrainingArguments(**training_config)

        # Get output_prefix from config if not provided via command line
        if output_prefix is None:
            output_prefix = config.get("output_prefix")

        # Get csv_filename from config if not provided via command line
        if csv_filename is None:
            csv_filename = config.get("csv_filename")

        # Get answer_only from config if not provided via command line (default to True to maintain current behavior)
        # If answer_only is None (not set), check config, otherwise use True as default
        if answer_only is None:
            answer_only = config.get("answer_only", True)
        # If answer_only was explicitly set via command line (True or False), keep that value

        # Get temperature from config if not provided via command line (default to 0.1)
        if temperature is None:
            temperature = config.get("temperature", 0.1)

        # Get ablate_latent from config if not provided via command line
        if ablate_latent is None:
            ablate_latent = config.get("ablate_latent")
    else:
        # Remove --output_prefix and its value from sys.argv if present (before HfArgumentParser)
        if "--output_prefix" in sys.argv:
            idx = sys.argv.index("--output_prefix")
            sys.argv.pop(idx)  # Remove --output_prefix
            if idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
                sys.argv.pop(idx)  # Remove output_prefix value

        # Remove --csv_filename and its value from sys.argv if present (before HfArgumentParser)
        if "--csv_filename" in sys.argv:
            idx = sys.argv.index("--csv_filename")
            sys.argv.pop(idx)  # Remove --csv_filename
            if idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
                sys.argv.pop(idx)  # Remove csv_filename value

        # Remove --skip_thinking from sys.argv if present (before HfArgumentParser)
        if "--skip_thinking" in sys.argv:
            sys.argv.remove("--skip_thinking")

        # Remove --verbalize_cot from sys.argv if present (before HfArgumentParser)
        if "--verbalize_cot" in sys.argv:
            sys.argv.remove("--verbalize_cot")

        # Remove --answer_only and its value from sys.argv if present (before HfArgumentParser)
        if "--answer_only" in sys.argv:
            idx = sys.argv.index("--answer_only")
            sys.argv.pop(idx)  # Remove --answer_only
            if idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
                sys.argv.pop(idx)  # Remove answer_only value (e.g., "True" or "False")

        # Remove --temperature and its value from sys.argv if present (before HfArgumentParser)
        if "--temperature" in sys.argv:
            idx = sys.argv.index("--temperature")
            sys.argv.pop(idx)  # Remove --temperature
            if idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
                sys.argv.pop(idx)  # Remove temperature value

        # Remove --ablate_latent and its value from sys.argv if present (before HfArgumentParser)
        if "--ablate_latent" in sys.argv:
            idx = sys.argv.index("--ablate_latent")
            sys.argv.pop(idx)  # Remove --ablate_latent
            if idx < len(sys.argv) and not sys.argv[idx].startswith("-"):
                sys.argv.pop(idx)  # Remove ablate_latent value

        # Use HfArgumentParser for command-line arguments
        parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        )
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        # answer_only defaults to True if not set via command line (maintains current behavior)
        if answer_only is None:
            answer_only = True

        # temperature defaults to 0.1 if not set via command line
        if temperature is None:
            temperature = 0.1

    accu_list = []
    all_detailed_results = []  # Store detailed results from all iterations
    # Use command-line values
    skip_thinking = config_args.skip_thinking
    verbalize_cot = config_args.verbalize_cot
    # answer_only was already extracted and processed in the if/else branches above
    print(f"verbalize_cot: {verbalize_cot}")
    print(f"skip_thinking: {skip_thinking}")
    print(f"answer_only: {answer_only}")
    print(f"temperature: {temperature}")
    print(f"ablate_latent: {ablate_latent}")
    for i in range(training_args.inf_num_iterations):
        accu, detailed_results = evaluation(
            model_args,
            data_args,
            training_args,
            iteration=i,
            output_prefix=output_prefix,
            skip_thinking=skip_thinking,
            verbalize_cot=verbalize_cot,
            answer_only=answer_only,
            temperature=temperature,
            ablate_latent=ablate_latent,
        )
        accu_list.append(accu)
        # Add iteration number to detailed results
        for result in detailed_results:
            result["iteration"] = i + 1
        all_detailed_results.extend(detailed_results)

    # Calculate mean and std dev if multiple iterations
    mean_acc = sum(accu_list) / len(accu_list)
    std_dev_acc = statistics.stdev(accu_list) if len(accu_list) > 1 else 0.0

    print(
        f"Average accuracy over {training_args.inf_num_iterations} sampling: {mean_acc:.2f}%"
    )
    if training_args.inf_num_iterations > 1:
        print(f"Standard deviation: {std_dev_acc:.2f}%")

    # Save results to CSV
    if csv_filename is None:
        # Auto-generate filename if not provided
        filename_parts = []
        if output_prefix:
            filename_parts.append(output_prefix)
        # Get data_name from data_args
        data_name = (
            data_args.data_names[0]
            if isinstance(data_args.data_names, list) and len(data_args.data_names) > 0
            else data_args.data_names
        )
        filename_parts.append(data_name)
        csv_filename = f"results_{'_'.join(filename_parts)}.csv"

    # Use output_dir if provided
    if hasattr(training_args, "output_dir") and training_args.output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(training_args.output_dir, exist_ok=True)
        # Join output_dir with csv_filename
        csv_filepath = os.path.join(training_args.output_dir, csv_filename)
    else:
        # Use current directory if output_dir not provided
        csv_filepath = csv_filename

    # Write CSV with accuracy values
    with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        if training_args.inf_num_iterations > 1:
            # Multiple iterations: save each accuracy, mean, and std dev
            fieldnames = ["iteration", "accuracy"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, acc in enumerate(accu_list):
                writer.writerow({"iteration": i + 1, "accuracy": f"{acc:.2f}"})
            writer.writerow({"iteration": "mean", "accuracy": f"{mean_acc:.2f}"})
            writer.writerow({"iteration": "std_dev", "accuracy": f"{std_dev_acc:.2f}"})
        else:
            # Single iteration: save only accuracy
            fieldnames = ["accuracy"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"accuracy": f"{accu_list[0]:.2f}"})

    print(f"Results saved to {csv_filepath}")

    # Save detailed results (answer, ground truth, correctness) to separate CSV
    detailed_csv_filename = (
        csv_filename.replace(".csv", "_detailed.csv")
        if csv_filename.endswith(".csv")
        else f"{csv_filename}_detailed.csv"
    )
    if hasattr(training_args, "output_dir") and training_args.output_dir:
        detailed_csv_filepath = os.path.join(
            training_args.output_dir, detailed_csv_filename
        )
    else:
        detailed_csv_filepath = detailed_csv_filename

    with open(detailed_csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["iteration", "question", "answer", "ground_truth", "is_correct"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_detailed_results:
            writer.writerow(
                {
                    "iteration": result["iteration"],
                    "question": result["question"],
                    "answer": result["answer"],
                    "ground_truth": result["ground_truth"],
                    "is_correct": result["is_correct"],
                }
            )

    print(f"Detailed results saved to {detailed_csv_filepath}")
