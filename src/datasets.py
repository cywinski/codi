## datasets.py
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset

from datasets import load_dataset

IGNORE_INDEX = -100
ANS_TOKEN = "<ans>"  # Define the special answer token


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=256,  # training_args.model_max_length,
            truncation=True,
            return_attention_mask=False,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    segment = [sentence]
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r"-?\d+\.?\d*", pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError:
            pred_answer = float("inf")
    return pred_answer


def get_answer_token_position(tokens, answer_token_id, tokenizer):
    """
    Finds the position of the <ans> token in the token sequence.
    """
    try:
        # Find indices where the token equals the answer_token_id
        match_indices = (tokens == answer_token_id).nonzero(as_tuple=True)[0]
        if len(match_indices) > 0:
            # Return the first occurrence
            return match_indices[0].item()
        else:
            # Fallback (should not happen if data is formatted correctly)
            return len(tokens) - 1
    except Exception:
        breakpoint()


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    answers: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    bot_id: int,
    eot_id: int,
    ans_id: int,  # Added ans_id
    remove_eos: bool,
) -> Dict:
    print("Tokenizing inputs... This may take some time...")
    sources_id = _tokenize_fn(sources, tokenizer)["input_ids"]
    cot_id = _tokenize_fn(targets, tokenizer)["input_ids"]
    answers_id = _tokenize_fn(answers, tokenizer)["input_ids"]

    # add eos token to accomodate pretrained model's format
    if not remove_eos:
        sources_id = [
            torch.tensor(
                x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long
            )
            for x in sources_id
        ]
        cot_id = [
            torch.tensor(
                x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long
            )
            for x in cot_id
        ]
    answers_id = [
        torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long)
        for x in answers_id
    ]

    if cot_id[0][0] == tokenizer.bos_token_id:
        cot_id = [x[1:] for x in cot_id]
        answers_id = [x[1:] for x in answers_id]

    # Reference Input (Teacher): Question + CoT + <ans> + Answer
    ref_input_ids = [
        torch.cat([x, y, z]).to(torch.long)
        for x, y, z in zip(sources_id, cot_id, answers_id)
    ]
    ref_labels = []
    for x, y in zip(ref_input_ids, sources_id):
        z = x.clone()
        z[: len(y)] = -100
        ref_labels.append(z)

    # Student Encoder Input: Question + <bot>
    sources_id = [
        torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long)
        for x in sources_id
    ]

    # Student Decoder Input (Labels): <ans> + Answer
    # Note: answers string already starts with <ans> due to SupervisedDataset formatting,
    # so answers_id[0] should be <ans> (or close to it depending on tokenizer).
    
    # We locate the specific <ans> token ID to be sure where alignment happens
    ref_answer_position = [
        get_answer_token_position(x, ans_id, tokenizer)
        for i, x in enumerate(ref_input_ids)
    ]
    model_answer_position = [
        get_answer_token_position(x, ans_id, tokenizer) for x in answers_id
    ]

    ref_eos_position = [len(x) - 1 for x in ref_input_ids]
    model_eos_position = [len(x) - 1 for x in answers_id]
    
    return dict(
        encoder_input_ids=sources_id,
        decoder_input_ids=answers_id,
        ref_input_ids=ref_input_ids,
        labels=answers_id,
        ref_answer_position=ref_answer_position,
        model_answer_position=model_answer_position,
        ref_eos_position=ref_eos_position,
        model_eos_position=model_eos_position,
        ref_labels=ref_labels,
    )


class SupervisedDataset(Dataset):
    QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
    QUESTION_DA_PROMPT = (
        "\nAnswer the above question. Answer the final number directly in one number.\n"
    )

    def __init__(
        self,
        data_name,
        raw_data,
        tokenizer,
        bot,
        eot,
        ans,  # Added ans token ID
        training_args,
        data_args,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Formatting inputs...")

        self.data_name = data_name
        questions, cots, answers = [], [], []
        token_nums = []

        for num_iter, example in enumerate(raw_data):
            if training_args.exp_mode and num_iter > training_args.exp_data_num:
                break
            if (
                data_args.max_samples is not None
                and len(questions) >= data_args.max_samples
            ):
                break
            
            question = f"{example['question']}"
            
            # GSM8k-Aug-NL / icot-full
            if "icot" in self.data_name and "full" in self.data_name:
                if example["answer"] is None:
                    continue

                token_num = len(
                    tokenizer.encode(
                        example["question"] + example["cot"] + example["answer"]
                    )
                )
                if token_num > training_args.max_token_num:
                    continue

                cot = f"{example['cot']}".split(". ")
                if not (training_args.include_last_cot):
                    cot = cot[:-1]

                answer = example["answer"].split(" ")[-1]
                if not answer[0].isdigit():
                    continue
                
                # CHANGED: Use <ans> token instead of natural language prompt
                answer = f"{ANS_TOKEN} {answer}"
                answer = answer.replace("####", "")
                
                questions.append(question)

                if cot:
                    cot = ". ".join(cot) + ".\n"
                else:
                    cot = ""
                cots.append(cot)
                answers.append(answer)
                token_nums.append(token_num)

            # GSM8k-Aug / icot
            elif "icot" in self.data_name:
                token_num = len(
                    tokenizer.encode(
                        example["question"] + example["cot"] + example["answer"]
                    )
                )
                if token_num > training_args.max_token_num:
                    continue

                cot_list = []
                cot = f"{example['cot']}".split(" ")
                if not training_args.include_last_cot:
                    cot = cot[:-1]

                len_cot = len(cot)
                for i in range(training_args.num_latent):
                    cot_list.append(" ".join(cot[: max(0, len_cot - i)]))
                
                answer = example["answer"].split(" ")[-1]
                if not answer[0].isdigit():
                    continue

                # CHANGED: Use <ans> token
                answer = f"{ANS_TOKEN} {answer}"
                answer = answer.replace("####", "")
                
                questions.append(question)
                cots.append(" ".join(cot))
                answers.append(answer)
                token_nums.append(token_num)
            
            # Other datasets (commonsense, strategy, prontoqa)
            elif "commonsense" in self.data_name or "strategy" in self.data_name or "prontoqa" in self.data_name:
                if "prontoqa" in self.data_name:
                    cot = "\n".join(example["steps"][:-1]) + "\n"
                else:
                    cot = example["cot"].strip() + "\n"
                
                question = example["question"].strip() + "\n"
                # CHANGED: Use <ans> token
                answer = f"{ANS_TOKEN} {str(example['answer']).strip()}"

                token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                if token_num > training_args.max_token_num:
                    continue
                questions.append(question)
                cots.append(cot)
                answers.append(answer)
                token_nums.append(token_num)
            else:
                raise NotImplementedError

        if training_args.exp_mode:
            questions = questions[: training_args.exp_data_num]
            cots = cots[: training_args.exp_data_num]
            answers = answers[: training_args.exp_data_num]
            token_nums = token_nums[: training_args.exp_data_num]

        print(f"{len(cots)} data in total...")
        logging.warning("Tokenizing inputs... This may take some time...")

        self.data_dict = preprocess(
            questions, cots, answers, tokenizer, bot, eot, ans, training_args.remove_eos
        )
        self.keys = list(self.data_dict.keys())

        # Print total number of training tokens
        total_tokens = sum(token_nums)
        print(f"Total number of training tokens: {total_tokens:,}")

    def __len__(self):
        return len(self.data_dict["encoder_input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {key: self.data_dict[key][i] for key in self.keys}


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        (
            encoder_input_ids,
            decoder_input_ids,
            ref_input_ids,
            labels,
            ref_answer_position,
            model_answer_position,
            ref_labels,
        ) = tuple(
            [instance[key] for instance in instances]
            for key in (
                "encoder_input_ids",
                "decoder_input_ids",
                "ref_input_ids",
                "labels",
                "ref_answer_position",
                "model_answer_position",
                "ref_labels",
            )
        )

        # pad left
        reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
        encoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            reversed_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(1)

        # pad
        ref_input_ids = torch.nn.utils.rnn.pad_sequence(
            ref_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        ref_labels = torch.nn.utils.rnn.pad_sequence(
            ref_labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            ref_input_ids=ref_input_ids,
            labels=labels,
            encoder_attention_mask=encoder_input_ids.ne(self.tokenizer.pad_token_id),
            ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
            model_answer_position=torch.tensor(model_answer_position, dtype=torch.long),
            ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
            ref_labels=ref_labels,
        )


def concatenate_datasets(datasets: list) -> SupervisedDataset:
    """Concatenate multiple SupervisedDataset instances into one."""
    if not datasets:
        raise ValueError("Cannot concatenate empty list of datasets")

    if len(datasets) == 1:
        return datasets[0]

    keys = datasets[0].keys

    for dataset in datasets[1:]:
        if dataset.keys != keys:
            raise ValueError(
                f"Datasets have different keys. First: {keys}, Current: {dataset.keys}"
            )

    concatenated_data_dict = {}
    for key in keys:
        concatenated_data_dict[key] = []
        for dataset in datasets:
            concatenated_data_dict[key].extend(dataset.data_dict[key])

    concatenated_dataset = SupervisedDataset.__new__(SupervisedDataset)
    concatenated_dataset.data_dict = concatenated_data_dict
    concatenated_dataset.keys = keys
    concatenated_dataset.data_name = "+".join([ds.data_name for ds in datasets])

    total_length = sum(len(dataset) for dataset in datasets)
    print(f"Concatenated {len(datasets)} datasets: {total_length} samples in total")

    return concatenated_dataset


def load_single_dataset(
    data_name: str, tokenizer, model, training_args, data_args
) -> SupervisedDataset:
    """Load a single dataset by name and return a SupervisedDataset instance."""
    logging.warning(f"Loading dataset: {data_name}")

    if "icot" in data_name:
        if "full" in data_name:
            dataset = load_dataset("zen-E/GSM8k-Aug-NL")["train"]
        else:
            dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
        
        train_dataset = SupervisedDataset(
            data_name=data_name,
            raw_data=dataset,
            tokenizer=tokenizer,
            bot=model.bot_id,
            eot=model.eot_id,
            ans=model.ans_id,  # Pass the ANS token ID
            training_args=training_args,
            data_args=data_args,
        )
        return train_dataset
    elif "strategy" in data_name:
        dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
        train_dataset = SupervisedDataset(
            data_name=data_name,
            raw_data=dataset,
            tokenizer=tokenizer,
            bot=model.bot_id,
            eot=model.eot_id,
            ans=model.ans_id,
            training_args=training_args,
            data_args=data_args,
        )
        return train_dataset
    elif "commonsense" in data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
        train_dataset = SupervisedDataset(
            data_name=data_name,
            raw_data=dataset,
            tokenizer=tokenizer,
            bot=model.bot_id,
            eot=model.eot_id,
            ans=model.ans_id,
            training_args=training_args,
            data_args=data_args,
        )
        return train_dataset
    elif "prontoqa" in data_name:
        with open("/home/ubuntu/coconut/data/prontoqa_train.json") as f:
            dataset = json.load(f)
        train_dataset = SupervisedDataset(
            data_name=data_name,
            raw_data=dataset,
            tokenizer=tokenizer,
            bot=model.bot_id,
            eot=model.eot_id,
            ans=model.ans_id,
            training_args=training_args,
            data_args=data_args,
        )
        return train_dataset
    else:
        raise NotImplementedError(f"Dataset {data_name} is not supported.")


def make_supervised_data_module(tokenizer, data_args, model, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")

    if not data_args.data_names:
        raise ValueError("data_names must be a non-empty list of dataset names")

    datasets = []
    for data_name in data_args.data_names:
        dataset = load_single_dataset(
            data_name, tokenizer, model, training_args, data_args
        )
        datasets.append(dataset)

    train_dataset = concatenate_datasets(datasets)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )