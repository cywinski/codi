import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset

IGNORE_INDEX = -100


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


def get_answer_token_position(tokens, answer_prompts, tokenizer):
    # answer_prompt = torch.tensor([464, 3280, 318, 25])
    try:
        match_indices = (
            (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0])
            .all(dim=1)
            .nonzero(as_tuple=True)[0]
            .item()
        )
        answer_token_id = match_indices + len(answer_prompts[0])
        return answer_token_id
    except Exception:
        breakpoint()


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    answers: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    bot_id: int,
    eot_id: int,
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
        torch.tensor(
            x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long
        )
        for x in answers_id
    ]

    if cot_id[0][0] == tokenizer.bos_token_id:
        cot_id = [x[1:] for x in cot_id]
        answers_id = [x[1:] for x in answers_id]

    ref_input_ids = [
        torch.cat([x, y, z]).to(torch.long)
        for x, y, z in zip(sources_id, cot_id, answers_id)
    ]
    ref_labels = []
    for x, y in zip(ref_input_ids, sources_id):
        z = x.clone()
        z[: len(y)] = -100
        ref_labels.append(z)

    # add eot to source
    sources_id = [
        torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long)
        for x in sources_id
    ]
    # add eot and eos
    if remove_eos:
        answers_id = [
            torch.tensor([eot_id] + x.numpy().tolist(), dtype=torch.long)
            for x in answers_id
        ]
    else:
        answers_id = [
            torch.tensor(
                [eot_id, tokenizer.eos_token_id] + x.numpy().tolist(),
                dtype=torch.long,
            )
            for x in answers_id
        ]

    answer_prompts = [
        torch.tensor(tokenizer.encode("The answer is:")),
        torch.tensor(tokenizer.encode("The next step result is:")),
    ]
    if answer_prompts[0][0] == tokenizer.bos_token_id:  # remove the bos
        answer_prompts[0] = answer_prompts[0][1:]
        answer_prompts[1] = answer_prompts[1][1:]

    ref_answer_position = [
        get_answer_token_position(x, answer_prompts, tokenizer)
        for i, x in enumerate(ref_input_ids)
    ]
    model_answer_position = [
        get_answer_token_position(x, answer_prompts, tokenizer) for x in answers_id
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
    QUESTION_DA_PROMPT = "\nAnswer the above question. Answer the final number directly in one number.\n"

    def __init__(
        self,
        data_name,
        raw_data,
        tokenizer,
        bot,
        eot,
        training_args,
        data_args,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Formatting inputs...")

        self.data_name = data_name
        questions, cots, answers = [], [], []
        num_ops_list = []
        operators = ["+", "-", "*", "/"]

        token_nums = []
        for num_iter, example in enumerate(raw_data):
            if training_args.exp_mode and num_iter > training_args.exp_data_num:
                break
            question = f"{example['question']}"
            if (
                "icot" in self.data_name and "full" in self.data_name
            ):  # icot-full (GSM8k-Aug-NL)
                # bad data
                if example["answer"] is None:  # or example["response"] is None:
                    continue

                # avoid OOM: remove very long data
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
                answer = f"The answer is: {answer}"
                answer = answer.replace("####", "")
                questions.append(question)

                if cot:
                    cot = ". ".join(cot) + ".\n"
                else:
                    cot = ""
                cots.append(cot)
                answers.append(answer)
            elif "icot" in self.data_name:  # icot (GSM8k-Aug)
                # avoid OOM: remove very long data
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

                # some answers startwith the negative sign (-), bringing distillation problems for LLaMA
                if not answer[0].isdigit():
                    continue

                answer = f"The answer is: {answer}"
                answer = answer.replace("####", "")
                questions.append(question)
                cots.append(" ".join(cot))
                answers.append(answer)
            elif "commonsense" in self.data_name or "strategy" in self.data_name:
                question = example["question"].strip() + "\n"
                cot = example["cot"].strip() + "\n"
                answer = f"The answer is: {str(example['answer']).strip()}"

                # avoid OOM: remove very long data
                token_num = len(
                    tokenizer.encode(question + " " + cot + " " + answer)
                )
                if token_num > training_args.max_token_num:
                    continue
                questions.append(question)
                cots.append(cot)
                answers.append(answer)
            elif "prontoqa" in data_args.data_name:
                question = example["question"].strip() + "\n"
                cot = "\n".join(example["steps"][:-1]) + "\n"
                answer = f"The answer is: {str(example['answer']).strip()}"

                # avoid OOM: remove very long data
                token_num = len(
                    tokenizer.encode(question + " " + cot + " " + answer)
                )
                if token_num > training_args.max_token_num:
                    continue
                questions.append(question)
                cots.append(cot)
                answers.append(answer)
            else:
                raise NotImplementedError
        if training_args.exp_mode:
            questions = questions[: training_args.exp_data_num]
            cots = cots[: training_args.exp_data_num]
            answers = answers[: training_args.exp_data_num]

        print(f"{len(cots)} data in total...")
        logging.warning("Tokenizing inputs... This may take some time...")

        self.data_dict = preprocess(
            questions, cots, answers, tokenizer, bot, eot, training_args.remove_eos
        )
        self.keys = list(self.data_dict.keys())

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
            encoder_attention_mask=encoder_input_ids.ne(
                self.tokenizer.pad_token_id
            ),
            ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
            model_answer_position=torch.tensor(
                model_answer_position, dtype=torch.long
            ),
            ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
            ref_labels=ref_labels,
        )


def make_supervised_data_module(tokenizer, data_args, model, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    if "icot" in data_args.data_name:
        if "full" in data_args.data_name:
            dataset = load_dataset("zen-E/GSM8k-Aug-NL")["train"]
        else:
            dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
        train_dataset = SupervisedDataset(
            data_name=data_args.data_name,
            raw_data=dataset,
            tokenizer=tokenizer,
            bot=model.bot_id,
            eot=model.eot_id,
            training_args=training_args,
            data_args=data_args,
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )
    elif "strategy" in data_args.data_name:
        dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
        train_dataset = SupervisedDataset(
            data_name=data_args.data_name,
            raw_data=dataset,
            tokenizer=tokenizer,
            bot=model.bot_id,
            eot=model.eot_id,
            training_args=training_args,
            data_args=data_args,
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )
    elif "commonsense" in data_args.data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
        train_dataset = SupervisedDataset(
            data_name=data_args.data_name,
            raw_data=dataset,
            tokenizer=tokenizer,
            bot=model.bot_id,
            eot=model.eot_id,
            training_args=training_args,
            data_args=data_args,
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )
    elif "prontoqa" in data_args.data_name:
        with open("/home/ubuntu/coconut/data/prontoqa_train.json") as f:
            dataset = json.load(f)
        train_dataset = SupervisedDataset(
            data_name=data_args.data_name,
            raw_data=dataset,
            tokenizer=tokenizer,
            bot=model.bot_id,
            eot=model.eot_id,
            training_args=training_args,
            data_args=data_args,
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )
    else:
        raise NotImplementedError(
            f"Dataset {data_args.data_name} is not supported."
        )
