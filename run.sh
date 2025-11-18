#!/bin/bash

torchrun --nproc_per_node=4 train.py configs/llama3b_gsm8k-aug-nl.yaml
torchrun --nproc_per_node=4 train.py configs/llama1b_gsm8k-aug-nl.yaml
torchrun --nproc_per_node=4 train_cot_sft.py configs/cot_sft_llama1b_gsm8k-aug-nl.yaml
torchrun --nproc_per_node=4 train_cot_sft.py configs/cot_sft_llama3b_gsm8k-aug-nl.yaml
