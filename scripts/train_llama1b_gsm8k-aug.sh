CONFIG_PATH="configs/llama1b_gsm8k-aug.yaml"

# For single GPU: python train.py "$CONFIG_PATH"
# For multi-GPU (e.g., 4 GPUs): torchrun --nproc_per_node=4 train.py "$CONFIG_PATH"

# Auto-detect number of GPUs and use torchrun if multiple GPUs available
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Detected $NUM_GPUS GPUs. Using torchrun for distributed training."
    torchrun --nproc_per_node="$NUM_GPUS" train.py "$CONFIG_PATH"
else
    echo "Using single GPU training."
    python train.py "$CONFIG_PATH"
fi
