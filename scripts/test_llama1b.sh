DATASET_NAME="multi-arith"
SAVE_DIR="results_answer_only/$DATASET_NAME"
CKPT_DIR="bcywinski/codi_llama1b-answer_only"

python test.py \
	--data_names "$DATASET_NAME" \
	--output_dir "$SAVE_DIR" \
	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
	--seed 11 \
	--model_max_length 512 \
	--bf16 \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--batch_size 128 \
	--greedy False \
	--num_latent 6 \
	--use_prj True \
	--prj_dim 2048 \
	--prj_no_ln False \
	--prj_dropout 0.0 \
	--inf_latent_iterations 6 \
	--inf_num_iterations 3 \
	--remove_eos False \
	--use_lora True \
	--answer_only True \
	--temperature 1.0 \
	--ckpt_dir "$CKPT_DIR" \
	--csv_filename latent_cot_mean.csv \
	--ablate_latent mean  # Options: zero, random, mean
	# --skip_thinking
	# --verbalize_cot

# python test.py \
# 	--data_names "$DATASET_NAME" \
# 	--output_dir "$SAVE_DIR" \
# 	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
# 	--seed 11 \
# 	--model_max_length 512 \
# 	--bf16 \
# 	--lora_r 128 --lora_alpha 32 --lora_init \
# 	--batch_size 128 \
# 	--greedy False \
# 	--num_latent 6 \
# 	--use_prj True \
# 	--prj_dim 2048 \
# 	--prj_no_ln False \
# 	--prj_dropout 0.0 \
# 	--inf_latent_iterations 6 \
# 	--inf_num_iterations 1 \
# 	--remove_eos False \
# 	--use_lora True \
# 	--answer_only True \
# 	--temperature 1.0 \
# 	--ckpt_dir "$CKPT_DIR" \
# 	--csv_filename no_cot.csv \
# 	--skip_thinking
# 	# --verbalize_cot
# 	# --ablate_latent zero  # Options: zero, random, mean


# python test.py \
# 	--data_names "$DATASET_NAME" \
# 	--output_dir "$SAVE_DIR" \
# 	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
# 	--seed 11 \
# 	--model_max_length 512 \
# 	--bf16 \
# 	--lora_r 128 --lora_alpha 32 --lora_init \
# 	--batch_size 128 \
# 	--greedy False \
# 	--num_latent 6 \
# 	--use_prj True \
# 	--prj_dim 2048 \
# 	--prj_no_ln False \
# 	--prj_dropout 0.0 \
# 	--inf_latent_iterations 6 \
# 	--inf_num_iterations 3 \
# 	--remove_eos False \
# 	--use_lora True \
# 	--answer_only True \
# 	--temperature 1.0 \
# 	--ckpt_dir "$CKPT_DIR" \
# 	--csv_filename verbal_cot.csv \
# 	--verbalize_cot
# 	# --ablate_latent zero  # Options: zero, random, mean
