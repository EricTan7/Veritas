export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NPROC_PER_NODE=6
export USE_AUG=false

RUN_NAME=$1
DATASET=$2


swift rlhf \
    --rlhf_type grpo \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs patternacc unifiedprm multi_reason_format \
    --reward_weights 1.0 1.0 0.25 \
    --model /path/to/model \
    --dataset ${DATASET}_rl \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --use_vllm true \
    --vllm_device auto \
    --vllm_max_model_len 8192 \
    --num_infer_workers 4 \
    --train_type lora \
    --freeze_vit false \
    --lora_rank 128 \
    --lora_alpha 256 \
    --torch_dtype bfloat16 \
    --num_generations 4 \
    --temperature 1.0 \
    --deepspeed zero2 \
    --log_completions true \
    --learning_rate 1e-6 \
    --weight_decay 0.01 \
    --eval_strategy  "no" \
    --save_steps 100 \
    --save_total_limit 50 \
    --save_only_model true \
    --logging_steps 10 \
    --max_length 4096 \
    --max_completion_length 4096 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --output_dir ./grpo/internvl3-8b-${RUN_NAME} \
    --dataloader_num_workers 8 \
    --dataset_num_proc 16 \
    --logging_dir ./internvl3-8b-${RUN_NAME} \
    --beta 0.0
