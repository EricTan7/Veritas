export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export USE_AUG=false
export MASTER_PORT=12352

RUN_NAME=$1
DATASET=$2


swift rlhf \
    --rlhf_type dpo \
    --model /path/to/model \
    --dataset ${DATASET} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --train_type lora \
    --freeze_vit false \
    --lora_rank 64 \
    --lora_alpha 128 \
    --torch_dtype bfloat16 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --eval_strategy  "no" \
    --save_steps 50 \
    --save_total_limit 50 \
    --save_only_model true \
    --logging_steps 10 \
    --max_length 4096 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --output_dir ./mipo/internvl3-8b-${RUN_NAME} \
    --dataloader_num_workers 8 \
    --dataset_num_proc 16 \
    --logging_dir ./internvl3-8b-${RUN_NAME} \
    --rpo_alpha 1.0 \
    --beta 0.0
