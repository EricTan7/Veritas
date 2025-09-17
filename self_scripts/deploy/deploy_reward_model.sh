model_path=/path/to/UnifiedReward-qwen-3b
CUDA_VISIBLE_DEVICES=6,7 vllm serve ${model_path} \
--port 8003 \
--host 0.0.0.0 \
--dtype bfloat16 \
--tensor-parallel-size 2 \
--pipeline-parallel-size 1 \
--limit-mm-per-prompt image=5
