model_path=/path/to/your/model
MODEL_PATH=$1
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ${MODEL_PATH} \
--port 8000 \
--host 0.0.0.0 \
--dtype bfloat16 \
--tensor-parallel-size 4 \
--pipeline-parallel-size 1 \
--limit-mm-per-prompt image=5,video=2
