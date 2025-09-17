export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MASTER_PORT=12354


MODEL_PATH=$1


### in-domain
for dataset in id_ff id_facevid2vid id_Hallo2 id_StyleGAN id_Midjourney
do
    swift infer \
        --val_dataset ${dataset} \
        --model ${MODEL_PATH} \
        --infer_backend pt \
        --max_model_len 8192 \
        --max_new_tokens 2048 \
        --max_batch_size 8 \
        --metric self_acc_tags \
        --dataset_num_proc 16
done


### cross-model
for dataset in cm_AdobeFirefly cm_Flux11Pro cm_StarryAI cm_MAGI cm_hart cm_Infinity
do
    swift infer \
        --val_dataset ${dataset} \
        --model ${MODEL_PATH} \
        --infer_backend pt \
        --max_model_len 8192 \
        --max_new_tokens 2048 \
        --max_batch_size 8 \
        --metric self_acc_tags \
        --dataset_num_proc 16
done


### cross-forgery
for dataset in cf_starganv2 cf_iclight cf_codeformer cf_infiniteyou cf_pulid cf_faceadapter
do
    swift infer \
        --val_dataset ${dataset} \
        --model ${MODEL_PATH} \
        --infer_backend pt \
        --max_model_len 8192 \
        --max_new_tokens 2048 \
        --max_batch_size 8 \
        --metric self_acc_tags \
        --dataset_num_proc 16
done


### cross-domain
for dataset in cd_deepfacelab cd_infiniteyou cd_dreamina cd_hailuo cd_gpt4o cd_FFIW
do
    swift infer \
        --val_dataset ${dataset} \
        --model ${MODEL_PATH} \
        --infer_backend pt \
        --max_model_len 8192 \
        --max_new_tokens 2048 \
        --max_batch_size 8 \
        --metric self_acc_tags \
        --dataset_num_proc 16
done

