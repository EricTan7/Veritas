### in-domain
for dataset in id/FaceForensics++ id/facevid2vid_ff id/Hallo2_test id/StyleGAN id/Midjourney
do
    python self_scripts/infer/infer_vllm.py \
    --data_file ${dataset}
done


### cross-model
for dataset in cm/AdobeFirefly cm/Flux11Pro cm/MAGI cm/HART cm/Infinity
do
    python self_scripts/infer/infer_vllm.py \
    --data_file ${dataset}
done


### cross-forgery
for dataset in cf/starganv2 cf/iclight cf/codeformer cf/infiniteyou cf/pulid cf/faceadapter
do
    python self_scripts/infer/infer_vllm.py \
    --data_file ${dataset}
done


### cross-domain
for dataset in cd/deepfacelab cd/infiniteyou cd/dreamina cd/hailuo cd/gpt4o cd/FFIW
do
    python self_scripts/infer/infer_vllm.py \
    --data_file ${dataset}
done

