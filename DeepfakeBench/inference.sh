# inference and visualize  0226
# 第一组: coop下，无freqfit / freqfit，face / general
CUDA_VISIBLE_DEVICES=6 python training/test.py \
--detector_cfg ./training/config/detector/clip_freq.yaml \
--dataset_cfg ./training/config/dataset/general_unifd_hd.yaml \
--weights_path /homedata/tanhao/logs/AIGCD/debug/AIGCD-general_unifd_freqfit-2d-wossf_lora_coop/clip_freq_ViT-L-14_2025-02-25-10-40-26/val/avg/ckpt_best.pth \
--wandb_tags freqfit-2d-wossf_lora_coop model_config.freqfit_type 2d-wossf \
save_feat True save_ckpt False \
model_config.forward_type coop 
CUDA_VISIBLE_DEVICES=7 python training/test.py \
--detector_cfg ./training/config/detector/clip_freq.yaml \
--dataset_cfg ./training/config/dataset/general_unifd_hd.yaml \
--weights_path /homedata/tanhao/logs/AIGCD/debug/AIGCD-general_unifd_lora_coop/clip_freq_ViT-L-14_2025-02-25-10-40-23/val/avg/ckpt_best.pth \
--wandb_tags lora_coop model_config.use_freqfit False \
save_feat True save_ckpt False \
model_config.forward_type coop

CUDA_VISIBLE_DEVICES=6 python training/test.py \
--detector_cfg ./training/config/detector/clip_freq.yaml \
--dataset_cfg ./training/config/dataset/face_common_hd.yaml \
--weights_path /homedata/tanhao/logs/AIGCD/debug/AIGCD-face_common_freqfit-2d-wossf_lora_coop/clip_freq_ViT-L-14_2025-02-25-10-40-22/val/avg/ckpt_best.pth \
--wandb_tags freqfit-2d-wossf_lora_coop model_config.freqfit_type 2d-wossf \
save_feat True save_ckpt False \
model_config.forward_type coop
CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/clip_freq.yaml \
--dataset_cfg ./training/config/dataset/face_common_hd.yaml \
--weights_path /homedata/tanhao/logs/AIGCD/debug/AIGCD-face_common_freqfit-2d_lora_coop/clip_freq_ViT-L-14_2025-02-25-10-40-20/val/avg/ckpt_best.pth \
--wandb_tags freqfit-2d_lora_coop model_config.freqfit_type 2d \
save_feat True save_ckpt False \
model_config.forward_type coop
CUDA_VISIBLE_DEVICES=7 python training/test.py \
--detector_cfg ./training/config/detector/clip_freq.yaml \
--dataset_cfg ./training/config/dataset/face_common_hd.yaml \
--weights_path /homedata/tanhao/logs/AIGCD/debug/AIGCD-face_common_lora_coop/clip_freq_ViT-L-14_2025-02-25-10-40-19/val/avg/ckpt_best.pth \
--wandb_tags lora_coop model_config.use_freqfit False \
save_feat True save_ckpt False \
model_config.forward_type coop

# 第二组: lp下。无freqfit / freqfit / freqfit-wossf，face / general（目前只训了face）
CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/clip_freq.yaml \
--dataset_cfg ./training/config/dataset/face_common_hd.yaml \
--weights_path /homedata/tanhao/logs/AIGCD/debug/AIGCD-face_common_freqfit-2d-wossf_lora_lp/clip_freq_ViT-L-14_2025-02-25-10-40-18/val/avg/ckpt_best.pth \
--wandb_tags freqfit-2d-wossf_lora_coop model_config.freqfit_type 2d-wossf \
save_feat True save_ckpt False
CUDA_VISIBLE_DEVICES=2 python training/test.py \
--detector_cfg ./training/config/detector/clip_freq.yaml \
--dataset_cfg ./training/config/dataset/face_common_hd.yaml \
--weights_path /homedata/tanhao/logs/AIGCD/debug/AIGCD-face_common_freqfit-2d_lora_lp/clip_freq_ViT-L-14_2025-02-25-10-40-15/val/avg/ckpt_best.pth \
--wandb_tags freqfit-2d_lora_lp model_config.freqfit_type 2d \
save_feat True save_ckpt False
CUDA_VISIBLE_DEVICES=3 python training/test.py \
--detector_cfg ./training/config/detector/clip_freq.yaml \
--dataset_cfg ./training/config/dataset/face_common_hd.yaml \
--weights_path /homedata/tanhao/logs/AIGCD/debug/AIGCD-face_common_lora_lp/clip_freq_ViT-L-14_2025-02-25-10-40-13/val/avg/ckpt_best.pth \
--wandb_tags lora_lp model_config.use_freqfit False \
save_feat True save_ckpt False





#### inference
# real ff_dfdcp
CUDA_VISIBLE_DEVICES=7 python training/test.py \
--detector_cfg ./training/config/detector/convnextv2_peft.yaml \
--dataset_cfg ./training/config/dataset/df40_30k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/convnextv2/convnextv2_base/lora_linear_train_30k_2025-05-11-14-47-51/val/avg/ckpt_e6.pth \
save_feat True
# real ff_cdf
CUDA_VISIBLE_DEVICES=6 python training/test.py \
--detector_cfg ./training/config/detector/convnextv2_peft.yaml \
--dataset_cfg ./training/config/dataset/df40_30k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/convnextv2/convnextv2_base/lora_linear_train_real_ff_cdf_30k_2025-05-11-21-03-07/val/avg/ckpt_e8.pth \
save_feat True
# real dfdcp
CUDA_VISIBLE_DEVICES=5 python training/test.py \
--detector_cfg ./training/config/detector/convnextv2_peft.yaml \
--dataset_cfg ./training/config/dataset/df40_30k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/convnextv2/convnextv2_base/lora_linear_train_real_dfdcp_30k_2025-05-11-14-47-51/val/avg/ckpt_e8.pth \
save_feat True
# fake blendface
CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/convnextv2_peft.yaml \
--dataset_cfg ./training/config/dataset/df40_30k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/convnextv2/convnextv2_base/lora_linear_train_fake_blendface_30k_2025-05-12-16-03-51/val/avg/ckpt_e7.pth \
save_feat True
# fake FS
CUDA_VISIBLE_DEVICES=4 python training/test.py \
--detector_cfg ./training/config/detector/convnextv2_peft.yaml \
--dataset_cfg ./training/config/dataset/df40_30k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/convnextv2/convnextv2_base/lora_linear_train_fake_FS_30k_2025-05-12-16-07-57/val/avg/ckpt_e10.pth \
save_feat True
# real丰富
CUDA_VISIBLE_DEVICES=4 python training/test.py \
--detector_cfg ./training/config/detector/convnextv2_peft.yaml \
--dataset_cfg ./training/config/dataset/df40_30k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/convnextv2/convnextv2_base/lora_linear_train_real_v1_30k_aug_2025-05-14-11-52-00/val/avg/ckpt_e7.pth \
save_feat True
# real ff
CUDA_VISIBLE_DEVICES=4 python training/test.py \
--detector_cfg ./training/config/detector/convnextv2_peft.yaml \
--dataset_cfg ./training/config/dataset/df40_30k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/convnextv2/convnextv2_base/lora_linear_train_real_ff_30k_2025-05-11-21-35-44/val/avg/ckpt_e7.pth \
save_feat True
# fake hyper  +aug
CUDA_VISIBLE_DEVICES=4 python training/test.py \
--detector_cfg ./training/config/detector/convnextv2_peft.yaml \
--dataset_cfg ./training/config/dataset/df40_30k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/convnextv2/convnextv2_base/lora_linear_train_fake_hyperreenact_30k_aug_2025-05-12-17-25-56/val/avg/ckpt_e8.pth \
save_feat True


# CLIP
# real ff_dfdcp
CUDA_VISIBLE_DEVICES=3 python training/test.py \
--detector_cfg ./training/config/detector/clip_peft.yaml \
--dataset_cfg ./training/config/dataset/df40_30k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip/clip-vit-base-patch16/lora_linear_train_30k_2025-05-11-14-47-51/val/avg/ckpt_e7.pth \
save_feat True
# real ff_cdf
CUDA_VISIBLE_DEVICES=2 python training/test.py \
--detector_cfg ./training/config/detector/clip_peft.yaml \
--dataset_cfg ./training/config/dataset/df40_30k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip/clip-vit-base-patch16/lora_linear_train_real_ff_cdf_30k_2025-05-11-21-03-06/val/avg/ckpt_e9.pth \
save_feat True
# real dfdcp
CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/clip_peft.yaml \
--dataset_cfg ./training/config/dataset/df40_30k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip/clip-vit-base-patch16/lora_linear_train_real_dfdcp_30k_2025-05-11-14-47-51/val/avg/ckpt_e6.pth \
save_feat True




## benchv1
CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/clip_peft.yaml \
--dataset_cfg ./training/config/dataset/benchv1_50k_selectedv1_20k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip/clip-vit-base-patch16/lora_linear_50k_selectedv1_20k2025-06-26-10-44-28/val/avg/ckpt_e7.pth \
data_aug.test_resize resize

CUDA_VISIBLE_DEVICES=7 python training/test.py \
--detector_cfg ./training/config/detector/convnextv2_peft.yaml \
--dataset_cfg ./training/config/dataset/benchv1_50k_selectedv1_20k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/convnextv2/convnextv2_base/lora_linear_50k_selectedv1_20k2025-06-26-10-44-32/val/avg/ckpt_e7.pth \
data_aug.test_resize resize


## aeroblade
sudo cp /mnt/vlr/duanxian/pretrain/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/

CUDA_VISIBLE_DEVICES=3 python training/test.py \
--detector_cfg ./training/config/detector/aeroblade.yaml \
--dataset_cfg ./training/config/dataset/benchv1_50k_selectedv1_20k.yaml

## NPR
CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/npr.yaml \
--dataset_cfg ./training/config/dataset/benchv1_50k_selectedv1_20k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/NPR/subset_dm_rn50_2025_07_03_17_04_59/model_epoch_49.pth


## freqnet
CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/freqnet.yaml \
--dataset_cfg ./training/config/dataset/benchv1_50k_selectedv1_20k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/freqnet/benchv1_50k_selectedv1_20k2025_06_27_11_44_40/model_epoch_5.pth


## AIDE
CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/aide.yaml \
--dataset_cfg ./training/config/dataset/benchv1_50k_selectedv1_20k.yaml \
--weights_path /mnt/r-contentsecurity-p/common/datas_yl/duanxian/ckpt_tmp/classification/AIDE/50k_selectedv1_20k/2025-07-03-17-15-02/checkpoint-19.pth

CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/aide.yaml \
--dataset_cfg ./training/config/dataset/benchv1_50k_selectedv1_20k.yaml \
--weights_path /mnt/r-contentsecurity-p/common/datas_yl/duanxian/ckpt_tmp/classification/AIDE/50k_selectedv1_20k/2025-07-03-17-15-02/checkpoint-15.pth


for ckpt in 18 17 16 15 14 13 12 11 10
do
    CUDA_VISIBLE_DEVICES=5 python training/test.py \
    --detector_cfg ./training/config/detector/aide.yaml \
    --dataset_cfg ./training/config/dataset/benchv2_110k.yaml \
    --weights_path /mnt/r-contentsecurity-p/common/datas_yl/duanxian/ckpt_tmp/classification/AIDE/benchv2_100k/2025-07-15-21-46-45/checkpoint-${ckpt}.pth
done


CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/clip_peft.yaml \
--dataset_cfg ./training/config/dataset/benchv2_50k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip/clip-vit-large-patch14/lora_linear_subset_50k_aug/2025-07-23-23-27-44/val/avg/ckpt_e8.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14

CUDA_VISIBLE_DEVICES=2 python training/test.py \
--detector_cfg ./training/config/detector/unifd_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_50k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/unifd/clip-vit-large-patch14/freeze_all_subset_50k_aug/2025-07-23-23-27-39/val/avg/ckpt_e10.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14



CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/iid_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_36k_iid.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/iid/xception-b5690688/lora_linear_subset_36k_aug/2025-07-29-14-56-42/val/avg/ckpt_e6.pth


CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/prodet_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_36k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/prodet/efficientnet-b4-6ed6700e/lora_linear_subset_36k_aug/2025-07-29-14-36-59/val/avg/ckpt_e10.pth


CUDA_VISIBLE_DEVICES=2 python training/test.py \
--detector_cfg ./training/config/detector/d3.yaml \
--dataset_cfg ./training/config/dataset/benchv2_36k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/d3/train_d3/model_epoch_99.pth







## infer on single image
CUDA_VISIBLE_DEVICES=1 python training/infer.py \
--detector_cfg ./training/config/detector/clip_peft.yaml \
--dataset_cfg ./training/config/dataset/benchv1_50k_selectedv1_20k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip/clip-vit-large-patch14/lora_linear_subset_50k_aug/2025-07-23-23-27-44/val/avg/ckpt_e8.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14

CUDA_VISIBLE_DEVICES=1 python training/infer.py \
--detector_cfg ./training/config/detector/clip_peft.yaml \
--dataset_cfg ./training/config/dataset/benchv1_50k_selectedv1_20k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip/clip-vit-base-patch16/lora_linear_subset_50k_aug/2025-07-23-23-27-43/val/avg/ckpt_e10.pth \
data_aug.test_resize resize


CUDA_VISIBLE_DEVICES=2 python training/infer.py \
--detector_cfg ./training/config/detector/unifd_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv1_50k_selectedv1_20k.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/unifd/clip-vit-large-patch14/freeze_all_subset_50k_aug/2025-07-23-23-27-39/val/avg/ckpt_e10.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14


### Prev sota

CUDA_VISIBLE_DEVICES=4 python training/test.py \
--detector_cfg ./training/config/detector/npr.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/NPR/all_wodup_rn50_2025_08_19_18_46_49/model_epoch_49.pth

CUDA_VISIBLE_DEVICES=4 python training/test.py \
--detector_cfg ./training/config/detector/npr.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/NPR/all_wodup_rn101_2025_08_19_18_46_50/model_epoch_49.pth




CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/aide.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/r-contentsecurity-p/common/datas_yl/duanxian/ckpt_tmp/classification/AIDE/all_wodup/2025-08-20-13-37-49/checkpoint-49.pth


CUDA_VISIBLE_DEVICES=15 python training/test.py \
--detector_cfg ./training/config/detector/cospy.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/cospy/all_wodup/cospy_calibrate/epoch_4.pth


# 重跑precision、recall
d3 freqnet 1
effort unifd cospy f3net iid npr prodet AIDE
/mnt/vlr/duanxian/logs/DeepfakeDet/classification/f3net/xception-b5690688/lora_linear_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e7.pth   1
/mnt/vlr/duanxian/logs/DeepfakeDet/classification/unifd/clip-vit-large-patch14/freeze_all_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e10.pth   1
/mnt/vlr/duanxian/logs/DeepfakeDet/classification/iid/xception-b5690688/lora_linear_all_wodup_aug/2025-08-20-10-34-34/val/avg/ckpt_e7.pth   1
/mnt/vlr/duanxian/logs/DeepfakeDet/classification/prodet/efficientnet-b4-6ed6700e/lora_linear_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e8.pth   1
/mnt/vlr/duanxian/logs/DeepfakeDet/classification/NPR/all_wodup_rn50_2025_08_19_18_46_49/model_epoch_47.pth   1
/mnt/vlr/duanxian/logs/DeepfakeDet/classification/cospy/all_wodup/cospy_calibrate/epoch_4.pth   1
/mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip_effort/clip-vit-large-patch14/effort_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e6.pth   1
/mnt/r-contentsecurity-p/common/datas_yl/duanxian/ckpt_tmp/classification/AIDE/all_wodup/2025-08-20-13-37-49/checkpoint-49.pth   1
/mnt/vlr/duanxian/logs/DeepfakeDet/classification/d3/all_wodup_0820_2/model_epoch_99.pth 1


CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/f3net_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/f3net/xception-b5690688/lora_linear_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e7.pth

CUDA_VISIBLE_DEVICES=11 python training/test.py \
--detector_cfg ./training/config/detector/unifd_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/unifd/clip-vit-large-patch14/freeze_all_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e10.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14

CUDA_VISIBLE_DEVICES=12 python training/test.py \
--detector_cfg ./training/config/detector/iid_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all_iid.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/iid/xception-b5690688/lora_linear_all_wodup_aug/2025-08-20-10-34-34/val/avg/ckpt_e7.pth

CUDA_VISIBLE_DEVICES=14 python training/test.py \
--detector_cfg ./training/config/detector/prodet_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/prodet/efficientnet-b4-6ed6700e/lora_linear_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e8.pth

CUDA_VISIBLE_DEVICES=15 python training/test.py \
--detector_cfg ./training/config/detector/clip_effort_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip_effort/clip-vit-large-patch14/effort_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e6.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14


CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/npr.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/NPR/all_wodup_rn50_2025_08_19_18_46_49/model_epoch_47.pth

CUDA_VISIBLE_DEVICES=2 python training/test.py \
--detector_cfg ./training/config/detector/aide.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/r-contentsecurity-p/common/datas_yl/duanxian/ckpt_tmp/classification/AIDE/all_wodup/2025-08-20-13-37-49/checkpoint-49.pth

CUDA_VISIBLE_DEVICES=3 python training/test.py \
--detector_cfg ./training/config/detector/cospy.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/cospy/all_wodup/cospy_calibrate/epoch_4.pth



## robustness
# blur
CUDA_VISIBLE_DEVICES=3 python training/test.py \
--detector_cfg ./training/config/detector/cospy.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/cospy/all_wodup/cospy_calibrate/epoch_4.pth \
gaussian_sigma 1.0 

CUDA_VISIBLE_DEVICES=14 python training/test.py \
--detector_cfg ./training/config/detector/prodet_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/prodet/efficientnet-b4-6ed6700e/lora_linear_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e8.pth \
gaussian_sigma 1.0 

CUDA_VISIBLE_DEVICES=12 python training/test.py \
--detector_cfg ./training/config/detector/iid_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all_iid.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/iid/xception-b5690688/lora_linear_all_wodup_aug/2025-08-20-10-34-34/val/avg/ckpt_e7.pth \
gaussian_sigma 1.0 

CUDA_VISIBLE_DEVICES=11 python training/test.py \
--detector_cfg ./training/config/detector/unifd_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/unifd/clip-vit-large-patch14/freeze_all_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e10.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14 \
gaussian_sigma 1.0 

CUDA_VISIBLE_DEVICES=15 python training/test.py \
--detector_cfg ./training/config/detector/clip_effort_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip_effort/clip-vit-large-patch14/effort_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e6.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14 \
gaussian_sigma 1.0 


# jpeg
CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/cospy.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/cospy/all_wodup/cospy_calibrate/epoch_4.pth \
jpeg_quality 70

CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/prodet_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/prodet/efficientnet-b4-6ed6700e/lora_linear_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e8.pth \
jpeg_quality 70

CUDA_VISIBLE_DEVICES=2 python training/test.py \
--detector_cfg ./training/config/detector/iid_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all_iid.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/iid/xception-b5690688/lora_linear_all_wodup_aug/2025-08-20-10-34-34/val/avg/ckpt_e7.pth \
jpeg_quality 70

CUDA_VISIBLE_DEVICES=3 python training/test.py \
--detector_cfg ./training/config/detector/unifd_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/unifd/clip-vit-large-patch14/freeze_all_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e10.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14 \
jpeg_quality 70

CUDA_VISIBLE_DEVICES=4 python training/test.py \
--detector_cfg ./training/config/detector/clip_effort_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip_effort/clip-vit-large-patch14/effort_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e6.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14 \
jpeg_quality 70


# hybrid




### Test on aigibench
CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/f3net_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/f3net/xception-b5690688/lora_linear_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e7.pth

CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/unifd_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/unifd/clip-vit-large-patch14/freeze_all_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e10.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14

CUDA_VISIBLE_DEVICES=2 python training/test.py \
--detector_cfg ./training/config/detector/iid_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all_iid.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/iid/xception-b5690688/lora_linear_all_wodup_aug/2025-08-20-10-34-34/val/avg/ckpt_e7.pth

CUDA_VISIBLE_DEVICES=3 python training/test.py \
--detector_cfg ./training/config/detector/prodet_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/prodet/efficientnet-b4-6ed6700e/lora_linear_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e8.pth

CUDA_VISIBLE_DEVICES=4 python training/test.py \
--detector_cfg ./training/config/detector/clip_effort_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip_effort/clip-vit-large-patch14/effort_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e6.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14


CUDA_VISIBLE_DEVICES=5 python training/test.py \
--detector_cfg ./training/config/detector/cospy.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/cospy/all_wodup/cospy_calibrate/epoch_4.pth


CUDA_VISIBLE_DEVICES=6 python training/test.py \
--detector_cfg ./training/config/detector/d3.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/d3/all_wodup_0820_2/model_epoch_99.pth


## Test on DiffusionFace (hydrafake training)
CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/f3net_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/f3net/xception-b5690688/lora_linear_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e7.pth

CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/unifd_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/unifd/clip-vit-large-patch14/freeze_all_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e10.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14

CUDA_VISIBLE_DEVICES=2 python training/test.py \
--detector_cfg ./training/config/detector/iid_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all_iid.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/iid/xception-b5690688/lora_linear_all_wodup_aug/2025-08-20-10-34-34/val/avg/ckpt_e7.pth

CUDA_VISIBLE_DEVICES=3 python training/test.py \
--detector_cfg ./training/config/detector/prodet_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/prodet/efficientnet-b4-6ed6700e/lora_linear_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e8.pth

CUDA_VISIBLE_DEVICES=4 python training/test.py \
--detector_cfg ./training/config/detector/clip_effort_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip_effort/clip-vit-large-patch14/effort_all_wodup_aug/2025-08-19-17-51-43/val/avg/ckpt_e6.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14


CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/cospy.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/cospy/all_wodup/cospy_calibrate/epoch_4.pth


CUDA_VISIBLE_DEVICES=6 python training/test.py \
--detector_cfg ./training/config/detector/d3.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/d3/all_wodup_0820_2/model_epoch_99.pth


screen -S w
## Test on DiffusionFace (ff++ training)  00 模型待选
f3net e7   unifd e5   iid e17   prodet e14   effort e3
CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/f3net_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/f3net/xception-b5690688/lora_linear_ff_46k_aug/2025-09-05-17-46-30/val/avg/ckpt_e7.pth

CUDA_VISIBLE_DEVICES=1 python training/test.py \
--detector_cfg ./training/config/detector/unifd_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/unifd/clip-vit-large-patch14/freeze_all_ff_46k_aug/2025-09-05-17-46-30/val/avg/ckpt_e5.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14

CUDA_VISIBLE_DEVICES=2 python training/test.py \
--detector_cfg ./training/config/detector/iid_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all_iid.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/iid/xception-b5690688/lora_linear_ff_46k_aug/2025-09-05-17-46-45/val/avg/ckpt_e17.pth

CUDA_VISIBLE_DEVICES=3 python training/test.py \
--detector_cfg ./training/config/detector/prodet_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/prodet/efficientnet-b4-6ed6700e/lora_linear_ff_46k_aug/2025-09-05-17-46-40/val/avg/ckpt_e14.pth

CUDA_VISIBLE_DEVICES=4 python training/test.py \
--detector_cfg ./training/config/detector/clip_effort_aug.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/clip_effort/clip-vit-large-patch14/effort_ff_46k_aug/2025-09-05-17-45-02/val/avg/ckpt_e3.pth \
pretrained /mnt/vlr/duanxian/pretrain/clip-vit-large-patch14

## Test on all (ff++ training) 模型待选
CUDA_VISIBLE_DEVICES=5 python training/test.py \
--detector_cfg ./training/config/detector/cospy.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all_test_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/cospy/ff_train/cospy_calibrate/epoch_4.pth


CUDA_VISIBLE_DEVICES=6 python training/test.py \
--detector_cfg ./training/config/detector/d3.yaml \
--dataset_cfg ./training/config/dataset/benchv2_all_test_all.yaml \
--weights_path /mnt/vlr/duanxian/logs/DeepfakeDet/classification/d3/ff_train/model_epoch_99.pth


unifd iid cospy (hydrafake training)  DiffusionFace     # cospy等下面跑完
cospy d3 (ff++ training) all   1




### video dataset
CUDA_VISIBLE_DEVICES=5 python training/test.py \
--detector_cfg ./training/config/detector/d3video.yaml \
--dataset_cfg ./training/config/dataset/video_test.yaml

CUDA_VISIBLE_DEVICES=5 python training/test.py \
--detector_cfg ./training/config/detector/d3video.yaml \
--dataset_cfg ./training/config/dataset/video_img_test.yaml

## face video
CUDA_VISIBLE_DEVICES=5 python training/test.py \
--detector_cfg ./training/config/detector/d3video.yaml \
--dataset_cfg ./training/config/dataset/video_face_part1.yaml

CUDA_VISIBLE_DEVICES=0 python training/test.py \
--detector_cfg ./training/config/detector/d3video.yaml \
--dataset_cfg ./training/config/dataset/video_face_part1.yaml \
video_jpeg_quality 95
