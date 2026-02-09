# Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning

**ICLR 2026 Oral** [ [Paper](https://openreview.net/pdf?id=5VXJPS1HoM) | [Data](https://www.modelscope.cn/datasets/EricTanh/HydraFake) | [Model](https://www.modelscope.cn/models/EricTanh/Veritas) ]


In this work, we introduce:

> 📍**HydraFake Dataset**: A deepfake detection dataset with rigorous training and evaluation protocol.
>
> 📍**Veritas Model**: A reasoning model achieving remarkable generalization on OOD forgeries, capable of providing transparent and human-aligned decision process.




## News
- 🔥 `2026.2.6` The [**training data**](https://www.modelscope.cn/datasets/EricTanh/HydraFake/tree/master/jsons/train) is released. [**Veritas**](https://www.modelscope.cn/models/EricTanh/Veritas) and [**Veritas-Cold-Start**](https://www.modelscope.cn/models/EricTanh/Veritas-Cold-Start) are both released. We recommend using Veritas-Cold-Start to customize your own detector (see below for more details).
- 🔥 `2026.2.6` Veritas is selected as ICLR 2026 **Oral**.
- 🔥 `2026.1.26` Veritas has been accepted to **ICLR 2026**.
- 🔥 `2025.9.17` We release the inference code for MLLMs and small vision models.
- 🔥 `2025.9.17` We release the [HydraFake](https://www.modelscope.cn/datasets/EricTanh/HydraFake) dataset (train/val/test).

## Installation
```bash
conda create -n veritas python=3.10
conda activate veritas

# Install the dependencies
pip install -e .
```


## 🔥 Training
Download training data [here](https://www.modelscope.cn/datasets/EricTanh/HydraFake/tree/master/jsons/train), including `sft_36k.json`, `mipo_3k.json`, `pgrpo_8k.json`.

### 1. Cold-start (SFT)
```bash
sh self_scripts/train/train_sft.sh sft sft_36k
```

### 2. Cold-start (MiPO)
```bash
sh self_scripts/train/train_mipo.sh mipo mipo_3k
```


### 3. P-GRPO

1.Deploy reward model. Download reward model [here](https://huggingface.co/CodeGoat24/UnifiedReward-qwen-3b), and replace the path in `swift/plugin/prm.py` and `self_scripts/deploy/deploy_reward_model.sh`.

> Note: the choice of reward model is **flexible**. More powerful models may lead to better performance, e.g., UnifiedReward-qwen-7B, Qwen3-VL-8B or UnifiedReward-2.0-qwen3vl-8B.
```bash
sh self_scripts/deploy/deploy_reward_model.sh
```

2.P-GRPO training.
```bash
sh self_scripts/train/train_pgrpo.sh pgrpo pgrpo_8k
```


## 🚀 Customize your own detector

We recommend using Veritas-Cold-Start + P-GRPO for further customization:
- Veritas is fine-tuned on in-domain datasets, i.e., the latest generative model is SD-XL. As mentioned in our paper, for practical usage, you can further fine-tune [Veritas-Cold-Start](https://www.modelscope.cn/models/EricTanh/Veritas-Cold-Start) on your own collected data. 
- If you adopt our P-GRPO, the only thing you should prepare is the binary labels of your data. Arrange them similar to our `pgrpo_8k.json`.
- We also encourage the development of (1) novel GRPO-style algorithm, e.g., involving grounded reward design to deliver more precise cross-modal signals, and (2) collaborative framework with small vision models, which can be exciting furture works.



## ⌛ Test your model on HydraFake Dataset
### 1. Data Preparation
Download the [HydraFake](https://docs.google.com/forms/d/e/1FAIpQLSf0uMg4thR4YcsNwpRqdWtc4K4z-txy24ileQPaUQzRIuMDYg/viewform?usp=header) dataset and the json files. Put the json files under `./datasets`. The data structure should be like:
```
hydrafake
├── test                # testing images
|   ├── AdobeFirefly
|   |   ├── 0_real
|   |   │   └── *.png
|   |   ├── 1_fake
|   |   │   └── *.png
|   |── ...
├── val                 # validation images
|   ├── real
|   |   └── *.png
|   ├── fake
|   |   └── *.png
├── train               # training images
|   ├── fake
|   |   ├── FS
|   |   |   ├── blendface
|   |   |   │   └── *.png
|   |   |   ├── ...
|   |   ├── FR
|   |   |   ├── Aniportrait
|   |   |   │   └── *.png
|   |   |   ├── ...
|   |   ├── EFG
|   |   |   ├── Dall-E1
|   |   |   │   └── *.png
|   |   |   ├── ...
├── jsons
|   ├── test
|   |   ├── id
|   |   │   └── *.json 
|   |   ├── cm
|   |   │   └── *.json 
|   |   ├── cf
|   |   │   └── *.json 
|   |   ├── cd
|   |   │   └── *.json 
|   ├── val
|   |   └── *.json 
|   ├── train
|   |   ├── fake
|   |   |   ├── FS
|   |   |   │   └── *.json 
|   |   |   ├── FR
|   |   |   │   └── *.json 
|   |   |   ├── EFG
|   |   |   │   └── *.json 
|   |   ├── real
|   |   │   └── *.json 
```
You can also put the dataset in other places, then you should change the json file path in `./swift/llm/dataset/dataset/data_utils.py` and the image path in the json files.


### 2. Test your MLLMs
#### 2.1 Inference with **ms-swift**
Run inference on HydraFake:
```bash
sh self_scripts/infer/infer_hydrafake.sh /path/to/your/model
```
Inference on a specific subset:
```bash
swift infer \
    --val_dataset cd_gpt4o \
    --model /path/to/your/model \
    --infer_backend pt \
    --max_model_len 8192 \
    --max_new_tokens 2048 \
    --dataset_num_proc 16 \
    --max_batch_size 8 \
    --metric self_acc_tags
```

#### 2.2 Inference with **vLLM**
Step1: Deploy your model:
```bash
sh self_scripts/deploy/deploy_model.sh /path/to/your/model  # models/Qwen2.5-VL-7B-Instruct
```
Step2: Run inference (put your model path in `self_scripts/infer/infer_vllm.py`):
```bash
sh self_scripts/infer/infer_hydrafake_vllm.sh 
```

### 3. Test your vision models
We provide a script based on [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).
```bash
# Effort for example
python DeepfakeBench/training/test.py \
--detector_cfg DeepfakeBench/training/config/detector/effort.yaml \   
--dataset_cfg DeepfakeBench/training/config/dataset/hydrafake.yaml \
--weights_path /path/to/your/model
```

## Deployment
We recommend using `vLLM` for model deployment:
```bash
sh self_scripts/deploy/deploy_model.sh /path/to/your/model
```
Inference on a single image:
```bash
python self_scripts/infer/infer_vllm_single.py \
--image_path /path/to/your/image
```



## 🛡️ HydraFake Dataset

📍 **Overview:**

<p align="center">
    <img src="src/data.png" alt="Dataset" width="95%">
</p>


(a) We carefully collect and reimplement advanced deepfake techniques to construct our HydraFake dataset. Real images are collected from 8 datasets. Fake images are from classic datasets, high-quality public datasets and our self-constructed deepfake data. (b) We introduce a rigorous and hierarchical evaluation protocol. Training data contains abundant samples but limited forgery types. Evaluations are split into four distinct levels. (c) Illustration of the subsets in different evaluation splits. (d) The performance of prevailing detectors on our HydraFake dataset. **Most detectors shows strong generalization on Cross-Model setting but poor ability on Cross-Forgery and Cross-Domain scenarios.**

📍 **Statistics:**

<p align="center">
    <img src="src/data_overview.png" alt="Dataset overview" width="90%">
</p>

HydraFake contains 52K images in total for evaluation, with 14K in-domain testing, 11K cross-model testing, 12K cross-forgery testing and 15K cross-domain testing.



## 🛰️ Method

📍 We introduce a pattern-aware reasoning framework, including three basic thinking patterns (*fast judgement*, *reasoning*, *conclusion*) and two advanced patterns (*planning* and *self-reflection*).

📍 Two-stage training pipeline:

(1) **Pattern-guided Cold-Start** (SFT + MiPO): Internalize thinking patterns and align reasoning process

(2) **Pattern-aware Exploration** (P-GRPO): Scale up effective patterns, improve reflection quality.

<p align="center">
    <img src="src/method.png" alt="Training pipeline" width="70%">
</p>




## Citation
If you find our work useful, please cite our paper:
```
@article{tan2025veritas,
  title={Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning},
  author={Tan, Hao and Lan, Jun and Tan, Zichang and Liu, Ajian and Song, Chuanbiao and Shi, Senyuan and Zhu, Huijia and Wang, Weiqiang and Wan, Jun and Lei, Zhen},
  journal={arXiv preprint arXiv:2508.21048},
  year={2025}
}
```

## License
This repo is released under the [Apache 2.0 License](https://github.com/EricTan7/Veritas/blob/main/LICENSE).

## Acknowledgements
This repo benefits from [ms-swift](https://github.com/modelscope/ms-swift) and [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench). Thanks for their great works!