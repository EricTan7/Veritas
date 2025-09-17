import torch
import numpy as np
import random
import torch.nn as nn
import torch.distributed as dist
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import os
import errno
import pickle
from PIL import Image
from networks.clip import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from networks.dinov2.models.vision_transformer import vit_large, vit_base, vit_small, vit_giant2
device = 'cuda'



def set_seed(config):
    seed = config['manualSeed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if config['cudnn']:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def thread_flag(dist_train):
    flag = False
    if dist_train:
        if dist.get_rank() == 0:
            flag = True
    else:
        flag = True
    
    return flag


def check_trainable_params(model):
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")



def getModelSize(model):
    param_size = 0
    param_sum = 0
    grad_param_size = 0
    grad_param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
        if param.requires_grad == True:
            grad_param_size += param.nelement() * param.element_size()
            grad_param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('total number of params:{:.3f}M'.format(param_sum / 1000 / 1000))
    print('trainable number of params:{:.3f} ({:.5%})'.format(grad_param_sum, grad_param_sum/param_sum))

    return (param_size, param_sum, buffer_size, buffer_sum, grad_param_size)


def convert_params_to_value(params):
    if params[0] == -1:
        return [-1]
    elif params[-1] == -1:
        return list(range(params[0]))
    else:
        return params


def to_tuple(x):
    if isinstance(x, (tuple, list)):
        return x
    else:
        return (x, x)
    

def load_clip(config, design_details=None, zero_shot=False, 
    zero_shot_dpam=False, distill=False, remoteclip=False):
    if distill:
        backbone_name = config['model_config']['distill_backbone']
        zero_shot=True
    else:
        backbone_name = config['model_config']['backbone']
    if remoteclip:
        model_path = os.path.join("pretrain/openclip/remoteclip/", f"RemoteCLIP-{backbone_name}.pt")
    else:
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
    use_freqfit = config["model_config"].get("use_freqfit", False)
    freqfit_layers = config["model_config"].get("freqfit_layers", None)
    freqfit_type = config["model_config"].get("freqfit_type", None)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cuda").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cuda")

    print(f"================= Building CLIP: {backbone_name} =================")
    if zero_shot:
        if zero_shot_dpam:
            dpam_layer = [12, -1] if "ViT-B" in backbone_name else [24, -1]
            dpam_layer = convert_params_to_value(dpam_layer)
            design_details = {"trainer": config['model_config']['trainer'],
                            "vision_depth": [-1],
                            "vision_ctx": 0,
                            "language_depth": [-1],
                            "language_ctx": 0,
                            "insert_text_layer": [-1],
                            "DPAM_layer": dpam_layer,
                            "use_freqfit": use_freqfit,
                            "freqfit_layers": freqfit_layers,
                            "freqfit_type": config["model_config"]["freqfit_type"]}
        else:
            design_details = {"trainer": config['model_config']['trainer'],
                            "vision_depth": [-1],
                            "vision_ctx": 0,
                            "language_depth": [-1],
                            "language_ctx": 0,
                            "insert_text_layer": [-1],
                            "DPAM_layer": [-1],
                            "use_freqfit": use_freqfit,
                            "freqfit_layers": freqfit_layers,
                            "freqfit_type": config["model_config"]["freqfit_type"]}
        print("Building zero-shot CLIP model")
    elif design_details is None:
        prompt_depth_vision = convert_params_to_value(config['model_config']['prompt_depth_vision'])
        promtp_depth_text = convert_params_to_value(config['model_config']['prompt_depth_text'])
        insert_text_layer = convert_params_to_value(config['model_config']['insert_text_layer'])
        dpam_layer = convert_params_to_value(config['model_config']['dpam_layer'])
        if freqfit_layers is not None:
            freqfit_layers = convert_params_to_value(freqfit_layers)
        design_details = {"trainer": config['model_config']['trainer'],
                        "vision_depth": prompt_depth_vision,
                        "vision_ctx": config['model_config']['n_ctx_vision'],
                        "language_depth": promtp_depth_text,    # [-1]
                        "language_ctx": config['model_config']['n_ctx_text'],# 0
                        "insert_text_layer": insert_text_layer,
                        "DPAM_layer": dpam_layer,
                        "kernel_size": config['model_config']['kernel_size'],
                        "adapter_type": config['model_config']['adapter_type'],
                        "use_freqfit": use_freqfit,
                        "freqfit_layers": freqfit_layers,
                        "freqfit_type": freqfit_type}
    
    input_resolution = to_tuple(config['resolution'])
    model = clip.build_model(state_dict or model.state_dict(), input_resolution, design_details)

    return model.float()


model2path_dinov2 = {
    'vit_base_reg': 'dinov2_vitb14_reg4_pretrain.pth', 
    'vit_large_reg': 'dinov2_vitl14_reg4_pretrain.pth', 
}

def load_dinov2(config):
    backbone_name = config['model_config']['backbone']
    ckpt_path = os.path.join(config["pretrained"], model2path_dinov2[backbone_name])
    print(f"================= Building DINOv2: {backbone_name} =================")
    if backbone_name == 'vit_base_reg':
        backbone = vit_base(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            block_chunks=0,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        ).cuda()
    elif backbone_name == 'vit_large_reg':
        backbone = vit_large(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            block_chunks=0,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        ).cuda()
    elif backbone_name == 'vit_base':
        backbone = vit_base(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            block_chunks=0,
        ).cuda()
    elif backbone_name == 'vit_large':
        backbone = vit_large(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            block_chunks=0,
        ).cuda()
    else:
        raise NotImplementedError

    state_dict = torch.load(ckpt_path, map_location='cuda')
    backbone.load_state_dict(state_dict)

    return backbone
        
    
