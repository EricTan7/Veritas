import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
# from torch.utils.tensorboard import SummaryWriter

from utils.metrics import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from transformers import AutoProcessor, AutoModel, ConvNextV2Model, ViTModel, ViTConfig
from peft import LoraConfig, get_peft_model
from utils.model_utils import convert_params_to_value
import loralib as lora
import copy
import re
from loss.oc_softmax import OCSoftmax
from diffusers import AutoPipelineForImage2Image, AutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
import warnings
import lpips
from joblib.memory import Memory
mem = Memory(location="cache", compress=("lz4", 9), verbose=0)


### For inference only
@DETECTOR.register_module(module_name='aeroblade')
class AEROBLADEDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seed = 1
        self.vae, self.decode_dtype = {}, {}
        for repo_id in self.config["repo_ids"]:
            ae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae")
            ae.cuda()
            ae = torch.compile(ae)
            decode_dtype = next(iter(ae.parameters())).dtype
            self.vae[repo_id] = ae
            self.decode_dtype[repo_id] = decode_dtype

        self.num_workers = 1
        self.threshold = self.config["vae_threshold"]
        
    def build_backbone(self, config):
        pass
        
    def build_loss(self, config):
        pass
    
    def features(self, data_dict: dict) -> torch.tensor:
        pass

    def classifier(self, features: torch.tensor) -> torch.tensor:
        pass
    
    def get_losses(self, data_dict: dict, pred: dict) -> dict:
        pass

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def compute_reconstructions(self, data_dict, repo_id):
        ae = self.vae[repo_id]
        decode_dtype = self.decode_dtype[repo_id]
        
        image = data_dict["image"]
        generator = torch.Generator().manual_seed(self.seed)

        # normalize
        image = image.to(dtype=ae.dtype) * 2.0 - 1.0
        # encode
        latents = retrieve_latents(ae.encode(image), generator=generator)
        # decode
        reconstructions = ae.decode(
            latents.to(decode_dtype), return_dict=False
        )[0]
        # de-normalize
        reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)

        return reconstructions

    def compute_distance(self, dist_metric, img, rec_img):
        _, net, layer = dist_metric.split("_")
        layer = int(layer)

        result = compute_lpips(
            img, rec_img,
            model_kwargs={"net": net},
            num_workers=self.num_workers,
        )
        out = -result[layer]
        out = out.mean((2, 3))      

        return out

    def forward(self, data_dict: dict, inference=False) -> dict:
        distances = []
        # iterate over VAE
        for repo_id in self.config["repo_ids"]:
            # compute reconstruction
            rec_img = self.compute_reconstructions(data_dict, repo_id)

            # iterate over distance_metrics
            for dist_metric in self.config['distance_metrics']:
                dist = self.compute_distance(dist_metric, data_dict['image'], rec_img)
                distances.append(dist)      # TODO shape of dist

        # determine maximum distance
        distances = torch.stack(distances).cuda()
        max_dist = torch.max(distances, dim=0)[0]   # [B, 1]
        max_dist = max_dist[:, 0]

        prob = torch.full_like(max_dist, 0.1)
        prob[max_dist > self.threshold] = 0.9
        pred = torch.zeros_like(max_dist)
      
        loss_dict = {"overall": torch.mean(max_dist)}

        pred_dict = {'cls': pred, 'prob': prob, 'loss': loss_dict}
        return pred_dict



@mem.cache(ignore=["num_workers"])       # cache vgg network
def compute_lpips(
    img,
    rec_img,
    model_kwargs: dict,
    batch_size: int=1,
    num_workers: int=1,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _PatchedLPIPS(spatial=True, **model_kwargs).cuda()

    torch.compile(model.net)

    lpips_layers = [[] for _ in range(1 + len(model.chns))]

    sum_batch, layers_batch = model(
        img,
        rec_img,
        retPerLayer=True,
        normalize=True,
    )
    lpips_layers[0].append(sum_batch.to(device="cpu", dtype=torch.float16))
    for i, layer_result in enumerate(layers_batch):
        lpips_layers[i + 1].append(
            layer_result.to(device="cpu", dtype=torch.float16)
        )

    lpips_layers = [torch.cat(lpips_layer) for lpips_layer in lpips_layers]
    return lpips_layers


def postprocess(self, result, net, layer):
        """Handle layer selection and resizing."""
        out = {}
        if layer == -1:
            for i, tensor in enumerate(result):
                out[f"lpips_{net}_{i}"] = -tensor
        else:
            out[f"lpips_{net}_{layer}"] = -result[self.layer]

        for layer, tensor in out.items():
            out[layer] = tensor.mean((2, 3), keepdim=True)

        return out


class _PatchedLPIPS(lpips.LPIPS):
    """Patched version of LPIPS which returns layer-wise output without upsampling."""

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if (
            normalize
        ):  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1))
            if self.version == "0.1"
            else (in0, in1)
        )
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = (
                lpips.normalize_tensor(outs0[kk]),
                lpips.normalize_tensor(outs1[kk]),
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res_no_up = [self.lins[kk](diffs[kk]) for kk in range(self.L)]
                res = [
                    lpips.upsample(res_no_up[kk], out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    lpips.spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
                    for kk in range(self.L)
                ]
                res_no_up = res
        else:
            if self.spatial:
                res_no_up = [diffs[kk].sum(dim=1, keepdim=True) for kk in range(self.L)]
                res = [
                    lpips.upsample(res_no_up[kk], out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    lpips.spatial_average(
                        diffs[kk].sum(dim=1, keepdim=True), keepdim=True
                    )
                    for kk in range(self.L)
                ]
                res_no_up = res

        val = 0
        for layer in range(self.L):
            val += res[layer]

        if retPerLayer:
            return (val, res_no_up)
        else:
            return val


