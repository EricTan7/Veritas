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
from detectors.utils import clip


### For inference only
@DETECTOR.register_module(module_name='d3')
class D3Detector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = CLIPModelShuffleAttentionPenultimateLayer(
            "ViT-L/14",    # "CLIP:ViT-L-14", 
            shuffle_times=1, 
            original_times=1,
            patch_size=14
        )
        # self.backbone.load_state_dict(torch.load(config['pretrained']), strict=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
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
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the prediction by classifier
        pred = self.backbone(data_dict["image"])
        # get the probability of the pred
        prob = pred.sigmoid()       # TODO
        # get loss
        # loss = self.loss_fn(pred.squeeze(1), data_dict['label'].float())
        loss = torch.tensor(0.)
        loss_dict = {'overall': loss}
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'loss': loss_dict}
        return pred_dict

    def load_ckpt(self, state_dict):
        self.backbone.attention_head.load_state_dict(state_dict, strict=True)



CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "ViT-L/14-penultimate" : 1024,
    "ViT-L-14" : 768,
}


class CLIPModelShuffleAttentionPenultimateLayer(nn.Module):
    def __init__(self, name, num_classes=1,shuffle_times=1, patch_size=32, original_times=1):
        super(CLIPModelShuffleAttentionPenultimateLayer, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.shuffle_times = shuffle_times
        self.original_times = original_times
        self.patch_size = patch_size
        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class
        self.register_hook()
        self.attention_head = TransformerAttention(CHANNELS[name+"-penultimate"], shuffle_times + original_times, last_dim=num_classes)
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def register_hook(self):
        
        def hook(module, input, output):
            self.features = torch.clone(output)
        for name, module in self.model.visual.named_children():
            if name == "ln_post":
                module.register_forward_hook(hook)
        return 
    
    def shuffle_patches(self, x, patch_size):
        B, C, H, W = x.size()
        # Unfold the input tensor to extract non-overlapping patches
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size, dilation=1)
        # Reshape the patches to (B, C, patch_H, patch_W, num_patches)
        shuffled_patches = patches[:, :, torch.randperm(patches.size(-1))]
        # Fold the shuffled patches back into images
        shuffled_images = F.fold(shuffled_patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
        return shuffled_images


    def forward(self, x, return_feature=False):
        features = []
        with torch.no_grad():
            for i in range(self.shuffle_times):
                # print(type(x))
                # print(self.patch_size)
                self.model.encode_image(self.shuffle_patches(x, patch_size=self.patch_size))
                features.append(self.features)

            self.model.encode_image(x)
            for i in range(self.original_times):
                features.append(self.features.clone())
        features = self.attention_head(torch.stack(features, dim=-2))

        return features


class TransformerAttention(nn.Module):
    def __init__(self, input_dim, output_dim, last_dim=1):
        super(TransformerAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim*output_dim, last_dim)
        # self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.unsqueeze(dim=1)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1)))
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output

