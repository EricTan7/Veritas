'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the CLIPDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
'''

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



@DETECTOR.register_module(module_name='dinov2')
class DINODetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        fea_dim = self.backbone.layernorm.weight.shape[0]
        self.head = nn.Linear(fea_dim, 2)
        self.loss_func = self.build_loss(config)

        self.freeze(config)
        
    def build_backbone(self, config):
        # prepare the backbone
        model = AutoModel.from_pretrained(config['pretrained'])
        return model

    def freeze(self, config):       # always train the head, control the trainable backbone
        transfer_type = config["model_config"]["transfer_type"].lower()
        tuned_layers = convert_params_to_value(self.config["model_config"]["tuned_layers"]) if self.config["model_config"].get("tuned_layers", False) else [-1]
        if transfer_type == "no_freeze":
            return
        elif transfer_type == "freeze_all":
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
        elif transfer_type == "vpt":
            for name, param in self.backbone.named_parameters():
                param.requires_grad = True if "VPT" in name else False
        elif transfer_type == "adapter":
            for name, param in self.backbone.named_parameters():
                param.requires_grad = True if "adapter" in name or "VPT" in name else False
        elif transfer_type == 'lora':
            # all_attn_layers = [n for n, m in self.backbone.named_modules() if isinstance(m, nn.MultiheadAttention)]
            self.backbone.attn_replace()
            all_lora_layers = list()
            for n, m in self.backbone.named_modules():
                if 'q_proj' in n and 'q_proj' in self.config["model_config"]["lora_config"]["lora_layers"]:
                    all_lora_layers.append(n)
                if 'k_proj' in n and 'k_proj' in self.config["model_config"]["lora_config"]["lora_layers"]:
                    all_lora_layers.append(n)
                if 'v_proj' in n and 'v_proj' in self.config["model_config"]["lora_config"]["lora_layers"]:
                    all_lora_layers.append(n)
                if 'out_proj' in n and 'out_proj' in self.config["model_config"]["lora_config"]["lora_layers"]:
                    all_lora_layers.append(n)
                if 'mlp' in self.config["model_config"]["lora_config"]["lora_layers"]:
                    if 'c_fc' in n or 'c_proj' in n:
                        all_lora_layers.append(n)
            loraconfig = LoraConfig(
                r=self.config["model_config"]["lora_config"]["r"],
                lora_alpha=self.config["model_config"]["lora_config"]["alpha"],
                lora_dropout=self.config["model_config"]["lora_config"]["dropout"],
                bias="none",
                target_modules=all_lora_layers,
            )
            self.backbone = get_peft_model(self.backbone, loraconfig)
        elif transfer_type == 'lora_linear':
            all_lora_layers = list()
            for n, m in self.backbone.named_modules():
                if isinstance(m, nn.Linear):
                    all_lora_layers.append(n)
            loraconfig = LoraConfig(
                r=self.config["model_config"]["lora_config"]["r"],
                lora_alpha=self.config["model_config"]["lora_config"]["alpha"],
                lora_dropout=self.config["model_config"]["lora_config"]["dropout"],
                bias="none",
                target_modules=all_lora_layers,
            )
            self.backbone = get_peft_model(self.backbone, loraconfig)
        else:
            raise NotImplementedError
        
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']   # [B, dim]
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred: dict) -> dict:
        label = data_dict['label']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # get loss
        loss_dict = self.get_losses(data_dict, pred)
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'loss': loss_dict}
        return pred_dict


