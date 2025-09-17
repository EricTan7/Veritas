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
from peft import LoraConfig, get_peft_model
from utils.model_utils import convert_params_to_value
import open_clip
from torchvision import transforms
from diffusers import StableDiffusionPipeline, AutoencoderKL



@DETECTOR.register_module(module_name='cospy')
class COSPYDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = Detector(config)
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
        pred = self.backbone.predict(data_dict["image"])
        prob = pred
        loss = torch.tensor(0.)
        loss_dict = {'overall': loss}
        pred_dict = {'cls': pred, 'prob': prob, 'loss': loss_dict}
        return pred_dict

    def load_ckpt(self, state_dict):
        print("For CO-SPY: load checkpoints from 'semantic_weights_path', 'artifact_weights_path' and 'classifier_weights_path'!")




class Detector():
    def __init__(self, args):
        super(Detector, self).__init__()

        # Device
        self.device = "cuda"

        # Initialize the detector
        self.model = CospyCalibrateDetector(
            semantic_weights_path=args["semantic_weights_path"],
            artifact_weights_path=args["artifact_weights_path"])

        # Load the pre-trained weights
        self.model.load_weights(args["classifier_weights_path"])   # calibrate权重
        self.model.eval()

        # Put the model on the device
        self.model.to(self.device)

    # Prediction function
    def predict(self, inputs):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        prediction = outputs.sigmoid().flatten()    # .tolist()
        return prediction





# CO-SPY Calibrate Detector (Calibrate the integration of semantic and artifact detectors)
class CospyCalibrateDetector(torch.nn.Module):
    def __init__(self, semantic_weights_path, artifact_weights_path, num_classes=1):
        super(CospyCalibrateDetector, self).__init__()

        # Load the semantic detector
        self.sem = SemanticDetector()
        self.sem.load_weights(semantic_weights_path)

        # Load the artifact detector
        self.art = ArtifactDetector()
        self.art.load_weights(artifact_weights_path)

        # Freeze the two pre-trained models
        for param in self.sem.parameters():
            param.requires_grad = False
        for param in self.art.parameters():
            param.requires_grad = False

        # Classifier
        self.fc = torch.nn.Linear(2, num_classes)

        # Transformations inside the forward function
        # Including the normalization and resizing (only for the artifact detector)
        self.sem_transform = transforms.Compose([
            transforms.Normalize(self.sem.mean, self.sem.std)
        ])
        self.art_transform = transforms.Compose([
            transforms.Resize(self.art.cropSize, antialias=False),
            transforms.Normalize(self.art.mean, self.art.std)
        ])

        # Resolution
        self.loadSize = 384
        self.cropSize = 384

        # Data augmentation
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(70, 96))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Pre-processing
        crop_func = transforms.RandomCrop(self.cropSize)
        rz_func = transforms.Resize(self.loadSize)

        self.test_transform = transforms.Compose([
            rz_func,
            crop_func,
            transforms.ToTensor(),
        ])

    def forward(self, x):
        x_sem = self.sem_transform(x)
        x_art = self.art_transform(x)
        pred_sem = self.sem(x_sem)
        pred_art = self.art(x_art)
        x = torch.cat([pred_sem, pred_art], dim=1)
        x = self.fc(x)
        return x

    def save_weights(self, weights_path):
        save_params = {"fc.weight": self.fc.weight.cpu(), "fc.bias": self.fc.bias.cpu()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.fc.weight.data = weights["fc.weight"]
        self.fc.bias.data = weights["fc.bias"]





# Semantic Detector (Extract semantic features using CLIP)
class SemanticDetector(torch.nn.Module):
    def __init__(self, dim_clip=1152, num_classes=1):
        super(SemanticDetector, self).__init__()

        # Get the pre-trained CLIP
        # model_name = "ViT-SO400M-14-SigLIP-384"
        # version = "webli"
        # self.clip, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=version)
        model_name = "local-dir:/pretrain/ViT-SO400M-14-SigLIP-384"
        self.clip, _, _ = open_clip.create_model_and_transforms(model_name)
        # Freeze the CLIP visual encoder
        self.clip.requires_grad_(False)

        # Classifier
        self.fc = torch.nn.Linear(dim_clip, num_classes)

        # Normalization
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        # Resolution
        self.loadSize = 384
        self.cropSize = 384

        # Data augmentation
        self.blur_prob = 0.5
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(30, 101))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Pre-processing
        crop_func = transforms.RandomCrop(self.cropSize)
        rz_func = transforms.Resize(self.loadSize)

        self.test_transform = transforms.Compose([
            rz_func,
            crop_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def forward(self, x, return_feat=False):
        feat = self.clip.encode_image(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    def save_weights(self, weights_path):
        save_params = {"fc.weight": self.fc.weight.cpu(), "fc.bias": self.fc.bias.cpu()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.fc.weight.data = weights["fc.weight"]
        self.fc.bias.data = weights["fc.bias"]




# Artifact Detector (Extract artifact features using VAE)
class ArtifactDetector(torch.nn.Module):
    def __init__(self, dim_artifact=512, num_classes=1):
        super(ArtifactDetector, self).__init__()
        # Load the pre-trained VAE
        model_id = "pretrain/stable-diffusion-v1-4"
        # vae = StableDiffusionPipeline.from_pretrained(model_id).vae
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae"
        )
        # Freeze the VAE visual encoder
        vae.requires_grad_(False)
        self.artifact_encoder = VAEReconEncoder(vae)

        # Classifier
        self.fc = torch.nn.Linear(dim_artifact, num_classes)

        # Normalization
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]

        # Resolution
        self.loadSize = 256
        self.cropSize = 224

        # Data augmentation
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(70, 96))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Pre-processing
        crop_func = transforms.RandomCrop(self.cropSize)
        rz_func = transforms.Resize(self.loadSize)
        self.test_transform = transforms.Compose([
            rz_func,
            crop_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def forward(self, x, return_feat=False):
        feat = self.artifact_encoder(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    def save_weights(self, weights_path):
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.load_state_dict(weights)






def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class VAEReconEncoder(nn.Module):
    def __init__(self, vae, block=Bottleneck):
        super(VAEReconEncoder, self).__init__()

        # Define the ResNet model
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 is [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64 , 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        # self.layer3 = self._make_layer(block, 256, 6, stride=2)
        # self.layer4 = self._make_layer(block, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Load the VAE model
        self.vae = vae

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reconstruct(self, x):
        with torch.no_grad():
            # `.sample()` means to sample a latent vector from the distribution
            # `.mean` means to use the mean of the distribution
            latent = self.vae.encode(x).latent_dist.mean
            decoded = self.vae.decode(latent).sample
        return decoded

    def forward(self, x):
        # Reconstruct
        x_recon = self.reconstruct(x)
        # Compute the artifacts
        x = x - x_recon

        # Scale the artifacts
        x = x / 7. * 100.

        # Forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
