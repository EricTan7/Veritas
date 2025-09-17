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


### For inference only
@DETECTOR.register_module(module_name='freqnet')
class FreqNetDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = freqnet(num_classes=1)
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
        prob = pred.sigmoid()
        # get loss
        loss = self.loss_fn(pred.squeeze(1), data_dict['label'].float())
        loss_dict = {'overall': loss}
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'loss': loss_dict}
        return pred_dict

    def load_ckpt(self, state_dict):
        self.backbone.load_state_dict(state_dict, strict=True)



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class FreqNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4], num_classes=1, zero_init_residual=False):
        super(FreqNet, self).__init__()



        self.weight1 = nn.Parameter(torch.randn((64, 3, 1, 1)).cuda())
        self.bias1   = nn.Parameter(torch.randn((64,)).cuda())
        self.realconv1 = conv1x1(64, 64, stride=1)
        self.imagconv1 = conv1x1(64, 64, stride=1)

        self.weight2 = nn.Parameter(torch.randn((64, 64, 1, 1)).cuda())
        self.bias2   = nn.Parameter(torch.randn((64,)).cuda())
        self.realconv2 = conv1x1(64, 64, stride=1)
        self.imagconv2 = conv1x1(64, 64, stride=1)


        self.weight3 = nn.Parameter(torch.randn((256, 256, 1, 1)).cuda())
        self.bias3   = nn.Parameter(torch.randn((256,)).cuda())
        self.realconv3 = conv1x1(256, 256, stride=1)
        self.imagconv3 = conv1x1(256, 256, stride=1)

        self.weight4 = nn.Parameter(torch.randn((256, 256, 1, 1)).cuda())
        self.bias4   = nn.Parameter(torch.randn((256,)).cuda())
        self.realconv4 = conv1x1(256, 256, stride=1)
        self.imagconv4 = conv1x1(256, 256, stride=1)
        
        self.inplanes = 64 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])            # layer1 of resnet50
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # layer2 of resnet50
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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

    def hfreqWH(self, x, scale):
        assert scale>2
        #print(f'input shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        b,c,h,w = x.shape
        x[:,:,h//2-h//scale:h//2+h//scale,w//2-w//scale:w//2+w//scale ] = 0.0
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
        #print(f'output shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        #print()
        return x

    def hfreqC(self, x, scale):
        assert scale>2
        # print(f'input shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        x = torch.fft.fft(x, dim=1, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=1) 
        b,c,h,w = x.shape

        x[:,c//2-c//scale:c//2+c//scale,:,:] = 0.0

        x = torch.fft.ifftshift(x, dim=1)
        x = torch.fft.ifft(x, dim=1, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
 
        # print(f'output shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        # print()
        return x
        
        
    def forward(self, x):
        
        ### HFRI
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight1, self.bias1, stride=1, padding=0)
        x = F.relu(x, inplace=True)
        
        ### HFRFC
        x = self.hfreqC(x, 4)
        
        ### FCL
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv1(x.real), self.imagconv1(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        ### HFRFS
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight2, self.bias2, stride=2, padding=0)
        x = F.relu(x, inplace=True)

        ### HFRFC
        x = self.hfreqC(x, 4)
        
        ### FCL
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv2(x.real), self.imagconv2(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)


        x = self.maxpool(x)
        x = self.layer1(x)# in64 out256
        
        
        ### HFRFS
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight3, self.bias3, stride=1, padding=0)
        x = F.relu(x, inplace=True)
        
        
        ### FCL
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv3(x.real), self.imagconv3(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        ### HFRFS
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight4, self.bias4, stride=2, padding=0)
        x = F.relu(x, inplace=True)

        ### FCL
        x = torch.fft.fft2(x, norm="ortho")#,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1]) 
        x = torch.complex(self.realconv4(x.real), self.imagconv4(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

def freqnet(**kwargs):

    return FreqNet()
    
    