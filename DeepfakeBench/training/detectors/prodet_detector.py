

import os
import datetime
import logging
import numpy as np
import yaml
from sklearn import metrics
from typing import Union
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from detectors.utils.prodet_api import feature_operation, feature_operation_v2
from utils.metrics import calculate_metrics_for_train

from detectors.base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import random
import torch.nn.init as init


logger = logging.getLogger(__name__)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class DriftBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_process=nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, 1, 1),
            # #nn.BatchNorm2d(512),
            # nn.LeakyReLU(inplace=True),
        )
        self.concat_process=nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # #nn.BatchNorm2d(512),
            # nn.LeakyReLU(inplace=True),
        )

    def conv_init(self,m):
        # 初始化
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def init_block(self):
        self.concat_process.apply(self.conv_init)
        self.noise_process.apply(self.conv_init)

    def forward(self,x):

        noise = torch.randn(x.shape[0],512,8,8).cuda()
        noise_processed = self.noise_process(noise)
        trans_res = self.concat_process(torch.cat((x,noise_processed),dim=1)) # cat or add?
        return trans_res

class FeatureAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.inverse_conv1 =  nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            Lambda(lambda x: x.view(-1, 512, 1, 1)),  # Reshape
            nn.ConvTranspose2d(512, 512, kernel_size=(7, 7))
        )
        self.inverse_conv2 =  nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            Lambda(lambda x: x.view(-1, 512, 1, 1)),  # Reshape
            nn.ConvTranspose2d(512, 512, kernel_size=(7, 7))
        )
        self.inverse_conv3 =  nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            Lambda(lambda x: x.view(-1, 512, 1, 1)),  # Reshape
            nn.ConvTranspose2d(512, 512, kernel_size=(7, 7))
        )
        self.attention = nn.Sequential(
            nn.Conv2d(512*4, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), # here are two types: same size or 1 size
            nn.Sigmoid(),
        )

    def forward(self,x1,x2,x3,feature):
        # import ipdb
        # ipdb.set_trace()
        x1 = self.inverse_conv1(x1)
        x2 = self.inverse_conv2(x2)
        x3 = self.inverse_conv3(x3)
        
        att = self.attention(torch.cat((x1,x2,x3,feature),dim=1))
        feature = feature*att
        return feature






@DETECTOR.register_module(module_name='prodet')
class ProDetDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.real2sbi=DriftBlock() # 这个drift对应的就是transition
        self.sbi2bi=DriftBlock()
        self.bi2fake=DriftBlock()

    def linear_init(self,m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                init.constant_(m.bias.data, 0)
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        model_config['pretrained'] = self.config['pretrained']
        backbone = backbone_class(model_config)
        self.adjust_feature = nn.Conv2d(1792, 512, 1)
        self.blend_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 2),
        )
        self.fea_att = FeatureAttentionBlock()
        self.ID_inconsistency_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 2),
        )
        self.deepfake_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 2),
        )
        self.final_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 2),
        )
        if self.config['linear_init']:
            self.deepfake_classifier.apply(self.linear_init)
            self.ID_inconsistency_classifier.apply(self.linear_init)
            self.blend_classifier.apply(self.linear_init)
            self.concat_fc.apply(self.linear_init)

        if config['comb_fake']['cat_manner'] == 're-class': # ignore
            self.re_classifier = nn.Linear(1792, 2)
        if config['pretrained'] != 'None':
            logger.info('Load pretrained model successfully!')
        else:
            logger.info('No pretrained model.')
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        self.fea_loss = nn.MSELoss()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        feature_raw = self.backbone.features(data_dict['image'])
        x = self.adjust_feature(feature_raw)
        return x

    def classifier(self, features: torch.tensor) -> torch.tensor:
        df_pred=self.deepfake_classifier(features)
        bld_pred = self.blend_classifier(features)
        bi_pred = self.ID_inconsistency_classifier(features)
        return df_pred,bld_pred,bi_pred

    def generate_labels(self,label,pred_dict):

        blend_label = torch.clip(label,0,1)
        df_label = torch.where(label == 2, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        id_diff_label = torch.where(label>=2, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        if 'rb_ratios' in pred_dict:
            rb_label = pred_dict['rb_ratios'].cuda()
            bb_label = pred_dict['bb_ratios'].cuda()
            bf_label = pred_dict['bf_ratios'].cuda()
            blend_label = torch.cat((blend_label,rb_label,torch.ones_like(bb_label),torch.ones_like(bf_label)),dim=0)
            id_diff_label = torch.cat((id_diff_label,torch.zeros_like(rb_label),bb_label,torch.ones_like(bf_label)),dim=0)
            df_label = torch.cat((df_label, torch.zeros_like(rb_label),torch.zeros_like(bb_label),bf_label), dim=0)
        return blend_label,df_label,id_diff_label

    def get_drift_loss(self,pred_dict):
        if 'r2b' in pred_dict:
            r2b = pred_dict['r2b']
            b2bi = pred_dict['b2b']
            b2f = pred_dict['b2f']
            bgt = pred_dict['bgt']
            bigt = pred_dict['bigt']
            fgt = pred_dict['fgt']
            r2b_loss = self.fea_loss(r2b,bgt)
            b2bi_loss = self.fea_loss(b2bi,bigt)
            b2f_loss = self.fea_loss(b2f,fgt)
        else:
            r2b_loss,b2bi_loss,b2f_loss = 0,0,0
        return r2b_loss,b2bi_loss,b2f_loss,


    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        df_pred = pred_dict['df_pred']
        bld_pred = pred_dict['bld_pred']
        bi_pred = pred_dict['bi_pred']
        bld_label,df_label,id_diff_label=self.generate_labels(label,pred_dict)
        loss_df = self.loss_func(df_pred, df_label)
        loss_bld = self.loss_func(bld_pred, bld_label)
        loss_idd = self.loss_func(bi_pred, id_diff_label)
        if self.config['feature_drift']:
            r2b_loss,b2b_loss,b2f_loss = self.get_drift_loss(pred_dict)
            loss = loss_bld+loss_df+loss_idd+r2b_loss+b2b_loss+b2f_loss
            loss_dict = {'overall': loss,'loss_df':loss_df,'loss_bld':loss_bld,'loss_iid':loss_idd,'r2b_loss':r2b_loss,'b2b_loss':b2b_loss,'b2f_loss':b2f_loss}
        else:
            loss = loss_bld+loss_df+loss_idd
            loss_dict = {'overall': loss,'loss_df':loss_df,'loss_bld':loss_bld,'loss_iid':loss_idd}
        if self.config['comb_fake']['cat_manner'] == 'linear' or self.config['comb_fake']['cat_manner'] == 're-class':
            loss_out = self.loss_func(pred_dict['cls'], bld_label)
            loss+=loss_out
            loss_dict['loss_linear']=loss_out
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = torch.ceil(data_dict['label'].clamp(max=1).float()).long()
        pred = pred_dict['cls']
        # compute metrics for batch data
        if len(label) != len(pred):
            pred = pred[:len(label)]
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def get_test_metrics(self):
        pass

    def feature_drift(self,features):
        single_len = len(features) // 4
        real, blend, bi, fake = features[:single_len], features[single_len:single_len * 2], features[single_len * 2:single_len * 3],features[single_len * 3:]
        real2blend=self.real2sbi(real)
        blend2bi=self.sbi2bi(blend)
        bi2fake=self.bi2fake(bi)
        return real2blend,blend2bi,bi2fake,blend,bi,fake

    def CAM_forward(self, images,):
        feature_raw = self.backbone.features(images)
        features = self.adjust_feature(feature_raw)
        df_pred, bld_pred, bi_pred = self.classifier(features)
        att_features = self.fea_att(df_pred,bld_pred,bi_pred,features)
        pred = self.final_classifier(att_features)
        return pred

    def forward(self, data_dict: dict, inference=False) -> dict:
        pred_dict={}
        # input: real blend bi fake

        # get the features by backbone
        # import ipdb
        # ipdb.set_trace()
        features = self.features(data_dict)
        
        if not inference:
            if self.config['feature_drift']:
                r2b, b2b, b2f, bgt, bigt, fgt = self.feature_drift(features)
                pred_dict['r2b'] = r2b
                pred_dict['b2b'] = b2b
                pred_dict['b2f'] = b2f
                pred_dict['bgt'] = bgt
                pred_dict['bigt'] = bigt
                pred_dict['fgt'] = fgt
            if self.config.get('feature_operation',False) and self.epoch>=self.config.get('Op_start',0):
                real_blends, blend_bis,bi_fakes, rb_ratios,bb_ratios, bf_ratios = feature_operation_v2(features)
                pred_dict['rb_ratios']=rb_ratios
                pred_dict['bb_ratios']=bb_ratios
                pred_dict['bf_ratios']=bf_ratios
                features=torch.cat((features, real_blends,blend_bis,bi_fakes), dim=0)

        # get the prediction by each classifier
        df_pred,bld_pred,bi_pred = self.classifier(features)

        att_features = self.fea_att(df_pred,bld_pred,bi_pred,features)
        pred = self.final_classifier(att_features)

        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict.update({'cls': pred, 'prob': prob,'df_pred':df_pred,'bld_pred':bld_pred,'bi_pred':bi_pred,'feat': features,'att_features':att_features})
        loss_dict = self.get_losses(data_dict, pred_dict)
        pred_dict["loss"] = loss_dict
        return pred_dict



