# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
# from torch.utils.tensorboard import SummaryWriter
from utils.metrics import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from utils.metrics import get_test_metrics
from utils.model_utils import thread_flag
from trainer.base_trainer import SimpleTrainer
import wandb
from utils.model_utils import set_seed, thread_flag, check_trainable_params, getModelSize
import ipdb


class Trainer(SimpleTrainer):
    def __init__(
        self,
        config,
        model,
        optimizer=None,
        scheduler=None,
        logger=None,
        metric_scoring='auc',
        log_dir=None,
        swa_model=None,
        optimizer_prompt=None,
        scheduler_prompt=None
        ):
        super().__init__()
        # check if all the necessary components are implemented
        # if config is None or model is None or optimizer is None or logger is None:
        #     raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.optimizer_prompt = optimizer_prompt
        self.scheduler = scheduler
        self.scheduler_prompt = scheduler_prompt
        self.swa_model = swa_model
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()  # move model to GPU

        self.log_dir = log_dir

    def train_step(self, data_dict, epoch):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                # losses = self.model.get_losses(data_dict, predictions)
                losses = predictions['loss']
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:
            predictions = self.model(data_dict, epoch)
            losses = predictions['loss']
            self.optimizer.zero_grad()
            if self.optimizer_prompt is not None:
                self.optimizer_prompt.zero_grad()
            losses['overall'].backward()
            # 梯度裁剪
            if self.config["loss_config"]["clip_grad"]:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["loss_config"]["clip_grad_norm"])  # max_norm根据实际需求调整   
            self.optimizer.step()
            if self.optimizer_prompt is not None:
                self.optimizer_prompt.step()

            return losses, predictions

    def train_epoch(
        self,
        epoch,
        train_data_loader,
        val_data_loader=None,
        test_data_loaders=None,
        ):
        if epoch == 1:
            self.print_dataset_statistics(train_data_loader, val_data_loader, test_data_loaders)

        times_per_epoch = 1

        test_step = len(train_data_loader) // times_per_epoch    # test 10 times per epoch
        step_cnt = epoch * len(train_data_loader)
        tot_step = self.config['nEpochs'] * len(train_data_loader)

        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)
        batch_time_recoder = Recorder()

        for iteration, data_dict in enumerate(train_data_loader):
            # print(data_dict["label"])
            # ipdb.set_trace()
            start = time.time()
            self.setTrain()
            # more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if data_dict[key]!=None and key!='name' and key!='path':
                    data_dict[key]=data_dict[key].cuda()

            losses, predictions=self.train_step(data_dict, epoch)
            gpu_mem = torch.cuda.max_memory_allocated()/(1024.0**3)

            # update model using SWA
            if self.config['SWA'] and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            batch_time_recoder.update(time.time() - start)

            # logging
            if iteration % self.config['log_period'] == 0 and thread_flag(self.config['ddp']):
                # info for loss
                log_str = f'Epoch[{epoch}/{self.config["nEpochs"]}] Iter[{iteration}/{len(train_data_loader)}]'
                temp_loss_dict = {}
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    log_str += f" loss_{k}: {v_avg:.3f}"
                    temp_loss_dict[f"loss_{k}"] = v_avg
                # info for metric
                temp_metric_dict = {}
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    log_str += f" {k}: {v_avg:.2%}"
                    temp_metric_dict['train_'+k] = v_avg
                # other statistics
                speed = train_data_loader.batch_size/batch_time_recoder.average()
                eta_seconds = batch_time_recoder.average() * (tot_step - step_cnt)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                cur_lr = self.optimizer.param_groups[0]['lr']
                log_str += f' lr: {cur_lr:.2e}'
                if self.optimizer_prompt is not None:
                    log_str += f' prompt_lr: {self.optimizer_prompt.param_groups[0]["lr"]:.2e}'
                log_str += f' mem: {gpu_mem:.2f}GB'
                log_str += f' speed: {speed:.2f}[img/s]'
                log_str += f' ETA: {eta}'
                self.logger.info(log_str)

                if self.config['wandb']:
                    wandb.log({
                        **temp_loss_dict,
                        **temp_metric_dict,
                        'lr': cur_lr,
                        "epoch": epoch,
                        "iter": step_cnt,
                    })

                # clear recorder.
                # Note we only consider the current `log_period` batches for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()
                batch_time_recoder.clear()

            # run test (validation)
            if (step_cnt+1) % test_step == 0:
                if self.config['ddp']:
                    dist.barrier()
                if test_data_loaders is not None and thread_flag(self.config['ddp']):
                    self.logger.info(f"Validation Epoch-Iter[{epoch}-{iteration}]")
                    test_best_metric = self.test_epoch(
                        val_data_loader,
                        phase='val',
                        epoch=epoch,
                        iteration=iteration
                    )
                else:
                    test_best_metric = None
                if self.config['ddp']:
                    dist.barrier()

            step_cnt += 1
        if self.scheduler is not None:
            self.scheduler.step()
        if self.scheduler_prompt is not None:
            self.scheduler_prompt.step()

        return test_best_metric

    @torch.no_grad()
    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists=[]
        label_lists = []
        logit_lists = []
        path_lists = []
        for i, data_dict in enumerate(tqdm(data_loader)):
            # ipdb.set_trace()
            paths = [path for path in data_dict['path']]
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if key!='path' and data_dict[key]!=None:
                    data_dict[key]=data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            # feature_lists += list(predictions['feat'].cpu().detach().numpy())
            logit_lists += list(predictions['cls'].cpu().detach().numpy())
            path_lists += paths

            losses = predictions['loss']
            for name, value in losses.items():
                test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists), \
                np.array(feature_lists), np.array(logit_lists), np.array(path_lists)
    
    def test_epoch(self, test_data_loaders, phase='val', epoch=-1, iteration=-1):
        # set model to eval mode
        self.setEval()
        if epoch == -1:     # only print during inference
            self.print_dataset_statistics(test=test_data_loaders)

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'f1_real': 0, 'f1_fake': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0, 'dataset_dict':{}}  # average over multiple val set; `dataset_dict` save the auc metric for all val set
        # testing for all test data
        keys = test_data_loaders.keys()
        temp_metric_dict, temp_loss_dict = {}, {}
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps, logit_nps, path_nps = self.test_one_dataset(test_data_loaders[key])
            # feature_nps: [N, dim]
            losses_all_datasets[key] = losses_one_dataset_recorder

            if phase != 'test':
                pos_label = 0 if "_oc" in self.config["model_name"] else 1
                metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps, img_names=data_dict['image'], pos_label=pos_label)
                for metric_name, value in metric_one_dataset.items():
                    if metric_name in avg_metric:
                        if isinstance(value, str):
                            avg_metric[metric_name] += 0
                        else:
                            avg_metric[metric_name] += value
                avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]

                log_str = f'[{key}]'
                for k, v in metric_one_dataset.items():
                    if k == 'pred' or k == 'label' or k=='dataset_dict':
                        continue
                    if k == 'pred_real' or k == 'pred_fake' or k == "recall5_thresh":
                        log_str += f" {k}: {v}"
                    else:
                        if isinstance(v, str):
                            log_str += f" {k}: {v}"
                        else:
                            log_str += f" {k}: {v:.2%}"
                    temp_metric_dict[f"{phase}_{key}_{k}"] = v
                log_str += '\n'
                for k, v in losses_one_dataset_recorder.items():
                    v_avg = v.average()
                    log_str += f"loss_{k}: {v_avg:.3f} "
                    temp_loss_dict[f"{phase}_{key}_loss_{k}"] = v_avg
                log_str += '\n'
                if 'pred' in metric_one_dataset:
                    acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
                    log_str += f'acc_real:{acc_real:.2%}; acc_fake:{acc_fake:.2%}'
                    temp_metric_dict[f"{phase}_{key}_acc-real"] = acc_real
                    temp_metric_dict[f"{phase}_{key}_acc-fake"] = acc_fake
                self.logger.info(log_str)
                # self.save_best(key, metric_one_dataset, phase, epoch, iteration)

            if self.config['save_feat']:
                save_dir = os.path.join(self.log_dir, phase, key)
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, f"feature_nps.npy"), feature_nps)
                np.save(os.path.join(save_dir, f"logit_nps.npy"), logit_nps)
                np.save(os.path.join(save_dir, f"label_nps.npy"), label_nps)
                np.save(os.path.join(save_dir, f"path_nps.npy"), path_nps)
        
        if phase != "test":
            if len(keys)>0 and self.config.get('save_avg', False):
                log_str = '[avg]'
                # calculate avg value
                for key in avg_metric:
                    if key != 'dataset_dict':
                        avg_metric[key] /= len(keys)
                        log_str += f" {key}: {avg_metric[key]:.2%}"
                        temp_metric_dict[f"{phase}_avg_{key}"] = avg_metric[key]
                self.logger.info(log_str)
                self.save_best('avg', avg_metric, phase, epoch, iteration)

            # save every test step
            self.save_ckpt(phase, 'avg', f"e{epoch}")

            temp_best_score_dict = {f"best-{phase}_{k}_{self.metric_scoring}": v[self.metric_scoring] for k, v in self.best_metrics_all_time.items()}
            if self.config['wandb']:
                wandb.log({
                    **temp_metric_dict,
                    **temp_loss_dict,
                    **temp_best_score_dict
                })

        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions



