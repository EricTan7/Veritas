import datetime
from copy import deepcopy
from abc import ABC, abstractmethod
import os
import pickle
import numpy as np
import json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseTrainer(ABC):
    """
        Basic Trainer
        Define some essential functions
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def speed_up(self):
        pass

    @abstractmethod
    def setTrain(self):
        pass

    @abstractmethod
    def setEval(self):
        pass

    @abstractmethod
    def load_ckpt(self, model_path):
        pass

    @abstractmethod
    def save_ckpt(self, dataset, epoch, iters, best=False):
        pass

    @abstractmethod
    def inference(self, data_dict):
        pass



class SimpleTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp']:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']], find_unused_parameters=True, output_device=self.config['local_rank'])
            #self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location=device)
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key, ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_{ckpt_info}.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp']:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
            # json.dump(metric_one_dataset, file)
        # self.logger.info(f"Metrics saved to {file_path}")

    def save_best(self, key, metric_one_dataset, phase, epoch, iteration):
        best_metric = self.best_metrics_all_time[key].get(
            self.metric_scoring, 
            float('-inf') if self.metric_scoring != 'eer' else float('inf')
        )
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']

            # Save checkpoint, feature, and metrics if specified in config
            # if self.config['save_ckpt'] and key not in FFpp_pool and phase == 'val':
            # if self.config['save_ckpt'] and key == 'avg' and phase == 'val':
            #     self.save_ckpt(phase, key, f"{epoch}+{iteration}")
            self.save_metrics(phase, metric_one_dataset, key)
    
    def get_respect_acc(self, prob, label, thresh=0.5):
        pred = np.where(prob > thresh, 1, 0)
        real_idx = label == 0
        fake_idx = label == 1
        real_pred = pred[real_idx]
        fake_pred = pred[fake_idx]
        acc_real = np.sum(real_pred == 0) / len(real_pred)
        acc_fake = np.sum(fake_pred == 1) / len(fake_pred)
        return acc_real, acc_fake

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions

    def get_imagedata_info(self, data):
        img_list = data.data_dict["image"]
        label_list = data.data_dict["label"]
        num_imgs = len(img_list)
        num_fake = sum([1 for label in label_list if label==1])
        num_real = sum([1 for label in label_list if label==0])
        return num_imgs, num_fake, num_real
    
    def print_dataset_statistics(self, train=None, val=None, test=None):
        if train is None and val is None:
            log_str = "Dataset statistics:\n"
            log_str += "  --------------------------------------------------------------\n"
            log_str += "  subset                     | # images | # fake   | # real\n"
            log_str += "  --------------------------------------------------------------\n"
            keys = test.keys()
            for key in keys:
                num_test_imgs, num_test_fake, num_test_real = self.get_imagedata_info(test[key].dataset)
                log_str += "  test  {:<20} | {:<8} | {:<8} | {:<8}\n".format(key, num_test_imgs, num_test_fake, num_test_real)
            log_str += "  --------------------------------------------------------------\n"

            self.logger.info(log_str)
        else:
            num_train_imgs, num_train_fake, num_train_real = self.get_imagedata_info(train.dataset)
            log_str = "Dataset statistics:\n"
            log_str += "  --------------------------------------------------------------\n"
            log_str += "  subset                     | # images | # fake   | # real\n"
            log_str += "  --------------------------------------------------------------\n"
            log_str += "  train {:<20} | {:<8} | {:<8} | {:<8}\n".format(", ".join(self.config['train_dataset']), num_train_imgs, num_train_fake, num_train_real)

            if isinstance(val, dict):
                keys = val.keys()
                for key in keys:
                    num_val_imgs, num_val_fake, num_val_real = self.get_imagedata_info(val[key].dataset)
                    log_str += "  val   {:<20} | {:<8} | {:<8} | {:<8}\n".format(key, num_val_imgs, num_val_fake, num_val_real)
            else:
                num_val_imgs, num_val_fake, num_val_real = self.get_imagedata_info(val.dataset)
                log_str += "  val      | {:8d} | {:<8} | {:<8}\n".format(num_val_imgs, num_val_fake, num_val_real)

            keys = test.keys()
            for key in keys:
                num_test_imgs, num_test_fake, num_test_real = self.get_imagedata_info(test[key].dataset)
                log_str += "  test  {:<20} | {:<8} | {:<8} | {:<8}\n".format(key, num_test_imgs, num_test_fake, num_test_real)
            log_str += "  --------------------------------------------------------------\n"

            self.logger.info(log_str)

