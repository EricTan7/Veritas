import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)
import numpy as np
import random
import h5py

from .I2G_dataset import I2GDataset
from .iid_dataset import IIDDataset, JSONIIDDataset
from .abstract_dataset import DeepfakeAbstractBaseDataset
from .ff_blend import FFBlendDataset
# from .fwa_blend import FWABlendDataset
from .lrl_dataset import LRLDataset
from .pair_dataset import pairDataset
from .sbi_dataset import SBIDataset
from .lsda_dataset import LSDADataset
from .tall_dataset import TALLDataset
from .json_dataset import JSONDataset

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler



def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if dataset.args.dataset == 'AVE':
        dataset.all_visual_pretrained_features = np.load(dataset.visual_pretrained_feature_path, allow_pickle=True).item()
    else:
        dataset.all_visual_pretrained_features = h5py.File(dataset.visual_pretrained_feature_path, 'r')
    dataset.all_audio_pretrained_features = np.load(dataset.audio_pretrained_feature_path, allow_pickle=True).item()



def prepare_training_data(config):
    # Only use the blending dataset class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)
        elif config['model_name'] == 'fwa':
            train_set = FWABlendDataset(config)
        elif config['model_name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')
        elif config['model_name'] == 'lsda':
            train_set = LSDADataset(config, mode='train')
        else:
            raise NotImplementedError(
                'Only facexray, fwa, sbi, and lsda are currently supported for blending dataset'
            )
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config, mode='train')  # Only use the pair dataset class in training
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set = LRLDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'json':
        train_set = JSONDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'json_iid':
        train_set = JSONIIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'video':
        train_set = VideoDataset(config, mode='train')
    else:
        train_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='train',
                )
    if config['model_name'] == 'lsda':
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5)
        train_data_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                sampler=custom_sampler, 
                collate_fn=train_set.collate_fn,
                prefetch_factor=2,
                pin_memory=True
            )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        if config.get('balanced_sampler', False):
            print("Using balanced sampler")
            label_list = np.array(train_set.label_list)
            ratio = np.bincount(label_list)
            w = 1. / torch.tensor(ratio, dtype=torch.float)
            sample_weights = w[label_list]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(label_list), replacement=True)
            train_data_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=config['train_batchSize'],
                    num_workers=int(config['workers']),
                    collate_fn=train_set.collate_fn,
                    sampler=sampler,
                    )
        else:
            train_data_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=config['train_batchSize'],
                    shuffle=True,
                    num_workers=int(config['workers']),
                    collate_fn=train_set.collate_fn,
                    )
    num_ids = len(set(train_set.data_dict["video_id"]))
    return train_data_loader, num_ids


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        if 'dataset_type' in config and config['dataset_type'] == 'lrl':
            test_set = LRLDataset(
                config=config,
                mode='test',
            )
        elif 'dataset_type' in config and config['dataset_type'] in ['json', 'json_iid']:
            test_set = JSONDataset(
                    config=config,
                    mode='test',
            )
        elif 'dataset_type' in config and config['dataset_type'] == 'video':
            test_set = VideoDataset(config, mode='test')
        else:
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
            )
            

        test_data_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['test_workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False,
                pin_memory=True
                # drop_last=(test_name=='DeepFakeDetection'),
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders



def prepare_val_data(config):
    def get_val_data_loader(config, val_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = val_name  # specify the current test dataset
        if 'dataset_type' in config and config['dataset_type'] == 'lrl':
            test_set = LRLDataset(
                config=config,
                mode='test',
            )
        elif 'dataset_type' in config and config['dataset_type'] in ['json', 'json_iid']:
            test_set = JSONDataset(
                    config=config,
                    mode='test',
            )
        elif 'dataset_type' in config and config['dataset_type'] == 'video':
            test_set = VideoDataset(config, mode='test')
        else:
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
            )

        test_data_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['test_workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False,
                # pin_memory=False,
                # persistent_workers=True,
                # prefetch_factor=2
                # drop_last=(val_name=='DeepFakeDetection'),
                # persistent_workers=True,
                # worker_init_fn=worker_init_fn
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['val_dataset']:
        test_data_loaders[one_test_name] = get_val_data_loader(config, one_test_name)
    return test_data_loaders

