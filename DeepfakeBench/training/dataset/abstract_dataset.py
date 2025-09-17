# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys

import lmdb
import io

sys.path.append('.')

import os
import math
import yaml
import glob
import json
import pickle

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
# from timm.data.random_erasing import RandomErasing
from .frequency_analysis import LGA, LVP

import albumentations as A
from .albu import IsotropicResize
import warnings
warnings.filterwarnings("ignore")
import ipdb

FFpp_pool = ['FaceForensics++','FaceShifter','DeepFakeDetection','FF-DF','FF-F2F','FF-FS','FF-NT']
general_datasets_pool = ['progan_4class', 'progan_20class', 'biggan', 'crn', 'cyclegan', 'deepfake',
                         'gaugan', 'imle', 'progan', 'san', 'seeingdark', 'stargan', 'stylegan2', 'stylegan', 'whichfaceisreal',
                         'dalle', 'glide_100_10', 'glide_100_27', 'glide_50_27', 'guided', 'ldm_100', 'ldm_200', 'ldm_200_cfg',
                         'wukong', 'VQDM', 'stable_diffusion_v_1_5', 'stable_diffusion_v_1_4', 'Midjourney', 'glide', 'ADM', 'BigGAN',
                         'stable_diffusion_v_1_4_sub10x', 'stable_diffusion_v_1_4_sub20x', 'progan_4class_sub5x', 'progan_20class_sub5x',
                         'progan_4class_sub10x', 'progan_20class_sub10x', 'progan_20class_sub20x']
genimage_datasets_pool = ['wukong', 'VQDM', 'stable_diffusion_v_1_5', 'stable_diffusion_v_1_4', 'Midjourney', 'glide', 'ADM', 'BigGAN']
# weights = [np.random.uniform(0, 10) for _ in range(8)]
weights = np.arange(1, 9)
kernel_size = 5
sigma = 1.0
constant = 1.0
lga = LGA(kernel_size, sigma, constant)
lvp = LVP(weights)

# def random_crop(image, crop_height, crop_width):
#     height, width = image.shape
#     max_x = width - crop_width
#     max_y = height - crop_height

#     # 随机选择裁剪的起始点
#     x = np.random.randint(0, max_x)
#     y = np.random.randint(0, max_y)

#     # 裁剪图像
#     crop = image[y:y + crop_height, x:x + crop_width]
#     return crop

# def center_crop(image, crop_height, crop_width):
#     height, width = image.shape

#     # 计算裁剪的起始点
#     start_x = (width - crop_width) // 2
#     start_y = (height - crop_height) // 2

#     # 裁剪图像
#     crop = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
#     return crop

def random_crop(image, crop_height, crop_width):
    height, width = image.shape

    # 如果图像大小小于裁剪尺寸，先调整图像大小
    if height < crop_height or width < crop_width:
        # 方法1：调整图像大小
        # image = cv2.resize(image, (max(width, crop_width), max(height, crop_height)))

        # 方法2：填充图像
        pad_height = max(crop_height - height, 0)
        pad_width = max(crop_width - width, 0)
        image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        height, width = image.shape

    # 随机选择裁剪的起始点
    max_x = width - crop_width
    max_y = height - crop_height
    x = np.random.randint(0, max_x + 1)
    y = np.random.randint(0, max_y + 1)

    # 裁剪图像
    crop = image[y:y + crop_height, x:x + crop_width]
    return crop


def center_crop(image, crop_height, crop_width):
    height, width = image.shape

    # 如果图像大小小于裁剪尺寸，先调整图像大小
    if height < crop_height or width < crop_width:
        # 方法1：调整图像大小
        # image = cv2.resize(image, (max(width, crop_width), max(height, crop_height)))

        # 方法2：填充图像
        pad_height = max(crop_height - height, 0)
        pad_width = max(crop_width - width, 0)
        image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        height, width = image.shape

    # 计算裁剪的起始点
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    # 裁剪图像
    crop = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    return crop


def all_in_pool(inputs,pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class DeepfakeAbstractBaseDataset(Dataset):
    """
    Abstract base class for all deepfake datasets.
    """
    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """
        
        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        # Check if 'video_mode' exists in config, otherwise set video_level to False
        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.lmdb = config.get('lmdb', False)
        self.lmdb_test = config.get('lmdb_test', False)
        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        
        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                if self.config.get("use_jsonl", False):
                    tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset_jsonl(one_data)
                else:
                    tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
            if self.lmdb:
                if len(dataset_list)>1:
                    if all_in_pool(dataset_list, FFpp_pool):
                        lmdb_path = os.path.join(config['lmdb_dir'], f"FaceForensics++_lmdb")
                        self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                    else:
                        raise ValueError('Training with multiple dataset and lmdb is not implemented yet.')
                else:
                    lmdb_path = os.path.join(config['lmdb_dir'], f"{dataset_list[0] if dataset_list[0] not in FFpp_pool else 'FaceForensics++'}_lmdb")
                    self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        elif mode in ['test', 'val']:
            one_data = config['test_dataset']
            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list, name_list = self.collect_img_and_label_for_one_dataset(one_data)
            if self.lmdb_test:
                lmdb_path = os.path.join(config['lmdb_dir'], f"{one_data}_lmdb" if one_data not in FFpp_pool else 'FaceForensics++_lmdb')
                self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list

        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }
        
        self.transform = self.init_data_aug_method() if self.mode == 'train' and self.config['use_data_augmentation'] else None
        self.train_normalize, self.test_normalize = self.build_normalize()

    def build_normalize(self):
        # train normalize
        norm_trans_tr = [
            T.ToTensor(),
            T.Normalize(mean=self.config['mean'], std=self.config['std']),
        ]
        if "random_erasing" in self.config['data_aug']['choices']:
            # scale: erasing面积的范围  ratio: erasing长宽比的范围
            norm_trans_tr += [T.RandomErasing(scale=(0.1, 0.4), ratio=(0.5, 2.0), value=0, p=self.config['data_aug']['cutout_prob'])]
        train_normalize = T.Compose(norm_trans_tr)

        # test normalize
        if self.config['data_aug']['test_resize'] == 'crop':
            norm_trans = [T.CenterCrop(self.config['resolution'])]
        elif self.config['data_aug']['test_resize'] == 'resize':
            norm_trans = [T.Resize((self.config['resolution'], self.config['resolution']), interpolation=InterpolationMode.BICUBIC)]   # NEAREST  BILINEAR  BICUBIC
        else:
            raise NotImplementedError
        norm_trans += [
            T.ToTensor(),
            T.Normalize(mean=self.config['mean'], std=self.config['std']),
        ]
        test_normalize = T.Compose(norm_trans)

        return train_normalize, test_normalize

    def init_data_aug_method(self):
        trans = []
        choices = self.config['data_aug']['choices']
        if 'resize' in choices:
            # trans += [
            #     A.OneOf([                
            #         IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            #         IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            #         IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            #     ], p = 0 if self.config['with_landmark'] else 1)
            # ]
            trans += [A.Resize(height=self.config['resolution'], width=self.config['resolution'], interpolation=cv2.INTER_CUBIC)]
        if 'random_crop' in choices:
            trans += [A.RandomCrop(height=self.config['resolution'], width=self.config['resolution'], p=1)]
        if 'random_resize_crop' in choices:
            trans += A.RandomResizedCrop(size=self.config['resolution'], scale=(0.7, 1.0), ratio=(0.75, 1.333), p=1)
        if 'resize_then_crop' in choices:
            trans += [
                A.Resize(height=256, width=256, interpolation=cv2.INTER_CUBIC),
                A.RandomCrop(height=self.config['resolution'], width=self.config['resolution'], p=1)
            ]
        if 'random_flip' in choices:
            trans += [A.HorizontalFlip(p=self.config['data_aug']['flip_prob'])]
        if 'random_rotate' in choices:
            trans += [A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob'])]
        if 'colorjitter' in choices:
            trans += [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=self.config['data_aug']['colorjitter_prob'])]
        if 'blur' in choices:
            trans += [A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob'])]
        if 'brightness_contrast' in choices:
            trans += [
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                    A.FancyPCA(),
                    A.HueSaturationValue()
                ], p=0.5)
            ]
        if 'compression' in choices:
            trans += [A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)]
        # if 'cutout' in choices:
        #     trans += [A.CoarseDropout(num_holes_range=self.config['data_aug']['num_holes'], 
        #                               hole_height_range=self.config['data_aug']['size_holes'], 
        #                               hole_width_range=self.config['data_aug']['size_holes'], 
        #                               p=self.config['data_aug']['cutout_prob'])]
        # if 'random_erasing' in choices:
        #     # scale: erasing面积的范围  ratio: erasing长宽比的范围
        #     trans += [A.Erasing(scale=(0.1, 0.4), ratio=(0.5, 2.0), fill_value=0, p=self.config['data_aug']['erasing_prob'])]
        # # Other types of cutout: MaskDropout, GridDropout, PixelDropout

        trans = A.Compose(
            trans,
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def rescale_landmarks(self, landmarks, original_size=256, new_size=224):
        scale_factor = new_size / original_size
        rescaled_landmarks = landmarks * scale_factor
        return rescaled_landmarks


    def collect_img_and_label_for_one_dataset_jsonl(self, dataset_name: str):
        label_list, frame_path_list = [], []
        try:
            with open(os.path.join(self.config['dataset_jsonl_folder'], dataset_name + '.jsonl'), 'r') as f:
                for line in f:
                    item = json.loads(line)
                    if item["split"] != "train":
                        continue
                    img_paths = item['image']
                    img_label = item['forgery_type']

                    if isinstance(img_paths, str):
                        frame_path_list.append(img_paths)
                        label = 0 if 'real' in img_label.lower() else 1
                        label_list.append(label)
                    elif isinstance(img_paths, list):
                        for path in img_paths:
                            frame_path_list.append(path)
                            label = 0 if 'original_sequences' in path.lower() else 1
                            label_list.append(label)
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image'])}")
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        shuffled = list(zip(label_list, frame_path_list))
        random.shuffle(shuffled)
        label_list, frame_path_list = zip(*shuffled)
        
        return frame_path_list, label_list, None


    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.
        
        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []
        
        # Record video name for video-level metrics
        video_name_list = []

        # Try to get the dataset information from the JSON file
        try:
            with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        # Get the information for the current dataset
        for label in dataset_info[dataset_name]:
            if self.mode=='val':
                # try:
                #     sub_dataset_info = dataset_info[dataset_name][label][self.mode]
                # except:
                sub_dataset_info = dataset_info[dataset_name][label]['test']
            else:
                sub_dataset_info = dataset_info[dataset_name][label][self.mode]
            # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
            if dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter']:
                sub_dataset_info = sub_dataset_info[self.compression]

            # Iterate over the videos in the dataset
            for video_name, video_info in sub_dataset_info.items():
                try:
                    # Unique video name
                    unique_video_name = video_info['label'] + '_' + video_name
                except:
                    ipdb.set_trace()
                # Get the label and frame paths for the current video
                if video_info['label'] not in self.config['label_dict']:
                    self.config['label_dict'][video_info['label']] = 1 if '_fake' in video_info['label'].lower() else 0
                    # raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
                label = self.config['label_dict'][video_info['label']]
                frame_paths = video_info['frames']
                # sorted video path to the lists
                if dataset_name not in general_datasets_pool:
                    try:
                        if '\\' in frame_paths[0]:
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                        else:
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                    except:
                        pass

                # Consider the case when the actual number of frames (e.g., 270) is larger than the specified (i.e., self.frame_num=32)
                # In this case, we select self.frame_num frames from the original 270 frames
                total_frames = len(frame_paths)
                if self.mode == 'train' and not isinstance(self.frame_num, int):
                    real_frame_num, fake_frame_num = self.frame_num["real"], self.frame_num["fake"]
                    if label == 0:
                        # real
                        if real_frame_num < total_frames:
                            total_frames = real_frame_num
                            if self.video_level:
                                # Select clip_size continuous frames
                                start_frame = random.randint(0, total_frames - real_frame_num) if self.mode == 'train' else 0
                                frame_paths = frame_paths[start_frame:start_frame + real_frame_num]  # update total_frames
                            else:
                                # Select self.frame_num frames evenly distributed throughout the video
                                step = total_frames // real_frame_num
                                frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:real_frame_num]
                    else:
                        # fake
                        if fake_frame_num < total_frames:
                            total_frames = fake_frame_num
                            if self.video_level:
                                # Select clip_size continuous frames
                                start_frame = random.randint(0, total_frames - fake_frame_num) if self.mode == 'train' else 0
                                frame_paths = frame_paths[start_frame:start_frame + fake_frame_num]  # update total_frames
                            else:
                                # Select self.frame_num frames evenly distributed throughout the video
                                step = total_frames // fake_frame_num
                                frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:fake_frame_num]
                else:
                    if self.frame_num < total_frames:
                        total_frames = self.frame_num
                        if self.video_level:
                            # Select clip_size continuous frames
                            start_frame = random.randint(0, total_frames - self.frame_num) if self.mode == 'train' else 0
                            frame_paths = frame_paths[start_frame:start_frame + self.frame_num]  # update total_frames
                        else:
                            # Select self.frame_num frames evenly distributed throughout the video
                            step = total_frames // self.frame_num
                            frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]
                
                # If video-level methods, crop clips from the selected frames if needed
                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError('clip_size must be specified when video_level is True.')
                    # Check if the number of total frames is greater than or equal to clip_size
                    if total_frames >= self.clip_size:
                        # Initialize an empty list to store the selected continuous frames
                        selected_clips = []

                        # Calculate the number of clips to select
                        num_clips = total_frames // self.clip_size

                        if num_clips > 1:
                            # Calculate the step size between each clip
                            clip_step = (total_frames - self.clip_size) // (num_clips - 1)

                            # Select clip_size continuous frames from each part of the video
                            for i in range(num_clips):
                                # Ensure start_frame + self.clip_size - 1 does not exceed the index of the last frame
                                start_frame = random.randrange(i * clip_step, min((i + 1) * clip_step, total_frames - self.clip_size + 1)) if self.mode == 'train' else i * clip_step
                                continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                                assert len(continuous_frames) == self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                                selected_clips.append(continuous_frames)

                        else:
                            start_frame = random.randrange(0, total_frames - self.clip_size + 1) if self.mode == 'train' else 0
                            continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                            assert len(continuous_frames)==self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                            selected_clips.append(continuous_frames)

                        # Append the list of selected clips and append the label
                        label_list.extend([label] * len(selected_clips))
                        frame_path_list.extend(selected_clips)
                        # video name save
                        video_name_list.extend([unique_video_name] * len(selected_clips))

                    else:
                        print(f"Skipping video {unique_video_name} because it has less than clip_size ({self.clip_size}) frames ({total_frames}).")
                
                # Otherwise, extend the label and frame paths to the lists according to the number of frames
                else:
                    # Extend the label and frame paths to the lists according to the number of frames
                    label_list.extend([label] * total_frames)
                    frame_path_list.extend(frame_paths)
                    # video name save
                    video_name_list.extend([unique_video_name] * len(frame_paths))
            
        # Shuffle the label and frame path lists in the same order
        # random.seed(100)
        shuffled = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list = zip(*shuffled)
        
        return frame_path_list, label_list, video_name_list
     
    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution'] # if self.mode == "train" else self.config['resolution']
        use_lmdb = self.lmdb if self.mode == 'train' else self.lmdb_test
        if not use_lmdb:
            if not file_path[0] == '.':
                tmp_path = os.path.join(f'{self.config["rgb_dir"]}', file_path).replace('\\', '/')
                if os.path.exists(tmp_path):
                    file_path = tmp_path
                else:
                    file_path = os.path.join(f'{self.config["rgb_dir2"]}', file_path).replace('\\', '/')
            img = Image.open(file_path).convert('RGB')
            img = np.array(img)

            if img is None:
                raise ValueError('Loaded image is None: {}'.format(file_path))
        elif use_lmdb:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')

                image_bin = txn.get(file_path.encode())
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if not self.lmdb:
            if not file_path[0] == '.':
                file_path = os.path.join(f'{self.config["rgb_dir"]}', file_path).replace('\\', '/')
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')

                image_bin = txn.get(file_path.encode())
                if image_bin is None:
                    mask = np.zeros((size, size,3))
                else:
                    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                    # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
                    mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if not self.lmdb:
            if not file_path[0] == '.':
                # file_path =  f'{self.config["rgb_dir"]}/'+file_path
                file_path = os.path.join(f'{self.config["rgb_dir"]}', file_path).replace('\\', '/')
            if os.path.exists(file_path):
                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')
                binary = txn.get(file_path.encode())
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
                landmark=self.rescale_landmarks(np.float32(landmark), original_size=256, new_size=self.config['resolution'])
        return landmark

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Set the seed for the random number generator
        # if augmentation_seed is not None:
        #     random.seed(augmentation_seed)
        #     np.random.seed(augmentation_seed)

        # Create a dictionary of arguments
        kwargs = {'image': img}
        
        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0:
                kwargs['mask'] = mask

        # Apply data augmentation
        self.transform.set_random_seed(augmentation_seed)
        transformed = self.transform(**kwargs)
        
        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask', mask)

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        # Reset the seeds to ensure different transformations for different videos
        # if augmentation_seed is not None:
        #     random.seed()
        #     np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]  # for the image-level IO, only one frame is used

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)
            else:
                augmentation_seed = random.randint(0, 2**32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

            # Load the image
            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                print(f"Error loading image at index {index}: {e}")
                return self.__getitem__(0)

            # Load mask and landmark (if needed)
            if self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

            # prepare frequency inputs
            if self.config["model_config"]["use_frequency"]:
                img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                # print(img_gray.shape)
                if self.mode == 'train':
                    img_gray = random_crop(img_gray, 224, 224)
                else:
                    img_gray = center_crop(img_gray, 224, 224)
                # print(img_gray.shape)
                # print("/n")
                img_gray = img_gray.astype(np.float32)
                img_lga = lga(img_gray)
                # img_lvp = lvp(img_gray)
                img_lga = torch.tensor(img_lga).unsqueeze(0)
                # img_lvp = torch.tensor(img_lvp).unsqueeze(0)
    
            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(np.array(image), landmarks, mask, augmentation_seed)
                image_trans = self.train_normalize(image_trans)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)
                image_trans = self.test_normalize(image_trans)
            
            # To tensor and normalize
            if not no_norm:
                # image_trans = self.normalize(self.to_tensor(image_trans))
                if self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(landmarks)
                if self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)
            # Stack landmark and mask tensors along a new dimension (time)
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]
            # Get the first landmark and mask tensors
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]

        if self.config["model_config"]["use_frequency"]:
            return image_tensors, label, landmark_tensors, mask_tensors, image_paths, img_lga
        else:
            return image_tensors, label, landmark_tensors, mask_tensors, image_paths
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        try:
            images, labels, landmarks, masks, image_paths, img_lga = zip(*batch)
        except:
            images, labels, landmarks, masks, image_paths = zip(*batch)
            img_lga = None
        
        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        if img_lga is not None:
            img_lga = torch.stack(img_lga, dim=0)
            # img_lvp = torch.stack(img_lvp, dim=0)
        labels = torch.LongTensor(labels)
        
        # Special case for landmarks and masks if they are None
        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmarks):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        data_dict['path'] = image_paths
        if img_lga is not None:
            data_dict['img_lga'] = img_lga
            # data_dict['img_lvp'] = img_lvp
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)
    
    # def to_tensor(self, img):
    #     """
    #     Convert an image to a PyTorch tensor.
    #     """
    #     return T.ToTensor()(img)

    # def normalize(self, img):
    #     """
    #     Normalize an image.
    #     """
    #     mean = self.config['mean']
    #     std = self.config['std']
    #     normalize = T.Normalize(mean=mean, std=std)
    #     return normalize(img)


if __name__ == "__main__":
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/video_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
                config = config,
                mode = 'train', 
            )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        # print(iteration)
        ...
        # if iteration > 10:
        #     break
