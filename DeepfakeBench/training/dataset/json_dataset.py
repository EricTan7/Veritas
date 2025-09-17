
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
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO
import requests



class JSONDataset(Dataset):
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

        # Check if 'video_mode' exists in config, otherwise set video_level to False
        self.video_level = config.get('video_mode', False)
        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        self.video_id_list = []
        
        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            video_id_list = []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_video_id, tmp_name = self.collect_img_and_label_for_one_dataset_json(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
                video_id_list.extend(tmp_video_id)
            shuffled = list(zip(label_list, image_list, video_id_list))
            random.shuffle(shuffled)
            label_list, image_list, video_id_list = zip(*shuffled)

        elif mode in ['test', 'val']:
            one_data = config['test_dataset']
            image_list, label_list, tmp_video_id, name_list = self.collect_img_and_label_for_one_dataset_json(one_data)
            video_id_list = None
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        if video_id_list is not None:
            label_map = {}
            result = []
            counter = 0
            for item in video_id_list:
                if item not in label_map:
                    label_map[item] = counter
                    counter += 1
                result.append(label_map[item])
            video_id_list = result

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list
        self.video_id_list = video_id_list

        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list,
            'video_id': self.video_id_list
        }
        
        self.transform = self.init_data_aug_method() if self.mode == 'train' and self.config['use_data_augmentation'] else None
        self.train_normalize, self.test_normalize = self.build_normalize()
        if "test_transform_cospy" in self.config:
            self.test_normalize = self.config["test_transform_cospy"]
            print("*"*50, "using cospy test transform")

        if self.config["model_name"] == "aide":
            self.dct = DCT_base_Rec_Module()

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

        trans = A.Compose(
            trans,
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def rescale_landmarks(self, landmarks, original_size=256, new_size=224):
        scale_factor = new_size / original_size
        rescaled_landmarks = landmarks * scale_factor
        return rescaled_landmarks

    def collect_img_and_label_for_one_dataset_json(self, dataset_name: str):
        label_list, frame_path_list = [], []
        video_id_list = []
        try:
            with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                data = json.load(f)
                for item in data:
                    img_paths = item['images']
                    img_label = item['label']

                    if isinstance(img_paths, str):
                        frame_path_list.append(img_paths)
                        label_list.append(img_label)
                    elif isinstance(img_paths, list):
                        for path in img_paths:
                            frame_path_list.append(path)
                            label_list.append(img_label)
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image'])}")
                    try:
                        video_id_list.append(item["video_id"])
                    except:
                        video_id_list.append("0")
        except Exception as e:
            print(e)
            print(dataset_name)
            raise ValueError(f'dataset {dataset_name} not exist!')

        shuffled = list(zip(label_list, frame_path_list, video_id_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_id_list = zip(*shuffled)
        
        return frame_path_list, label_list, video_id_list, None

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
        size = self.config['resolution']
        try:
            img = Image.open(file_path).convert('RGB')
        except:
            response = requests.get(file_path)
            response.raise_for_status() 
            image_bytes = response.content
            image_file_in_memory = io.BytesIO(image_bytes)
            img = Image.open(image_file_in_memory)
        if img is None:
            raise ValueError('Loaded image is None: {}'.format(file_path))
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

        if not file_path[0] == '.':
            file_path = os.path.join(f'{self.config["rgb_dir"]}', file_path).replace('\\', '/')
        if os.path.exists(file_path):
            mask = cv2.imread(file_path, 0)
            if mask is None:
                mask = np.zeros((size, size))
        else:
            return np.zeros((size, size, 1))
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
        if not file_path[0] == '.':
            # file_path =  f'{self.config["rgb_dir"]}/'+file_path
            file_path = os.path.join(f'{self.config["rgb_dir"]}', file_path).replace('\\', '/')
        if os.path.exists(file_path):
            landmark = np.load(file_path)
        else:
            return np.zeros((81, 2))
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

        if self.config["model_name"] == "aide":
            image_path = image_paths[0]
            image = self.load_rgb(image_path)
            image = transform_before_test(image)

            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)

            x_0 = transform_train(image)
            x_minmin = transform_train(x_minmin) 
            x_maxmax = transform_train(x_maxmax)

            x_minmin1 = transform_train(x_minmin1) 
            x_maxmax1 = transform_train(x_maxmax1)

            image_tensors = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0)
            landmark_tensors = torch.tensor(0)
            mask_tensors = torch.tensor(0)

            
            return image_tensors, label, landmark_tensors, mask_tensors, image_paths

        else:
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
        
                # Do Data Augmentation
                if self.mode == 'train' and self.config['use_data_augmentation']:
                    image_trans, landmarks_trans, mask_trans = self.data_aug(np.array(image), landmarks, mask, augmentation_seed)
                    image_trans = self.train_normalize(image_trans)
                else:
                    # robustness
                    if self.config["gaussian_sigma"] is not None:
                        sigma = self.config["gaussian_sigma"]
                        sigma = float(sigma)
                        image = np.array(image)
                        gaussian_filter(image[:,:,0], output=image[:,:,0], sigma=sigma)
                        gaussian_filter(image[:,:,1], output=image[:,:,1], sigma=sigma)
                        gaussian_filter(image[:,:,2], output=image[:,:,2], sigma=sigma)
                        image = Image.fromarray(image)
                    if self.config["jpeg_quality"] is not None:
                        quality = self.config["jpeg_quality"]
                        quality = int(quality)
                        out = BytesIO()
                        image.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
                        image = Image.open(out)
                        # load from memory before ByteIO closes
                        image = np.array(image)
                        out.close()
                        image = Image.fromarray(image)

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
    


transform_before_test = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform_train = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)



def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class DCT_base_Rec_Module(nn.Module):
    """_summary_

    Args:
        x: [C, H, W] -> [C*level, output, output]
    """
    def __init__(self, window_size=32, stride=16, output=256, grade_N=6, level_fliter=[0]):
        super().__init__()
        
        assert output % window_size == 0
        assert len(level_fliter) > 0
        
        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
        self.N = (output // window_size) * (output // window_size)
        
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)
        
        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), stride=stride
        )
        self.fold0 = nn.Fold(
            output_size=(window_size, window_size), 
            kernel_size=(window_size, window_size), 
            stride=window_size
        )
        
        lm, mh = 2.82, 2
        level_f = [
            Filter(window_size, 0, window_size * 2)
        ]
        
        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList([Filter(window_size, window_size * 2. / grade_N * i, window_size * 2. / grade_N * (i+1), norm=True) for i in range(grade_N)])
        
        
    def forward(self, x):
        
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        C, W, H = x.shape
        x_unfold = self.unfold(x.unsqueeze(0)).squeeze(0)  
        

        _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(0, 1).reshape(L, C, window_size, window_size) 
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T
        
        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=1)
        
        grade = torch.zeros(L).to(x.device)
        w, k = 1, 2
        for _ in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[_](_x)
            _x = torch.sum(_x, dim=[1,2,3])
            grade += w * _x            
            w *= k
        
        _, idx = torch.sort(grade)
        max_idx = torch.flip(idx, dims=[0])[:N]
        maxmax_idx = max_idx[0]
        if len(max_idx) == 1:
            maxmax_idx1 = max_idx[0]
        else:
            maxmax_idx1 = max_idx[1]

        min_idx = idx[:N]
        minmin_idx = idx[0]
        if len(min_idx) == 1:
            minmin_idx1 = idx[0]
        else:
            minmin_idx1 = idx[1]

        x_minmin = torch.index_select(level_x_unfold, 0, minmin_idx)
        x_maxmax = torch.index_select(level_x_unfold, 0, maxmax_idx)
        x_minmin1 = torch.index_select(level_x_unfold, 0, minmin_idx1)
        x_maxmax1 = torch.index_select(level_x_unfold, 0, maxmax_idx1)

        x_minmin = x_minmin.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)
        x_maxmax = x_maxmax.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)
        x_minmin1 = x_minmin1.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)
        x_maxmax1 = x_maxmax1.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)

        x_minmin = self.fold0(x_minmin)
        x_maxmax = self.fold0(x_maxmax)
        x_minmin1 = self.fold0(x_minmin1)
        x_maxmax1 = self.fold0(x_maxmax1)

       
        return x_minmin, x_maxmax, x_minmin1, x_maxmax1


        


