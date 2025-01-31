import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

def pad_to_square(img):
    width, height = img.size
    max_size = max(width, height)
    
    pad_left = (max_size - width) // 2
    pad_top = (max_size - height) // 2
    pad_right = max_size - width - pad_left
    pad_bottom = max_size - height - pad_top
    
    return transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)(img)


def to_square(img):
    width, height = img.size
    max_size_of_pic = max(width, height)

    pad_left = (max_size_of_pic - width)//2
    pad_top = (max_size_of_pic - height)//2
    pad_right = max_size_of_pic - width - pad_left
    pad_bottom = max_size_of_pic - height - pad_top

    return transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0)(img)


# аугментации
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.15, hue=(-0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.Lambda(to_square),
    transforms.Resize(200, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()])

class SignDataset(Dataset):
    def __init__(self, paths: List[Path], transform=None):
        self.paths = paths
        self.transform = transform # если есть аугментации
        labels = sorted(set(str(x).split('/')[-2] for x in paths))
        # labels = sorted(set(str(x).split(os.sep)[-2] for x in paths))
        # labels = sorted(set(x.parent.name for x in paths))
        self.one_hot_encoding = {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.paths[idx]))
        label = str(self.paths[idx]).split('/')[-2]
        # image = cv2.resize(image, (200, 200))
        # image = np.transpose(image, (2, 0, 1))
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = cv2.resize(image, (200, 200))
            image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image).float(), torch.tensor(self.one_hot_encoding[label])


def get_sign_dataloader(
        path_train, path_val, batch_size, shuffle=True, num_workers=1,
    ):
    train_dataset = SignDataset(paths=[*Path(path_train).rglob('*.jpg')])
    val_dataset = SignDataset(paths=[*Path(path_val).rglob('*.jpg')])

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    return DataLoader(train_dataset, **loader_args), DataLoader(val_dataset, **loader_args)


def get_sign_test_dataloader(
        path_test, batch_size, num_workers=1,
    ):
    test_dataset = SignDataset(paths=[*Path(path_test).rglob('*.jpg')])

    loader_args = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers
    }
    return DataLoader(test_dataset, **loader_args)
