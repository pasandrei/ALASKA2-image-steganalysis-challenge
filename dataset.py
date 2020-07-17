from pathlib import Path

from albumentations import (
    Resize,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    CoarseDropout,
    ToGray,
    Compose,
    RandomRotate90
)

import cv2

import numpy as np

import torch
from torch.utils.data import Dataset


class ALASKA2Dataset(Dataset):
    def __init__(self, annotations, root_dir, augmented):
        """
        Args:
            annotations [[string, int]]
            root_dir (string): Directory with all the images.
            augmented (bool): True if we want to augment the dataset with flips, rotate, etc
        """

        self.annotations = annotations
        self.root_dir = root_dir
        self.augmented = augmented

        if (augmented):
            self.augmentations = Compose([HorizontalFlip(), VerticalFlip(), RandomRotate90(),
                                          CoarseDropout(max_holes=16, max_height=32, max_width=32)])
        else:
            self.augmentations = Compose([])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = Path(self.root_dir) / self.annotations[idx][0]

        image = cv2.imread(str(img_name))

        image = np.array(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented_image = self.augmentations(image=image)
        image = augmented_image['image']

        image = np.float32(image.transpose(2, 0, 1)) / 255 # HxWxC to CxHxW

        sample = {'image': image, 'image_path': str(img_name),
                  'ground_truth': self.annotations[idx][1]}

        return sample
