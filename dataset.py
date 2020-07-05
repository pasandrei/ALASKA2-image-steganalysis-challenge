from pathlib import Path

from albumentations import (
	Resize,
	RandomResizedCrop,
	HorizontalFlip,
	VerticalFlip,
	Rotate,
	CoarseDropout,
	GaussNoise,
	RandomBrightnessContrast,
	RandomGamma,
	ToGray,
	Compose,
	Blur,
	OpticalDistortion,
	RGBShift
)

import cv2

import numpy as np

import random

import torch
from torch.utils.data import Dataset
import os


class ALASKA2Dataset(Dataset):
	def __init__(self, annotations, root_dir, augmented):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""

		self.annotations = annotations
		self.root_dir = root_dir
		self.augmented = augmented
		self.weights = []

		for i in range(len(annotations)):
			self.weights.append(float(annotations[i][2]))

		if (augmented):
			# self.augmentations = Compose([RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
			# 							  HorizontalFlip(), VerticalFlip(),
			# 							  CoarseDropout(max_holes=12, max_height=20, max_width=20)], p=1.0)
			self.augmentations = Compose([Resize(height=224, width=224), CoarseDropout(max_holes=16, max_height=32, max_width=32)])
		else:
			self.augmentations = Compose([Resize(height=224, width=224)])


	def get_weights(self):
		return self.weights

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

		image = np.float32(image.transpose(2, 0, 1)) / 255
		# image = np.float32(image) / 255

		# cv2.imshow("", cv2.cvtColor(augmented_image['image'], cv2.COLOR_RGB2BGR))
		# cv2.waitKey()

		sample = {'image': image, 'image_path': str(img_name), 'ground_truth': self.annotations[idx][1]}

		return sample
