import torch
import numpy as np
import cv2
import os
from configs.system_device import device
from pathlib import Path


def construct_dataset(root_dir):
	train = []
	test = []

	root_dir = Path(root_dir)

	for folder in os.listdir(root_dir):
		folder_path = root_dir / folder

		if not os.path.isdir(folder_path):
			continue

		if folder == "Test":
			for image in os.listdir(folder_path):
				test.append([folder + '/' + image, 0, 1.0])
		else:
			if folder == "Cover":
				label = 0
				weight = 1.0
			elif folder == "JMiPOD":
				label = 1
				weight = 1.0
			elif folder == "JUNIWARD":
				label = 2
				weight = 1.0
			else:
				label = 3
				weight = 1.0

			for image in os.listdir(folder_path):
				train.append([folder + '/' + image, label, weight])

	return train, test


def generate_mean_std(amp, _3d=False, mean=None, std=None):
	if mean is None or std is None:
		mean_val = [0.485, 0.456, 0.406]  # RGB
		std_val = [0.229, 0.224, 0.225]  # RGB
	else:
		mean_val = mean
		std_val = std

	# mean_val = [0.406, 0.456, 0.485]  # BGR
	# std_val = [0.225, 0.224, 0.229]   # BGR

	mean = torch.tensor(mean_val)
	std = torch.tensor(std_val)

	mean = mean.cuda()
	std = std.cuda()

	view = [1, len(mean_val), 1, 1]
	if _3d:
		view = [1, len(mean_val), 1, 1, 1]

	mean = mean.view(*view)
	std = std.view(*view)

	if amp:
		mean = mean.half()
		std = std.half()

	return mean, std


def compute_dataset_mean_std():
	pixel_mean = np.zeros(3)
	pixel_std = np.zeros(3)
	k = 1
	# for nbatch, data in enumerate(tqdm(train_dataloader)):
	# 	image = data['image'].numpy()
	#
	# 	pixels = image.reshape((-1, image.shape[3]))
	#
	# 	for pixel in pixels:
	# 		diff = pixel - pixel_mean
	# 		pixel_mean += diff / k
	# 		pixel_std += diff * (pixel - pixel_mean)
	# 		k += 1
	#
	# 	print(pixel_mean)
	# 	print(np.sqrt(pixel_std / (k - 2)))

	pixel_std = np.sqrt(pixel_std / (k - 2))
	print(pixel_mean)
	print(pixel_std)

	return pixel_mean, pixel_std


def convert_image(image, size=(224, 224)):
	"""
	Returns and RGB image as a numpy array where of the form C x H x W with values in range [0,1].
	"""

	image = cv2.resize(image, size)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = np.float32(image.transpose(2, 0, 1)) / 255

	return image
