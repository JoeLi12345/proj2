import sys, os, time, glob, time, pdb, cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
plt.switch_backend('agg') # for servers not supporting display

sys.path.insert(0,'..')

# import neccesary libraries for defining the optimizers
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassF1Score
from unet import UNet
from unet_nn import UNet_Fine
from datasets import WESADDatasetFine

import yaml
config_path = "config_wesad_supervised.yaml"  if len(sys.argv) < 2 else sys.argv[-1]
with open(config_path, 'r') as file:
	config = yaml.safe_load(file) 
from torch.utils.data import DataLoader
import wandb

def init_wandb(name=None):
	wandb.init(project="unet_ssl", entity="jli505", config=config, reinit=True, name=name)
	global cfg
	cfg = wandb.config
	global checkpoints_dir
	checkpoints_dir = f"{wandb.run.dir}/checkpoints"
	os.makedirs(checkpoints_dir, exist_ok = True)

def train(subject, weight=0.5, remove_percent=0.0, train_dataset=None, seed=None):
	torch.manual_seed(1347)
	if train_dataset is None:
		train_dataset = WESADDatasetFine(split="train", subject=subject, train_type="personalized", remove_percent=remove_percent, seed=seed)
	val_dataset = WESADDatasetFine(split="validation", subject=subject, train_type="personalized", remove_percent=remove_percent, transform=None)
	sz1, sz2 = len(train_dataset), len(val_dataset)
	print("train, val = ",len(train_dataset), len(val_dataset))
	unet_pretrained = UNet(in_channels=train_dataset.in_channels, n_classes=train_dataset.in_channels, depth=config['depth'], wf=2, padding=True)
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	print('device: ', device)

	script_time = time.time()
	#load the training dataset with a weighted sampler
	batch_size = cfg.batch_size

	train_samples_weight = []
	for i in range(len(train_dataset)):
		if train_dataset.activities[i] == 1:
			train_samples_weight.append(1.)
		else:
			train_samples_weight.append(0.)

	train_samples_weight = torch.from_numpy(np.array(train_samples_weight))
	train_sampler = torch.utils.data.WeightedRandomSampler(train_samples_weight, len(train_samples_weight), replacement=True)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, sampler=train_sampler)
	#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, sampler = sampler)

	val_samples_weight = []
	t1, t2 = 0, 0
	for i in range(len(val_dataset)):
		if val_dataset.activities[i] == 1:
			t1 += 1
		elif val_dataset.activities[i] == 2:
			t2 += 1
	for i in range(len(val_dataset)):
		if val_dataset.activities[i] == 1:
			val_samples_weight.append(1./t1)
		elif val_dataset.activities[i] == 2:
			val_samples_weight.append(1./t2)
		else:
			val_samples_weight.append(0.)
	val_samples_weight = torch.from_numpy(np.array(val_samples_weight))
	val_sampler = torch.utils.data.WeightedRandomSampler(val_samples_weight, len(val_samples_weight), replacement=True)
	
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, sampler=val_sampler)

	for batch_idx, (imgs, activity) in enumerate(val_loader):
		activity = np.array(activity)
		cnt1, cnt2 = 0, 0
		for x in activity:
			if x == 1:
				cnt1 += 1
			elif x == 2:
				cnt2 += 1
		print(cnt1, cnt2)

init_wandb(name="verify_batch")
train(4, weight=1)
