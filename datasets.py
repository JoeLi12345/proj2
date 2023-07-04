import numpy as np
import os, glob, cv2, sys
import torch
import random
from pandas import read_csv
import pandas as pd
import pickle
from torchvision import transforms
import copy

class WESADDataset(torch.utils.data.Dataset):
	def __init__(self, split, subject, train_type, transform=None):
		self.split = split
		self.subject = subject
		self.train_type = train_type
		self.transform = transform
		self.data, self.activities, self.same_label = [], [], []
		if train_type == "personalized" or split == "test":
			self.load_dataset(self.subject)
			a, b, c = self.format_data()
			self.data += a
			self.activities += b
			self.same_label += c
		elif train_type == "generalized":
			for i in range(15):
				self.load_dataset(i)
				a, b, c = self.format_data()
				self.data += a
				self.activities += b
				self.same_label += c
		elif train_type == "generalized_minus":
			for i in range(15):
				if i == self.subject:
					continue
				self.load_dataset(i)
				a, b, c = self.format_data()
				self.data += a
				self.activities += b
				self.same_label += c

	#load the files from HAR directory

	def load_dataset(self, subject):
		vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
		x = vals[subject]
		df = pd.read_pickle('../WESAD/S'+str(x)+'/S'+str(x)+'.pkl')
		self.signals = []
		self.labels = []
		self.in_channels = 0
		for keys in df['signal']['chest']:
			self.in_channels += df['signal']['chest'][keys].shape[1]
		for keys in df['signal']['chest']:
			x = df['signal']['chest'][keys]
			for i in range(x.shape[1]):
				arr = []
				for j in range(x.shape[0]):
					arr.append(x[j][i])
				self.signals.append(arr)
		for x in df['label']:
			self.labels.append(x)
	
	#create "self.data" such that the data is in the order of train, val, and test
	#create windows with 64 data points each
	def format_data(self):
		dx, cur, inc, num_activities = 64, 0, 32, 3
		data, activities, same_label, category = [], [], [], []
		for i in range(num_activities):
			category.append([])
		#partition into windows of length dx
		while (cur+dx-1 < len(self.signals[0])):
			value = self.labels[cur]
			same = True
			valid = True
			freq = [0]*10
			for i in range(cur, cur+dx):
				if self.labels[i] < 1 or self.labels[i] > 3:
					valid = False;
					break
				if self.labels[i] != value:
					same = False
				freq[self.labels[i]] += 1
			if valid:
				value = np.argmax(freq)
				window = []
				for i in range(self.in_channels):
					window.append(self.signals[i][cur:cur+dx])
				category[value-1].append((window, same))
			cur += inc
		#70, 15, 15 split for training, validation, testing
		for i in range(len(category)):
			n = len(category[i])
			partition = [0, int(0.7*n), int(0.85*n), n]
			partition_index = -1
			if self.split == "train":
				partition_index = 0
			elif self.split == "validation":
				partition_index = 1
			elif self.split == "test":
				partition_index = 2
			else:
				print("invalid train/val/test arg")
				exit()
			for j in range(partition[partition_index], partition[partition_index+1]):
				data.append(category[i][j][0])
				activities.append(i)
				same_label.append(category[i][j][1])
		mean = np.expand_dims(np.array(data).mean(axis=(0, 2)), -1)
		std = np.expand_dims(np.array(data).std(axis=(0, 2)), -1)
		if self.transform is None:
			self.transform = lambda x: (x-mean)/std
		data = self.transform(np.array(data))
		data = list(data)
		return data, activities, same_label

	def __getitem__(self, index):
		reg = self.data[index]
		noised = copy.deepcopy(reg)
		#implement masking by randomly selecting some data points to mask out
		subset_indices = random.sample(range(len(reg[0])), int(self.mask_amount*len(reg[0])))
		cnt=0
		for i in subset_indices:
			for j in range(len(noised)):
				noised[j][i] = self.mask_value
				cnt += 1
		reg = torch.tensor(reg, dtype=torch.float)
		noised = torch.tensor(noised, dtype=torch.float)
		#print("-1 count", ( torch.abs(noised+1) < 0.01).sum())
		#change later
		#return reg, reg
		return reg, noised

	def __len__(self):
		return len(self.data)

#data loader for the transfer learning task
class WESADDatasetFine(WESADDataset):
	def __init__(self, split, subject, train_type, remove_percent=0.0, seed=None, transform=None):
		super().__init__(split, subject, train_type, transform)
		self.seed = seed
		self.remove_points()
		self.remove_percent = remove_percent
		if split == "train":
			self.remove_labeled()
		self.calc_freq()

	def calc_freq(self):
		freq = [0]*3
		for x in self.activities:
			freq[x] += 1
		print(self.split, "freq: ", freq)

	def remove_points(self):
		erase_indices = []
		for i in range(len(self.same_label)):
			if self.same_label[i] == False:
				erase_indices.append(i)
		erase_indices.reverse()
		for ind in erase_indices:
			self.data.pop(ind)
			self.activities.pop(ind)
			self.same_label.pop(ind)

	def remove_labeled(self):
		if self.seed is not None:
			random.seed(self.seed)
		for i in range(3):
			potential_indices = []
			for j in range(len(self.activities)):
				if self.activities[j] == i:
					potential_indices.append(j)
			erase_indices = random.sample(potential_indices, int(self.remove_percent*len(potential_indices)))
			erase_indices.sort()
			erase_indices.reverse()
			for ind in erase_indices:
				self.data.pop(ind)
				self.activities.pop(ind)
				self.same_label.pop(ind)
				
		'''erase_indices = random.sample(range(len(self.data)), int(self.remove_percent*len(self.data)))
		erase_indices.sort()
		erase_indices.reverse()
		for ind in erase_indices:
			self.data.pop(ind)
			self.activities.pop(ind)
			self.same_label.pop(ind)'''

	def __getitem__(self, index):
		X = self.data[index]
		X = torch.tensor(X, dtype=torch.float)
		y = self.activities[index]
		y = torch.tensor(y, dtype=torch.int64)
		return X, y

'''dataset = WESADDataset(split="train", subject=8)
#dataset.__getitem__(3)
import matplotlib.pyplot as plt
for i in range(8):
	x = range(len(dataset.signals[i][2000000:2100000]))
	y = dataset.signals[i][2000000:2100000]
	plt.plot(x, y, color="red")
	plt.ylabel()
	plt.show()'''
