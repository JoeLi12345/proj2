import torch
from torch import nn
import torch.nn.functional as F
from unet import UNet

class UNet_Fine(nn.Module):
	def __init__(
		self,
		model,
		num_classes = 8,
		hidden_dim = 30,
		window_length = 64
	):
		
		"""
		Implementation of
		U-Net: Convolutional Networks for Biomedical Image Segmentation
		(Ronneberger et al., 2015)
		https://arxiv.org/abs/1505.04597
		Using the default arguments will yield the exact version used
		in the original paper
		Args:
			in_channels (int): number of input channels
			n_classes (int): number of output channels
			depth (int): depth of the network
			wf (int): number of filters in the first layer is 2**wf
			padding (bool): if True, apply padding such that the input shape
							is the same as the output.
							This may introduce artifacts
			batch_norm (bool): Use BatchNorm after layers with an
							   activation function
			up_mode (str): one of 'upconv' or 'upsample'.
						   'upconv' will use transposed convolutions for
						   learned upsampling.
						   'upsample' will use bilinear upsampling.
		"""
		super(UNet_Fine, self).__init__()
		self.foundation_model = model
		self.bottleneck_dim = model.bottleneck_channels*window_length//(2 ** (model.depth - 1))
		#self.input_dim = window_length*in_channels
		self.layer1 = nn.Linear(self.bottleneck_dim, hidden_dim)
		self.layer2 = nn.Linear(hidden_dim, num_classes)
		self.silu = nn.SiLU()
		self.flatten = nn.Flatten()
		self.head = nn.Sequential(self.flatten, self.silu, self.layer1, self.silu, self.layer2, self.silu)
		#self.window_length = window_length
		#self.in_channels = in_channels

	def forward(self, x):
		#x=self.flatten(x)
		x=self.foundation_model.bottleneck(x)
		x=self.head(x)
		#x=torch.reshape(x, (-1, self.in_channels, self.window_length))
		return x
