import os
import logging
import pickle

import torch
import torch.nn as nn
import numpy as np
# import pandas as pd
import math
import glob
import re
from shutil import copyfile
import sklearn as sk
import subprocess
import datetime

def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_uniform_(m.weight, gain=1)
			nn.init.constant_(m.bias, val=0)

def init_kaiming_weights(net):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu') #'leaky_relu'
			# nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
			nn.init.constant_(m.bias, val=0)



def split_last_dim(data):
	last_dim = data.size()[-1]
	last_dim = last_dim // 2

	if len(data.size()) == 3:
		res = data[:, :, :last_dim], data[:, :, last_dim:]

	if len(data.size()) == 2:
		res = data[:, :last_dim], data[:, last_dim:]
	return res


def linspace_vector(start, end, n_points):
	# start is either one value or a vector
	size = np.prod(start.size())

	assert(start.size() == end.size())
	if size == 1:
		# start and end are 1d-tensors
		res = torch.linspace(start, end, n_points)
	else:
		# start and end are vectors
		res = torch.Tensor()
		for i in range(0, start.size(0)):
			res = torch.cat((res,
				torch.linspace(start[i], end[i], n_points)),0)
		res = torch.t(res.reshape(start.size(0), n_points))
	return res

def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()

