import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Learner(nn.Module):
	def __init__(self, n_way):
		super(Learner, self).__init__()
		self.n_way = n_way
		self.vars = nn.ParameterList() # this dict contains all tensors needed to be optimized
		self.vars_bn = nn.ParameterList() # running mean and running variance

		self.config = [
			('conv2d', [64, 1, 3, 3, 2, 0]), # (1, 28, 28) -> (64, 15, 15)
			('relu', [True]),
			('bn', [64]),
			('conv2d', [64, 64, 3, 3, 2, 0]), # (64, 15, 15) -> (64, 6, 6)
			('relu', [True]),
			('bn', [64]),
			('conv2d', [64, 64, 3, 3, 2, 0]), # (64, 6, 6) -> (64, 2, 2)
			('relu', [True]),
			('bn', [64]),
			('conv2d', [64, 64, 2, 2, 1, 0]), # (64, 2, 2) -> (64, 1, 1)
			('relu', [True]),
			('bn', [64]),
			('flatten', []),
			('linear', [self.n_way, 64])
		]

		for i, (name, param) in enumerate(self.config):
			if name == 'conv2d': # (ch_out, ch_in, kernel_size, kernel_size, stride, padding)
				w = nn.Parameter(torch.ones(*param[:4]))
				torch.nn.init.kaiming_normal_(w)
				self.vars.append(w)
				self.vars.append(nn.Parameter(torch.zeros(param[0])))
			elif name == 'linear':
				w = nn.Parameter(torch.ones(*param))
				torch.nn.init.kaiming_normal_(w)
				self.vars.append(w)
				self.vars.append(nn.Parameter(torch.zeros(param[0])))
			elif name == 'bn':
				w = nn.Parameter(torch.ones(param[0]))
				self.vars.append(w)
				self.vars.append(nn.Parameter(torch.zeros(param[0])))

				# set requires_grad = False
				running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
				running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
				self.vars_bn.extend([running_mean, running_var])
			elif name in ['relu', 'flatten']:
				continue

	def forward(self, x, vars = None, bn_training=True):

		if vars is None:
			vars = self.vars

		idx = 0
		bn_idx = 0

		for name, param in self.config:
			if name == 'conv2d':
				w, b = vars[idx], vars[idx+1]
				x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
				idx += 2
			elif name == 'linear':
				w, b = vars[idx], vars[idx+1]
				x = F.linear(x, w, b)
				idx += 2
			elif name == 'bn':
				w, b = vars[idx], vars[idx+1]
				running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
				x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
				idx += 2
				bn_idx += 2
			elif name == 'flatten':
				x = x.view(x.size(0), -1)
			elif name == 'relu':
				x = F.relu(x, inplace=param[0])

		return x

	def parameters(self):
		return self.vars

	def zero_grad(self, vars=None):
		with torch.no_grad():
			if vars is None:
				for p in self.vars:
					if p.grad is not None:
						p.grad.zero_()
			else:
				for p in vars:
					if p.grad is not None:
						p.grad.zero_()






