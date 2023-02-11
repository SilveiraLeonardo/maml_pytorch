import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import optim
import numpy as np

from learning import Learner
from copy import deepcopy

class Meta(nn.Module):
	def __init__(self, outer_lr, inner_lr, n_way, k_support, k_query, batch_size,
		update_step=5, update_step_test=10):
		super(Meta, self).__init__()
		self.update_lr = inner_lr
		self.meta_lr = outer_lr
		self.n_way = n_way
		self.k_support = k_support
		self.k_query = k_query
		self.task_num = batch_size
		self.update_step = update_step
		self.update_step_test = update_step_test

		self.net = Learner(n_way = self.n_way).to(device)
		self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

	def forward(self, x_support, y_support, x_query, y_query):
		# x_support: [b, support_size, c, h, w]
		# y_support: [b, support_size]
		# x_query: [b, query_size, c, h, w]
		# y_query: [b, query_size]

		task_num, support_size, c, h, w = x_support.size()
		query_size = x_query.size(1)

		losses_q = [0 for _ in range(self.update_step+1)]
		corrects = [0 for _ in range(self.update_step+1)]

		for i in range(task_num):
			logits = self.net(x_support[i], vars=None, bn_training=True)
			loss = F.cross_entropy(logits, y_support[i])
			grad = torch.autograd.grad(loss, self.net.parameters())
			fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

			# loss and accuracy before first update
			with torch.no_grad():
				logits_q = self.net(x_query[i], self.net.parameters(), bn_training=True)
				loss_q = F.cross_entropy(logits_q, y_query[i])
				losses_q[0] = losses_q[0] + loss_q

				pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
				correct = torch.eq(pred_q, y_query[i]).sum().item()
				corrects[0] = corrects[0] + correct

			with torch.no_grad():
				logits_q = self.net(x_query[i], fast_weights, bn_training=True)
				loss_q = F.cross_entropy(logits_q, y_query[i])
				losses_q[1] = losses_q[1] + loss_q

				pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
				correct = torch.eq(pred_q, y_query[i]).sum().item()
				corrects[1] = corrects[1] + correct

			for k in range(1, self.update_step):
				logits = self.net(x_support[i], vars=fast_weights, bn_training=True)
				loss = F.cross_entropy(logits, y_support[i])
				grad = torch.autograd.grad(loss, fast_weights)
				fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

				logits_q = self.net(x_query[i], fast_weights, bn_training=True)
				loss_q = F.cross_entropy(logits_q, y_query[i])
				losses_q[k+1] = losses_q[k+1] + loss_q

				with torch.no_grad():
					pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
					correct = torch.eq(pred_q, y_query[i]).sum().item()
					corrects[k+1] = corrects[k+1] + correct

		loss_q = losses_q[-1] / task_num

		self.meta_optim.zero_grad()
		loss_q.backward()
		self.meta_optim.step()

		accs = np.array(corrects)/(query_size * task_num)

		return accs[-1], loss_q.item()

	def finetuning(self, x_support, y_support, x_query, y_query):
		# x_support: [support_size, c, h, w]
		# y_support: [support_size]
		# x_query: [query_size, c, h, w]
		# y_query: [query_size]		

		query_size = x_query.size(0)
		corrects = [0 for _ in range(self.update_step_test+1)]

		net = deepcopy(self.net) # fine tune on the copied model

		logits = net(x_support)
		loss = F.cross_entropy(logits, y_support)
		grad = torch.autograd.grad(loss, net.parameters())
		fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

		# loss and accuracy before first update
		with torch.no_grad():
			logits_q = net(x_query, net.parameters(), bn_training=True)
			pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
			correct = torch.eq(pred_q, y_query).sum().item()
			corrects[0] = corrects[0] + correct
		# loss and accuracy after the update
		with torch.no_grad():
			logits_q = net(x_query, fast_weights, bn_training=True)
			pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
			correct = torch.eq(pred_q, y_query).sum().item()
			corrects[1] = corrects[1] + correct

		for k in range(1, self.update_step_test):
			logits = net(x_support, vars=fast_weights, bn_training=True)
			loss = F.cross_entropy(logits, y_support)
			grad = torch.autograd.grad(loss, fast_weights)
			fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

			logits_q = net(x_query, fast_weights, bn_training=True)
			loss_q = F.cross_entropy(logits_q, y_query)

			with torch.no_grad():
				pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
				correct = torch.eq(pred_q, y_query).sum().item()
				corrects[k+1] = corrects[k+1] + correct

		del net
		accs = np.array(corrects) / query_size

		return accs[-1], loss_q.item()









