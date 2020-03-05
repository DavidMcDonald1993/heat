from __future__ import print_function

import random
import numpy as np
import scipy as sp
import networkx as nx

from keras.utils import Sequence

# def hyperbolic_distance(u, v):
# 	mink_dp = u[:,:-1].dot(v[:,:-1].T) - u[:,-1:] * v[:,-1:].T
# 	mink_dp = np.maximum(-mink_dp, 1 + 1e-15)
# 	return np.arccosh(mink_dp)

class TrainingDataGenerator(Sequence):

	def __init__(self, 
		positive_samples, 
		probs, 
		model,
		graph, 
		args):
		idx = np.random.permutation(len(positive_samples))
		self.positive_samples = positive_samples[idx]
		self.probs = probs
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples
		self.model = model

		# self.sps = nx.floyd_warshall_numpy(graph, 
		# 	nodelist=sorted(graph))
		
	def get_training_sample(self, batch_positive_samples):
		num_negative_samples = self.num_negative_samples
		probs = self.probs

		batch_negative_samples = np.array([
			np.searchsorted(probs[u], 
			# np.searchsorted(probs, 
				np.random.rand(num_negative_samples))
			for u, v in batch_positive_samples
		], dtype=np.int32)

		batch_nodes = np.concatenate([batch_positive_samples, 
			batch_negative_samples], axis=1)

		return batch_nodes

	def __len__(self):
		return int(np.ceil(len(self.positive_samples) / \
			float(self.batch_size)))

	def __getitem__(self, batch_idx):
		batch_size = self.batch_size
		positive_samples = self.positive_samples
		batch_positive_samples = positive_samples[
			batch_idx * batch_size : 
			(batch_idx + 1) * batch_size]
		training_sample = self.get_training_sample(
			batch_positive_samples)

		# for (u, v), row in zip(batch_positive_samples, 
		# 	training_sample):

		# 	for w in row[2:]:
		# 		assert self.sps[u, v] < self.sps[u, w]
		
		target = np.zeros((training_sample.shape[0], 1, 1), 
			dtype=np.int64)
		
		return training_sample, target

	def on_epoch_end(self):
		positive_samples = self.positive_samples
		idx = np.random.permutation(len(positive_samples))
		self.positive_samples = positive_samples[idx]
