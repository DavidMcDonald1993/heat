from __future__ import print_function

import random
import numpy as np
import scipy as sp

from keras.utils import Sequence

# def hyperbolic_distance(u, v):
# 	mink_dp = u[:,:-1].dot(v[:,:-1].T) - u[:,-1:] * v[:,-1:].T
# 	mink_dp = np.maximum(-mink_dp, 1 + 1e-15)
# 	return np.arccosh(mink_dp)

class TrainingDataGenerator(Sequence):

	def __init__(self, positive_samples, probs, model, args):
		idx = np.random.permutation(len(positive_samples))
		self.positive_samples = positive_samples[idx]
		self.positive_samples = positive_samples
		self.probs = probs
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples
		self.model = model

		# embedding = self.model.get_weights()[-1]
		# dists = hyperbolic_distance(embedding, embedding)
		# probs = np.exp(-dists) * self.probs_
		# probs /= probs.sum(axis=-1, keepdims=True)
		# self.probs = probs.cumsum(-1)

	def get_training_sample(self, batch_positive_samples):
		num_negative_samples = self.num_negative_samples
		probs = self.probs
		
		input_nodes = batch_positive_samples[:,0]

		batch_negative_samples = np.array([
			np.searchsorted(probs[u], np.random.rand(num_negative_samples))
			for u in input_nodes
		], dtype=np.int32)

		batch_nodes = np.concatenate([batch_positive_samples, batch_negative_samples], axis=1)

		return batch_nodes

	def __len__(self):
		return int(np.ceil(len(self.positive_samples) / float(self.batch_size)))

	def __getitem__(self, batch_idx):
		batch_size = self.batch_size
		positive_samples = self.positive_samples
		batch_positive_samples = positive_samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
		training_sample = self.get_training_sample(batch_positive_samples)
		
		target = np.zeros((training_sample.shape[0], 1, 1), dtype=np.int32)
		
		return training_sample, target

	def on_epoch_end(self):
		positive_samples = self.positive_samples
		idx = np.random.permutation(len(positive_samples))
		self.positive_samples = positive_samples[idx]

		# embedding = self.model.get_weights()[-1]
		# dists = hyperbolic_distance(embedding, embedding)
		# probs = np.exp(-dists) * self.probs_
		# probs /= probs.sum(axis=-1, keepdims=True)
		# self.probs = probs.cumsum(-1)
