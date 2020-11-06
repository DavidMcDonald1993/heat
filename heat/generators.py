import numpy as np

from keras.utils import Sequence

import os

class TrainingDataGenerator(Sequence):

	def __init__(self, 
		positive_samples, 
		probs, 
		model,
		graph, 
		args):
		assert isinstance(positive_samples, np.ndarray)
		self.num_positive_samples = len(positive_samples)
		idx = np.random.permutation(self.num_positive_samples)
		self.positive_samples = positive_samples[idx]
		self.probs = probs
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples
		self.model = model

	def get_training_sample(self, batch_positive_samples):
		num_negative_samples = self.num_negative_samples
		probs = self.probs

		batch_negative_samples = np.array([
			np.searchsorted(probs[u], 
				np.random.rand(num_negative_samples))
			for u, v in batch_positive_samples
		], dtype=np.int64)

		batch_nodes = np.concatenate(
			[batch_positive_samples, batch_negative_samples], 
			axis=1)

		return batch_nodes

	def __len__(self):
		return int(np.ceil(self.num_positive_samples / \
			float(self.batch_size)))

	def __getitem__(self, batch_idx):
		batch_size = self.batch_size
		positive_samples = self.positive_samples
		batch_positive_samples = positive_samples[
			batch_idx * batch_size : 
			(batch_idx + 1) * batch_size]
		training_sample = self.get_training_sample(
			batch_positive_samples)

		target = np.zeros((training_sample.shape[0], 1, 1), 
			dtype=np.int64)
		
		return training_sample, target

	def on_epoch_end(self):
		positive_samples = self.positive_samples
		idx = np.random.permutation(self.num_positive_samples)
		self.positive_samples = positive_samples[idx]
