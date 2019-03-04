from __future__ import print_function

import random
import numpy as np
import scipy as sp

from utils import get_training_sample

from keras.utils import Sequence

class TrainingSequence(Sequence):

	def __init__(self, positive_samples, negative_samples, probs, alias_dict, args):
		assert isinstance(positive_samples, list)
		self.positive_samples = positive_samples
		# self.neighbourhood_samples = neighbourhood_samples
		self.negative_samples = negative_samples
		self.probs = probs
		self.alias_dict = alias_dict
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples

		# self.num_neighbours = 0
		# self.neighbour_weight = 0.01

	def alias_draw(self, J, q, size=1):
	    '''
	    Draw sample from a non-uniform discrete distribution using alias sampling.
	    '''
	    K = len(J)
	    kk = np.floor(np.random.uniform(high=K, size=size)).astype(np.int)
	    r = np.random.uniform(size=size)
	    idx = r >= q[kk]
	    kk[idx] = J[kk[idx]]
	    return kk

	def get_training_sample(self, batch_positive_samples):
		# neighbourhood_samples = self.neighbourhood_samples
		negative_samples = self.negative_samples
		num_negative_samples = self.num_negative_samples
		alias_dict = self.alias_dict
		probs = self.probs
		# num_neighbours = self.num_neighbours

		input_nodes = batch_positive_samples[:,0]

		# batch_neighbour_samples = np.array([
		# 	np.random.choice(neighbourhood_samples[u], size=num_neighbours, replace=True)
		# 	for u in input_nodes
		# ], dtype=np.int64)

		batch_negative_samples = np.array([
			# negative_samples[u][self.alias_draw(alias_dict[u][0], alias_dict[u][1], size=num_negative_samples)]
			np.random.choice(negative_samples[u], size=num_negative_samples, replace=True, p=probs[u])
			for u in input_nodes
		], dtype=np.int64)

		batch_nodes = np.concatenate([batch_positive_samples, batch_negative_samples], axis=1)

		return batch_nodes

	def __len__(self):
		return int(np.ceil(len(self.positive_samples) / float(self.batch_size)))

	def __getitem__(self, batch_idx):
		batch_size = self.batch_size
		positive_samples = self.positive_samples
		batch_positive_samples = np.array(
			positive_samples[batch_idx * batch_size : (batch_idx + 1) * batch_size], dtype=np.int64)
		training_sample = self.get_training_sample(batch_positive_samples)
		
		# target = np.zeros((training_sample.shape[0], training_sample.shape[1]-1, 1 ))
		# target[:,0] = 1. - self.num_negative_samples * 0.0
		# target[:,1:] = 0.0
		target = np.zeros((training_sample.shape[0],  ))
		
		return training_sample, target

	def on_epoch_end(self):
		random.shuffle(self.positive_samples)
