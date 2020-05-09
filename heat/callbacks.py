from __future__ import print_function

import re
import sys
import os
import glob
import numpy as np
import pandas as pd

from .utils import hyperboloid_to_poincare_ball

from keras.callbacks import Callback

def minkowski_dot(u):
	return ((u[...,:-1] ** 2).sum(axis=-1, keepdims=True) 
		- u[...,-1:] ** 2)

class Checkpointer(Callback):

	def __init__(self, 
		epoch,
		nodes,
		embedding_directory,
		history=1,
		):
		self.epoch = epoch
		self.nodes = nodes
		self.embedding_directory = embedding_directory
		self.history = history

	def on_epoch_end(self, batch, logs={}):
		self.epoch += 1
		print ("\nEpoch {} complete".format(self.epoch)) 
		self.remove_old_models()
		self.save_model()

	def remove_old_models(self):
		embedding_directory = self.embedding_directory
		history = self.history
		old_model_paths = sorted(filter(
			re.compile("[0-9]+\_embedding\.csv\.gz").match, 
			os.listdir(embedding_directory)))
		if history > 0:
			old_model_paths = old_model_paths[:-history]
		for old_model_path in old_model_paths:
			print ("removing model: {}".format(old_model_path))
			os.remove(os.path.join(embedding_directory, 
				old_model_path))

	def save_model(self):
		filename = os.path.join(self.embedding_directory, 
			"{:05d}_embedding.csv.gz".format(self.epoch))
		embedding = self.model.get_weights()[0]
		print ("saving current embedding to {}".format(filename))

		assert np.allclose(minkowski_dot(embedding,), -1, )

		embedding_df = pd.DataFrame(embedding, index=self.nodes)
		embedding_df.to_csv(filename, compression="gzip")

		poincare_embedding = hyperboloid_to_poincare_ball(
			embedding)
		norms = np.linalg.norm(poincare_embedding, axis=-1)
		print ("MIN", norms.min(), 
			"MEAN", norms.mean(),
			"MAX", norms.max())
		print (np.linalg.norm(poincare_embedding.mean(0)))