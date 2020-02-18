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
	return (u[:,:-1] ** 2).sum(axis=-1, keepdims=True) - u[:,-1:] ** 2

class Checkpointer(Callback):

	def __init__(self, 
		epoch,
		nodes,
		embedding_directory,
		):
		self.epoch = epoch
		self.nodes = nodes
		self.embedding_directory = embedding_directory

	def on_epoch_end(self, batch, logs={}):
		self.epoch += 1
		print ("Epoch {} complete".format(self.epoch)) 
		self.remove_old_models()
		self.save_model()

	def remove_old_models(self):
		embedding_directory = self.embedding_directory
		for old_model_path in filter(
			re.compile("[0-9]+_embedding.csv.gz").match, 
			os.listdir(embedding_directory)):
			print ("removing model: {}".format(old_model_path))
			os.remove(os.path.join(embedding_directory, 
				old_model_path))

	def save_model(self):
		filename = os.path.join(self.embedding_directory, 
			"{:05d}_embedding.csv.gz".format(self.epoch))
		embedding = self.model.get_weights()[0]
		print ("saving current embedding to {}".format(filename))

		embedding_df = pd.DataFrame(embedding, index=self.nodes)
		embedding_df.to_csv(filename, compression="gzip")

		embedding_poincare = hyperboloid_to_poincare_ball(embedding)

		norms = np.linalg.norm(embedding_poincare, axis=-1)
		print ("min", norms.min(), "mean", norms.mean(), 
			"max", norms.max())