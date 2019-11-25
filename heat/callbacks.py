from __future__ import print_function

import re
import sys
import os
import glob
import numpy as np
import pandas as pd

from keras.callbacks import Callback

def minkowski_dot(u):
	return (u[:,:-1] ** 2).sum(axis=-1, keepdims=True) - u[:,-1:] ** 2

class Checkpointer(Callback):

	def __init__(self, 
		epoch,
		nodes,
		embedding_directory,
		history=3
		):
		self.epoch = epoch
		self.nodes = nodes
		self.embedding_directory = embedding_directory
		self.history = history

	def on_epoch_end(self, batch, logs={}):
		self.epoch += 1
		print ("Epoch {} complete".format(self.epoch)) 
		self.remove_old_models()
		self.save_model()

	def remove_old_models(self):
		for old_model_path in sorted(
			glob.iglob(os.path.join(self.embedding_directory, "*.csv.gz")))[:-self.history]:
			print ("removing model: {}".format(old_model_path))
			os.remove(old_model_path)

	def save_model(self):
		filename = os.path.join(self.embedding_directory, 
			"{:05d}_embedding.csv.gz".format(self.epoch))
		embedding = self.model.get_weights()[0]
		print ("saving current embedding to {}".format(filename))

		embedding_df = pd.DataFrame(embedding, index=self.nodes)
		embedding_df.to_csv(filename, compression="gzip")