import os
import glob

import tensorflow as tf

from keras.layers import Input, Layer
from keras.models import Model

import keras.backend as K

from heat.utils import load_embedding


def hyperboloid_initializer(shape, r_max=1e-3):

	def poincare_ball_to_hyperboloid(X, append_t=True):
		x = 2 * X
		t = 1. + K.sum(K.square(X), axis=-1, keepdims=True)
		if append_t:
			x = K.concatenate([x, t], axis=-1)
		return 1 / (1. - K.sum(K.square(X), axis=-1, keepdims=True)) * x

	w = tf.random_uniform(shape=shape, minval=-r_max, 
		maxval=r_max, dtype=K.floatx())
	return poincare_ball_to_hyperboloid(w)

class HyperboloidEmbeddingLayer(Layer):
	
	def __init__(self, 
		num_nodes, 
		embedding_dim, 
		**kwargs):
		super(HyperboloidEmbeddingLayer, self).__init__(**kwargs)
		self.num_nodes = num_nodes
		self.embedding_dim = embedding_dim

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.embedding = self.add_weight(name='embedding', 
		  shape=(self.num_nodes, self.embedding_dim),
		  initializer=hyperboloid_initializer,
		  trainable=True)
		super(HyperboloidEmbeddingLayer, self).build(input_shape)

	def call(self, idx):
		return tf.gather(self.embedding, idx)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1], 
			self.embedding_dim + 1)
	
	def get_config(self):
		base_config = super(HyperboloidEmbeddingLayer, self).\
			get_config()
		base_config.update({"num_nodes": self.num_nodes, 
			"embedding_dim": self.embedding_dim})
		return base_config

def build_model(num_nodes, args):

	x = Input(shape=(1 + 1 + args.num_negative_samples, ), 
		name="model_input", 
		dtype=tf.int64)
	y = HyperboloidEmbeddingLayer(num_nodes, 
		args.embedding_dim, 
		name="embedding_layer")(x)
	model = Model(x, y)

	return model


def load_weights(model, args):

	previous_models = sorted(glob.iglob(
		os.path.join(args.embedding_path, "*.csv.gz")))
	if len(previous_models) > 0:
		model_file = previous_models[-1]
		initial_epoch = int(model_file.split("/")[-1].split("_")[0])
		print ("previous models found in directory -- loading from file {} and resuming from epoch {}".format(model_file, initial_epoch))
		embedding_df = load_embedding(model_file)
		embedding = embedding_df.reindex(sorted(embedding_df.index)).values
		model.layers[1].set_weights([embedding])
	else:
		print ("no previous model found in {}".format(args.embedding_path))
		initial_epoch = 0

	return model, initial_epoch
