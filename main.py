from __future__ import print_function

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import random
import numpy as np
import networkx as nx
import pandas as pd


from keras import backend as K
from keras.callbacks import Callback, TerminateOnNaN, EarlyStopping

import tensorflow as tf
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import math_ops, control_flow_ops
# from tensorflow.python.training import optimizer

from heat.utils import hyperboloid_to_poincare_ball, load_data, load_embedding
from heat.utils import determine_positive_and_negative_samples
from heat.losses import  hyperbolic_softmax_loss
from heat.generators import TrainingDataGenerator
from heat.visualise import draw_graph, plot_degree_dist
from heat.callbacks import Checkpointer
from heat.models import build_model, load_weights
from heat.optimizers import ReimannianOptimizer

K.set_floatx("float64")
K.set_epsilon(1e-15)

np.set_printoptions(suppress=True)

# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
# config.gpu_options.per_process_gpu_memory_fraction = 0.5

config.log_device_placement=False
config.allow_soft_placement=True

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="HEAT algorithm for feature learning on complex networks")

	parser.add_argument("--edgelist", dest="edgelist", type=str, default=None,
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, default=None,
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, default=None,
		help="path to labels.")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")
	parser.add_argument("--lr", dest="lr", type=np.float64, default=1.,
		help="Learning rate (default is 1.).")

	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=5,
		help="The number of epochs to train for (default is 5).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=512, 
		help="Batch size for training (default is 512).")
	parser.add_argument("--nneg", dest="num_negative_samples", type=int, default=10, 
		help="Number of negative samples for training (default is 10).")
	parser.add_argument("--context-size", dest="context_size", type=int, default=3,
		help="Context size for generating positive samples (default is 3).")
	parser.add_argument("--patience", dest="patience", type=int, default=10,
		help="The number of epochs of no improvement in loss before training is stopped. (Default is 10)")

	parser.add_argument("-d", "--dim", dest="embedding_dim", type=int,
		help="Dimension of embeddings for each layer (default is 2).", default=2)

	parser.add_argument("-p", dest="p", type=float, default=1.,
		help="node2vec return parameter (default is 1.).")
	parser.add_argument("-q", dest="q", type=float, default=1.,
		help="node2vec in-out parameter (default is 1.).")
	parser.add_argument('--num-walks', dest="num_walks", type=int, default=10, 
		help="Number of walks per source (default is 10).")
	parser.add_argument('--walk-length', dest="walk_length", type=int, default=80, 
		help="Length of random walk from source (default is 80).")

	parser.add_argument("--sigma", dest="sigma", type=np.float64, default=1.,
		help="Width of gaussian (default is 1).")

	parser.add_argument("--alpha", dest="alpha", type=float, default=0, 
		help="Probability of randomly jumping to a similar node when walking.")

	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", 
		help="Use this flag to set verbosity of training.")
	parser.add_argument('--workers', dest="workers", type=int, default=2, 
		help="Number of worker threads to generate training patterns (default is 2).")

	parser.add_argument("--walks", dest="walk_path", default=None, 
		help="path to save random walks.")

	parser.add_argument("--embedding", dest="embedding_path", default=None, 
		help="path to save embedings.")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument('--use-generator', action="store_true", help='flag to train using a generator')

	parser.add_argument('--visualise', action="store_true", 
		help='flag to visualise embedding (embedding_dim must be 2)')

	parser.add_argument('--no-walks', action="store_true", 
		help='flag to only train on edgelist (no random walks)')

	parser.add_argument('--all-negs', action="store_true", 
		help='flag to only train using all nodes as negative samples')

	args = parser.parse_args()
	return args

def configure_paths(args):
	'''
	build directories on local system for output of model after each epoch
	'''
	if not args.no_walks:
		if not os.path.exists(args.walk_path):
			os.makedirs(args.walk_path)
			print ("making {}".format(args.walk_path))
		print ("saving walks to {}".format(args.walk_path))
		# walk filename 
		args.walk_filename = os.path.join(args.walk_path, 
			"num_walks={}-walk_len={}-p={}-q={}.walk".format(
				args.num_walks, args.walk_length, args.p, args.q))

	if not os.path.exists(args.embedding_path):
		os.makedirs(args.embedding_path)
		print ("making {}".format(args.embedding_path))
	print ("saving embedding to {}".format(args.embedding_path))

def main():

	args = parse_args()

	assert not (args.visualise and args.embedding_dim > 2), "Can only visualise two dimensions"
	assert args.embedding_path is not None, "you must specify a path to save embedding"
	if not args.no_walks:
		assert args.walk_path is not None, "you must specify a path to save walks"

	random.seed(args.seed)
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	graph, features, node_labels = load_data(args)
	print ("Loaded dataset")

	configure_paths(args)
	print ("Configured paths")

	# build model
	num_nodes = len(graph)
	
	model = build_model(num_nodes, args)
	model, initial_epoch = load_weights(model, args)
	optimizer = ReimannianOptimizer(lr=args.lr)
	loss = hyperbolic_softmax_loss(sigma=args.sigma)
	model.compile(optimizer=optimizer, 
		loss=loss, 
		target_tensors=[tf.placeholder(dtype=tf.int64)])
	model.summary()

	callbacks = [
		TerminateOnNaN(),
		EarlyStopping(monitor="loss", 
			patience=args.patience, 
			verbose=True),
		Checkpointer(epoch=initial_epoch, 
			nodes=sorted(graph.nodes()), 
			embedding_directory=args.embedding_path)
	]			

	positive_samples, negative_samples, probs = \
		determine_positive_and_negative_samples(graph, 
		features, args)

	del features # remove features reference to free up memory

	if args.use_generator:
		print ("Training with data generator with {} worker threads".format(args.workers))
		training_generator = TrainingDataGenerator(
			positive_samples,  
			probs,
			model,
			graph,
			args)

		model.fit_generator(
			training_generator, 
			workers=args.workers,
			# max_queue_size=50, 
			use_multiprocessing=False,
			epochs=args.num_epochs, 
			steps_per_epoch=len(training_generator),
			initial_epoch=initial_epoch, 
			verbose=args.verbose,
			callbacks=callbacks
		)

	else:
		print ("Training without data generator")

		train_x = np.append(positive_samples, 
			negative_samples, axis=-1)
		train_y = np.zeros([len(train_x), 1, 1], dtype=np.int64 )

		model.fit(train_x, train_y,
			shuffle=True,
			batch_size=args.batch_size, 
			epochs=args.num_epochs, 
			initial_epoch=initial_epoch, 
			verbose=args.verbose,
			callbacks=callbacks
		)

	print ("Training complete")

	if args.visualise:
		embedding = model.get_weights()[0]
		if embedding.shape[1] == 3:
			print ("projecting to poincare ball")
			embedding = hyperboloid_to_poincare_ball(embedding)
		draw_graph(graph, 
			embedding, 
			node_labels, 
			path="2d-poincare-disk-visualisation.png")

if __name__ == "__main__":
	main()