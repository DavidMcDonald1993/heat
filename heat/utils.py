from __future__ import print_function

import os
import fcntl
import functools
import numpy as np
import networkx as nx

import random

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import pandas as pd

import pickle as pkl

from .node2vec_sampling import Graph 

from multiprocessing.pool import Pool 

import matplotlib.pyplot as plt

from collections import Counter

def load_data(args):

	edgelist_filename = args.edgelist
	features_filename = args.features
	labels_filename = args.labels

	print ("reading edgelist from", edgelist_filename)

	graph = nx.read_weighted_edgelist(edgelist_filename, delimiter="\t", nodetype=int,
		create_using=nx.DiGraph() if args.directed else nx.Graph())

	print ("removing all edges with zero weight")
	zero_weight_edges = [(u, v) 
		for u, v, w in graph.edges(data="weight") if w == 0.]
	print ("found", len(zero_weight_edges), "edges with zero weight")
	graph.remove_edges_from(zero_weight_edges)

	print ("ensuring all weights are positive")
	nx.set_edge_attributes(graph, name="weight", values={edge: abs(weight) 
		for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})

	for u in range(len(graph)):
		assert u in graph

	print ("number of nodes: {}".format(len(graph)))
	print ("number of edges: {}".format(len(graph.edges())))

	if features_filename is not None:

		print ("loading features from {}".format(features_filename))

		if features_filename.endswith(".csv") or features_filename.endswith(".csv.gz"):
			features = pd.read_csv(features_filename, index_col=0, sep=",")
			features = features.reindex(sorted(graph.nodes())).values
			features = StandardScaler().fit_transform(features) # input features are standard scaled
		else:
			raise Exception

		print ("features shape is {}\n".format(features.shape))

	else: 
		features = None

	if labels_filename is not None:

		print ("loading labels from {}".format(labels_filename))

		if labels_filename.endswith(".csv") or labels_filename.endswith(".csv.gz"):
			labels = pd.read_csv(labels_filename, index_col=0, sep=",")
			labels = labels.reindex(sorted(graph.nodes())).values.astype(int)#.flatten()
			assert len(labels.shape) == 2
		elif labels_filename.endswith(".pkl"):
			with open(labels_filename, "rb") as f:
				labels = pkl.load(f)
			labels = np.array([labels[n] for n in sorted(graph.nodes())], dtype=np.int)
		else:
			raise Exception

		print ("labels shape is {}\n".format(labels.shape))

	else:
		labels = None

	return graph, features, labels

def load_embedding(embedding_filename):
	assert embedding_filename.endswith(".csv.gz")
	embedding_df = pd.read_csv(embedding_filename, index_col=0)
	return embedding_df

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def poincare_ball_to_hyperboloid(X):
	x = 2 * X
	t = 1. + np.sum(np.square(X), axis=-1, keepdims=True)
	x = np.concatenate([x, t], axis=-1)
	return 1 / (1. - np.sum(np.square(X), axis=-1, keepdims=True)) * x

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''

	n, probs = probs

	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return n, (J, q)

def alias_draw(J, q, size=1):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = np.floor(np.random.uniform(high=K, size=size)).astype(np.int)
	r = np.random.uniform(size=size)
	idx = r >= q[kk]
	kk[idx] = J[kk[idx]]
	return kk


def determine_positive_and_negative_samples(graph, features, args):

	graph = graph.to_undirected() # we perform walks on undirected matrix

	nodes = graph.nodes()

	if not isinstance(nodes, set):
		nodes = set(nodes)

	def determine_positive_samples_and_probs(graph, features, args):

		N = len(graph)
		negative_samples = np.ones((N, N), dtype=bool)
		np.fill_diagonal(negative_samples, 0)

		positive_samples = list(graph.edges())
		positive_samples += [(v, u) 
			for u, v in positive_samples]

		if args.no_walks:
			counts = np.array([graph.degree(u)
				for u in sorted(graph)])

			if not args.all_negs:
				for n in nodes:
					negative_samples[n, list(graph.neighbors(n))] = 0
	
		else:
			counts = np.zeros(N)

			print ("determining positive and negative samples", 
				"using random walks")

			walks = perform_walks(graph, features, args)

			context_size = args.context_size

			for num_walk, walk in enumerate(walks):
				for i in range(len(walk)):
					u = walk[i]
					counts[u] += 1
					for j in range(context_size):

						if i+j+1 >= len(walk):
							break
						v = walk[i+j+1]
						if u == v:
							continue

						positive_samples.append((u, v))
						positive_samples.append((v, u))

						if not args.all_negs:
							negative_samples[u, v] = 0
							negative_samples[v, u] = 0

				if num_walk % 1000 == 0:  
					print ("processed walk {:04d}/{}".format(
						num_walk, len(walks)))

		print ("DETERMINED POSITIVE AND NEGATIVE SAMPLES")
		print ("found {} positive sample pairs".format(
			len(positive_samples)))

		counts = counts ** 0.75
		probs = counts[None, :] 
		probs = probs * negative_samples
		assert (probs > 0).any(axis=-1).all(), \
			"a node in the network does not have any negative samples"
		probs /= probs.sum(axis=-1, keepdims=True)
		probs = probs.cumsum(axis=-1)

		assert np.allclose(probs[:,-1], 1)

		print ("PREPROCESSED NEGATIVE SAMPLE PROBABILTIES")

		positive_samples = np.array(positive_samples)

		if not args.use_generator:
			print ("SORTING POSITIVE SAMPLES")
			idx = positive_samples[:,0].argsort()
			positive_samples = positive_samples[idx]
			print ("SORTED POSITIVE SAMPLES")

		return positive_samples, probs

	def select_negative_samples(positive_samples, probs, num_negative_samples):

		# with Pool(processes=None) as p:
		# 	negative_samples = p.map(functools.partial(choose_negative_samples,
		# 	num_negative_samples=num_negative_samples), 
		# 	((u, count, probs[u]) for u, count in Counter(positive_samples[:,0]).items()))
		negative_samples = (choose_negative_samples(x, num_negative_samples) 
			for x in ((u, count, probs[u]) 
			for u, count in sorted(Counter(positive_samples[:,0]).items(), key=lambda x: x[0])))
		negative_samples = np.concatenate([arr for _, arr in 
			sorted(negative_samples, key=lambda x: x[0])], axis=0,)

		print ("selected negative samples")

		return positive_samples, negative_samples

	positive_samples, probs = \
		determine_positive_samples_and_probs(
			graph, features, args)

	if not args.use_generator:
		print("Training without generator -- selecting negative samples before training")
		positive_samples, negative_samples = select_negative_samples(
			positive_samples, probs, args.num_negative_samples)
		probs = None
	else:
		print ("Training with data generator -- skipping selection of negative samples")
		negative_samples = None 

	return positive_samples, negative_samples, probs

def choose_negative_samples(x, num_negative_samples):
		u, count, probs = x
		return u, np.searchsorted(probs, np.random.rand(count, num_negative_samples)).astype(np.int32)

def perform_walks(graph, features, args):

	def save_walks_to_file(walks, walk_file):
		with open(walk_file, "w") as f:
			for walk in walks:
				f.write(",".join([str(n) for n in walk]) + "\n")

	def load_walks_from_file(walk_file, ):

		walks = []

		with open(walk_file, "r") as f:
			for line in (line.rstrip() for line in f.readlines()):
				walks.append([int(n) for n in line.split(",")])
		return walks

	def make_feature_sim(features):

		if features is not None:
			feature_sim = cosine_similarity(features)
			np.fill_diagonal(feature_sim, 0) # remove diagonal
			feature_sim[feature_sim < 1e-15] = 0
			feature_sim /= np.maximum(feature_sim.sum(axis=-1, keepdims=True), 1e-15) # row normalize
		else:
			feature_sim = None

		return feature_sim

	walk_file = args.walk_filename

	if not os.path.exists(walk_file):

		feature_sim = make_feature_sim(features)

		if args.alpha > 0:
			assert features is not None

		node2vec_graph = Graph(graph=graph, 
			is_directed=False,
			p=args.p, 
			q=args.q,
			alpha=args.alpha, 
			feature_sim=feature_sim, 
			seed=args.seed)
		node2vec_graph.preprocess_transition_probs()
		walks = node2vec_graph.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
		
		if args.save_walks: 
			save_walks_to_file(walks, walk_file)
			print ("saved walks to {}".format(walk_file))

	else:
		print ("loading walks from {}".format(walk_file))
		walks = load_walks_from_file(walk_file, )

	return walks

def lock_method(lock_filename):
	''' Use an OS lock such that a method can only be called once at a time. '''

	def decorator(func):

		@functools.wraps(func)
		def lock_and_run_method(*args, **kwargs):

			# Hold program if it is already running 
			# Snippet based on
			# http://linux.byexamples.com/archives/494/how-can-i-avoid-running-a-python-script-multiple-times-implement-file-locking/
			fp = open(lock_filename, 'r+')
			done = False
			while not done:
				try:
					fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
					done = True
				except IOError:
					pass
			return func(*args, **kwargs)

		return lock_and_run_method

	return decorator 

def threadsafe_fn(lock_filename, fn, *args, **kwargs ):
	lock_method(lock_filename)(fn)(*args, **kwargs)

def save_test_results(filename, seed, data, ):
	d = pd.DataFrame(index=[seed], data=data)
	if os.path.exists(filename):
		test_df = pd.read_csv(filename, sep=",", index_col=0)
		test_df = d.combine_first(test_df)
	else:
		test_df = d
	test_df.to_csv(filename, sep=",")

def threadsafe_save_test_results(lock_filename, filename, seed, data):
	threadsafe_fn(lock_filename, save_test_results, filename=filename, seed=seed, data=data)
