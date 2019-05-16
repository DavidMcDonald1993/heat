from __future__ import print_function

import os
import fcntl
import functools
import numpy as np
import networkx as nx

import random
# import bisect

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

	graph = nx.read_weighted_edgelist(edgelist_filename, delimiter="\t", nodetype=int,
		create_using=nx.DiGraph() if args.directed else nx.Graph())

	# remove self loops as they slow down random walk
	graph.remove_edges_from(graph.selfloop_edges())
	print ("removing all self loop edges")

	print ("number of nodes: {}\nnumber of edges: {}\n".format(len(graph), len(graph.edges())))

	if features_filename is not None:

		print ("loading features from {}".format(features_filename))

		if features_filename.endswith(".csv"):
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

		if labels_filename.endswith(".csv"):
			labels = pd.read_csv(labels_filename, index_col=0, sep=",")
			labels = labels.reindex(sorted(graph.nodes())).values#.flatten()
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
	assert embedding_filename.endswith(".csv")
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

def convert_edgelist_to_dict(edgelist, undirected=True, self_edges=False):
	if edgelist is None:
		return None
	if undirected:
		edgelist += [(v, u) for u, v in edgelist]
	edge_dict = {}
	for u, v in edgelist:
		if self_edges:
			default = set(u)
		else:
			default = set()
		edge_dict.setdefault(u, default).add(v)
	edge_dict = {k: list(v) for k, v in edge_dict.items()}

	return edge_dict

def get_training_sample(samples, num_negative_samples, ):
	positive_sample_pair, probs = samples
	# negative_samples_ = negative_samples[alias_draw(probs[0], probs[1], num_negative_samples)]
	negative_samples_ = np.random.choice(len(probs), replace=True, size=num_negative_samples, p=probs)
	return np.append(positive_sample_pair, negative_samples_, )

def build_training_samples(positive_samples, negative_samples, num_negative_samples, alias_dict):
	input_nodes = positive_samples[:,0]
	print ("Building training samples")
	N = len(alias_dict)

	# batch_negative_samples = np.array([
	# 	np.random.choice(N, replace=True, size=num_negative_samples, p=probs)
	# 	for probs in (alias_dict[u] for u in input_nodes)
	# ], dtype=np.int64)
	# batch_nodes = np.append(positive_samples, batch_negative_samples, axis=1)
	# return batch_nodes

	# batch_negative_samples = np.array([
	# 	negative_samples[u][alias_draw(alias_dict[u][0], alias_dict[u][1], num_negative_samples)]
	# 	for u in input_nodes
	# ], dtype=np.int64)
	# batch_nodes = np.append(positive_samples, batch_negative_samples, axis=1)
	# return batch_nodes
	
	with Pool(processes=2) as p:
		training_samples = p.map(functools.partial(get_training_sample,
			num_negative_samples=num_negative_samples,
			),
			zip(positive_samples,
				# (negative_samples[u] for u in input_nodes),
				(alias_dict[u] for u in input_nodes)))
	return np.array(training_samples)

def create_second_order_topology_graph(topology_graph, args):

	adj = nx.adjacency_matrix(topology_graph).A
	adj_sim = cosine_similarity(adj)
	adj_sim -= np.identity(len(topology_graph))
	adj_sim [adj_sim  < args.rho] = 0
	second_order_topology_graph = nx.from_numpy_matrix(adj_sim)

	print ("Created second order topology graph graph with {} edges".format(len(second_order_topology_graph.edges())))

	return second_order_topology_graph

def create_feature_graph(features, args):

	features_sim = cosine_similarity(features)
	features_sim -= np.identity(len(features))
	features_sim [features_sim  < args.rho] = 0
	feature_graph = nx.from_numpy_matrix(features_sim)

	print ("Created feature correlation graph with {} edges".format(len(feature_graph.edges())))

	return feature_graph

# def determine_negative_samples(positive_samples, nodes):
# 	n, all_positive_samples = positive_samples
# 	return n, np.array(sorted(nodes.difference(all_positive_samples)))

def determine_positive_and_negative_samples(graph, features, args):

	nodes = graph.nodes()

	if not isinstance(nodes, set):
		nodes = set(nodes)

	def determine_positive_samples_and_probs(graph, features, args):

		N = len(graph)
		negative_samples = np.ones((N, N))
		np.fill_diagonal(negative_samples, 0)

		if args.no_walks:

			print ("using only edges as positive samples")

			positive_samples = list(graph.edges())
			positive_samples += [(v, u) for (u, v) in positive_samples]

			for n in sorted(graph.nodes()):
				negative_samples[n, list(graph.neighbors(n))] = 0
			# all_positive_samples = {n: set([n] + list(graph.neighbors(n))) for n in sorted(nodes)}

			counts = np.array([graph.degree(n) for n in sorted(nodes)])

		else:

			print ("determining positive and negative samples using random walks")

			walks = perform_walks(graph, features, args)

			context_size = args.context_size
			directed = args.directed
			
			# all_positive_samples = {n: {n} for n in sorted(nodes)}

			positive_samples = []

			counts = {n: 0. for n in sorted(nodes)}

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

						negative_samples[((u,v), (v,u))] = 0

						# bisect.insort_left(positive_samples, (u, v))
						# bisect.insort_left(positive_samples, (v, u))

						# all_positive_samples[u].add(v)
						# all_positive_samples[v].add(u)

				if num_walk % 1000 == 0:  
					print ("processed walk {:04d}/{}".format(num_walk, len(walks)))

			counts = np.array([counts[n] for n in sorted(nodes)])

		print ("DETERMINING NEGATIVE SAMPLE PROBS")

		# negative_samples = np.ones((len(nodes), len(nodes)))
		# for n in sorted(nodes):
		# 	negative_samples[n, list(all_positive_samples[n])] = 0

		# with Pool(processes=None) as p:
		# 	negative_samples = p.map(functools.partial(determine_negative_samples, nodes=nodes), 
		# 		((n, all_positive_samples[n]) for n in sorted(nodes)))
		# negative_samples = {n: s for n, s in negative_samples}
		# # negative_samples = {n: np.array(sorted(nodes.difference(all_positive_samples[n]))) for n in sorted(nodes)}
		# for u, neg_samples in negative_samples.items():
		# 	assert len(neg_samples) > 0, "node {} does not have any negative samples".format(u)
		# 	# print ("node {} has {} negative samples".format(u, len(neg_samples)))

		print ("DETERMINED POSITIVE AND NEGATIVE SAMPLES")
		print ("found {} positive sample pairs".format(len(positive_samples)))

		counts = counts ** 0.75
		probs = counts[None, :] 

		probs = probs * negative_samples
		probs /= probs.sum(axis=-1, keepdims=True)


		# # prob_dict = {n: probs[negative_samples[n]] for n in sorted(nodes)}
		# # prob_dict = {n: p / p.sum() for n, p in prob_dict.items()}
		# prob_dict = {n: probs[n] for n in sorted(nodes)}
		# print ("PREPROCESSING NEGATIVE SAMPLE PROBABILTIES")

		# with Pool(processes=None) as p:
		# 	alias_dict = p.map(alias_setup, prob_dict.items())

		# alias_dict = {n: p for n, p in alias_dict}
		probs = probs.cumsum(-1)

		print ("PREPROCESSED NEGATIVE SAMPLE PROBABILTIES")

		return positive_samples, probs

	def select_negative_samples(positive_samples, probs):

		print ("SORTING POSITIVE SAMPLES")
		positive_samples = np.array(sorted(positive_samples), dtype=np.int32)
		print ("SORTED POSITIVE SAMPLES")

		with Pool(processes=None) as p:
			negative_samples = p.map(functools.partial(choose_negative_samples,
			num_negative_samples=args.num_negative_samples), 
			((u, count, probs[u]) for u, count in  Counter(positive_samples[:,0]).items()))

		negative_samples = np.concatenate([arr for _, arr in sorted(negative_samples, key=lambda x: x[0])], axis=0,)

		return positive_samples, negative_samples

	positive_samples, probs = determine_positive_samples_and_probs(graph, features, args)

	if not args.use_generator:
		print("training without generator -- selecting negative samples before training")
		positive_samples, negative_samples = select_negative_samples(positive_samples, probs)
		probs = None
	else:
		print ("training with generator -- skipping selecting negative samples")
		negative_samples = None 


	return positive_samples, negative_samples, probs# negative_samples, alias_dict

def choose_negative_samples(x, num_negative_samples):
		u, count, probs = x
		# return u, np.random.multinomial(1, probs, size=(count, num_negative_samples)).argmax(-1)
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
