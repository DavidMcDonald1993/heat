from __future__ import print_function

import os
import fcntl
import functools
import numpy as np
import networkx as nx
from node2vec_sampling import Graph 

import random

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd




def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
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

	return J, q

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
		# if undirected:
		# 	edge_dict.setdefault(v, default).add(u)

	# for u, v in edgelist:
	# 	assert v in edge_dict[u]
	# 	if undirected:
	# 		assert u in edge_dict[v]
	# raise SystemExit
	edge_dict = {k: list(v) for k, v in edge_dict.items()}

	return edge_dict

def get_training_sample(batch_positive_samples, negative_samples, num_negative_samples, probs, alias_dict):
	input_nodes = batch_positive_samples[:,0]

	batch_negative_samples = np.array([
		# np.random.choice(negative_samples[u], 
		# replace=True, size=(num_negative_samples,), 
		# p=probs[u] if probs is not None else probs
		# )
		negative_samples[u][alias_draw(alias_dict[u][0], alias_dict[u][1], num_negative_samples)]
		for u in input_nodes
	], dtype=np.int64)
	batch_nodes = np.append(batch_positive_samples, batch_negative_samples, axis=1)
	return batch_nodes

# def make_validation_data(val_edges, val_non_edges, negative_samples, alias_dict, args):

# 	val_edges = val_edges + [(v, u) for u, v in val_edges]

# 	if not isinstance(val_edges, np.ndarray):
# 		val_edges = np.array(val_edges)
# 	idx = np.arange(len(val_edges))
# 	positive_samples = val_edges[idx]

# 	val_negative_samples = np.array([
# 		negative_samples[u][alias_draw(alias_dict[u][0], alias_dict[u][1], args.num_negative_samples)]
# 		for u in positive_samples[:,0]
# 	])

# 	x = np.append(positive_samples, val_negative_samples, axis=-1)

# 	y = np.zeros(len(x))
# 	# y = np.zeros((len(x), args.num_positive_samples + args.num_negative_samples, 1))
# 	# y[:,0] = 1.

# 	return x, y


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

def split_edges(edges, non_edges, args, val_split=0.05, test_split=0.1, neg_mul=1):
	
	num_val_edges = int(np.ceil(len(edges) * val_split))
	num_test_edges = int(np.ceil(len(edges) * test_split))

	random.seed(args.seed)
	random.shuffle(edges)
	random.shuffle(non_edges)

	val_edges = edges[:num_val_edges]
	test_edges = edges[num_val_edges:num_val_edges+num_test_edges]
	train_edges = edges[num_val_edges+num_test_edges:]

	val_non_edges = non_edges[:num_val_edges*neg_mul]
	test_non_edges = non_edges[num_val_edges*neg_mul:num_val_edges*neg_mul+num_test_edges*neg_mul]

	return train_edges, (val_edges, val_non_edges), (test_edges, test_non_edges)

def determine_positive_and_negative_samples(nodes, walks, context_size, directed=False):

	print ("determining positive and negative samples")

	if not isinstance(nodes, set):
		nodes = set(nodes)
	
	all_positive_samples = {n: set() for n in sorted(nodes)}
	# neighbourhood_samples = {n: set() for n in sorted(nodes)}
	negative_samples = {n: set() for n in sorted(nodes)}

	positive_samples = []

	counts = {n: 0. for n in sorted(nodes)}

	for num_walk, walk in enumerate(walks):
		for i in range(len(walk)):
			u = walk[i]
			counts[u] += 1	
			# for j in range(len(walk) - i):
			for j in range(context_size):
				if i+j+1 >= len(walk):
					break
				v = walk[i+j+1]
				if u == v:
					continue
				n = 1
				# n = context_size - j
				if j < context_size: 
					positive_samples.extend([(u, v)] * n)
					positive_samples.extend([(v, u)] * n)

					all_positive_samples[u].add(v)
					all_positive_samples[v].add(u)

				# elif j < 3:
				# 	neighbourhood_samples[u].add(v)
				# 	neighbourhood_samples[v].add(u)
				# else: 
				# 	negative_samples[u].add(v)
					# negative_samples[v].add(u)


		if num_walk % 1000 == 0:  
			print ("processed walk {:05d}/{}".format(num_walk, len(walks)))

	# neighbourhood_samples = {n : list(v) for n, v in neighbourhood_samples.items()}
	# negative_samples = {n : np.array(list(v)) for n, v in negative_samples.items()}

	negative_samples = {n: np.array(sorted(nodes.difference(all_positive_samples[n]))) for n in sorted(nodes)}
	# negative_samples = {n: np.array(sorted(all_positive_samples[n])) for n in sorted(nodes)}
	# negative_samples = {n : np.array(sorted(nodes)) for n in sorted(nodes)}
	# negative_samples = {n: np.array(sorted(neg_samples)) for n, neg_samples in negative_samples.items()}
	for u, neg_samples in negative_samples.items():
		# print ("node {} has {} negative samples".format(u, len(neg_samples)))
		assert len(neg_samples) > 0, "node {} does not have any negative samples".format(u)

	print ("DETERMINED POSITIVE AND NEGATIVE SAMPLES")
	print ("found {} positive sample pairs".format(len(positive_samples)))

	counts = np.array(list(counts.values())) ** 0.75
	probs = counts #/ counts.sum()
	# probs = np.ones_like(counts)

	prob_dict = {n: probs[negative_samples[n]] for n in sorted(nodes)}
	prob_dict = {n: p / p.sum() for n, p in prob_dict.items()}

	alias_dict = {n: alias_setup(p) for n, p in prob_dict.items()}

	print ("PREPROCESSED NEGATIVE SAMPLE PROBS")

	return positive_samples, negative_samples, prob_dict, alias_dict

def determine_walk_file():

	if args.alpha > 0:
		assert features is not None
		walk_file = os.path.join(args.walk_path, "add_attributes_alpha={}".format(args.alpha))
		A = nx.adjacency_matrix(graph).A
		np.fill_diagonal(A, 1)
		A /= np.maximum(A.sum(axis=-1, keepdims=True), 1e-15)
		# assert ((A.sum(-1) - 1)<1e-7).all(), A
		# assert ((feature_sim.sum(-1) - 1) < 1e-7).all()
		adj = (1. - args.alpha) * A + args.alpha * feature_sim
		# assert ((adj.sum(axis=-1) - 1) < 1e-7).all()
		g = nx.from_numpy_matrix(adj)
	elif args.multiply_attributes:
		assert features is not None
		walk_file = os.path.join(args.walk_path, "multiply_attributes")
		A = nx.adjacency_matrix(graph).A
		g = nx.from_numpy_matrix(A * feature_sim)
	elif args.jump_prob > 0:
		assert features is not None
		walk_file = os.path.join(args.walk_path, "jump_prob={}".format(args.jump_prob))
		g = graph
	else:
		walk_file = os.path.join(args.walk_path, "no_attributes")
		g = graph
	walk_file += "_num_walks={}-walk_len={}-p={}-q={}.walk".format(args.num_walks, 
				args.walk_length, args.p, args.q)


def perform_walks(graph, features, args):

	def save_walks_to_file(walks, walk_file):
		with open(walk_file, "w") as f:
			for walk in walks:
				f.write(",".join([str(n) for n in walk]) + "\n")

	def load_walks_from_file(walk_file, walk_length):

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
			A = nx.adjacency_matrix(graph).A
			np.fill_diagonal(A, 1)
			A /= np.maximum(A.sum(axis=-1, keepdims=True), 1e-15)
			adj = (1. - args.alpha) * A + args.alpha * feature_sim
			g = nx.from_numpy_matrix(adj)
		elif args.multiply_attributes:
			assert features is not None
			A = nx.adjacency_matrix(graph).A
			g = nx.from_numpy_matrix(A * feature_sim)
		elif args.jump_prob > 0:
			assert features is not None
			g = graph
		else:
			g = graph

		node2vec_graph = Graph(graph=g, is_directed=False, p=args.p, q=args.q,
			jump_prob=args.jump_prob, feature_sim=feature_sim, seed=args.seed)
		node2vec_graph.preprocess_transition_probs()
		walks = node2vec_graph.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
		save_walks_to_file(walks, walk_file)
		print ("saved walks to {}".format(walk_file))

	else:
		print ("loading walks from {}".format(walk_file))
		walks = load_walks_from_file(walk_file, args.walk_length)
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
	# try:
	if os.path.exists(filename):
		test_df = pd.read_csv(filename, sep=",", index_col=0)
		test_df = d.combine_first(test_df)
	# except EmptyDataError:
	else:
		test_df = d
	test_df.to_csv(filename, sep=",")

def threadsafe_save_test_results(lock_filename, filename, seed, data):
	threadsafe_fn(lock_filename, save_test_results, filename=filename, seed=seed, data=data)
