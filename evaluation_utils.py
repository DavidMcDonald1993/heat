import os

import numpy as np
import pandas as pd

import glob

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import functools
import fcntl

import random

def euclidean_distance(u, v):
	assert len(u.shape) == len(v.shape) 
	return np.linalg.norm(u - v, axis=-1)

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape) 
	return np.sum(x[...,:-1] * y[...,:-1], axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def hyperbolic_distance_hyperboloid(u, v):
	assert len(u.shape) == len(v.shape)
	mink_dp = -minkowski_dot(u, v)
	mink_dp = np.maximum(mink_dp - 1, 1e-15)
	return np.squeeze(np.arccosh(1 + mink_dp), axis=-1)

def hyperbolic_distance_poincare(u, v):
	assert len(u.shape) == len(v.shape)
	norm_u = np.linalg.norm(u, keepdims=False, axis=-1)
	norm_u = np.minimum(norm_u, np.nextafter(1,0, ))
	norm_v = np.linalg.norm(v, keepdims=False, axis=-1)
	norm_v = np.minimum(norm_v, np.nextafter(1,0, ))
	uu = np.linalg.norm(u - v, keepdims=False, axis=-1, ) ** 2
	dd = (1 - norm_u**2) * (1 - norm_v**2)
	return np.arccosh(1 + 2 * uu / dd)

def logarithmic_map(p, x):

	alpha = -minkowski_dot(p, x)

	alpha = np.maximum(alpha, 1+1e-15)

	return np.arccosh(alpha) * (x - alpha * p) / \
		np.sqrt(alpha ** 2 - 1) 
		

def parallel_transport(p, q, x):

	assert len(p.shape) == len(q.shape) == len(x.shape)
	alpha = -minkowski_dot(p, q)
	return x + minkowski_dot(q - alpha * p, x) * (p + q) / \
		(alpha + 1) 


def kullback_leibler_divergence_euclidean(
	source_mus,
	source_sigmas,
	target_mus,
	target_sigmas):

	dim = source_mus.shape[1] 

	# project to tangent space

	sigma_ratio = target_sigmas / source_sigmas

	x_minus_mu = target_mus - source_mus

	trace = np.sum(sigma_ratio, 
		axis=-1, keepdims=True)

	mu_sq_diff = np.sum(x_minus_mu ** 2 / \
		source_sigmas, 
		axis=-1, keepdims=True) # assume sigma is diagonal

	log_det = np.sum(np.log(sigma_ratio), 
		axis=-1, keepdims=True)

	return np.squeeze(
		0.5 * (trace + mu_sq_diff - dim - log_det), 
		axis=-1)

def kullback_leibler_divergence_hyperboloid(source_mus,
	source_sigmas,
	target_mus,
	target_sigmas):

	dim = source_mus.shape[1] - 1

	to_tangent_space = logarithmic_map(source_mus, 
		target_mus)

	# parallel transport to mu zero
	mu_zero = np.zeros((1, dim + 1))
	mu_zero[..., -1] = 1
	
	to_tangent_space_mu_zero = parallel_transport(source_mus,
		mu_zero, 
		to_tangent_space)

	# mu is zero vector
	# ignore zero t coordinate
	x_minus_mu = to_tangent_space_mu_zero[...,:-1]

	sigma_ratio = target_sigmas / source_sigmas

	trace_fac = np.sum(sigma_ratio,
		axis=-1, keepdims=True)

	mu_sq_diff = np.sum(x_minus_mu ** 2 / \
		source_sigmas,
		axis=-1, keepdims=True) # assume sigma inv is diagonal

	log_det = np.sum(np.log(sigma_ratio), 
		axis=-1, keepdims=True)

	return np.squeeze(
		0.5 * (trace_fac + mu_sq_diff - dim - log_det), 
		axis=-1
	)

def load_file(filename, header="infer", sep=","):
	print ("reading from", filename)
	df = pd.read_csv(filename, index_col=0, 
		header=header, sep=sep)
	idx = sorted(df.index)
	df = df.reindex(idx)
	print ("embedding shape is", df.shape)
	return df.values

def load_hyperboloid(embedding_directory):
	files = sorted(glob.iglob(os.path.join(embedding_directory, 
		"*_embedding.csv.gz")))
	embedding_filename = files[-1]
	embedding = load_file(embedding_filename)

	return embedding

def load_poincare(embedding_directory):
	files = sorted(glob.iglob(os.path.join(embedding_directory, 
		"*embedding.csv.gz")))
	embedding_filename = files[-1]
	embedding = load_file(embedding_filename)

	return embedding

def load_euclidean(embedding_directory):
	files = sorted(glob.iglob(os.path.join(embedding_directory, 
		"*.csv.gz")))
	embedding_filename = files[-1]
	embedding = load_file(embedding_filename, 
		header=None, sep=" ")
	return embedding

def load_klh(embedding_directory):
	embedding_filename = os.path.join(embedding_directory, 
			"final_embedding.csv.gz")
	variance_filename = os.path.join(embedding_directory,
		"final_variance.csv.gz")

	embedding = load_file(embedding_filename)
	variance = load_file(variance_filename)

	return embedding, variance

def load_kle(embedding_directory):
	embedding_filename = os.path.join(embedding_directory, 
			"mu.csv.gz")
	variance_filename = os.path.join(embedding_directory,
		"sigma.csv.gz")

	embedding = load_file(embedding_filename)
	variance = load_file(variance_filename)

	return embedding, variance

def load_st(embedding_directory):
	source_filename = os.path.join(embedding_directory, 
		"source.csv.gz")
	target_filename = os.path.join(embedding_directory,
			"target.csv.gz")

	source = load_file(source_filename)
	target = load_file(target_filename)
	return source, target

def load_embedding(dist_fn, embedding_directory):

	if dist_fn == "hyperboloid":
		embedding = load_hyperboloid(embedding_directory)
		return embedding
	elif dist_fn == "poincare":
		embedding = load_poincare(embedding_directory)
		return embedding
	elif dist_fn == "euclidean":
		embedding = load_euclidean(embedding_directory)
		return embedding
	elif dist_fn == "klh":
		embedding, variance = load_klh(embedding_directory)
		return embedding, variance
	elif dist_fn == "kle":
		embedding, variance = load_kle(embedding_directory)
		return embedding, variance
	elif dist_fn == "st":
		source, target = load_st(embedding_directory)
		return source, target

def compute_scores(u, v, dist_fn):

	if dist_fn == "hyperboloid":
		scores = -hyperbolic_distance_hyperboloid(u, v)
	elif dist_fn == "poincare":
		scores = -hyperbolic_distance_poincare(u, v)
	elif dist_fn == "euclidean":
		scores = -euclidean_distance(u, v)
	elif dist_fn == "klh":
		assert isinstance(u, tuple)
		assert isinstance(v, tuple)
		scores = -kullback_leibler_divergence_hyperboloid(
			u[0], u[1], v[0], v[1])
	elif dist_fn == "kle":
		assert isinstance(u, tuple)
		assert isinstance(v, tuple)
		scores = -kullback_leibler_divergence_euclidean(
			u[0], u[1], v[0], v[1])
	elif dist_fn == "st":
		assert not isinstance(u, tuple)
		assert not isinstance(v, tuple)
		scores = -euclidean_distance(u, v)

	return scores

# def evaluate_precision_at_k(embedding, 
# 	edgelist,  
# 	dist_fn,
# 	k=10):

# 	edgelist_dict = {}
# 	for u, v in edgelist:
# 		if u not in edgelist_dict:
# 			edgelist_dict.update({u: set()})
# 		edgelist_dict[u].add(v)

# 	precisions = []
# 	for u in edgelist_dict:

# 		true_neighbours = edgelist_dict[u]
# 		if len(true_neighbours) < k:
# 			continue

# 		if isinstance(embedding, tuple):
# 			scores = compute_scores(
# 				(embedding[0][u:u+1], embedding[1][u:u+1]), 
# 				embedding,
# 				dist_fn)
# 		else:
# 			scores = compute_scores(
# 				embedding[u:u+1], 
# 				embedding,
# 				dist_fn)
# 		assert len(scores.shape) == 1
# 		nodes_sorted = scores.argsort()
# 		nodes_sorted = nodes_sorted[nodes_sorted != u][-k:]
# 		s = np.mean([u in true_neighbours for u in nodes_sorted])
# 		precisions.append(s)

# 	return np.mean(precisions)

def evaluate_mean_average_precision(
	embedding, 
	edgelist, 
	dist_fn,
	graph_edges=None,
	ks=(1,3,5,10),
	max_non_neighbours=1000
	):

	if isinstance(embedding, tuple):
		N, _  = embedding[0].shape
	else:
		N, _  = embedding.shape

	all_nodes = set(range(N))

	edgelist_dict = {}
	for u, v in edgelist:
		if u not in edgelist_dict:
			edgelist_dict.update({u: set()})
		edgelist_dict[u].add(v)

	if graph_edges:
		graph_edgelist_dict = {}
		for u, v in graph_edges:
			if u not in graph_edgelist_dict:
				graph_edgelist_dict.update({u: set()})
			if u in edgelist_dict and v not in edgelist_dict[u]:
				graph_edgelist_dict[u].add(v)

	precisions = []
	pks = {k: [] for k in ks}
	for i, u in enumerate(edgelist_dict):

		true_neighbours = edgelist_dict[u]
		non_neighbours = all_nodes - {u} - true_neighbours
		if graph_edges and u in graph_edgelist_dict:
			non_neighbours -= graph_edgelist_dict[u]
		
		true_neighbours = list(true_neighbours)
		non_neighbours = list(non_neighbours)

		if len(non_neighbours) > max_non_neighbours:
			non_neighbours = random.sample(non_neighbours, 
				k=max_non_neighbours,)

		neighbours = true_neighbours + non_neighbours

		if dist_fn in ("kle", "klh"):
			assert isinstance(embedding, tuple)
			means, variances = embedding
			scores = compute_scores(
				(means[u:u+1], variances[u:u+1]), 
				(means[neighbours], 
					variances[neighbours]), 
				dist_fn)
		elif dist_fn == "st":
			assert isinstance(embedding, tuple)
			source, target = embedding
			scores = compute_scores(
				source[u:u+1], 
				target[neighbours],  
				dist_fn)
		else:
			scores = compute_scores(
				embedding[u:u+1], 
				embedding[neighbours],
				dist_fn)
		assert len(scores.shape) == 1

		labels = np.append(np.ones_like(true_neighbours),
			np.zeros_like(non_neighbours))

		# true_neighbours = edgelist_dict[u]
		# labels = np.array([n in true_neighbours 
		# 	for n in range(N)])
		# mask = np.array([n!=u or n in true_neighbours
		# 	for n in range(N)]) # ignore self loops
		# if graph_edges and u in graph_edgelist_dict:
		# 	mask *= np.array([n not in graph_edgelist_dict[u]
		# 		for n in range(N)]) # ignore training edges
		# 	assert mask.sum() > 0
		# 	assert labels[mask].sum() > 0
		# s = average_precision_score(labels[mask], scores[mask])
		
		s = average_precision_score(labels, scores)
		precisions.append(s)

		nodes_sorted = scores.argsort()

		for k in ks:
			if len(true_neighbours) < k:
				continue
			nodes_sorted_ = nodes_sorted[-k:]
			s = np.mean([neighbours[u] in true_neighbours 
				for u in nodes_sorted_])
			pks[k].append(s)

		if i % 1000 == 0:
			print ("completed", i, "/", len(edgelist_dict))


	mAP = np.mean(precisions)
	print ("MAP", mAP)

	pks = {k: (np.mean(v) if len(v) > 0 else 0)
			for k, v in pks.items()}

	return mAP, pks

def evaluate_rank_AUROC_AP(
	embedding,
	test_edges, 
	test_non_edges, 
	dist_fn,
	):

	edge_scores = get_scores(
		embedding,
		test_edges, 
		dist_fn)

	non_edge_scores = get_scores(
		embedding,
		test_non_edges, 
		dist_fn)

	assert len(edge_scores.shape) == 1
	assert len(non_edge_scores.shape) == 1

	labels = np.append(np.ones_like(edge_scores), 
		np.zeros_like(non_edge_scores))
	scores_ = np.append(edge_scores, non_edge_scores)
	ap_score = average_precision_score(labels, scores_) # macro by default
	auc_score = roc_auc_score(labels, scores_)

	idx = (-non_edge_scores).argsort()
	ranks = np.searchsorted(-non_edge_scores, 
		-edge_scores, sorter=idx) + 1
	ranks = ranks.mean()

	print ("MEAN RANK =", ranks, "AP =", ap_score, 
		"AUROC =", auc_score)

	return ranks, ap_score, auc_score

def get_scores(embedding, edges, dist_fn):
	if dist_fn in ("kle", "klh"):
		means, variances = embedding

		embedding_u = (means[edges[:,0]], 
			variances[edges[:,0]])
		embedding_v = (means[edges[:,1]], 
			variances[edges[:,1]])

	elif dist_fn == "st":
		source, target = embedding
		embedding_u = source[edges[:,0]]
		embedding_v = target[edges[:,1]]

	else:

		embedding_u = embedding[edges[:,0]]
		embedding_v = embedding[edges[:,1]]

	return compute_scores(embedding_u, embedding_v, dist_fn)

def touch(path):
	with open(path, 'a'):
		os.utime(path, None)

def read_edgelist(fn):
	edges = []
	with open(fn, "r") as f:
		for line in (l.rstrip() for l in f.readlines()):
			edge = tuple(int(i) for i in line.split("\t"))
			edges.append(edge)
	return edges

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

def check_complete(test_results_filename, seed):
	if os.path.exists(test_results_filename):
		existing_results = pd.read_csv(test_results_filename, index_col=0)
		if seed in existing_results.index:
			print (test_results_filename, ": seed=", seed, "complete --terminating")
			return True 
	return False
	