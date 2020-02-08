import os

import numpy as np
import pandas as pd

import glob

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import euclidean_distances

import functools
import fcntl

def minkowski_dot(x, y):
	assert len(x.shape) == len(y.shape) 
	return np.sum(x[...,:-1] * y[...,:-1], 
		axis=-1, keepdims=True) - x[...,-1:] * y[...,-1:]

def hyperbolic_distance_hyperboloid(X):
	u = np.expand_dims(X, axis=1)
	v = np.expand_dims(X, axis=0)
	mink_dp = -minkowski_dot(u, v)
	mink_dp = np.maximum(mink_dp - 1, np.nextafter(0, 1))
	return np.squeeze(np.arccosh(1 + mink_dp), axis=-1)

def hyperbolic_distance_poincare(X):
	norm_X = np.linalg.norm(X, keepdims=True, axis=-1)
	norm_X = np.minimum(norm_X, np.nextafter(1,0, ))
	uu = euclidean_distances(X) ** 2
	dd = (1 - norm_X**2) * (1 - norm_X**2).T
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


def kullback_leibler_divergence_euclidean(mu_sigmas):

	mus, sigmas = mu_sigmas

	dim = mus.shape[1] - 1

	# project to tangent space
	source_mus = np.expand_dims(mus, axis=1)
	target_mus = np.expand_dims(mus, axis=0)

	source_sigmas = np.expand_dims(sigmas, axis=1)
	target_sigmas = np.expand_dims(sigmas, axis=0)

	x_minus_mu = target_mus - source_mus

	trace = np.sum(target_sigmas / \
		source_sigmas, 
		axis=-1, keepdims=True)

	uu = np.sum(x_minus_mu ** 2 / \
		source_sigmas, 
		axis=-1, keepdims=True) # assume sigma is diagonal

	log_det = np.sum(np.log(target_sigmas), 
		axis=-1, keepdims=True) - \
		np.sum(np.log(source_sigmas), 
		axis=-1, keepdims=True)

	return np.squeeze(0.5 * (trace + uu - dim - log_det), axis=-1)

def kullback_leibler_divergence_hyperboloid(mu_sigmas):

	mus, sigmas = mu_sigmas

	dim = mus.shape[1] - 1

	# project to tangent space
	source_mus = np.expand_dims(mus, axis=1)
	target_mus = np.expand_dims(mus, axis=0)

	to_tangent_space = logarithmic_map(source_mus, 
		target_mus)

	# parallel transport to mu zero
	mu_zero = np.zeros((1, 1, dim + 1))
	mu_zero[..., -1] = 1
	
	to_tangent_space_mu_zero = parallel_transport(source_mus,
		mu_zero, 
		to_tangent_space)

	source_sigmas = np.expand_dims(sigmas, axis=1)
	target_sigmas = np.expand_dims(sigmas, axis=0)

	# mu is zero vector
	# ignore zero t coordinate
	x_minus_mu = to_tangent_space_mu_zero[...,:-1]

	sigma_ratio = target_sigmas / source_sigmas

	trace_fac = np.sum(sigma_ratio,
		axis=-1, keepdims=True)

	mu_sq_diff = np.sum(x_minus_mu ** 2 / \
		source_sigmas,
		axis=-1, keepdims=True) # assume sigma inv is diagonal

	log_det = np.sum(np.log(sigma_ratio), axis=-1, keepdims=True)

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
	return df.values


def load_hyperboloid(embedding_directory):
	files = sorted(glob.iglob(os.path.join(embedding_directory, 
		"*embedding.csv.gz")))
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
		"*embedding.csv.gz")))
	embedding_filename = files[-1]
	embedding = load_file(embedding_filename, header=None, sep=" ")
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

def compute_scores(embedding, dist_fn):

	if dist_fn == "hyperboloid":
		scores = -hyperbolic_distance_hyperboloid(embedding)
	elif dist_fn == "poincare":
		scores = -hyperbolic_distance_poincare(embedding)
	elif dist_fn == "euclidean":
		scores = -euclidean_distances(embedding)
	elif dist_fn == "klh":
		scores = -kullback_leibler_divergence_hyperboloid(embedding)
	elif dist_fn == "kle":
		scores = -kullback_leibler_divergence_euclidean(embedding)
	elif dist_fn == "st":
		scores = -euclidean_distances(embedding[0], embedding[1])

	return scores

def evaluate_precision_at_k(scores, 
	edgelist,  
	k=10):

	edgelist_dict = {}
	for u, v in edgelist:
		if u not in edgelist_dict:
			edgelist_dict.update({u: []})
		edgelist_dict[u].append(v)

	precisions = []
	for u in edgelist_dict:
		scores_ = scores[u]
		true_neighbours = edgelist_dict[u]
		nodes_sorted = scores_.argsort()
		nodes_sorted = nodes_sorted[nodes_sorted != u][-k:]
		s = np.mean([u in true_neighbours for u in nodes_sorted])
		precisions.append(s)

	return np.mean(precisions)

def evaluate_mean_average_precision(scores, 
	edgelist, 
	graph_edges=None
	):
	N, _  = scores.shape
	edgelist_dict = {}
	for u, v in edgelist:
		if u not in edgelist_dict:
			edgelist_dict.update({u: []})
		edgelist_dict[u].append(v)
	if graph_edges:
		graph_edgelist_dict = {}
		for u, v in graph_edges:
			if u not in graph_edgelist_dict:
				graph_edgelist_dict.update({u: []})
			if u in edgelist_dict and v not in edgelist_dict[u]:
				graph_edgelist_dict[u].append(v)

	precisions = []
	for u in edgelist_dict:
		scores_ = scores[u]
		true_neighbours = edgelist_dict[u]
		labels = np.array([n in true_neighbours 
			for n in range(N)])
		mask = np.array([n!=u
			for n in range(N)]) # ignore self loops
		if graph_edges and u in graph_edgelist_dict:
			mask *= np.array([n not in graph_edgelist_dict[u]
				for n in range(N)]) # ignore training edges
			assert mask.sum() > 0
			assert labels[mask].sum() > 0
		s = average_precision_score(labels[mask], scores_[mask])
		precisions.append(s)

	return np.mean(precisions)

def evaluate_rank_AUROC_AP(scores, 
	edgelist, 
	non_edgelist):
	assert not isinstance(edgelist, dict)
	assert (scores <= 0).all()

	if not isinstance(edgelist, np.ndarray):
		edgelist = np.array(edgelist)

	if not isinstance(non_edgelist, np.ndarray):
		non_edgelist = np.array(non_edgelist)

	edge_scores = scores[edgelist[:,0], edgelist[:,1]]
	non_edge_scores = scores[non_edgelist[:,0], non_edgelist[:,1]]

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



	