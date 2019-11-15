import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"


import numpy as np
import networkx as nx
import pandas as pd

import argparse

from heat.utils import load_embedding, load_data

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances

import functools
import fcntl

def minkowki_dot(u, v):
	"""
	`u` and `v` are vectors in Minkowski space.
	"""
	rank = u.shape[-1] - 1
	euc_dp = u[:,:rank].dot(v[:,:rank].T)
	return euc_dp - u[:,rank, None] * v[:,rank]

def hyperbolic_distance_hyperboloid(u, v):
	mink_dp = minkowki_dot(u, v)
	mink_dp = np.maximum(-1 - mink_dp, 1e-15)
	return np.arccosh(1 + mink_dp)

def hyperbolic_distance_poincare(X):
	norm_X = np.linalg.norm(X, keepdims=True, axis=-1)
	norm_X = np.minimum(norm_X, np.nextafter(1,0, ))
	uu = euclidean_distances(X) ** 2
	dd = (1 - norm_X**2) * (1 - norm_X**2).T
	return np.arccosh(1 + 2 * uu / dd)

def euclidean_distance(X):
	return euclidean_distances(X)

def evaluate_precision_at_k(scores, edgelist, non_edgelist, k=10):
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
	non_edgelist):
	edgelist_dict = {}
	for u, v in edgelist:
		if u not in edgelist_dict:
			edgelist_dict.update({u: []})
		edgelist_dict[u].append(v)

	precisions = []
	for u in edgelist_dict:
		scores_ = scores[u]
		true_neighbours = edgelist_dict[u]
		labels = np.array([n in true_neighbours 
			for n in range(len(scores))])
		mask = np.array([n!=u for n in range(len(scores))])
		s = average_precision_score(labels[mask], scores_[mask])
		precisions.append(s)

	return np.mean(precisions)

def evaluate_rank_and_AP(scores, 
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


def parse_args():

	parser = argparse.ArgumentParser(description='Load Hyperboloid Embeddings and evaluate reconstruction')
	
	parser.add_argument("--edgelist", dest="edgelist", type=str, 
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")

	parser.add_argument("--embedding", dest="embedding_filename",  
		help="path of embedding to load.")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')


	parser.add_argument("--test-results-dir", dest="test_results_dir",  
		help="path to save results.")

	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--dist_fn", dest="dist_fn", type=str,
	choices=["poincare", "hyperboloid", "euclidean"])


	return parser.parse_args()


def main():

	args = parse_args()

	graph, _, _ = load_data(args)
	print ("Loaded dataset")
	
	dist_fn = args.dist_fn

	sep = ","
	header = "infer"
	if dist_fn == "euclidean":
		sep = " "
		header = None

	embedding_df = pd.read_csv(args.embedding_filename,
		sep=sep, header=header, index_col=0)
	embedding_df = embedding_df.reindex(sorted(embedding_df.index))	
	# row 0 is embedding for node 0
	# row 1 is embedding for node 1 etc...
	embedding = embedding_df.values


	if dist_fn == "poincare":
		dists = hyperbolic_distance_poincare(embedding)
	elif dist_fn == "hyperboloid":
		dists = hyperbolic_distance_hyperboloid(embedding, embedding)
	else:
		dists = euclidean_distance(embedding)

	test_edges = np.array(list(graph.edges()))
	test_non_edges = np.array(list(nx.non_edges(graph)))

	np.random.seed(args.seed)
	idx = np.random.permutation(len(test_non_edges))[:len(test_edges)]
	test_non_edges = test_non_edges[idx]

	test_results = dict()

	scores = -dists

	(mean_rank_recon, ap_recon, 
	roc_recon) = evaluate_rank_and_AP(scores, 
		test_edges, test_non_edges)

	test_results.update({"mean_rank_recon": mean_rank_recon, 
		"ap_recon": ap_recon,
		"roc_recon": roc_recon})

	map_reconstruction = evaluate_mean_average_precision(scores,
		test_edges, test_non_edges)

	test_results.update({"map_recon": map_reconstruction})

	precisions_at_k = [(k, 
		evaluate_precision_at_k(scores,  
			test_edges, test_non_edges, k=k))
			for k in (1, 3, 5, 10)]
	for k, pk in precisions_at_k:
		print ("precision at", k, pk)
	test_results.update({"p@{}".format(k): pk
		for k, pk in precisions_at_k})

	test_results_dir = args.test_results_dir
	if not os.path.exists(test_results_dir):
		os.makedirs(test_results_dir)
	test_results_filename = os.path.join(test_results_dir, "test_results.csv")
	test_results_lock_filename = os.path.join(test_results_dir, "test_results.lock")
	touch(test_results_lock_filename)

	print ("saving test results to {}".format(test_results_filename))

	threadsafe_save_test_results(test_results_lock_filename, test_results_filename, args.seed, data=test_results )

	print ("done")


if __name__ == "__main__":
	main()