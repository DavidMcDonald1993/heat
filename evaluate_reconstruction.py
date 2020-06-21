import os

import random
import numpy as np
import networkx as nx
import pandas as pd

import argparse

import pickle as pkl

from heat.utils import load_data
from evaluation_utils import check_complete, load_embedding, compute_scores, evaluate_rank_AUROC_AP, evaluate_mean_average_precision, touch, threadsafe_save_test_results
from remove_utils import sample_non_edges

def parse_args():

	parser = argparse.ArgumentParser(description='Load Embeddings and evaluate reconstruction')
	
	parser.add_argument("--edgelist", dest="edgelist", type=str, 
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument("--embedding", dest="embedding_directory",  
		help="directory of embedding to load.")

	parser.add_argument("--test-results-dir", dest="test_results_dir",  
		help="path to save results.")

	parser.add_argument("--seed", type=int, default=0)
	
	parser.add_argument("--dist_fn", dest="dist_fn", type=str,
		choices=["poincare", "hyperboloid", "euclidean", 
			"kle", "klh", "st"])

	return parser.parse_args()

def main():

	args = parse_args()

	test_results_dir = args.test_results_dir
	if not os.path.exists(test_results_dir):
		os.makedirs(test_results_dir, exist_ok=True)
	
	test_results_filename = os.path.join(test_results_dir, 
		"{}.pkl".format(args.seed))

	# if check_complete(test_results_filename, args.seed):
	# 	return

	# test_results_lock_filename = os.path.join(test_results_dir, 
	# 	"test_results.lock")
	# touch(test_results_lock_filename)

	graph, _, _ = load_data(args)
	assert not args.directed 
	assert not nx.is_directed(graph)
	print ("Loaded dataset")
	print ()

	random.seed(args.seed)
	
	test_edges = list(graph.edges())

	test_edges += [(v, u) for u, v in test_edges]

	num_edges = len(test_edges)

	test_non_edges = sample_non_edges(graph, 
		set(test_edges),
		num_edges)

	test_edges = np.array(test_edges)
	test_non_edges = np.array(test_non_edges)


	print ("number of test edges:", len(test_edges))
	print ("number of test non edges:", len(test_non_edges))


	embedding = load_embedding(args.dist_fn, 
		args.embedding_directory)
	
	test_results = dict()

	(mean_rank_recon, ap_recon, 
		roc_recon) = evaluate_rank_AUROC_AP(
			embedding,
			test_edges, 
			test_non_edges,
			args.dist_fn)

	test_results.update({"mean_rank_recon": mean_rank_recon, 
		"ap_recon": ap_recon,
		"roc_recon": roc_recon})

	map_recon, precisions_at_k = evaluate_mean_average_precision(
		embedding, 
		test_edges,
		args.dist_fn)
	test_results.update({"map_recon": map_recon})

	for k, pk in precisions_at_k.items():
		print ("precision at", k, pk)
	test_results.update({"p@{}".format(k): pk
		for k, pk in precisions_at_k.items()})

	print ("saving test results to {}".format(
		test_results_filename))

	test_results = pd.Series(test_results)
	with open(test_results_filename, "wb") as f:
		pkl.dump(test_results, f, pkl.HIGHEST_PROTOCOL)

	# threadsafe_save_test_results(test_results_lock_filename, 
	# 	test_results_filename, args.seed, data=test_results )

	print ("done")


if __name__ == "__main__":
	main()