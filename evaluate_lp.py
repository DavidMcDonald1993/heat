import os

import numpy as np
import networkx as nx

import argparse

from heat.utils import load_data
from evaluation_utils import check_complete, load_embedding, compute_scores, evaluate_rank_AUROC_AP, evaluate_mean_average_precision, touch, threadsafe_save_test_results, read_edgelist

import random

def parse_args():

	parser = argparse.ArgumentParser(description='Load Embeddings and evaluate link prediction')
	
	parser.add_argument("--edgelist", dest="edgelist", type=str, 
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")
	parser.add_argument("--removed_edges_dir", dest="removed_edges_dir", type=str, 
		help="path to load removed edges")
	
	parser.add_argument("--embedding", dest="embedding_directory",  
		help="directory of embedding to load.")

	parser.add_argument("--test-results-dir", dest="test_results_dir",  
		help="path to save results.")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

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
		"test_results.csv")

	if check_complete(test_results_filename, args.seed):
		return

	test_results_lock_filename = os.path.join(test_results_dir, 
		"test_results.lock")
	touch(test_results_lock_filename)

	graph, _, _ = load_data(args)
	assert not nx.is_directed(graph)
	print ("Loaded dataset")
	print ()

	seed = args.seed
	random.seed(seed)

	removed_edges_dir = args.removed_edges_dir

	test_edgelist_fn = os.path.join(removed_edges_dir, 
		"test_edges.tsv")
	test_non_edgelist_fn = os.path.join(removed_edges_dir, 
		"test_non_edges.tsv")

	print ("loading test edges from {}".format(test_edgelist_fn))
	print ("loading test non-edges from {}".format(test_non_edgelist_fn))

	test_edges = read_edgelist(test_edgelist_fn)
	test_non_edges = read_edgelist(test_non_edgelist_fn)

	test_edges = np.array(test_edges)
	test_non_edges = np.array(test_non_edges)

	print ("number of test edges:", len(test_edges))
	print ("number of test non edges:", len(test_non_edges))

	embedding = load_embedding(args.dist_fn, 
		args.embedding_directory)

	test_results = dict()

	(mean_rank_lp, ap_lp, 
		roc_lp) = evaluate_rank_AUROC_AP(
			embedding,
			test_edges, 
			test_non_edges,
			args.dist_fn)

	test_results.update(
		{"mean_rank_lp": mean_rank_lp, 
		"ap_lp": ap_lp,
		"roc_lp": roc_lp})

	map_lp, precisions_at_k = evaluate_mean_average_precision(
		embedding, 
		test_edges,
		args.dist_fn, 
		graph_edges=graph.edges()
		)

	test_results.update({"map_lp": map_lp})

	for k, pk in precisions_at_k.items():
		print ("precision at", k, pk)
	test_results.update({"p@{}".format(k): pk
		for k, pk in precisions_at_k.items()})

	print ("saving test results to {}".format(
		test_results_filename))

	threadsafe_save_test_results(test_results_lock_filename, 
		test_results_filename, seed, data=test_results )

	print ("done")


if __name__ == "__main__":
	main()