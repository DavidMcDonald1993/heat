import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"

import numpy as np
import networkx as nx

import argparse

from heat.utils import load_data
from evaluation_utils import load_embedding, compute_scores, evaluate_rank_AUROC_AP, evaluate_mean_average_precision, evaluate_precision_at_k, touch, threadsafe_save_test_results, read_edgelist

def parse_args():

	parser = argparse.ArgumentParser(description='Load Hyperboloid Embeddings and evaluate link prediction')
	
	parser.add_argument("--edgelist", dest="edgelist", type=str, 
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")
	parser.add_argument("--output", dest="output", type=str, 
		help="path to load training and removed edges")
	
	parser.add_argument("--embedding", dest="embedding_directory",  
		help="directory of embedding to load.")

	parser.add_argument("--test-results-dir", dest="test_results_dir",  
		help="path to save results.")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--dist_fn", dest="dist_fn", type=str,
		choices=["poincare", "hyperboloid", "euclidean", "kle", "klh"])

	return parser.parse_args()

def main():

	args = parse_args()

	args.directed = True

	graph, _, _ = load_data(args)
	assert nx.is_directed(graph)
	print ("Loaded dataset")
	print ()

	seed= args.seed
	removed_edges_dir = os.path.join(args.output, 
		"seed={:03d}".format(seed), "removed_edges")

	test_edgelist_fn = os.path.join(removed_edges_dir, 
		"test_edges.tsv")
	test_non_edgelist_fn = os.path.join(removed_edges_dir, 
		"test_non_edges.tsv")

	print ("loading test edges from {}".format(test_edgelist_fn))
	print ("loading test non-edges from {}".format(test_non_edgelist_fn))

	test_edges = read_edgelist(test_edgelist_fn)
	test_non_edges = read_edgelist(test_non_edgelist_fn)

	print ("number of test edges:", len(test_edges))
	print ("number of test non edges:", len(test_non_edges))

	embedding = load_embedding(args.dist_fn, args.embedding_directory)

	scores = compute_scores(embedding, args.dist_fn)

	test_results = dict()

	(mean_rank_lp, ap_lp, 
		roc_lp) = evaluate_rank_AUROC_AP(scores, 
		test_edges, 
		test_non_edges)

	test_results.update({"mean_rank_lp": mean_rank_lp, 
		"ap_lp": ap_lp,
		"roc_lp": roc_lp})

	map_lp = evaluate_mean_average_precision(scores, 
		test_edges, 
		graph_edges=graph.edges()
	)

	print ("MAP lp", map_lp)

	test_results.update({"map_lp": map_lp})

	test_results_dir = args.test_results_dir
	if not os.path.exists(test_results_dir):
		os.makedirs(test_results_dir, exist_ok=True)
	test_results_filename = os.path.join(test_results_dir, 
		"test_results.csv")
	test_results_lock_filename = os.path.join(test_results_dir, 
		"test_results.lock")
	touch(test_results_lock_filename)

	print ("saving test results to {}".format(test_results_filename))

	threadsafe_save_test_results(test_results_lock_filename, 
		test_results_filename, seed, data=test_results )

	print ("done")


if __name__ == "__main__":
	main()