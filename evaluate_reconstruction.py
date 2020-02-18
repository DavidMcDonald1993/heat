import sys
sys.path.append("../")

# print (sys.path)
# raise SystemExit

import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"

import random
import numpy as np
import networkx as nx

import argparse

from heat.utils import load_data
from evaluation_utils import check_complete, load_embedding, compute_scores, evaluate_rank_AUROC_AP, evaluate_mean_average_precision, evaluate_precision_at_k, touch, threadsafe_save_test_results
from remove_utils import sample_non_edges

def parse_args():

	parser = argparse.ArgumentParser(description='Load Hyperboloid Embeddings and evaluate reconstruction')
	
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
		"test_results.csv")

	if check_complete(test_results_filename, args.seed):
		return

	test_results_lock_filename = os.path.join(test_results_dir, 
		"test_results.lock")
	touch(test_results_lock_filename)

	graph, _, _ = load_data(args)
	print ("Loaded dataset")
	print ()

	random.seed(args.seed)
	
	test_edges = list(graph.edges())
	test_edges += [(v, u) for u, v in graph.edges()]
	num_edges = len(test_edges)

	test_non_edges = sample_non_edges(graph, 
		set(test_edges),
		num_edges)

	embedding = load_embedding(args.dist_fn, 
		args.embedding_directory)
	scores = compute_scores(embedding, args.dist_fn)

	test_results = dict()

	(mean_rank_recon, ap_recon, 
	roc_recon) = evaluate_rank_AUROC_AP(scores, 
		test_edges, test_non_edges)

	test_results.update({"mean_rank_recon": mean_rank_recon, 
		"ap_recon": ap_recon,
		"roc_recon": roc_recon})

	map_recon = evaluate_mean_average_precision(scores, 
		test_edges)
	test_results.update({"map_recon": map_recon})

	precisions_at_k = [(k, 
		evaluate_precision_at_k(scores,  
			test_edges, k=k))
			for k in (1, 3, 5, 10)]
	for k, pk in precisions_at_k:
		print ("precision at", k, pk)
	test_results.update({"p@{}".format(k): pk
		for k, pk in precisions_at_k})

	# test_results_dir = args.test_results_dir
	# if not os.path.exists(test_results_dir):
	# 	os.makedirs(test_results_dir)
	# test_results_filename = os.path.join(test_results_dir, "test_results.csv")
	# test_results_lock_filename = os.path.join(test_results_dir, "test_results.lock")
	# touch(test_results_lock_filename)

	print ("saving test results to {}".format(test_results_filename))

	threadsafe_save_test_results(test_results_lock_filename, 
		test_results_filename, args.seed, data=test_results )

	print ("done")


if __name__ == "__main__":
	main()