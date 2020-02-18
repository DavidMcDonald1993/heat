import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"

import random
import numpy as np
import networkx as nx

import argparse

from heat.utils import load_data

def write_edgelist_to_file(edgelist, file):
	with open(file, "w+") as f:
		for u, v in edgelist:
			f.write("{}\t{}\n".format(u, v))

def split_edges(edges, non_edges, seed, val_split=0.00, test_split=0.15, neg_mul=1):
	
	num_val_edges = int(np.ceil(len(edges) * val_split))
	num_test_edges = int(np.ceil(len(edges) * test_split))

	random.seed(seed)
	random.shuffle(edges)
	random.shuffle(non_edges)

	val_edges = edges[:num_val_edges]
	test_edges = edges[num_val_edges:num_val_edges+num_test_edges]
	train_edges = edges[num_val_edges+num_test_edges:]

	val_non_edges = non_edges[:num_val_edges*neg_mul]
	test_non_edges = non_edges[num_val_edges*neg_mul:num_val_edges*neg_mul+num_test_edges*neg_mul]

	return train_edges, (val_edges, val_non_edges), (test_edges, test_non_edges)

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Script to remove edges for link prediction experiments")

	parser.add_argument("--edgelist", dest="edgelist", type=str, 
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")
	parser.add_argument("--output", dest="output", type=str,
		help="path to save training and removed edges")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument("--seed", type=int, default=0)

	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	seed= args.seed
	training_edgelist_dir = os.path.join(args.output, "seed={:03d}".format(seed), "training_edges")
	removed_edges_dir = os.path.join(args.output, "seed={:03d}".format(seed), "removed_edges")

	if not os.path.exists(training_edgelist_dir):
		os.makedirs(training_edgelist_dir, exist_ok=True)
	if not os.path.exists(removed_edges_dir):
		os.makedirs(removed_edges_dir, exist_ok=True)

	training_edgelist_fn = os.path.join(training_edgelist_dir, "edgelist.tsv")
	val_edgelist_fn = os.path.join(removed_edges_dir, "val_edges.tsv")
	val_non_edgelist_fn = os.path.join(removed_edges_dir, "val_non_edges.tsv")
	test_edgelist_fn = os.path.join(removed_edges_dir, "test_edges.tsv")
	test_non_edgelist_fn = os.path.join(removed_edges_dir, "test_non_edges.tsv")
	
	graph, _, _ = load_data(args)
	print("loaded dataset")

	edges = list(graph.edges())
	non_edges = list(nx.non_edges(graph))

	_, (val_edges, val_non_edges), (test_edges, test_non_edges) = split_edges(edges, non_edges, seed)

	for edge in test_edges:
		assert edge in graph.edges() or edge[::-1] in graph.edges()

	graph.remove_edges_from(val_edges + test_edges)
	graph.add_edges_from(((u, u, {"weight": 0}) for u in graph.nodes())) # ensure that every node appears at least once by adding self loops

	print ("removed edges")

	nx.write_edgelist(graph, training_edgelist_fn, delimiter="\t", data=["weight"])
	write_edgelist_to_file(val_edges, val_edgelist_fn)
	write_edgelist_to_file(val_non_edges, val_non_edgelist_fn)
	write_edgelist_to_file(test_edges, test_edgelist_fn)
	write_edgelist_to_file(test_non_edges, test_non_edgelist_fn)

	print ("done")

if __name__ == "__main__":
	main()
