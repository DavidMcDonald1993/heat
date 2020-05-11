import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"

import random
import numpy as np
import networkx as nx

import argparse

from heat.utils import load_data
from remove_utils import sample_non_edges, write_edgelist_to_file

def split_edges(graph, 
	edges, 
	seed,
	val_split=0.05, 
	test_split=0.10, 
	neg_mul=1,
	cover=True):
	
	assert isinstance(graph, nx.DiGraph)
	assert isinstance(edges, list)

	num_nodes = len(graph)
	num_edges = len(edges)

	edge_set = set(edges)

	num_val_edges = int(np.ceil(num_edges * val_split))
	num_test_edges = int(np.ceil(num_edges * test_split))

	random.seed(seed)
	random.shuffle(edges)

	# ensure every node appears in edgelist
	nodes = set(graph)

	cover_edges  = []

	if cover:
		for u, v in edges:
			if u in nodes or v in nodes:
				nodes -= {u, v}
				cover_edges.append((u, v))
			if len(nodes) == 0:
				break
		
		print ("determined cover")

		edges = filter(lambda edge: 
			edge not in cover_edges, edges)
		print ("filtered cover out of edges")

	val_edges = []
	test_edges = []
	train_edges = []
	for edge in edges:
		if len(val_edges) < num_val_edges:
			val_edges.append(edge)
		elif len(test_edges) < num_test_edges:
			test_edges.append(edge)
		else:
			train_edges.append(edge)

	train_edges += cover_edges
	print ("determined edge split")

	val_non_edges = sample_non_edges(graph, 
		edge_set, 
		num_val_edges*neg_mul)
	print ("determined val non edges")
	test_non_edges = sample_non_edges(graph,
		edge_set.union(val_non_edges),
		num_test_edges*neg_mul)
	print ("determined test non edges")

	return (train_edges, (val_edges, val_non_edges), 
		(test_edges, test_non_edges))

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

	parser.add_argument('--directed', 
		action="store_true", help='flag to train on directed graph')

	parser.add_argument("--seed", type=int, default=0)

	args = parser.parse_args()
	return args

def main():

	args = parse_args()
	args.directed = True

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
	assert nx.is_directed(graph)
	N = len(graph)

	for u in range(N):
		assert u in graph

	edges = list(graph.edges())
	print ("enumerated edges")

	(training_edges, (val_edges, val_non_edges), 
		(test_edges, test_non_edges)) = split_edges(graph, 
			edges, 
			seed, 
			val_split=0)

	assert len(nx.DiGraph(training_edges)) == N

	print ("number of val edges", len(val_edges), 
		"number of val non edges", len(val_edges))
	print ("number of test edges", len(test_edges), 
		"number of test non edges", len(test_edges))

	# remove val and test edges
	graph.remove_edges_from(val_edges + test_edges) 

	print ("removed edges")

	nx.write_edgelist(graph, training_edgelist_fn, 
		delimiter="\t",
		data=["weight"])
	write_edgelist_to_file(val_edges, val_edgelist_fn)
	write_edgelist_to_file(val_non_edges, val_non_edgelist_fn)
	write_edgelist_to_file(test_edges, test_edgelist_fn)
	write_edgelist_to_file(test_non_edges, test_non_edgelist_fn)

	print ("done")
	
if __name__ == "__main__":
	main()
