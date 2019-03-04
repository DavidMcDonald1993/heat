from __future__ import print_function

import os
import re
import argparse
import numpy as np
import networkx as nx
import pandas as pd

from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib.pyplot as plt

import h5py

from data_utils import load_g2g_datasets

def minkowski_dot_pairwise(u, v):
	"""
	`u` and `v` are vectors in Minkowski space.
	"""
	euc_dp = u[:,:-1].dot(v[:,:-1].T)
	return euc_dp - u[:,-1, None] * v[:,-1]

def hyperbolic_distance_hyperboloid_pairwise(X, Y):
	inner_product = minkowski_dot_pairwise(X, Y)
	inner_product = np.minimum(inner_product, -(1 + 1e-32))
	return np.arccosh(-inner_product)

def load_embedding(filename):
	with h5py.File(filename, 'r') as f:
		embedding = np.array(f.get("embedding_layer/embedding_layer/embedding:0"))
	return embedding

def evaluate_greedy_routing(graph, dists, args):

	print ("Evaluating greedy routing")

	np.random.seed(args.seed)

	lcc = max(nx.connected_component_subgraphs(graph), key=len)

	num_routing = args.num_routing
	N = len(dists)

	starts = []
	ends = []

	path_lengths = []

	for r in range(num_routing):
		start = np.random.choice(lcc.nodes())
		end = np.random.choice(lcc.nodes())
		while start == end:
			end = np.random.choice(lcc.nodes())
		starts.append(start)
		ends.append(end)


		failure = False

		path = [start]
		cur = path[-1]
		while cur != end:
			neighbours = graph.neighbors(cur)
			neighbour_dists = dists[end, neighbours]
			next_ = neighbours[neighbour_dists.argmin()]

			if next_ in path:
				failure = True
				path_lengths.append(-1)
				break

			path.append(next_)
			cur = path[-1]

		if not failure:
			assert len(path) > 0
			path_lengths.append(len(path) - 1) # number of hops

	path_lengths = np.array(path_lengths)
	complete = path_lengths >= 0
	mean_complete = np.mean(complete)

	print ("Determining shortest path lengths")
	true_sp_length = np.array([nx.shortest_path_length(graph, source=start, 
		target=end, weight="weight") for start, end in zip(starts, ends)])
	hop_stretch = path_lengths[complete] / true_sp_length[complete]
	mean_hop_stretch = np.mean(hop_stretch)

	print ("mean complete = ", mean_complete, "mean hop stretch = ", mean_hop_stretch)

	return mean_complete, mean_hop_stretch


def parse_model_filename(args):

	dataset = args.dataset
	directory = os.path.join(dataset, "dim={:03d}".format(args.embedding_dim), "seed={:03d}/".format(args.seed))

	if args.only_lcc:
		directory += "lcc/"
	else:
		directory += "all_components/"

	if args.evaluate_link_prediction:
		directory += "eval_lp/"
		if args.add_non_edges:
			directory += "add_non_edges/"
		else:
			directory += "no_non_edges/"
	elif args.evaluate_class_prediction:
		directory += "eval_class_pred/"
	else: 
		directory += "no_lp/"


	if args.softmax:
		directory += "softmax_loss/"
	elif args.sigmoid:
		directory += "sigmoid_loss/"
	elif args.euclidean:
		directory += "euclidean_loss/"
	else:
		directory += "hyperbolic_distance_loss/r={}_t={}/".format(args.r, args.t)


	
	if args.multiply_attributes:
		directory += "multiply_attributes/"
	elif args.alpha>0:
		directory += "add_attributes_alpha={}/".format(args.alpha, )
	elif args.jump_prob > 0:
		directory += "jump_prob={}/".format(args.jump_prob)
	else:
		directory += "no_attributes/"

	args.model_path = os.path.join(args.model_path, directory)

	# saved_models = sorted([f for f in os.listdir(args.model_path) 
	# 	if re.match(r"^[0-9][0-9][0-9][0-9]*", f)])
	# assert len(saved_models) > 0
	# return os.path.join(args.model_path, saved_models[-1])
	return os.path.join(args.model_path, "best_model.h5")

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Greedy routing on hyperbolic embeddings of Complex Networks")

	parser.add_argument("--data-directory", dest="data_directory", type=str, default="/data/",
		help="The directory containing data files (default is '/data/').")

	parser.add_argument("--dataset", dest="dataset", type=str, default="cora_ml",
		help="The dataset to load. Must be one of [wordnet, cora, citeseer, pubmed,\
		AstroPh, CondMat, GrQc, HepPh, karate]. (Default is cora_ml)")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")

	parser.add_argument("-r", dest="r", type=float, default=3.,
		help="Radius of hypercircle (default is 3).")
	parser.add_argument("-t", dest="t", type=float, default=1.,
		help="Steepness of logistic function (defaut is 1).")

	parser.add_argument("-d", "--dim", dest="embedding_dim", type=int,
		help="Dimension of embeddings for each layer (default is 2).", default=2)

	parser.add_argument("--alpha", dest="alpha", type=float, default=0,
		help="weighting of attributes (default is 0).")

	parser.add_argument("--no-attributes", action="store_true", 
		help="Use this flag to not use attributes.")
	parser.add_argument("--multiply-attributes", action="store_true", 
		help="Use this flag to multiply attribute sim to adj.")
	parser.add_argument("--jump-prob", dest="jump_prob", type=float, default=0, 
		help="Probability of randomly jumping to a similar node when walking.")

	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", 
		help="Use this flag to set verbosity of training.")
	parser.add_argument('--workers', dest="workers", type=int, default=2, 
		help="Number of worker threads to generate training patterns (default is 2).")

	parser.add_argument("--model", dest="model_path", default="models/", 
		help="path to save model after each epoch (default is 'models/)'.")

	parser.add_argument("--sigmoid", dest="sigmoid", action="store_true", 
		help="Use this flag to use sigmoid loss.")
	parser.add_argument("--softmax", dest="softmax", action="store_true", 
		help="Use this flag to use softmax loss.")
	parser.add_argument("--euclidean", dest="euclidean", action="store_true", 
		help="Use this flag to use euclidean negative sampling loss.")
	
	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')


	parser.add_argument('--only-lcc', action="store_true", help='flag to train on only lcc')

	parser.add_argument('--evaluate-class-prediction', action="store_true", help='flag to evaluate class prediction')
	parser.add_argument('--evaluate-link-prediction', action="store_true", help='flag to evaluate link prediction')

	parser.add_argument('--no-non-edges', action="store_true", help='flag to not add non edges to training graph')
	parser.add_argument('--add-non-edges', action="store_true", help='flag to add non edges to training graph')

	parser.add_argument('--num-routing', dest="num_routing", type=int, default=1000, 
		help="Number of source-target pairs to evaluate (default is 1000).")

	args = parser.parse_args()
	return args


def main():

	args = parse_args() # parse args from command line 
	args.softmax = True

	np.random.seed(args.seed)

	dataset = args.dataset
	if dataset in ["cora", "cora_ml", "pubmed", "citeseer"]:
		# topology_graph, features, labels = load_labelled_attributed_network(dataset, args)
		topology_graph, features, labels, label_info = load_g2g_datasets(dataset, args)
	else:
		raise Exception

	if not args.evaluate_link_prediction:
		args.evaluate_class_prediction = labels is not None

	model_filename = parse_model_filename(args)

	embedding = load_embedding(model_filename)

	dists = hyperbolic_distance_hyperboloid_pairwise(embedding, embedding)

	mean_complete, mean_hop_stretch = evaluate_greedy_routing(topology_graph, dists, args)


if __name__ == "__main__":
	main()