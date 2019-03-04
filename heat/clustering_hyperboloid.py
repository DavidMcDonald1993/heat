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

from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

# from data_utils import load_karate, load_g2g_datasets, load_ppi, load_tf_interaction
from tree import TopologyConstrainedTree

def minkowki_dot(u, v):
	"""
	`u` and `v` are vectors in Minkowski space.
	"""
	rank = u.shape[-1] - 1
	euc_dp = u[:,:rank].dot(v[:,:rank].T)
	return euc_dp - u[:,rank, None] * v[:,rank]

def hyperbolic_distance(u, v):
	mink_dp = minkowki_dot(u, v)
	mink_dp = np.minimum(mink_dp, -(1 + 1e-32))
	return np.arccosh(-mink_dp)

def load_embedding(filename):
	with h5py.File(filename, 'r') as f:
		embedding = np.array(f.get("embedding_layer/embedding_layer/embedding:0"))
	return embedding

def perform_clustering(dists, eps):
	dbsc = DBSCAN(metric="precomputed", eps=eps, n_jobs=-1, min_samples=3)
	labels = dbsc.fit_predict(dists)
	return labels

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def convert_module_to_directed_module(module, ranks):
	# module = nx.Graph(module)
	idx = module.nodes()

	directed_modules = []

	for connected_component in nx.connected_component_subgraphs(module):
		nodes = connected_component.nodes()
		root = nodes[ranks[nodes].argmin()]
		directed_edges = [(u, v) if ranks[u] < ranks[v] else (v, u) for u ,v in connected_component.edges() if u != v]
		directed_module = nx.DiGraph(directed_edges)

		# print (len(connected_component), len(connected_component.edges()), len(directed_module), len(directed_edges), )

		if len(directed_module) > 0:
			directed_modules += [directed_module]

	return directed_modules

def grow_forest(data_train, directed_modules, ranks, feature_names, bootstrap=True, ):

	n = data_train.shape[0]


	forest = []
	all_oob_samples = []

	for directed_module in directed_modules:
		feats = directed_module.nodes()
		root = feats[ranks[feats].argmin()]
		assert nx.is_connected(directed_module.to_undirected())

		if bootstrap:
			idx = np.random.choice(n, size=n, replace=True)
			_data_train = data_train[idx]
		else:
			_data_train = data_train

		tree = TopologyConstrainedTree(parent_index=None, index=root, g=directed_module, 
			data=_data_train, feature_names=feature_names, depth=0, max_depth=np.inf, min_samples_split=2, min_neighbours=1)

		if bootstrap:
			oob_samples = list(set(range(n)) - set(idx))
			oob_samples = data_train[oob_samples]
			all_oob_samples.append(oob_samples)

			# oob_prediction = tree.predict(oob_samples)
			# oob_prediction_accuracy = tree.prediction_accuracy(oob_samples[:,-1], oob_prediction)
			# print (n, set(idx), len(oob_samples), oob_prediction_accuracy)

		forest.append(tree)
	return forest, all_oob_samples

def evaluate_modules_on_test_data(features, labels, directed_modules, ranks, feature_names,
	n_repeats=10, test_size=0.3, ):

	data = np.column_stack([features, labels])

	f1_micros = []

	sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=0)
	for split_train, split_test in sss.split(features, labels):

		data_train = data[split_train]
		data_test = data[split_test]


		forest, _ = grow_forest(data_train, directed_modules, ranks, feature_names)


		test_prediction = np.array([t.predict(data_test) for t in forest])
		test_prediction = test_prediction.mean(axis=0) > 0.5
		f1_micro = f1_score(data_test[:,-1], test_prediction, average="micro")
		f1_micros.append(f1_micro)

	return np.mean(f1_micros)

def determine_feature_importances(forest, all_oob_samples):

	n_trees = len(forest)
	n_features = all_oob_samples[0].shape[1] - 1
	feature_importances = np.zeros((n_trees, n_features),)
	feature_pair_importances = np.zeros((n_trees, n_features, n_features), )

	for i, tree, oob_samples in zip(range(n_trees), forest, all_oob_samples):
		oob_sample_prediction = tree.predict(oob_samples)
		oob_sample_accuracy = tree.prediction_accuracy(oob_samples[:,-1], oob_sample_prediction)

		for feature in range(n_features):
			_oob_samples = oob_samples.copy()
			np.random.shuffle(_oob_samples[:,feature])
			permuted_prediction = tree.predict(_oob_samples)
			permuted_prediction_accuracy = tree.prediction_accuracy(_oob_samples[:,-1], permuted_prediction)
			feature_importances[i, feature] = oob_sample_accuracy - permuted_prediction_accuracy

		# for f1 in range(n_features):
		# 	for f2 in range(n_features):
		# 		_oob_samples = oob_samples.copy()
		# 		np.random.shuffle(_oob_samples[:,f1])
		# 		np.random.shuffle(_oob_samples[:,f2])

		# 		permuted_prediction = tree.predict(_oob_samples)
		# 		permuted_prediction_accuracy = tree.prediction_accuracy(_oob_samples[:,-1], permuted_prediction)
		# 		feature_importances[i, feature] = oob_sample_accuracy - permuted_prediction_accuracy

	return feature_importances.mean(axis=0), feature_pair_importances.mean(axis=0)


def plot_disk_embeddings(edges, poincare_embedding, modules,):

	if not isinstance(edges, np.ndarray):
		edges = np.array(edges) 

	all_modules = sorted(set(modules))
	num_modules = len(all_modules)
	colors = np.random.rand(num_modules, 3)

	fig = plt.figure(figsize=[14, 7])
	
	ax = fig.add_subplot(111)
	plt.title("Poincare")
	ax.add_artist(plt.Circle([0,0], 1, fill=False))
	u_emb = poincare_embedding[edges[:,0]]
	v_emb = poincare_embedding[edges[:,1]]
	plt.plot([u_emb[:,0], v_emb[:,0]], [u_emb[:,1], v_emb[:,1]], c="k", linewidth=0.05, zorder=0)
	for i, m in enumerate(all_modules):
		idx = modules == m
		plt.scatter(poincare_embedding[idx,0], poincare_embedding[idx,1], s=10, 
			c=colors[i], label="module={}".format(m) if m > -1 else "noise", zorder=1)
	plt.xlim([-1,1])
	plt.ylim([-1,1])

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=4)

	# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=3)
	plt.show()
	# plt.savefig(path)
	plt.close()

# def load_graph(filename):
# 	pass

# def load_features(filename):
# 	pass

def load_tf_interaction(args, ):
	_dir = os.path.join(args.data_directory, "tissue_classification")
	interaction_df = pd.read_csv(os.path.join(_dir, "NIHMS177825-supplement-03-1.csv"), 
	    sep=",", skiprows=1).iloc[1:]
	topology_graph = nx.from_pandas_dataframe(interaction_df, "Gene 1 Symbol", "Gene 2 Symbol")

	features_df = pd.read_csv(os.path.join(_dir, "NIHMS177825-supplement-06-2.csv"), 
	    sep=",", skiprows=1, index_col="Symbol", ).iloc[:,2:]

	# remove nodes with no expression data
	for n in topology_graph.nodes():
	    if n not in features_df.index:
	        topology_graph.remove_node(n)

	# sort features by node order
	features_df = features_df.loc[topology_graph.nodes(),:]

	features = features_df.values
	tfs = features_df.index

	topology_graph = nx.convert_node_labels_to_integers(topology_graph, label_attribute="original_name")
	nx.set_edge_attributes(topology_graph, "weight", 1)

	return topology_graph, features, tfs

def load_labels(filename, label_column="Cell Line", label_of_interest="Mesoderm"):
	label_df = pd.read_csv(filename, index_col=0)
	if label_of_interest is not None:
		labels = np.array(label_df.loc[:,label_column]==label_of_interest, dtype=np.float)
		label_map = pd.Index(["other", label_of_interest])
	else:
		labels, label_map = pd.factorize(label_df[label_column])
	return labels, label_map


def parse_model_filename(args):

	dataset = args.dataset
	directory = "dim={}/seed={}/".format(args.embedding_dim, args.seed)

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

	args.model_path = os.path.join(args.model_path, dataset)
	args.model_path = os.path.join(args.model_path, directory)

	saved_models = sorted([f for f in os.listdir(args.model_path) 
		if re.match(r"^[0-9][0-9][0-9][0-9]*", f)])
	assert len(saved_models) > 0
	return os.path.join(args.model_path, saved_models[-1])

def parse_args():
	parser = argparse.ArgumentParser(description="Density-based clustering in hyperbolic space")

	parser.add_argument("--data-directory", dest="data_directory", type=str, default="/data/",
		help="The directory containing data files (default is '/data/').")

	parser.add_argument("--dataset", dest="dataset", type=str, default="tf_interaction",
		help="The dataset to load. Must be one of [wordnet, cora, citeseer, pubmed,\
		AstroPh, CondMat, GrQc, HepPh, karate]. (Default is karate)")
	
	
	parser.add_argument("-e", dest="max_eps", type=float, default=0.5,
		help="maximum eps.")

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


	# parser.add_argument("--second-order", action="store_true", 
	# 	help="Use this flag to use second order topological similarity information.")
	parser.add_argument("--no-attributes", action="store_true", 
		help="Use this flag to not use attributes.")
	# parser.add_argument("--add-attributes", action="store_true", 
	# 	help="Use this flag to add attribute sim to adj.")
	parser.add_argument("--multiply-attributes", action="store_true", 
		help="Use this flag to multiply attribute sim to adj.")
	parser.add_argument("--jump-prob", dest="jump_prob", type=float, default=0, 
		help="Probability of randomly jumping to a similar node when walking.")


	# parser.add_argument("--distance", dest="distance", action="store_true", 
	# 	help="Use this flag to use hyperbolic distance loss.")
	parser.add_argument("--sigmoid", dest="sigmoid", action="store_true", 
		help="Use this flag to use sigmoid loss.")
	parser.add_argument("--softmax", dest="softmax", action="store_true", 
		help="Use this flag to use softmax loss.")
	parser.add_argument("--euclidean", dest="euclidean", action="store_true", 
		help="Use this flag to use euclidean negative sampling loss.")

	parser.add_argument("--model", dest="model_path", default="models/", 
		help="path to save model after each epoch (default is 'models/)'.")

	parser.add_argument('--only-lcc', action="store_true", help='flag to train on only lcc')

	parser.add_argument('--evaluate-class-prediction', action="store_true", help='flag to evaluate class prediction')
	parser.add_argument('--evaluate-link-prediction', action="store_true", help='flag to evaluate link prediction')

	parser.add_argument('--no-non-edges', action="store_true", help='flag to not add non edges to training graph')
	parser.add_argument('--add-non-edges', action="store_true", help='flag to add non edges to training graph')


	args = parser.parse_args()
	return args

def main():
	args = parse_args()

	dataset = args.dataset
	assert dataset == "tf_interaction"
	if dataset == "karate":
		# topology_graph, features, labels = load_karate(args)
		raise Exception
	# elif dataset in ["cora", "cora_ml", "pubmed", "citeseer"]:
	# 	# topology_graph, features, labels = load_labelled_attributed_network(dataset, args)
	# 	topology_graph, features, labels, label_info = load_g2g_datasets(dataset, args)
	# elif dataset == "ppi":
	# 	topology_graph, features, labels = load_ppi(args)
	elif dataset == "tf_interaction":
		topology_graph, features, tfs = load_tf_interaction(args, )
	else:
		raise Exception

	features = features.T

	label_filename = os.path.join(args.data_directory, "tissue_classification/cell_lines.csv")
	print ("label_filename={}".format(label_filename))
	labels = load_labels(label_filename)

	model_filename = parse_model_filename(args)
	print ("loading model from {}".format(model_filename))

	embedding = load_embedding(model_filename)
	poincare_embedding = hyperboloid_to_poincare_ball(embedding)
	ranks = np.sqrt(np.sum(np.square(poincare_embedding), axis=-1, keepdims=False))
	assert (ranks<1).all()
	assert (ranks.argsort() == embedding[:,-1].argsort()).all()

	# klein_embedding = hyperboloid_to_klein(embedding)
	dists = hyperbolic_distance(embedding, embedding)
	# sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
	# split_train, split_test = next(sss.split(features, labels))

	# data = np.column_stack([features, labels])
	# data_train = data[split_train]
	# data_test = data[split_test]

	best_eps = -1
	best_f1 = 0
	# for eps in [2.3]:
	for eps in np.arange(0.1, args.max_eps, 0.1):
		modules = perform_clustering(dists, eps)
		num_modules = len(set(modules) - {-1})
		print ("discovered {} modules with eps = {}".format(num_modules, eps))

		print ("fraction nodes in modules: {}".format((modules > -1).sum() / float(len(modules))))

		num_connected = 0
		directed_modules = []
		for m in range(num_modules):
			idx = np.where(modules == m)[0]
			module = topology_graph.subgraph(idx)
			print ("module =", m, "number of nodes =", len(module), 
				"number of edges =", len(module.edges()))
			num_connected += nx.is_connected(module)
			directed_modules += convert_module_to_directed_module(module, ranks)
		print ("created {} directed_modules".format(len(directed_modules)))
		# print ("number connected modules = {}".format(num_connected))

		if len(directed_modules) > 0:
			mean_f1_micro = evaluate_modules_on_test_data(features, labels, directed_modules, ranks, feature_names=tfs, 
				n_repeats=25, test_size=0.3)
			print ("f1={}".format(mean_f1_micro, ))
			if mean_f1_micro > best_f1:
				print ("best f1={}".format(mean_f1_micro))
				best_f1 = mean_f1_micro
				best_eps = eps
				# best_forest = forest
		print ()

	modules = perform_clustering(dists, best_eps)
	num_modules = len(set(modules) - {-1})

	directed_modules = []
	for m in range(num_modules):
		idx = np.where(modules == m)[0]
		module = topology_graph.subgraph(idx)
		print ("module =", m, "number of nodes =", len(module), 
			"number of edges =", len(module.edges()))
		num_connected += nx.is_connected(module)

	for m in range(num_modules):
		idx = np.where(modules == m)[0]
		module = topology_graph.subgraph(idx)
		directed_modules += convert_module_to_directed_module(module, ranks)
		print ("module {} contrains {} nodes and {} edges and connected={}".format(m, 
			len(module), len(module.edges()), nx.is_connected(module)))

	data = np.column_stack([features, labels])

	forest, all_oob_samples = grow_forest(data, directed_modules, ranks, feature_names=tfs)

	print ()
	for tree in forest[:1]:
		print (tree)
		print ()

	print ("determining feature_importances")
	feature_importances, feature_pair_importances = determine_feature_importances(forest, all_oob_samples)
	
	for tree in forest:
		root = tree.index

		edges = [(t.parent_index, t.index) for t in tree.postorder() if t.parent_index is not None and t.index is not None]
		edges += [(t.parent_index, -(i+1)) for i, t in enumerate(tree.postorder()) if t.is_leaf]

		inner_nodes = [t.index for t in tree.postorder() if not t.is_leaf] 
		leaves = [-(i+1) for i, t in enumerate(tree.postorder()) if t.is_leaf]
		print (edges)
		print (inner_nodes)
		print (leaves)
		raise SystemExit

		# idx = [t.index for t in tree.postorder()]
		# print (len(idx))
		module = nx.DiGraph(edges)
		nx.set_node_attributes(module, "original_name", {k:v["original_name"] 
			for k, v in topology_graph.nodes(data=True)if k in module.nodes()})
		# nx.set_node_attributes(module, "rank", {n: "{:.3f}".format(ranks[n]) for n in module.nodes()})
		idx = module.nodes()
		print (root)
		print (edges)
		print (len(module))
		# module = topology_graph.subgraph(idx)

		# pos = poincare_embedding[idx]
		# plt.axis("equal")
		# pos = poincare_embedding[:,:2]
		pos = graphviz_layout(module, prog="dot", root=root)
		nx.draw_networkx_nodes(module, pos=pos, node_size=feature_importances[idx]*30000)
		nx.draw_networkx_edges(module, pos=pos)
		nx.draw_networkx_labels(module, pos=pos, 
			labels={n : "{}\n{:.3f}".format(module.node[n]["original_name"],
			feature_importances[n]) for n in module.nodes()})#nx.get_node_attributes(module, "original_name"))
		plt.show()

	
	print ("Best eps was {}, best f1={}".format(best_eps, best_f1))
	plot_disk_embeddings(topology_graph.edges(), poincare_embedding, modules)

if __name__ == "__main__":
	main()