from __future__ import print_function

import sys
import os
import json
import random
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph

import pickle as pkl

from sklearn.preprocessing import StandardScaler

def load_data(args):

	edgelist_filename = args.edgelist
	features_filename = args.features
	labels_filename = args.labels

	assert not edgelist_filename == "none", "you must specify and edgelist file"

	graph = nx.read_edgelist(edgelist_filename, delimiter="\t", nodetype=int,
		create_using=nx.DiGraph() if args.directed else nx.Graph())

	if not features_filename == "none":

		if features_filename.endswith(".csv"):
			features = pd.read_csv(features_filename, index_col=0, sep=",")
			features = features.reindex(graph.nodes()).values
			features = StandardScaler().fit_transform(features)
		else:
			raise Exception

	else: 
		features = None

	if not labels_filename == "none":

		if labels_filename.endswith(".csv"):
			labels = pd.read_csv(labels_filename, index_col=0, sep=",")
			labels = labels.reindex(graph.nodes()).values.flatten()
			assert len(labels.shape) == 1
		else:
			raise Exception

	else:
		labels = None

	graph = nx.convert_node_labels_to_integers(graph)

	return graph, features, labels

def load_karate(args):

	_dir = os.path.join(args.data_directory, "karate")

	graph = nx.read_edgelist(os.path.join(_dir, "karate.edg"))

	label_df = pd.read_csv(os.path.join(_dir, "mod-based-clusters.txt"), sep=" ", index_col=0, header=None,)
	label_df.index = [str(idx) for idx in label_df.index]
	label_df = label_df.reindex(graph.nodes())

	labels = label_df.iloc[:,0].values

	graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")
	nx.set_edge_attributes(G=graph, name="weight", values=1.)

	features = np.genfromtxt(os.path.join(_dir, "/data/karate/feats.csv"), delimiter=",")

	return graph, features, labels

def load_g2g_datasets(dataset_str, args, scale=True):

	"""Load a graph from a Numpy binary file.
	Parameters
	----------
	file_name : str
		Name of the file to load.
	Returns
	-------
	graph : dict
		Dictionary that contains:
			* 'A' : The adjacency matrix in sparse matrix format
			* 'X' : The attribute matrix in sparse matrix format
			* 'z' : The ground truth class labels
			* Further dictionaries mapping node, class and attribute IDs
	"""
	file_name = os.path.join(args.data_directory,"g2g_datasets")
	file_name = os.path.join(file_name, dataset_str)
	if not file_name.endswith('.npz'):
		file_name += '.npz'
	with np.load(file_name) as loader:
		loader = dict(loader)
		A = sp.sparse.csr_matrix((loader['adj_data'], loader['adj_indices'],
						   loader['adj_indptr']), shape=loader['adj_shape'])

		X = sp.sparse.csr_matrix((loader['attr_data'], loader['attr_indices'],
						   loader['attr_indptr']), shape=loader['attr_shape'])
		z = loader.get('labels')

		graph_dict = {
			'A': A,
			'X': X,
			'z': z
		}

		idx_to_node = loader.get('idx_to_node')
		if idx_to_node:
			idx_to_node = idx_to_node.tolist()
			graph_dict['idx_to_node'] = idx_to_node

		idx_to_attr = loader.get('idx_to_attr')
		if idx_to_attr:
			idx_to_attr = idx_to_attr.tolist()
			graph_dict['idx_to_attr'] = idx_to_attr

		idx_to_class = loader.get('idx_to_class')
		if idx_to_class:
			idx_to_class = idx_to_class.tolist()
			graph_dict['idx_to_class'] = idx_to_class

	if args.directed:
		create_using = nx.DiGraph()
	else:
		create_using = nx.Graph() 

	graph = nx.from_scipy_sparse_matrix(graph_dict["A"], create_using=create_using)
	features = graph_dict["X"].A
	labels = graph_dict["z"]

	if args.only_lcc:
		graph = max(nx.connected_component_subgraphs(graph), key=len)
		features = features[graph.nodes()]
		labels = labels[graph.nodes()]
		graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")
		nx.set_edge_attributes(G=graph, name="weight", values=1.)

	if scale:
		scaler = StandardScaler()
		features = scaler.fit_transform(features)

	return graph, features, labels

def load_labelled_attributed_network(dataset_str, args, scale=True):
	"""Load data."""

	def parse_index_file(filename):
		"""Parse index file."""
		index = []
		for line in open(filename):
			index.append(int(line.strip()))
		return index

	def sample_mask(idx, l):
		"""Create mask."""
		mask = np.zeros(l)
		mask[idx] = 1
		return np.array(mask, dtype=np.bool)

	_dir = os.path.join(args.data_directory, "labelled_attributed_networks")

	names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
	objects = []
	for i in range(len(names)):
		with open(os.path.join(_dir, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
			if sys.version_info > (3, 0):
				objects.append(pkl.load(f, encoding='latin1'))
			else:
				objects.append(pkl.load(f))

	x, y, tx, ty, allx, ally, graph = tuple(objects)
	test_idx_reorder = parse_index_file(os.path.join(_dir, "ind.{}.test.index".format(dataset_str)))
	test_idx_range = np.sort(test_idx_reorder)

	if dataset_str == 'citeseer':
		assert scale
		# Fix citeseer dataset (there are some isolated nodes in the graph)
		# Find isolated nodes, add them as zero-vecs into the right position
		test_idx_range_full = list(range(min(test_idx_reorder), max(test_idx_reorder)+1))
		tx_extended = sp.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
		tx_extended[test_idx_range-min(test_idx_range), :] = tx
		tx = tx_extended
		ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
		ty_extended[test_idx_range-min(test_idx_range), :] = ty
		ty = ty_extended

	features = sp.sparse.vstack((allx, tx)).tolil()
	features[test_idx_reorder, :] = features[test_idx_range, :]
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

	labels = np.vstack((ally, ty))
	labels[test_idx_reorder, :] = labels[test_idx_range, :]
	labels = labels.argmax(axis=-1)

	# test_label_idx = test_idx_range.tolist()
	# train_label_idx = list(range(len(y)))
	# val_label_idx = list(range(len(y), len(y)+500))

	graph = nx.from_scipy_sparse_matrix(adj)
	graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")
	nx.set_edge_attributes(G=graph, name="weight", values=1.)

	if args.only_lcc:
		graph = max(nx.connected_component_subgraphs(graph), key=len)
		features = features[graph.nodes()]
		labels = labels[graph.nodes()]
		graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")
		nx.set_edge_attributes(G=graph, name="weight", values=1.)

	features = features.A
	if scale:
		scaler = StandardScaler()
		features = scaler.fit_transform(features)

	return graph, features, labels

def load_tf_interaction(args, scale=True):
	_dir = os.path.join(args.data_directory, "tissue_classification")
	interaction_df = pd.read_csv(os.path.join(_dir, "NIHMS177825-supplement-03-1.csv"), 
		sep=",", skiprows=1).iloc[1:]
	graph = nx.from_pandas_dataframe(interaction_df, "Gene 1 Symbol", "Gene 2 Symbol")

	features_df = pd.read_csv(os.path.join(_dir, "NIHMS177825-supplement-06-2.csv"), 
		sep=",", skiprows=1, index_col="Symbol", ).iloc[:,2:]

	# remove nodes with no expression data
	for n in graph.nodes():
		if n not in features_df.index:
			graph.remove_node(n)

	# sort features by node order
	features_df = features_df.loc[graph.nodes(),:]

	features = features_df.values

	if scale:
		features = StandardScaler().fit_transform(features)

	graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")
	nx.set_edge_attributes(graph, "weight", 1)
	labels = None
	label_info = None

	return graph, features, labels, label_info


def load_ppi(dataset, args, scale=True,):
	prefix = os.path.join(args.data_directory, "ppi/ppi")
	G_data = json.load(open(prefix + "-G.json"))
	graph = json_graph.node_link_graph(G_data)
	if isinstance(graph.nodes()[0], int):
		conversion = lambda n : int(n)
	else:
		conversion = lambda n : n

	if os.path.exists(prefix + "-feats.npy"):
		features = np.load(prefix + "-feats.npy")
	else:
		print("No features present.. Only identity features will be used.")
		features = None
	id_map = json.load(open(prefix + "-id_map.json"))
	id_map = {conversion(k):int(v) for k,v in id_map.items()}
	class_map = json.load(open(prefix + "-class_map.json"))
	if isinstance(list(class_map.values())[0], list):
		lab_conversion = lambda n : n
	else:
		lab_conversion = lambda n : int(n)

	class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

	## Remove all nodes that do not have val/test annotations
	## (necessary because of networkx weirdness with the Reddit data)
	broken_count = 0
	for node in graph.nodes():
		if not 'val' in graph.node[node] or not 'test' in graph.node[node]:
			graph.remove_node(node)
			broken_count += 1
	print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

	## Make sure the graph has edge train_removed annotations
	## (some datasets might already have this..)
	print("Loaded data.. now preprocessing..")
	for edge in graph.edges():
		if ( graph.node[edge[0]]['val'] or  graph.node[edge[1]]['val'] or
			 graph.node[edge[0]]['test'] or  graph.node[edge[1]]['test']):
			 graph[edge[0]][edge[1]]['train_removed'] = True
		else:
			 graph[edge[0]][edge[1]]['train_removed'] = False

	if scale and not features is None:
		from sklearn.preprocessing import StandardScaler
		train_ids = np.array([id_map[n] 
							  for n in  graph.nodes() 
							  if not graph.node[n]['val'] 
							  and not graph.node[n]['test']])
		# val_ids = np.array([id_map[n] 
		# 					  for n in  graph.nodes() 
		# 					  if graph.node[n]['val'] 
		# 					  and not graph.node[n]['test']])
		# test_ids = np.array([id_map[n] 
		# 					  for n in  graph.nodes() 
		# 					  if not graph.node[n]['val'] 
		# 					  and graph.node[n]['test']])
		train_feats = features[train_ids]
		scaler = StandardScaler()
		scaler.fit(train_feats)
		features = scaler.transform(features)
		
	labels = np.array([class_map[n] for n in graph.nodes()])
	nx.set_edge_attributes(G=graph, name="weight", values=1.)

	assert args.only_lcc
	if args.only_lcc:
		graph = max(nx.connected_component_subgraphs(graph), key=len)
		features = features[graph.nodes()]
		labels = labels[graph.nodes()]
		graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")
		nx.set_edge_attributes(G=graph, name="weight", values=1.)

	print (len(graph), len(graph.edges()), features.shape[1], labels.shape[1])
	raise SystemExit

	return graph, features, labels

def load_wordnet(args):

	'''
	testing link prediciton / reconstruction / lexical entailment
	'''

	graph = nx.read_edgelist(os.path.join(args.data_directory, "wordnet/noun_closure.tsv", ))
	graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")
	nx.set_edge_attributes(graph, name="weight", values=1)

	features = None
	labels = None
	label_info = None
	
	return graph, features, labels, label_info

def load_collaboration_network(args):
	'''
	'''
	dataset_str = args.dataset
	assert dataset_str in ["AstroPh", "CondMat", "GrQc", "HepPh"], "dataset string is not valid"

	graph = nx.read_edgelist(os.path.join(args.data_directory, "collaboration_networks/ca-{}.txt.gz".format(dataset_str)))
	graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")
	nx.set_edge_attributes(graph, name="weight", values=1)

	features = None
	labels = None
	label_info = None

	if args.only_lcc:
		print (len(graph))
		graph = max(nx.connected_component_subgraphs(graph), key=len)
		graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")
		nx.set_edge_attributes(G=graph, name="weight", values=1.)
		print (len(graph))
		raise SystemExit

	return graph, features, labels, label_info

def load_contact(args):

	data_dir = os.path.join("/home/david/Desktop")
	graph = nx.read_edgelist(os.path.join(data_dir, "contact.edgelist"), nodetype=int)

	print (len(graph), nx.number_connected_components(graph))

	features = pd.read_csv(os.path.join(data_dir, "feats.csv"), sep=",", index_col=0)
	print (features.shape)
	features = features.reindex(graph.nodes()).values
	labels = pd.read_csv(os.path.join(data_dir, "labels.csv"), sep=",", index_col=0).reindex(graph.nodes()).values.flatten()

	graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")
	nx.set_edge_attributes(graph, name="weight", values=1)




	label_info = None

	if args.only_lcc:
		graph = max(nx.connected_component_subgraphs(graph), key=len)
		graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_name")
		nx.set_edge_attributes(G=graph, name="weight", values=1.)

	return graph, features, labels, label_info


	