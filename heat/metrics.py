from __future__ import print_function

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
# from skmultilearn.model_selection import iterative_train_test_split

def evaluate_direction(embedding, directed_edges, ):

	if not isinstance(directed_edges, np.ndarray):
		directed_edges = np.array(directed_edges)

	# if not isinstance(directed_non_edges, np.ndarray):
	# 	directed_non_edges = np.array(directed_non_edges)
	labels = np.ones(len(directed_edges))
	ranks = embedding[:,-1]

	direction_predictions = ranks[directed_edges[:,0]] > ranks[directed_edges[:,1]]
	# non_edge_predictions = ranks[directed_non_edges[:,0]] > ranks[directed_non_edges[:,1]]

	scores = direction_predictions
	# scores = np.append(direction_predictions, non_edge_predictions)

	f1_micro = f1_score(labels, scores, average="micro")
	f1_macro = f1_score(labels, scores, average="macro")
	# ap_score = average_precision_score(labels, scores) # macro by default
	# auc_score = roc_auc_score(labels, scores)

	print ("F1 micro =", f1_micro, "F1 macro =", f1_macro, ) 
		# "AP =", ap_score, "ROC AUC =", auc_score)
	
	return f1_micro, f1_macro#, ap_score, auc_score

def evaluate_rank_and_MAP(dists, edgelist, non_edgelist):
	assert not isinstance(edgelist, dict)

	if not isinstance(edgelist, np.ndarray):
		edgelist = np.array(edgelist)

	if not isinstance(non_edgelist, np.ndarray):
		non_edgelist = np.array(non_edgelist)

	alpha = 0e+3

	edge_dists = dists[edgelist[:,0], edgelist[:,1]] ** 2
	edge_scores = -edge_dists 
	# edge_scores = -(1. + alpha * (poincare_ranks[edgelist[:,1]] - poincare_ranks[edgelist[:,0]])) * edge_dists ** 2

	non_edge_dists = dists[non_edgelist[:,0], non_edgelist[:,1]] ** 2
	non_edge_scores = -non_edge_dists 
	# non_edge_scores = -(1. + alpha * (poincare_ranks[non_edgelist[:,1]] - poincare_ranks[non_edgelist[:,0]])) * non_edge_dists ** 2

	labels = np.append(np.ones_like(edge_dists), np.zeros_like(non_edge_dists))
	# scores = -np.append(edge_dists, non_edge_dists)
	scores = np.append(edge_scores, non_edge_scores, )
	ap_score = average_precision_score(labels, scores) # macro by default
	auc_score = roc_auc_score(labels, scores)


	idx = non_edge_dists.argsort()
	ranks = np.searchsorted(non_edge_dists, edge_dists, sorter=idx) + 1
	ranks = ranks.mean()

	print ("MEAN RANK =", ranks, "AP =", ap_score, 
		"AUROC =", auc_score)

	return ranks, ap_score, auc_score

def evaluate_rank_and_MAP_fb(dists, edge_dict, non_edge_dict):

	assert isinstance(edge_dict, dict)

	ranks = []
	ap_scores = []
	roc_auc_scores = []
	
	for u, neighbours in edge_dict.items():
		# print (neighbours)
		# print (non_edge_dict[u])
		# raise SystemExit
		non_neighbours = non_edge_dict[u]
		_dists = dists[u, neighbours + non_neighbours]
		_labels = np.append(np.ones(len(neighbours)), np.zeros(len(non_neighbours)))
		# _dists = dists[u]
		# _dists[u] = 1e+12
		# _labels = np.zeros(dists.shape[0])
		# _dists_masked = _dists.copy()
		# _ranks = []
		# for v in neighbours:
		# 	_labels[v] = 1
		# 	_dists_masked[v] = np.Inf
		ap_scores.append(average_precision_score(_labels, -_dists))
		roc_auc_scores.append(roc_auc_score(_labels, -_dists))

		neighbour_dists = dists[u, neighbours]
		non_neighbour_dists = dists[u, non_neighbours]
		idx = non_neighbour_dists.argsort()
		_ranks = np.searchsorted(non_neighbour_dists, neighbour_dists, sorter=idx) + 1

		# _ranks = []
		# _dists_masked = _dists.copy()
		# _dists_masked[:len(neighbours)] = np.inf

		# for v in neighbours:
		# 	d = _dists_masked.copy()
		# 	d[v] = _dists[v]
		# 	r = np.argsort(d)
		# 	_ranks.append(np.where(r==v)[0][0] + 1)

		ranks.append(np.mean(_ranks))
	print ("MEAN RANK =", np.mean(ranks), "MEAN AP =", np.mean(ap_scores), 
		"MEAN ROC AUC =", np.mean(roc_auc_scores))
	return np.mean(ranks), np.mean(ap_scores), np.mean(roc_auc_scores)

def evaluate_multi_label_classification(klein_embedding, labels,
	train_idx, test_idx, n_repeats=10):
	pass


def evaluate_classification(klein_embedding, labels,
	label_percentages=np.arange(0.02, 0.11, 0.01), n_repeats=10):

	print ("Evaluating node classification")

	# assert len(labels.shape) == 1

	num_nodes, dim = klein_embedding.shape

	f1_micros = np.zeros((n_repeats, len(label_percentages)))
	f1_macros = np.zeros((n_repeats, len(label_percentages)))
	
	model = LogisticRegressionCV()
	split = StratifiedShuffleSplit

	if len(labels.shape) > 1: # multilabel classification
		model = OneVsRestClassifier(model)
		split = ShuffleSplit

	n = len(klein_embedding)

	for seed in range(n_repeats):
	
		for i, label_percentage in enumerate(label_percentages):

			# if len(labels.shape) > 1:

			# 	num_train = int(n * label_percentage)
			# 	idx = np.random.permutation(n)
			# 	split_train = idx[:num_train]
			# 	split_test = idx[num_train:]

			# 	# X_train, y_train, X_test, y_test = iterative_train_test_split(klein_embedding, labels, test_size=1-label_percentage)
			# 	# model.fit(klein_embedding[klein_embedding[train_idx]], labels[train_idx])
			# 	# predictions = model.predict(klein_embedding[test_idx])
			# 	# f1_micro = f1_score(labels[test_idx], predictions, average="micro")
			# 	# f1_macro = f1_score(labels[test_idx], predictions, average="macro")

			# else:
			sss = split(n_splits=1, test_size=1-label_percentage, random_state=seed)
			split_train, split_test = next(sss.split(klein_embedding, labels))
			model.fit(klein_embedding[split_train], labels[split_train])
			predictions = model.predict(klein_embedding[split_test])
			f1_micro = f1_score(labels[split_test], predictions, average="micro")
			f1_macro = f1_score(labels[split_test], predictions, average="macro")
			f1_micros[seed,i] = f1_micro
			f1_macros[seed,i] = f1_macro
		print ("completed repeat {}".format(seed+1))

	return label_percentages, f1_micros.mean(axis=0), f1_macros.mean(axis=0)

# def evaluate_lexical_entailment(embedding):

# 	def is_a_score(u, v, alpha=1e3):
# 		return -(1 + alpha * (np.linalg.norm(v, axis=-1) - np.linalg.norm(u, axis=-1))) * hyperbolic_distance(u, v)

# 	print ("evaluating lexical entailment")

# 	hyperlex_noun_idx_df = pd.read_csv("../data/wordnet/hyperlex_idx_ranks.txt", index_col=0, sep=" ")

# 	U = np.array(hyperlex_noun_idx_df["WORD1"], dtype=int)
# 	V = np.array(hyperlex_noun_idx_df["WORD2"], dtype=int)

# 	true_is_a_score = np.array(hyperlex_noun_idx_df["AVG_SCORE_0_10"])
# 	predicted_is_a_score = is_a_score(embedding[U], embedding[V])

# 	r, p = spearmanr(true_is_a_score, predicted_is_a_score)

# 	print (r, p)

# 	return r, p

