import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (roc_auc_score, 
        f1_score, 
        precision_score, 
        recall_score)
from sklearn.preprocessing import LabelBinarizer

from skmultilearn.model_selection import IterativeStratification

from heat.utils import (load_data, hyperboloid_to_klein, 
	poincare_ball_to_hyperboloid, hyperboloid_to_poincare_ball,
	hyperboloid_to_poincare_ball, poincare_ball_to_klein)
from evaluation_utils import check_complete, touch, threadsafe_save_test_results, load_embedding

import functools
import fcntl
import argparse

from collections import Counter

def compute_measures( 
    labels, 
    probs,
	threshold=.5,
	average="micro"):

	if len(labels.shape) == 1:
		labels = LabelBinarizer().fit_transform(labels)

	roc = roc_auc_score(labels, probs, average=average)
	pred = probs > threshold
	f1 = f1_score(labels, pred, average=average)
	precision = precision_score(labels, pred, average=average)
	recall = recall_score(labels, pred, average=average )

	return roc, f1, precision, recall

def evaluate_kfold_label_classification(
	embedding, 
	labels, 
	k=10):
	assert len(labels.shape) == 2
	
	# model = LogisticRegressionCV(
	# 	max_iter=1000, 
	# 	n_jobs=-1)
	model = SVC(probability=True)

	if labels.shape[1] == 1:
		print ("single label clasification")
		labels = labels.flatten()
		sss = StratifiedKFold(n_splits=k, 
			shuffle=True, 
			random_state=0)

	else:
		print ("multi-label classification")
		sss = IterativeStratification(n_splits=k, 
			order=1)
		model = OneVsRestClassifier(model, )
			
	k_fold_rocs = np.zeros(k)
	k_fold_f1s = np.zeros(k)
	k_fold_precisions = np.zeros(k)
	k_fold_recalls = np.zeros(k)

	for i, (split_train, split_test) in enumerate(\
		sss.split(embedding, labels, )):
		print ("Fold", i+1, "fitting model")
		model.fit(embedding[split_train], labels[split_train])	
		probs = model.predict_proba(embedding[split_test])

		(k_fold_rocs[i], 
			k_fold_f1s[i], 
			k_fold_precisions[i], 
			k_fold_recalls[i]) = compute_measures(
				labels[split_test],
				probs,)

		print ("Completed {}/{} folds".format(i+1, k))

	return (np.mean(k_fold_rocs), np.mean(k_fold_f1s),
		np.mean(k_fold_precisions), np.mean(k_fold_recalls))

def evaluate_node_classification(
	embedding, 
	labels,
	label_percentages=np.arange(0.02, 0.11, 0.01), 
	n_repeats=10):

	print ("Evaluating node classification")

	f1_micros = np.zeros((n_repeats, len(label_percentages)))
	f1_macros = np.zeros((n_repeats, len(label_percentages)))
	
	# model = LogisticRegressionCV(max_iter=1000,
	# 	n_jobs=-1)
	model = SVC(probability=True)

	if labels.shape[1] == 1:
		print ("single label clasification")
		labels = labels.flatten()

		split = StratifiedShuffleSplit
		for seed in range(n_repeats):
		
			for i, label_percentage in enumerate(label_percentages):
				print ("processing label percentage", i, 
					":", "{:.02f}".format(label_percentage))
				sss = split(n_splits=1, 
					test_size=1-label_percentage, 
					random_state=seed)
				split_train, split_test = next(sss.split(embedding, 
					labels))
				
				model.fit(embedding[split_train], labels[split_train])
				predictions = model.predict(embedding[split_test])

				f1_micro = f1_score(labels[split_test], predictions, 
					average="micro")
				f1_macro = f1_score(labels[split_test], predictions, 
					average="macro")
				print ("{:.02f}".format(label_percentage), 
					f1_micro, f1_macro)

				f1_micros[seed, i] = f1_micro
				f1_macros[seed, i] = f1_macro
			print ("completed repeat {}".format(seed+1))

	else: # multilabel classification
		print ("multilabel classification")
		model = OneVsRestClassifier(model)
		split = IterativeStratification

		for seed in range(n_repeats):
		
			for i, label_percentage in enumerate(label_percentages):
				print ("processing label percentage", i, 
					":", "{:.02f}".format(label_percentage))
				sss = split(n_splits=2, order=1, #random_state=seed,
					sample_distribution_per_fold=[1.0-label_percentage, label_percentage])
				split_train, split_test = next(sss.split(embedding, labels))
				model.fit(embedding[split_train], labels[split_train])
				predictions = model.predict(embedding[split_test])
				f1_micro = f1_score(labels[split_test], predictions, 
					average="micro")
				f1_macro = f1_score(labels[split_test], predictions, 
					average="macro")
				f1_micros[seed,i] = f1_micro
				f1_macros[seed,i] = f1_macro
			print ("completed repeat {}".format(seed+1))

	return label_percentages, f1_micros.mean(axis=0), f1_macros.mean(axis=0)

def parse_args():

	parser = argparse.ArgumentParser(description='Load Embeddings and evaluate node classification')
	
	parser.add_argument("--edgelist", dest="edgelist", type=str, 
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, 
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, 
		help="path to labels")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument("--embedding", dest="embedding_directory",  
		help="path of embedding to load.")

	parser.add_argument("--test-results-dir", dest="test_results_dir",  
		help="path to save results.")

	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--dist_fn", dest="dist_fn", type=str,
	choices=["poincare", "hyperboloid", "euclidean"])

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

	_, _, node_labels = load_data(args)
	print ("Loaded dataset")

	embedding = load_embedding(args.dist_fn, args.embedding_directory)

	min_count = 10
	if node_labels.shape[1] == 1: # remove any node belonging to an under-represented class
		label_counts = Counter(node_labels.flatten())
		mask = np.array([label_counts[l] >= min_count
			for l in node_labels.flatten()])
		embedding = embedding[mask]
		node_labels = node_labels[mask]
	else:
		assert node_labels.shape[1] > 1
		idx = node_labels.sum(0) >= min_count
		node_labels = node_labels[:, idx]
		idx = node_labels.any(-1)
		embedding = embedding[idx]
		node_labels = node_labels[idx]

	if args.dist_fn == "hyperboloid":
		print ("loaded a hyperboloid embedding")
		# print ("projecting from hyperboloid to klein")
		# embedding = hyperboloid_to_klein(embedding)
		print ("projecting from hyperboloid to poincare")
		embedding = hyperboloid_to_poincare_ball(embedding)
		print ("projecting from poincare to klein")
		embedding = poincare_ball_to_klein(embedding)

	elif args.dist_fn == "poincare":
		print ("loaded a poincare embedding")
		# print ("projecting from poincare to klein")
		# embedding = poincare_ball_to_hyperboloid(embedding)
		# embedding = hyperboloid_to_klein(embedding)
		print ("projecting from poincare to klein")
		embedding = poincare_ball_to_klein(embedding)

	test_results = {}
	
	label_percentages, f1_micros, f1_macros = \
		evaluate_node_classification(embedding, node_labels)

	for label_percentage, f1_micro, f1_macro in zip(label_percentages, f1_micros, f1_macros):
		print ("{:.2f}".format(label_percentage), 
			"micro = {:.2f}".format(f1_micro), 
			"macro = {:.2f}".format(f1_macro) )
		test_results.update({"{:.2f}_micro".format(label_percentage): f1_micro})
		test_results.update({"{:.2f}_macro".format(label_percentage): f1_macro})

	k = 10
	k_fold_roc, k_fold_f1, k_fold_precision, k_fold_recall = \
		evaluate_kfold_label_classification(embedding, node_labels, k=k)

	test_results.update({
		"{}-fold-roc".format(k): k_fold_roc, 
		"{}-fold-f1".format(k): k_fold_f1,
		"{}-fold-precision".format(k): k_fold_precision,
		"{}-fold-recall".format(k): k_fold_recall,
		})

	print ("saving test results to {}".format(test_results_filename))
	threadsafe_save_test_results(test_results_lock_filename, test_results_filename, args.seed, data=test_results )
	
if __name__ == "__main__":
	main()