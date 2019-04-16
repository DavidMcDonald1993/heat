from __future__ import print_function

import re
import sys
import os
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.utils.fixes import signature

import keras.backend as K
from keras.callbacks import Callback

from .utils import convert_edgelist_to_dict
from .metrics import evaluate_rank_and_MAP, evaluate_rank_and_MAP_fb, evaluate_classification, evaluate_direction
from .visualise import draw_graph

def gans_to_hyperboloid_np(x):
	t = np.sqrt(1. + np.sum(np.square(x), axis=-1, keepdims=True))
	return np.concatenate([x, t], axis=-1)

def minkowski_dot_np(x, y):
	assert len(x.shape) == 2
	rank = x.shape[1] - 1
	return np.sum(x[:,:rank] * y[:,:rank], axis=-1, keepdims=True) - x[:,rank:] * y[:,rank:]

def minkowski_dot_pairwise(u, v):
	"""
	`u` and `v` are vectors in Minkowski space.
	"""
	# rank = u.shape[-1] - 1
	euc_dp = u[:,:-1].dot(v[:,:-1].T)
	return euc_dp - u[:,-1, None] * v[:,-1]

def hyperbolic_distance_hyperboloid_pairwise(X, Y):
	inner_product = minkowski_dot_pairwise(X, Y)
	inner_product = np.minimum(inner_product, -(1 + 1e-32))
	return np.arccosh(-inner_product)

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def plot_euclidean_embedding(epoch, edges, euclidean_embedding, labels, 
	mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
	mean_rank_lp, map_lp, mean_roc_lp, path):

	# if len(labels.shape) > 1:
		# raise Exception	
		# unique_labels = np.unique(labels, axis=0)
		# labels = np.array([np.where((unique_labels == label).all(axis=-1))[0][0] for label in labels])

	if not isinstance(edges, np.ndarray):
		edges = np.array(edges)

	if labels is not None:
		num_classes = len(set(labels))
		colors = np.random.rand(num_classes, 3)

	print ("saving plot to {}".format(path))

	fig = plt.figure(figsize=[7, 7])
	title = "Epoch={:05d}, Mean_rank_recon={}, MAP_recon={}, Mean_AUC_recon={}".format(epoch, 
		mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction)
	if mean_rank_lp is not None:
		title += "\nMean_rank_lp={}, MAP_lp={}, Mean_AUC_lp={}".format(mean_rank_lp,
			map_lp, mean_roc_lp)
	plt.suptitle(title)
	
	plt.title("Euclidean")
	u_emb = euclidean_embedding[edges[:,0]]
	v_emb = euclidean_embedding[edges[:,1]]
	# for u, v in edges:
	# 	u_emb = poincare_embedding[u]
	# 	v_emb = poincare_embedding[v]
	# 	plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.05, zorder=0)
	plt.plot([u_emb[:,0], v_emb[:,0]], 
		[u_emb[:,1], v_emb[:,1]], c="k", linewidth=0.05, zorder=0)

	if labels is None:
		plt.scatter(euclidean_embedding[:,0], euclidean_embedding[:,1], s=10, c="r", zorder=1)

	else:

		for c in range(num_classes):
			idx = labels == c
			plt.scatter(euclidean_embedding[idx,0], euclidean_embedding[idx,1], s=10, c=colors[c], 
				label=None, zorder=1)
		
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
	plt.savefig(path)
	plt.close()


def plot_disk_embeddings(epoch, edges, poincare_embedding, labels, 
	mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
	mean_rank_lp, map_lp, mean_roc_lp, path,  ):

	if not isinstance(edges, np.ndarray):
		edges = np.array(edges)

	if labels is not None:
		num_classes = len(set(labels))
		colors = np.random.rand(num_classes, 3)

	print ("saving plot to {}".format(path))

	fig = plt.figure(figsize=[14, 7])
	title = "Epoch={:05d}, Mean_rank_recon={}, MAP_recon={}, Mean_AUC_recon={}".format(epoch, 
		mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction)
	if mean_rank_lp is not None:
		title += "\nMean_rank_lp={}, MAP_lp={}, Mean_AUC_lp={}".format(mean_rank_lp,
			map_lp, mean_roc_lp)
	plt.suptitle(title)
	
	ax = fig.add_subplot(111)
	plt.title("Poincare")
	ax.add_artist(plt.Circle([0,0], 1, fill=False))
	u_emb = poincare_embedding[edges[:,0]]
	v_emb = poincare_embedding[edges[:,1]]
	plt.plot([u_emb[:,0], v_emb[:,0]], [u_emb[:,1], v_emb[:,1]], c="k", linewidth=0.05, zorder=0)
	if labels is None:
		plt.scatter(poincare_embedding[:,0], poincare_embedding[:,1], s=10, c="r", zorder=1)
	else:
		plt.scatter(poincare_embedding[:,0], poincare_embedding[:,1], s=10, c=labels, ) 
	plt.xlim([-1,1])
	plt.ylim([-1,1])

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

	plt.savefig(path)
	plt.close()

def plot_precisions_recalls(dists, reconstruction_edges, non_edges, val_edges, val_non_edges, path):

	print ("saving precision recall curves to {}".format(path))

	fig = plt.figure(figsize=[7, 7])
	title = "Embedding quality precision-recall curve"
	plt.suptitle(title)

	reconstruction_edges = np.array(reconstruction_edges)
	non_edges = np.array(non_edges) 

	edge_dists = dists[reconstruction_edges[:,0], reconstruction_edges[:,1]]
	non_edge_dists = dists[non_edges[:,0], non_edges[:,1]]

	targets = np.append(np.ones_like(edge_dists), np.zeros_like(non_edge_dists))
	_dists = np.append(edge_dists, non_edge_dists)

	precisions, recalls, _ = precision_recall_curve(targets, -_dists)
	ap = average_precision_score(targets, -_dists)

	step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters else {})

	plt.step(recalls, precisions, color='r', alpha=0.2,
         where='post', zorder=1)
	plt.fill_between(recalls, precisions, alpha=0.2, color='r', zorder=1, **step_kwargs)


	legend = ["reconstruction AP={}".format(ap)]

	if val_edges is not None:
		val_edges = np.array(val_edges)
		val_edge_dists = dists[val_edges[:,0], val_edges[:,1]]

		val_non_edges = np.array(val_non_edges)
		val_non_edge_dists = dists[val_non_edges[:,0], val_non_edges[:,1]]

		targets = np.append(np.ones_like(val_edge_dists), np.zeros_like(val_non_edge_dists))
		_dists = np.append(val_edge_dists, val_non_edge_dists)

		precisions, recalls, _ = precision_recall_curve(targets, -_dists)
		ap = average_precision_score(targets, -_dists)

		plt.step(recalls, precisions, color='b', alpha=0.2,
         where='post', zorder=0)
		plt.fill_between(recalls, precisions, alpha=0.2, color='b', zorder=0, **step_kwargs)

		legend += ["link prediction AP={}".format(ap)]


	plt.xlabel("recall")
	plt.ylabel("precision")
	plt.legend(legend)
	plt.savefig(path)
	plt.close()


def plot_roc(dists, reconstruction_edges, non_edges, val_edges, val_non_edges, path):

	print ("saving roc plot to {}".format(path))

	fig = plt.figure(figsize=[7, 7])
	title = "Embedding quality ROC curve"
	plt.suptitle(title)

	reconstruction_edges = np.array(reconstruction_edges)
	non_edges = np.array(non_edges) 

	edge_dists = dists[reconstruction_edges[:,0], reconstruction_edges[:,1]]
	non_edge_dists = dists[non_edges[:,0], non_edges[:,1]]

	targets = np.append(np.ones_like(edge_dists), np.zeros_like(non_edge_dists))
	_dists = np.append(edge_dists, non_edge_dists)

	fpr, tpr, _ = roc_curve(targets, -_dists)
	auc = roc_auc_score(targets, -_dists)

	plt.plot(fpr, tpr, c="r")

	legend = ["reconstruction AUC={}".format(auc)]

	if val_edges is not None:
		val_edges = np.array(val_edges)
		val_edge_dists = dists[val_edges[:,0], val_edges[:,1]]

		val_non_edges = np.array(val_non_edges)
		val_non_edge_dists = dists[val_non_edges[:,0], val_non_edges[:,1]]

		targets = np.append(np.ones_like(val_edge_dists), np.zeros_like(val_non_edge_dists))
		_dists = np.append(val_edge_dists, val_non_edge_dists)

		fpr, tpr, _ = roc_curve(targets, -_dists)
		auc = roc_auc_score(targets, -_dists)

		plt.plot(fpr, tpr, c="b")

		legend += ["link prediction AUC={}".format(auc)]


	plt.xlabel("fpr")
	plt.ylabel("tpr")
	plt.legend(legend)
	plt.savefig(path)
	plt.close()

def plot_classification(label_percentages, f1_micros, f1_macros, path):

	print ("saving classification plot to {}".format(path))


	fig = plt.figure(figsize=[7, 7])
	title = "Node classification"
	plt.suptitle(title)
	
	plt.plot(label_percentages, f1_micros, c="r")
	plt.plot(label_percentages, f1_macros, c="b")
	plt.legend(["f1_micros", "f1_macros"])
	plt.xlabel("label_percentages")
	plt.ylabel("f1 score")
	plt.ylim([0,1])
	plt.savefig(path)
	plt.close()

class ValidationLogger(Callback):

	def __init__(self, 
		reconstruction_edges, 
		non_edges, 
		val_edges, 
		val_non_edges, 
		labels, 
		# alpha,
		directed_edges, 
		directed_non_edges,
		epoch, 
		plot_freq, 
		validate,
		args):
		self.reconstruction_edges = reconstruction_edges
		self.non_edges = non_edges
		# self.reconstruction_edge_dict = convert_edgelist_to_dict(reconstruction_edges)
		# self.non_edge_dict = convert_edgelist_to_dict(non_edges)

		self.val_edges = val_edges
		self.val_non_edges = val_non_edges
		# self.val_edge_dict = convert_edgelist_to_dict(val_edges)
		# self.val_non_edge_dict = convert_edgelist_to_dict(val_non_edges)
		self.labels = labels
		self.directed_edges = directed_edges
		self.directed_non_edges = directed_non_edges
		self.epoch = epoch
		self.plot_freq = plot_freq
		# self.alpha = alpha
		self.validate = validate
		self.args = args


	def on_epoch_end(self, batch, logs={}):
	
		self.epoch += 1

		if self.validate:

			hyperboloid_embedding = self.model.layers[-1].get_weights()[-1]
			print (hyperboloid_embedding)

			if self.args.euclidean:
				poincare_embedding = hyperboloid_embedding
				klein_embedding = hyperboloid_embedding
			else:
				poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
				klein_embedding = hyperboloid_to_klein(hyperboloid_embedding)

				ranks = np.sqrt(np.sum(np.square(poincare_embedding), axis=-1, keepdims=False), )
				assert (ranks < 1).all()

			if self.args.euclidean:
				dists = euclidean_distances(hyperboloid_embedding)
			else:
				dists = hyperbolic_distance_hyperboloid_pairwise(hyperboloid_embedding, hyperboloid_embedding)
			
			print (dists)
			print (dists.mean(), dists.max())
			print (minkowski_dot_np(hyperboloid_embedding, hyperboloid_embedding))


			print ("reconstruction")
			(mean_rank_reconstruction, map_reconstruction, 
				mean_roc_reconstruction) = evaluate_rank_and_MAP(dists, 
				self.reconstruction_edges, self.non_edges)

			logs.update({"mean_rank_reconstruction": mean_rank_reconstruction, 
				"map_reconstruction": map_reconstruction,
				"mean_roc_reconstruction": mean_roc_reconstruction})

			if self.args.evaluate_link_prediction:
				print ("link prediction")
				(mean_rank_lp, map_lp, 
				mean_roc_lp) = evaluate_rank_and_MAP(dists, 
				self.val_edges, self.val_non_edges)

				logs.update({"mean_rank_lp": mean_rank_lp, 
					"map_lp": map_lp,
					"mean_roc_lp": mean_roc_lp})

			if self.args.evaluate_class_prediction:
				label_percentages, f1_micros, f1_macros = evaluate_classification(klein_embedding, self.labels, )

				print (f1_micros)

				for label_percentage, f1_micro, f1_macro in zip(label_percentages, f1_micros, f1_macros):
					logs.update({"{}_micro".format(label_percentage): f1_micro})
					logs.update({"{}_macro".format(label_percentage): f1_macro})
				logs.update({"micro_sum" : np.sum(f1_micros)})

			if self.args.directed:
				print ("EVALUATING DIRECTION", len(self.directed_edges), len(self.directed_non_edges))
				directed_f1_micro, directed_f1_macro = evaluate_direction(hyperboloid_embedding, 
					self.directed_edges,)
				logs.update({"directed_f1_micro": directed_f1_micro, 
					"directed_f1_macro": directed_f1_macro, })
					# "directed_ap_score": directed_ap_score,
					# "directed_auc_score": directed_auc_score})


			if self.epoch % self.plot_freq == 0:

				if self.args.embedding_dim == 2:
					plot_path = os.path.join(self.args.plot_path, "epoch_{:05d}_plot.png".format(self.epoch) )
					if not self.args.euclidean:
						draw_graph(self.reconstruction_edges, poincare_embedding, self.labels, plot_path)

				roc_path = os.path.join(self.args.plot_path, "epoch_{:05d}_roc_curve.png".format(self.epoch) )
				plot_roc(dists, self.reconstruction_edges, self.non_edges, 
					self.val_edges, self.val_non_edges, roc_path)

				precision_recall_path = os.path.join(self.args.plot_path, "epoch_{:05d}_precision_recall_curve.png".format(self.epoch) )
				plot_precisions_recalls(dists, self.reconstruction_edges, self.non_edges, 
					self.val_edges, self.val_non_edges, precision_recall_path)

				if self.args.evaluate_class_prediction:
					f1_path = os.path.join(self.args.plot_path, "epoch_{:05d}_class_prediction_f1.png".format(self.epoch))
					plot_classification(label_percentages, f1_micros, f1_macros, f1_path)

		self.remove_old_models()
		self.save_model()

		sys.stdout.flush()

	def remove_old_models(self):
		old_models = sorted([f for f in os.listdir(self.args.model_path) 
			if re.match(r"^[0-9][0-9][0-9][0-9]*", f)])
		# for model in old_models[:-self.args.patience]:
		for model in old_models:
			old_model_path = os.path.join(self.args.model_path, model)
			print ("removing model: {}".format(old_model_path))
			os.remove(old_model_path)

	def save_model(self):
		filename = os.path.join(self.args.model_path, "{:05d}.h5".format(self.epoch))
		print("saving model to {}".format(filename))
		self.model.save_weights(filename)