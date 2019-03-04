import numpy as np
import networkx as nx

from sklearn.metrics import f1_score, log_loss

class TopologyConstrainedTree(object):
	
	def __init__(self, parent_index, index, g, data, feature_names, depth, max_depth, min_samples_split, min_neighbours):
		self.parent_index = parent_index
		self.index = index
		self.g = g
		self.data = data
		self.feature_names = feature_names
		self.val_data = None
		self.labels = data[:,-1]
		self.depth = depth
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_neighbours = min_neighbours
		
		# print ("creating tree at depth {} with parent_index={}".format(self.depth, self.parent_index))
		# print ("data shape is {}".format(self.data.shape))
		
		if (self.depth == max_depth 
			or data.shape[0] < min_samples_split 
			or len(set(self.labels)) == 1
		   or (self.parent_index is not None and len(g.neighbors(self.parent_index)) < self.min_neighbours)):
			self.is_leaf = True
			# if self.depth == max_depth:
			# 	print ("MAX DEPTH")
			# elif data.shape[0] < min_samples_split :
			# 	print ("TOO FEW SAMPLES")
			# elif len(set(self.labels)) == 1:
			# 	print ("ALL LABELS ARE THE SAME")
			# elif self.parent_index is not None and len(g.neighbors(self.parent_index)) < self.min_neighbours:
			# 	print ("TOO FEW NEIGHBOURS")
			# print ("this node is a leaf")
		else:
			self.is_leaf = False
			# print ("this node is not a leaf")
			self.get_split()
#         print()
	
	# Calculate the Gini index for a split dataset
	def gini_index(self, groups, classes):
		# count all samples at split point
		n_instances = float(sum([len(group) for group in groups]))
		# sum weighted Gini index for each group
		gini = 0.0
		for group in groups:
			size = float(len(group))
			# avoid divide by zero
			if size == 0:
				continue
			score = 0.0
			# score the group based on the score for each class
			cl = group[:,-1]
			for class_val in classes:
				p = (cl==class_val).sum() / size
				score += p * p
			# weight the group score by its relative size
			gini += (1.0 - score) * (size / n_instances)
		return gini
	
	def test_split(self, index, value, ):
		data = self.data
		idx = data[:,index] < value
		left = data[idx]
		right = data[~idx]
		return left, right
	
	# Select the best split point for a dataset
	def get_split(self):
#         print ("get split")
		class_values = set(self.labels)
		b_index, b_value, b_score, b_groups = 999, 999, 999, None
		if self.index is None:
			if self.parent_index is None:
				index_choices = range(data.shape[-1]-1)
			else:
				index_choices = self.g.neighbors(self.parent_index)
		else:
			index_choices = [self.index]
			
		for index in index_choices:
			for row in self.data:
				value = row[index]
				groups = self.test_split(index, value, )
				gini = self.gini_index(groups, class_values)
				if gini < b_score:
					b_index, b_value, b_score, b_groups = index, value, gini, groups
		self.index = b_index
		self.value = b_value
		self.score = b_score
		
		# print ("selected index={} and value={} with gini_score={}".format(self.index, self.value, self.score))

		self.left = TopologyConstrainedTree(self.index, None, self.g, b_groups[0], self.feature_names, self.depth + 1, 
						 self.max_depth, self.min_samples_split, self.min_neighbours)
		self.right = TopologyConstrainedTree(self.index, None, self.g, b_groups[1], self.feature_names, self.depth + 1, 
						  self.max_depth, self.min_samples_split, self.min_neighbours )
		
	def prediction_accuracy(self, y_true, y_pred):
		return f1_score(y_true, y_pred, average="micro")
		# return (y_true != y_pred).sum()
	
	def assign_val_data(self, data):
		if len(data) == 0:
			return
		if len(data.shape) == 1:
			data = np.expand_dims(data, 0)
		self.val_data = data
		
		if not self.is_leaf:
			idx = data[:,self.index] < self.value

			self.left.assign_val_data(data[idx])
			self.right.assign_val_data(data[~idx])
			
	def val_data_on_all_leaves(self):
		if self.is_leaf:
			return self.val_data is not None
		else:
			return self.val_data is not None and self.left.val_data_on_all_leaves() and self.right.val_data_on_all_leaves()
	
	def compute_gain(self, ):
		if self.is_leaf or not self.val_data_on_all_leaves():
			return -np.inf
		labels = self.val_data[:,-1]
		leaf_prediction = np.array([max(set(self.labels), key=list(self.labels).count)] * self.val_data.shape[0])
		tree_prediction = self.predict(self.val_data)
#         gain = self.evaluate_prediction(labels, leaf_prediction) - self.evaluate_prediction(labels, tree_prediction)
		leaf_num_misclassified = sum(labels != leaf_prediction)
		tree_num_misclassified = sum(labels != tree_prediction)
		
		gain = tree_num_misclassified - leaf_num_misclassified
		return gain
	
	def postorder(self):
		if self.is_leaf:
			return [self]
		else:
			return self.left.postorder() + self.right.postorder() + [self]
		
	def make_leaf(self):
		self.is_leaf = True
		self.left = None
		self.right = None
		
	def predict(self, data):
#         assert len(data.shape) > 1
		if len(data.shape) == 1:
			data = np.expand_dims(data, 0)
		if self.is_leaf:
			return np.array([max(set(self.labels), key=list(self.labels).count)] * data.shape[0])
		
		idx = data[:,self.index] < self.value   
	   
		left_pred = self.left.predict(data[idx])
		right_pred = self.right.predict(data[~idx])
		
		pred = np.zeros(data.shape[0])
		pred[idx] = left_pred
		pred[~idx] = right_pred
		return pred
	
	def __len__(self):
		if self.is_leaf:
			return 1
		else:
			return 1 + len(self.left) + len(self.right)
	
	def __str__(self):
		s = "|" * (self.depth - 1) + "-" * int(self.depth > 0)
		if not self.is_leaf:
			s += "index={}, value={}, gini_score={}, data_shape={}\n".format(self.index, 
							 self.value, self.score, self.data.shape)
			s += "|" * (self.depth - 1) + "-" * int(self.depth > 0) + "if {} < {}:\n".format(self.feature_names[self.index], self.value) +\
			"{}\n".format(self.left)+\
			"|" * (self.depth - 1) + "-" * int(self.depth > 0) + "if {} >= {}:\n".format(self.feature_names[self.index], self.value) +\
			"{}".format(self.right)
		else:
			s += "[LEAF] prediction={}".format(max(set(self.labels), key=list(self.labels).count))
		
		return s

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return (
				self.parent_index == other.parent_index 
				and self.index == other.index
				and self.g == other.g
				and (self.data == other.data).all()
				and self.depth == other.depth
				and self.max_depth == other.max_depth
				and self.min_samples_split == other.min_samples_split
				and self.min_neighbours == other.min_neighbours
				)
		return False

	def __ne__(self, other):
		return not self == other

	