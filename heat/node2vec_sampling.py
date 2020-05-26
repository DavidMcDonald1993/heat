'''
Source: https://github.com/aditya-grover/node2vec/blob/master/src/node2vec.py
'''

import numpy as np
import scipy as sp
import networkx as nx
import random

import functools
from multiprocessing.pool import Pool

class Graph():
	def __init__(self, 
		graph, 
		is_directed, 
		p, 
		q, 
		alpha=0, 
		feature_sim=None, 
		seed=0):
		assert not nx.is_directed(graph)
		self.graph = graph
		self.is_directed = is_directed
		self.p = p
		self.q = q
		self.alpha = alpha
		self.feature_sim = feature_sim 
		if self.feature_sim is not None:
			self.feature_sim = self.feature_sim.cumsum(-1)

		np.random.seed(seed)
		random.seed(seed)


	def node2vec_walk(self,  start_node, walk_length,):
		'''
		Simulate a random walk starting from start node.
		'''
		graph = self.graph
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges
		feature_sim = self.feature_sim

		jump = False
		preprocessed_edges = alias_edges is not None

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			# node2vec style random walk 
			cur_nbrs = sorted(graph.neighbors(cur))

			if (feature_sim is not None 
				and self.alpha > 0 
				and not (feature_sim[cur]<1e-15).all() 
				and (np.random.rand() < self.alpha or len(cur_nbrs) == 0)):
				# random jump based on attribute similarity
				next_ = np.searchsorted(feature_sim[cur],
					np.random.rand())
				walk.append(next_)
				jump = True

			elif len(cur_nbrs) > 0:
				if len(walk) == 1 or jump or not preprocessed_edges:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next_ = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next_)
				jump = False
			else:
				break

		return walk

	
	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		graph = self.graph
		walks = []
		nodes = sorted(graph.nodes())
		i = 0

		print ("PERFORMING WALKS")

		# with Pool(processes=2) as p:
		# 	nodes *= num_walks
		# 	walks = p.map(functools.partial(self.node2vec_walk, walk_length=walk_length), nodes)

		for _ in range(num_walks):
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(node, 
					walk_length=walk_length, ))

				if i % 1000 == 0:
					print ("performed walk {:04d}/{}".format(i, num_walks*len(graph)))
				i += 1

				# yield self.node2vec_walk(node, 
				# 	walk_length=walk_length, )

		return walks

	def get_alias_node(self, node):

		graph = self.graph

		unnormalized_probs = [abs(graph[node][nbr]['weight']) for nbr in sorted(graph.neighbors(node))]
		norm_const = sum(unnormalized_probs) + 1e-7
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return node, alias_setup(normalized_probs)

	def get_alias_edge(self, edge):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		src, dst = edge

		graph = self.graph
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(graph.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(abs(graph[dst][dst_nbr]['weight'])/p)
			elif graph.has_edge(dst_nbr, src):
				unnormalized_probs.append(abs(graph[dst][dst_nbr]['weight']))
			else:
				unnormalized_probs.append(abs(graph[dst][dst_nbr]['weight'])/q)
		norm_const = sum(unnormalized_probs) + 1e-7
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return edge, alias_setup(normalized_probs)


	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		print ("preprocessing transition probs")
		graph = self.graph
		is_directed = self.is_directed

		print ("preprocessing nodes")

		# with Pool(processes=None) as p:
		# 	alias_nodes = p.map(self.get_alias_node, graph.nodes())
		alias_nodes = (self.get_alias_node(node) for node in graph.nodes())
		alias_nodes = {node: alias_node for node, alias_node in alias_nodes}

		print ("preprocessed all nodes")
		self.alias_nodes = alias_nodes

		edges = list(graph.edges())
		if not is_directed:
			edges += [(v, u) for u, v in edges]

		if self.p != 1 or self.q != 1:
			print ("preprocessing edges")

			# with Pool(processes=None) as p:
				# alias_edges = p.map(self.get_alias_edge, edges)
			alias_edges = (self.get_alias_edge(edge) for edge in edges)
			alias_edges = {edge: alias_edge for edge, alias_edge in alias_edges}

			print ("preprocessed all edges")
		else:
			print ("p and q are both set to 1, skipping preprocessing edges")
			alias_edges = None
		self.alias_edges = alias_edges


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]

