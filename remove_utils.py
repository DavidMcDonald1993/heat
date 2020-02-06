import random

def write_edgelist_to_file(edgelist, filename):
	# g = nx.DiGraph(edgelist)
	# nx.write_edgelist(g, filename, delimiter="\t")
	with open(filename, "w") as f:
		for u, v in edgelist:
			f.write("{}\t{}\n".format(u, v))

def sample_non_edges(nodes, edges, sample_size):
	assert isinstance(edges, set)
	nodes = list(nodes)
	# edges = set(edges)
	print ("sampling", sample_size, "non edges")
	non_edges = set()
	while len(non_edges) < sample_size:
		non_edges_= {tuple(random.sample(nodes, k=2))
			for _ in range(sample_size - len(non_edges))}
		non_edges_ -= edges 
		non_edges = non_edges.union(non_edges_)
		# if edge not in edges + non_edges:
		# 	non_edges.append(edge)
	return list(non_edges)