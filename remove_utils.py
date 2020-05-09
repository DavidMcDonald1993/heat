import random

def write_edgelist_to_file(edgelist, filename, delimiter="\t"):
	with open(filename, "w") as f:
		for u, v in edgelist:
			f.write("{}{}{}\n".format(u, delimiter, v))

def sample_non_edges(nodes, edges, sample_size):
	assert isinstance(edges, set)
	nodes = list(nodes)
	print ("sampling", sample_size, "non edges")
	non_edges = set()
	while len(non_edges) < sample_size:
		non_edges_= {tuple(random.sample(nodes, k=2))
			for _ in range(sample_size - len(non_edges))}
		non_edges_ -= edges 
		non_edges = non_edges.union(non_edges_)
	return list(non_edges)