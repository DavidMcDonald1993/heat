import numpy as np
import pandas as pd
import networkx as nx


from heat.data_utils import load_ppi


def main():

	graph, features, labels = load_ppi(scale=False)

	# write edgelist
	nx.write_edgelist(graph, "datasets/ppi/edgelist.tsv", delimiter="\t", data=["weight"])

	# write features
	features = pd.DataFrame(features, index=graph.nodes())
	features.to_csv("datasets/ppi/feats.csv")

	# write labels
	labels = pd.DataFrame(labels, index=graph.nodes())
	labels.to_csv("datasets/ppi/labels.csv")


if __name__ == "__main__":
	main()