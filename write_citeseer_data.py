import numpy as np
import pandas as pd
import networkx as nx


from heat.data_utils import load_g2g_datasets


def main():

	graph, features, labels = load_g2g_datasets("citeseer", scale=False)

	# write edgelist
	nx.write_edgelist(graph, "datasets/citeseer/edgelist.tsv", delimiter="\t", data=["weight"])

	# write features
	features = pd.DataFrame(features, index=graph.nodes())
	features.to_csv("datasets/citeseer/feats.csv")

	# write labels
	labels = pd.DataFrame(labels, index=graph.nodes())
	labels.to_csv("datasets/citeseer/labels.csv")


if __name__ == "__main__":
	main()