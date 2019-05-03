import numpy as np
import pandas as pd
import argparse
import os
import itertools

from matplotlib import pyplot as plt

# import glob

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Collate results script")

	parser.add_argument("--test-results", dest="test_results_path", default="test_results/", 
		help="path to save test results (default is 'test_results/)'.")

	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	num_seeds = 30
	columns = range(num_seeds)

	datasets = ["cora_ml", "citeseer", "ppi"]

	exps = ["lp_experiment", "reconstruction_experiment", "nc_experiment"]
	dims = ["dim={:03}".format(dim) for dim in (5, 10, 25, 50)]
	alphas = ["nk"] + ["alpha={:.02f}".format(alpha) for alpha in (0, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0)]

	print ("NODE CLASSIFICATION")
	print ()

	columns = ["{:.02f}_micro".format(p) for p in np.arange(0.02, 0.11, 0.01)]

	for dataset in datasets:
		print (dataset)
		print ()

		# show only results for dim=010
		dim = dims[1]

		# show only f1 micro
		f1_micros = pd.DataFrame(0, index=alphas, columns=columns)

		for alpha in alphas:

			results_file = os.path.join(args.test_results_path, 
				dataset,
				"nc_experiment",
				alpha,
				dim,
				"test_results.csv")

			results_df = pd.read_csv(results_file, index_col=0, sep=",")
			assert results_df.shape[0] == 30, alpha

			results_df = results_df[columns]
			f1_micros.loc[alpha] = results_df.mean(axis=0)

		print (f1_micros.to_string())
		print ()

	print ("RECONSTRUCTION")
	print ()

	for dataset in datasets:
		print (dataset)
		print ()

		aurocs = pd.DataFrame(0, index=dims, columns=alphas)

		for dim in dims:

			for alpha in alphas:

				results_file = os.path.join(args.test_results_path, 
					dataset,
					"reconstruction_experiment",
					alpha,
					dim,
					"test_results.csv")

				results_df = pd.read_csv(results_file, index_col=0, sep=",")
				assert results_df.shape[0] == 30, alpha

				results_df = results_df["roc_recon"]

				aurocs.loc[dim, alpha] = results_df.mean(axis=0)

		print (aurocs.to_string())
		print ()

	print ("LINK PREDICTION")
	print ()

	for dataset in datasets:
		print (dataset)
		print ()

		aurocs = pd.DataFrame(0, index=dims, columns=alphas)

		for dim in dims:

			for alpha in alphas:

				results_file = os.path.join(args.test_results_path, 
					dataset,
					"lp_experiment",
					alpha,
					dim,
					"test_results.csv")

				results_df = pd.read_csv(results_file, index_col=0, sep=",")
				assert results_df.shape[0] == 30, alpha

				results_df = results_df["roc_lp"]

				aurocs.loc[dim, alpha] = results_df.mean(axis=0)

		print (aurocs.to_string())
		print ()



if __name__ == "__main__":
	main()