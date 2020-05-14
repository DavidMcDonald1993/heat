import pandas as pd
import argparse
import os

from scipy.stats import ttest_rel
import itertools

def make_dir(d):
	if not os.path.exists(d):
		print ("making directory", d)
		os.makedirs(d, exist_ok=True)

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Collate results script")

	parser.add_argument("--test-results", 
	dest="test_results_path", default="test_results/", 
		help="path to load test results (default is 'test_results/)'.")

	parser.add_argument("--exp", 
		dest="exp", default="reconstruction",
		choices=["reconstruction", "lp", "nc"],
		help="experiment to evaluate")

	parser.add_argument("--output", 
	dest="output", default="collated_results", 
		help="path to save collated test results (default is 'collated_results)'.")

	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	num_seeds = 30

	datasets = ["cora_ml", "citeseer", "ppi", "pubmed", "mit"]

	exp = "{}_experiment".format(args.exp)
	dims = ["dim={:03}".format(dim) for dim in (5, 10, 25, 50)]
	baseline_algs = ["nk"] + ["aane", "abrw", "tadw", 
		"attrpure", "deepwalk", "sagegcn"] 
	heat_algs = ["alpha={:.02f}".format(alpha) 
			for alpha in (0, 0.05, 0.1, 0.2, 0.5, 1.0)]

	algorithms = baseline_algs + heat_algs

	output_dir = os.path.join(args.output, exp) 
	make_dir(output_dir)


	for dim in dims:

		output_dir_ = os.path.join(output_dir, 
			dim)

		for dataset in datasets:

			mean_df = pd.DataFrame()
			sem_df = pd.DataFrame()

			dfs = dict()# store dfs for ttests

			for algorithm in algorithms:

				results_file = os.path.join(args.test_results_path, 
					dataset,
					exp,
					algorithm,
					dim,
					"test_results.csv")
				print ("reading", results_file)

				results_df = pd.read_csv(results_file, index_col=0, sep=",")
				assert results_df.shape[0] == num_seeds, \
					(dataset, dim, algorithm)

				dfs[algorithm] = results_df

				mean_df = mean_df.append(pd.Series(
					results_df.mean(0), name=algorithm
				))

				sem_df = sem_df.append(pd.Series(
					results_df.sem(0), name=algorithm
				))

			mean_filename = os.path.join(output_dir_,
				"{}_{}_means.csv".format(dataset, dim))
			print ("writing to", mean_filename)
			mean_df.to_csv(mean_filename)

			sem_filename = os.path.join(output_dir_,
				"{}_{}_sems.csv".format(dataset, dim))
			print ("writing to", sem_filename)
			sem_df.to_csv(sem_filename)

			# perform t tests
			for a1, a2 in itertools.product(heat_algs, 
				baseline_algs):

				ttest = ttest_rel(dfs[a1], dfs[a2])
				ttest_df = pd.DataFrame(ttest, 
					columns=dfs[a1].columns,
					index=["t-statistic", "p-value"])

				ttest_df_filename = os.path.join(output_dir_,
					"{}_{}_ttest-{}-{}.csv".format(dataset, dim,
						a1, a2))
				print ("writing ttests for", a1, "and", a2,
					"to", ttest_df_filename)
				ttest_df.to_csv(ttest_df_filename)


if __name__ == "__main__":
	main()