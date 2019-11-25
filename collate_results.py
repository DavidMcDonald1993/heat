import pandas as pd
import argparse
import os

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
	algorithms = ["nk"] + ["aane", "abrw", "tadw", 
		"attrpure", "deepwalk", "sagegcn"] + \
		["alpha={:.02f}".format(alpha) 
			for alpha in (0, 0.05, 0.1, 0.2, 0.5, 1.0)]

	output_dir = os.path.join(args.output, exp) 
	if not os.path.exists(output_dir):
		print ("making directory", output_dir)
		os.makedirs(output_dir, exist_ok=True)

	for dataset in datasets:

		for dim in dims:
	
			collated_df = pd.DataFrame()

			for algorithm in algorithms:

				results_file = os.path.join(args.test_results_path, 
					dataset,
					exp,
					algorithm,
					dim,
					"test_results.csv")
				print ("reading", results_file)

				results_df = pd.read_csv(results_file, index_col=0, sep=",")
				assert results_df.shape[0] == num_seeds, (dataset, 
					dim, algorithm)

				collated_df = collated_df.append(pd.Series(
					results_df.mean(0), name=algorithm
				))

			output_filename = os.path.join(output_dir,
				"{}_{}.csv".format(dataset, dim))
			print ("writing to", output_filename)
			collated_df.to_csv(output_filename)


if __name__ == "__main__":
	main()