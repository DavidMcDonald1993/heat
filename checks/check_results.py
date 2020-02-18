import pandas as pd 
import os
import itertools

from pandas.errors import EmptyDataError

def main():

    e = 5
    datasets = ["cora_ml", "citeseer", "ppi", "pubmed", "mit"]
    dims = (5, 10, 25, 50)
    alphas = (0, .05, .1, .2, .5, 1)
    exps = ["nc_experiment", "lp_experiment",
     "reconstruction_experiment"]

    for dataset, dim, alpha, exp in itertools.product(
        datasets, dims, alphas, exps
    ):
        embedding_directory = os.path.join(
            "test_results", dataset, exp, 
            "alpha={:.02f}".format(alpha),
            "dim={:03d}".format(dim), 

        )

        filename = os.path.join(embedding_directory, 
            "test_results.csv")

        try:
            results = pd.read_csv(filename, index_col=0)

            if not results.shape[0] == 30:
                print (filename, "has missing results")
                print (set(range(30) - set(results.index)))

        except EmptyDataError:
            print (filename, "is empty removing it")
            os.remove(filename)
        except IOError:
            print (filename, "does not exist")

if __name__ == "__main__":
    main()