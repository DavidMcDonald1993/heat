import pandas as pd 
import os
import itertools

from pandas.errors import EmptyDataError

def main():

    e = 5
    datasets = ["cora_ml", "citeseer", "ppi", "pubmed", "mit"]
    dims = (5, 10, 25, 50)
    alphas = (0, .05, .1, .2, .5, .8, 1)
    seeds = range(30)
    exps = ["nc_experiment", "lp_experiment"]

    for dataset, dim, alpha, seed, exp in itertools.product(
        datasets, dims, alphas, seeds, exps
    ):
        embedding_directory = os.path.join(
            "embeddings", dataset, exp, 
            "alpha={:.02f}".format(alpha),
            "seed={:03d}".format(seed),
            "dim={:02d}".format(dim), 

        )

        filename = os.path.join(embedding_directory, 
            "{:05d}_embedding.csv.gz".format(e))

        try:
            pd.read_csv(filename)
        except EmptyDataError:
            print (filename, "is empty removing it")
            os.remove(filename)
        except IOError:
            print (filename, "does not exist")

if __name__ == "__main__":
    main()