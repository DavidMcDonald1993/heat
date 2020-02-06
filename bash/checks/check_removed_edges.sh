#!/bin/bash

for dataset in cora_ml citeseer ppi pubmed mit
do
    for seed in {0..29}
    do
        training_dir=$(printf "edgelists/${dataset}/seed=%03d/training_edges" ${seed})
        edgelist=${training_dir}/edgelist.tsv

        if [ ! -f $edgelist ]
        then 
            echo no edgelist at $edgelist
        fi
    done
done