#!/bin/bash

for dataset in cora_ml citeseer ppi
do
	for seed in {0..29} 
	do
		edgelist=datasets/${dataset}/edgelist.tsv
		features=datasets/${dataset}/feats.csv
		labels=datasets/${dataset}/labels.csv
		output=edgelists/${dataset}/
		python remove_edges.py --edgelist=$edgelist --features=$features --labels=$labels --output=$output --seed $seed
	done
done