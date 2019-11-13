#!/bin/bash

# heat=/rds/projects/2018/hesz01/heat/main.py

e=5

# nc experiments
for dataset in cora_ml citeseer ppi pubmed mit
do
	for dim in 5 10 25 50
	do	
		for seed in {0..29}
		do

			data_dir=datasets/${dataset}
			edgelist=${data_dir}/edgelist.tsv
			features=${data_dir}/feats.csv
			labels=${data_dir}/labels.csv

			for alpha in 00 05 10 20 50 
			do

				embedding=$(printf "embeddings/${dataset}/nc_experiment/alpha=0.${alpha}/seed=%03d/dim=%03d/%05d_embedding.csv" ${seed} ${dim} ${e})

				if [ ! -f ${embedding} ]
				then
					echo no embedding at ${embedding}
				fi
				
			done

			embedding=$(printf "embeddings/${dataset}/nc_experiment/alpha=1.00/seed=%03d/dim=%03d/%05d_embedding.csv" ${seed} ${dim} ${e})

			if [ ! -f ${embedding} ]
			then
				echo no embedding at ${embedding}
			fi
		done
	done
done


# lp experiments
for dataset in cora_ml citeseer ppi pubmed mit
do
	for dim in 5 10 25 50
	do
		for seed in {0..29}
		do

			for alpha in 00 05 10 20 50 
			do

				embedding=$(printf "embeddings/${dataset}/lp_experiment/alpha=0.${alpha}/seed=%03d/dim=%03d/%05d_embedding.csv" ${seed} ${dim} ${e})

				if [ ! -f ${embedding} ]
				then
					echo no embedding at ${embedding}
				fi
			done

			
			embedding=$(printf "embeddings/${dataset}/lp_experiment/alpha=1.00/seed=%03d/dim=%03d/%05d_embedding.csv" ${seed} ${dim} ${e})

			if [ ! -f ${embedding} ]
			then
				echo no embedding at ${embedding}
			fi
		done
	done
done