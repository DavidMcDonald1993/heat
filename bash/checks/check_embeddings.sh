#!/bin/bash

e=5

# nc experiments
for dataset in cora_ml citeseer ppi pubmed mit
do
	for dim in 5 10 25 50
	do	
		for exp in nc_experiment lp_experiment
		do
			for seed in {0..29}
			do

				for alpha in 00 05 10 20 50 
				do

					embedding=$(printf \
					"embeddings/${dataset}/${exp}/alpha=0.${alpha}/seed=%03d/dim=%03d/%05d_embedding.csv" ${seed} ${dim} ${e})

					if [ -f ${embedding}.gz ]
					then
						continue
					elif [ -f ${embedding} ]
					then 
						gzip $embedding 
					else
						echo no embedding at ${embedding}
					fi
					
				done

				embedding=$(printf "embeddings/${dataset}/${exp}/alpha=1.00/seed=%03d/dim=%03d/%05d_embedding.csv" ${seed} ${dim} ${e})

				if [ -f ${embedding}.gz ]
					then
						continue
					elif [ -f ${embedding} ]
					then 
						gzip $embedding 
					else
						echo no embedding at ${embedding}
				fi
			done
		done
	done
done