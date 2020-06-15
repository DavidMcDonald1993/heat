#!/bin/bash

e=5

# nc experiments
for dataset in cora_ml citeseer ppi pubmed mit
do
	for dim in 5
	do	
		for exp in nc_experiment lp_experiment
		do
			for seed in {0..29}
			do

				for alpha in 00 05 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95
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