#!/bin/bash

heat=/rds/projects/2018/hesz01/heat/main.py

days=3
hrs=00
mem=10G

# nc experiments
for dataset in cora_ml citeseer ppi
do
	for dim in 5 10 25 50
	do
		for seed in {0..29}
		do

			data_dir=datasets/${dataset}
			edgelist=${data_dir}/edgelist.tsv
			features=${data_dir}/feats.csv
			labels=${data_dir}/labels.csv
			embedding_dir=embeddings/${dataset}/nc_experiment
			walks_dir=walks/${dataset}/nc_experiment

			modules=$(echo \
			module purge\; \
			module load bluebear\; \
			module load apps/python3/3.5.2\; \
			module load apps/keras/2.0.8-python-3.5.2
			)

			cmd=$(echo python ${heat} --edgelist ${edgelist} --features ${features} --labels ${labels} \
			--embedding ${embedding_dir} --walks ${walks_dir} --seed ${seed} --dim ${dim} -e 5)
				
			for alpha in 00 05 10 20 50 80
			do

				slurm_options=$(echo \
				--job-name=performEmbeddingsNC-${dataset}-${dim}-${seed}-0.${alpha}\
				--time=${days}-${hrs}:00:00 \
				--mem=${mem} \
				--output=performEmbeddingsNC-${dataset}-${dim}-${seed}-0.${alpha}.out \
				--error=performEmbeddingsNC-${dataset}-${dim}-${seed}-0.${alpha}.err
				)

				if [ ! -f $(printf "${embedding_dir}/alpha=0.${alpha}/seed=%03d/dim=%03d/embedding.csv" ${seed} ${dim}) ]
				then
					echo -e submitting NC '#!/bin/bash\n'${modules}'\n'${cmd}' --alpha 0.'${alpha}
					sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd}' --alpha 0.'${alpha})
				fi

			done

			slurm_options=$(echo \
			--job-name=performEmbeddingsNC-${dataset}-${dim}-${seed}-1.0\
			--time=${days}-${hrs}:00:00 \
			--mem=${mem} \
			--output=performEmbeddingsNC-${dataset}-${dim}-${seed}-1.0.out \
			--error=performEmbeddingsNC-${dataset}-${dim}-${seed}-1.0.err
			)

			if [ ! -f $(printf "${embedding_dir}/alpha=1.00/seed=%03d/dim=%03d/embedding.csv" ${seed} ${dim}) ]
			then
				echo -e submitting NC '#!/bin/bash\n'${modules}'\n'${cmd}' --alpha 1.0'
				sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd}' --alpha 1.0')
			fi
		done
	done
done

# lp experiments
for dataset in cora_ml citeseer ppi
do
	for dim in 5 10 25 50
	do
		for seed in {0..29}
		do

			data_dir=datasets/${dataset}
			training_dir=$(printf "edgelists/${dataset}/seed=%03d/training_edges" ${seed})
			edgelist=${training_dir}/edgelist.tsv
			features=${data_dir}/feats.csv
			labels=${data_dir}/labels.csv
			embedding_dir=embeddings/${dataset}/lp_experiment
			walks_dir=walks/${dataset}/lp_experiment

			modules=$(echo \
			module purge\; \
			module load bluebear\; \
			module load apps/python3/3.5.2\; \
			module load apps/keras/2.0.8-python-3.5.2
			)

			cmd=$(echo python ${heat} --edgelist ${edgelist} --features ${features} --labels ${labels} \
			--embedding ${embedding_dir} --walks ${walks_dir} --seed ${seed} --dim ${dim} -e 5)
				
			for alpha in 00 05 10 20 50 80
			do


				slurm_options=$(echo \
				--job-name=performEmbeddingsLP-${dataset}-${dim}-${seed}-0.${alpha}\
				--time=${days}-${hrs}:00:00 \
				--mem=${mem} \
				--output=performEmbeddingsLP-${dataset}-${dim}-${seed}-0.${alpha}.out \
				--error=performEmbeddingsLP-${dataset}-${dim}-${seed}-0.${alpha}.err
				)

				if [ ! -f $(printf "${embedding_dir}/alpha=0.${alpha}/seed=%03d/dim=%03d/embedding.csv" ${seed} ${dim}) ]
				then
					echo -e submitting LP '#!/bin/bash\n'${modules}'\n'${cmd}' --alpha 0.'${alpha}
					sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd}' --alpha 0.'${alpha})
				fi

			done

			slurm_options=$(echo \
			--job-name=performEmbeddingsLP-${dataset}-${dim}-${seed}-1.0\
			--time=${days}-${hrs}:00:00 \
			--mem=${mem} \
			--output=performEmbeddingsLP-${dataset}-${dim}-${seed}-1.0.out \
			--error=performEmbeddingsLP-${dataset}-${dim}-${seed}-1.0.err
			)

			if [ ! -f $(printf "${embedding_dir}/alpha=1.00/seed=%03d/dim=%03d/embedding.csv" ${seed} ${dim}) ]
			then
				echo -e submitting LP '#!/bin/bash\n'${modules}'\n'${cmd}' --alpha 1.0'
				sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd}' --alpha 1.0')
			fi
		done
	done
done