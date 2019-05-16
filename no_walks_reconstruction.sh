#!/bin/bash

heat=/rds/projects/2018/hesz01/heat/main.py

days=3
hrs=00
mem=10G

e=1500

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
			--embedding ${embedding_dir} --no-walks --seed ${seed} --dim ${dim} -e ${e} --nneg 50)

			slurm_options=$(echo \
			--job-name=performEmbeddingsNC-${dataset}-${dim}-${seed}-no-walks\
			--time=${days}-${hrs}:00:00 \
			--mem=${mem} \
			--output=performEmbeddingsNC-${dataset}-${dim}-${seed}-no-walks.out \
			--error=performEmbeddingsNC-${dataset}-${dim}-${seed}-no-walks.err
			)

			if [ ! -f $(printf "${embedding_dir}/no_walks/seed=%03d/dim=%03d/%05d_embedding.csv" ${seed} ${dim} ${e}) ]
			then
				echo -e submitting NC '#!/bin/bash\n'${modules}'\n'${cmd}
				sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd})
			fi
		done
	done
done