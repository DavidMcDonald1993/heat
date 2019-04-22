#!/bin/bash

heat=/rds/projects/2018/hesz01/heat/main.py

hrs=1
mem=3

# nc experiments
for dataset in cora_ml citeseer ppi
do
	for dim in 5 10 25 50
	do
		for seed in {0..29}
		do

			data_dir=datasets/${dataset}
			edgelist=${data_dir}/edgelist.tsv
			features=${data_dir}/features.csv
			labels=${data_dir}/labels.csv

			modules=$(echo \
			module purge\; \
			module load bluebear\; \
			module load apps/python3/3.5.2\; \
			)

			cmd_nc=$(echo python evaluate_nc.py --edgelist ${edgelist} --features ${features} --labels ${labels} \
			--seed ${seed})
			cmd_recon=$(echo python evaluate_reconstruction.py --edgelist ${edgelist} --features ${features} --labels ${labels} \
			--seed ${seed})
				

			embedding=$(printf "../poincare-embeddings/embeddings/${dataset}/dim=%02d/seed=%03d/nc_experiment/embedding.csv" ${dim} ${seed})
			test_results=$(printf "test_results/${dataset}/nc_experiment/nk/dim=%03d/" ${dim})

			slurm_options=$(echo \
			--job-name=evaluateEmbeddingsNC-${dataset}-${dim}-${seed}-nk\
			--time=${hrs}:00:00 \
			--mem=${mem}G \
			--output=evaluateEmbeddingsNC-${dataset}-${dim}-${seed}-nk.out \
			--error=evaluateEmbeddingsNC-${dataset}-${dim}-${seed}-nk.err
			)

			if [ -f ${embedding} ]
			then
				echo -e submitting eval_NC '#!/bin/bash\n'${modules}'\n'${cmd_nc}' --embedding '${embedding}' --test-results-dir '${test_results}
				sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd_nc}' --embedding '${embedding}' --test-results-dir '${test_results})
			else
				echo no embedding at ${embedding}
			fi
			
			test_results=$(printf "test_results/${dataset}/reconstruction_experiment/nk/dim=%03d/" ${dim})

			slurm_options=$(echo \
			--job-name=evaluateEmbeddingsRECON-${dataset}-${dim}-${seed}-nk\
			--time=${hrs}:00:00 \
			--mem=${mem}G \
			--output=evaluateEmbeddingsRECON-${dataset}-${dim}-${seed}-nk.out \
			--error=evaluateEmbeddingsRECON-${dataset}-${dim}-${seed}-nk.err
			)

			if [ -f ${embedding} ]
			then
				echo -e submitting eval_RECON '#!/bin/bash\n'${modules}'\n'${cmd_recon}' --embedding '${embedding}' --test-results-dir '${test_results}
				sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd_recon}' --embedding '${embedding}' --test-results-dir '${test_results})
			else 
				echo no embedding at ${embedding}
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

			output=edgelists/${dataset}

			modules=$(echo \
			module purge\; \
			module load bluebear\; \
			module load apps/python3/3.5.2\; \
			)

			cmd_lp=$(echo python evaluate_lp.py --output ${output} --seed ${seed} )
				
			embedding=$(printf "../poincare-embeddings/embeddings/${dataset}/dim=%02d/seed=%03d/lp_experiment/embedding.csv" ${dim} ${seed})
			test_results=$(printf "test_results/${dataset}/lp_experiment/nk/dim=%03d/" ${dim})

			slurm_options=$(echo \
			--job-name=evaluateEmbeddingsLP-${dataset}-${dim}-${seed}-nk\
			--time=${hrs}:00:00 \
			--mem=${mem}G \
			--output=evaluateEmbeddingsLP-${dataset}-${dim}-${seed}-nk.out \
			--error=evaluateEmbeddingsLP-${dataset}-${dim}-${seed}-nk.err
			)

			if [ -f ${embedding} ] 
			then
				echo -e submitting LP '#!/bin/bash\n'${modules}'\n'${cmd_lp}' --embedding '${embedding}' --test-results-dir '${test_results}
				sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd_lp}' --embedding '${embedding}' --test-results-dir '${test_results})
			else 
				echo no embedding at ${embedding}
			fi
		done
	done
done