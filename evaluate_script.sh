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
			walks_dir=walks/${dataset}/nc_experiment

			modules=$(echo \
			module purge\; \
			module load bluebear\; \
			module load apps/python3/3.5.2\; \
			module load apps/keras/2.0.8-python-3.5.2
			)

			cmd_nc=$(echo python evaluate_nc.py --edgelist ${edgelist} --features ${features} --labels ${labels} \
			--seed ${seed})
			cmd_recon=$(echo python evaluate_reconstruction.py --edgelist ${edgelist} --features ${features} --labels ${labels} \
			--seed ${seed})
				
			for alpha in 00 05 10 20 50 80
			do

				embedding=$(printf "embeddings/${dataset}/nc_experiment/alpha=0.${alpha}/seed=%03d/dim=%03d/embedding.csv" ${seed} ${dim})
				test_results=$(printf "test_results/${dataset}/nc_experiment/alpha=0.${alpha}/dim=%03d/" ${dim})

				slurm_options=$(echo \
				--job-name=evaluateEmbeddingsNC-${dataset}-${dim}-${seed}-0.${alpha}\
				--time=${hrs}:00:00 \
				--mem=${mem}G \
				--output=evaluateEmbeddingsNC-${dataset}-${dim}-${seed}-0.${alpha}.out \
				--error=evaluateEmbeddingsNC-${dataset}-${dim}-${seed}-0.${alpha}.err
				)

				echo -e submitting eval_NC '#!/bin/bash\n'${modules}'\n'${cmd_nc}' --embedding '${embedding}' --test-results-dir '${test_results}
				sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd_nc}' --embedding '${embedding}' --test-results-dir '${test_results})

				test_results=$(printf "test_results/${dataset}/reconstruction_experiment/alpha=0.${alpha}/dim=%03d/" ${dim})

				slurm_options=$(echo \
				--job-name=evaluateEmbeddingsRECON-${dataset}-${dim}-${seed}-0.${alpha}\
				--time=${hrs}:00:00 \
				--mem=${mem}G \
				--output=evaluateEmbeddingsRECON-${dataset}-${dim}-${seed}-0.${alpha}.out \
				--error=evaluateEmbeddingsRECON-${dataset}-${dim}-${seed}-0.${alpha}.err
				)

				echo -e submitting eval_RECON '#!/bin/bash\n'${modules}'\n'${cmd_recon}' --embedding '${embedding}' --test-results-dir '${test_results}
				sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd_recon}' --embedding '${embedding}' --test-results-dir '${test_results})


			done

			embedding=$(printf "embeddings/${dataset}/nc_experiment/alpha=1.0/seed=%03d/dim=%03d/embedding.csv" ${seed} ${dim})
			test_results=$(printf "test_results/${dataset}/nc_experiment/alpha=1.0/dim=%03d/" ${dim})

			slurm_options=$(echo \
			--job-name=evaluateEmbeddingsNC-${dataset}-${dim}-${seed}-1.0\
			--time=${hrs}:00:00 \
			--mem=${mem}G \
			--output=evaluateEmbeddingsNC-${dataset}-${dim}-${seed}-1.0.out \
			--error=evaluateEmbeddingsNC-${dataset}-${dim}-${seed}-1.0.err
			)

			echo -e submitting eval_NC '#!/bin/bash\n'${modules}'\n'${cmd_nc}' --embedding '${embedding}' --test-results-dir '${test_results}
			sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd_nc}' --embedding '${embedding}' --test-results-dir '${test_results})

			test_results=$(printf "test_results/${dataset}/reconstruction_experiment/alpha=1.0/dim=%03d/" ${dim})

			slurm_options=$(echo \
			--job-name=evaluateEmbeddingsRECON-${dataset}-${dim}-${seed}-1.0\
			--time=${hrs}:00:00 \
			--mem=${mem}G \
			--output=evaluateEmbeddingsRECON-${dataset}-${dim}-${seed}-1.0.out \
			--error=evaluateEmbeddingsRECON-${dataset}-${dim}-${seed}-1.0.err
			)

			echo -e submitting eval_RECON '#!/bin/bash\n'${modules}'\n'${cmd_recon}' --embedding '${embedding}' --test-results-dir '${test_results}
			sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd_recon}' --embedding '${embedding}' --test-results-dir '${test_results})

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
			module load apps/keras/2.0.8-python-3.5.2
			)

			cmd_lp=$(echo python evaluate_lp.py --output ${output} --seed ${seed} )
				
			for alpha in 00 05 10 20 50 80
			do

				embedding=$(printf "embeddings/${dataset}/lp_experiment/alpha=0.${alpha}/seed=%03d/dim=%03d/embedding.csv" ${seed} ${dim})
				test_results=$(printf "test_results/${dataset}/lp_experiment/alpha=0.${alpha}/dim=%03d/" ${dim})

				slurm_options=$(echo \
				--job-name=evaluateEmbeddingsLP-${dataset}-${dim}-${seed}-0.${alpha}\
				--time=${hrs}:00:00 \
				--mem=${mem}G \
				--output=evaluateEmbeddingsLP-${dataset}-${dim}-${seed}-0.${alpha}.out \
				--error=evaluateEmbeddingsLP-${dataset}-${dim}-${seed}-0.${alpha}.err
				)

				echo -e submitting LP '#!/bin/bash\n'${modules}'\n'${cmd_lp}' --embedding '${embedding}' --test-results-dir '${test_results}
				sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd_lp}' --embedding '${embedding}' --test-results-dir '${test_results})

			done

			
			embedding=$(printf "embeddings/${dataset}/lp_experiment/alpha=1.0/seed=%03d/dim=%03d/embedding.csv" ${seed} ${dim})
			test_results=$(printf "test_results/${dataset}/lp_experiment/alpha=1.0/dim=%03d/" ${dim})

			slurm_options=$(echo \
			--job-name=evaluateEmbeddingsLP-${dataset}-${dim}-${seed}-1.0\
			--time=${hrs}:00:00 \
			--mem=${mem}G \
			--output=evaluateEmbeddingsLP-${dataset}-${dim}-${seed}-1.0.out \
			--error=evaluateEmbeddingsLP-${dataset}-${dim}-${seed}-1.0.err
			)

			echo -e submitting LP '#!/bin/bash\n'${modules}'\n'${cmd_lp}' --embedding '${embedding}' --test-results-dir '${test_results}
			sbatch ${slurm_options} <(echo -e '#!/bin/bash\n'${modules}'\n'${cmd_lp}' --embedding '${embedding}' --test-results-dir '${test_results})

		done
	done
done