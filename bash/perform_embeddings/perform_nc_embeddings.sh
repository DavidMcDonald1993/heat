#!/bin/bash

#SBATCH --job-name=HEATNC
#SBATCH --output=HEATNC_%A_%a.out
#SBATCH --error=HEATNC_%A_%a.err
#SBATCH --array=0-3599
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G

e=5

datasets=(cora_ml citeseer ppi pubmed mit)
dims=(5 10 25 50)
seeds=({0..29})
alphas=(00 05 10 20 50 100)

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_alphas=${#alphas[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_alphas * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_alphas * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / num_alphas % num_seeds ))
alpha_id=$((SLURM_ARRAY_TASK_ID % (num_alphas) ))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
alpha=${alphas[$alpha_id]}

if [ $alpha -eq 100 ];
then
	alpha=1.00
else
	alpha=0.${alpha}
fi

data_dir=datasets/${dataset}
edgelist=${data_dir}/edgelist.tsv
features=${data_dir}/feats.csv
labels=${data_dir}/labels.csv
dir=$(printf "${dataset}/nc_experiment/alpha=${alpha}/seed=%03d" ${seed})
embedding_dir=$(printf "embeddings/${dir}/dim=%03d" ${dim} )
walks_dir=walks/${dir}

embedding_f=$(printf "${embedding_dir}/%05d_embedding.csv.gz" ${e})

if [ ! -f embedding_f ]
then 
	module purge
	module load bluebear
	module load apps/python3/3.5.2
	module load apps/keras/2.0.8-python-3.5.2

	args=$(echo --edgelist ${edgelist} --features ${features} --labels ${labels} \
	--embedding ${embedding_dir} --walks ${walks_dir} --seed ${seed} --dim ${dim} \
	--alpha ${alpha} -e ${e})

	python main.py ${args}

else
	echo $embedding_f already exists

fi