#!/bin/bash

#SBATCH --job-name=HEATNC
#SBATCH --output=HEATNC_%A_%a.out
#SBATCH --error=HEATNC_%A_%a.err
#SBATCH --array=0-1199
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

datasets=(cora_ml citeseer ppi pubmed mit)
dims=(5 10 25 50)
seeds=({0..29})
alphas=(00 20)
exp=nc_experiment

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
	alpha=0.$alpha
fi

echo $dataset $dim $seed $alpha

data_dir=datasets/${dataset}
edgelist=${data_dir}/edgelist.tsv.gz 
features=${data_dir}/feats.csv.gz 
labels=${data_dir}/labels.csv.gz 

embedding_dir=$(printf "embeddings/${dataset}/${exp}/alpha=${alpha}/seed=%03d/dim=%03d/" ${seed} ${dim})
echo $embedding_dir

test_results=$(printf "test_results/${dataset}/${exp}/alpha=${alpha}/dim=%03d/" ${dim})


if [ ! -f ${test_results}/${seed}.pkl ]
then
    module purge
    module load bluebear
    module load Python/3.6.3-iomkl-2018a
    pip install --user numpy pandas networkx scikit-learn scikit-multilearn matplotlib

    args=$(echo --edgelist ${edgelist} --labels ${labels} \
        --dist_fn hyperboloid \
        --embedding ${embedding_dir} --seed ${seed} \
        --test-results-dir ${test_results})
    echo $args

    python evaluate_nc.py ${args}
else 
    echo ${test_results}/${seed}.pkl already exists 
fi
