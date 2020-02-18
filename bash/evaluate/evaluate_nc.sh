#!/bin/bash

#SBATCH --job-name=HEATNC
#SBATCH --output=HEATNC_%A_%a.out
#SBATCH --error=HEATNC_%A_%a.err
#SBATCH --array=0-3599
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G

datasets=(cora_ml citeseer ppi pubmed mit)
dims=(5 10 25 50)
seeds=({0..29})
alphas=(00 05 10 20 50 100)
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
edgelist=${data_dir}/edgelist.tsv
features=${data_dir}/feats.csv
labels=${data_dir}/labels.csv
embedding_dir=embeddings/${dataset}/${exp}

test_results=$(printf "test_results/${dataset}/${exp}/alpha=${alpha}/dim=%03d/" ${dim})
embedding_dir=$(printf "${embedding_dir}/alpha=${alpha}/seed=%03d/dim=%03d/" ${seed} ${dim})
echo $embedding_dir

args=$(echo --edgelist ${edgelist} --labels ${labels} \
    --dist_fn hyperboloid \
    --embedding ${embedding_dir} --seed ${seed} \
    --test-results-dir ${test_results})
echo $args

module purge
module load bluebear
module load Python/3.6.3-iomkl-2018a
pip install --user numpy pandas networkx scikit-learn scikit-multilearn matplotlib

python evaluate_nc.py ${args}