#!/bin/bash

#SBATCH --job-name=evaluateLP
#SBATCH --output=evaluateLP_%A_%a.out
#SBATCH --error=evaluateLP_%A_%a.err
#SBATCH --array=0-599
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G

# heat=/rds/projects/2018/hesz01/heat/main.py
# e=5

datasets=({cora_ml,citeseer,ppi,pubmed,mit})
dims=(5 10 25 50)
seeds=({0..29})
exp=lp_experiment

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID % num_seeds ))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}

# data_dir=datasets/${dataset}
# edgelist=${data_dir}/edgelist.tsv
# features=${data_dir}/feats.csv
# labels=${data_dir}/labels.csv
embedding_dir=../poincare_embeddings/embeddings/${dataset}
# walks_dir=walks/${dataset}/lp_experiment
output=edgelists/${dataset}

test_results=$(printf "test_results/${dataset}/${exp}/nk/dim=%03d/" ${dim})
embedding_f=$(printf "${embedding_dir}/dim=%02d/seed=%03d/${exp}/embedding.csv" ${dim} ${seed})
echo $embedding_f

args=$(echo --output ${output} --dist_fn poincare \
    --embedding ${embedding_f} --seed ${seed} \
    --test-results-dir ${test_results})
echo $args

module purge
module load bluebear
module load apps/python3/3.5.2

python evaluate_lp.py ${args}