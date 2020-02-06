#!/bin/bash

#SBATCH --job-name=evaluateNC
#SBATCH --output=evaluateNC_%A_%a.out
#SBATCH --error=evaluateNC_%A_%a.err
#SBATCH --array=0-599
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G

datasets=({cora_ml,citeseer,ppi,pubmed,mit})
dims=(5 10 25 50)
seeds=({0..29})
exp=nc_experiment

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID % num_seeds ))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}

data_dir=datasets/${dataset}
edgelist=${data_dir}/edgelist.tsv
features=${data_dir}/feats.csv
labels=${data_dir}/labels.csv
embedding_dir=../poincare-embeddings/embeddings/${dataset}

test_results=$(printf "test_results/${dataset}/${exp}/nk/dim=%03d/" ${dim})
embedding_dir=$(printf "${embedding_dir}/dim=%02d/seed=%03d/${exp}" ${dim} ${seed})
echo $embedding_dir

args=$(echo --edgelist ${edgelist} --labels ${labels} \
    --dist_fn poincare \
    --embedding ${embedding_dir} --seed ${seed} \
    --test-results-dir ${test_results})
echo $args

module purge
module load bluebear
module load Python/3.6.3-iomkl-2018a
pip install --user numpy pandas networkx scikit-learn scikit-multilearn

python evaluate_nc.py ${args}