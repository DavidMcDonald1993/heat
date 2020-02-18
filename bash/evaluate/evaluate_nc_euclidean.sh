#!/bin/bash

#SBATCH --job-name=EUCLIDNC
#SBATCH --output=EUCLIDNC_%A_%a.out
#SBATCH --error=EUCLIDNC_%A_%a.err
#SBATCH --array=0-3599
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G

datasets=(cora_ml citeseer ppi pubmed mit)
dims=(5 10 25 50)
seeds=({0..29})
methods=(abrw attrpure deepwalk tadw aane sagegcn)
exp=nc_experiment

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_methods=${#methods[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_methods * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_methods * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / num_methods % num_seeds))
method_id=$((SLURM_ARRAY_TASK_ID % num_methods))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
method=${methods[$method_id]}

echo $dataset $dim $seed $method


data_dir=datasets/${dataset}
edgelist=${data_dir}/edgelist.tsv
features=${data_dir}/feats.csv
labels=${data_dir}/labels.csv
embedding_dir=$(echo ../OpenANE/embeddings/${dataset}/nc_experiment/${dim}/${method}/${seed})

test_results=$(printf "test_results/${dataset}/${exp}/${method}/dim=%03d/" ${dim})
echo $embedding_dir

args=$(echo --edgelist ${edgelist} --labels ${labels} \
    --dist_fn euclidean \
    --embedding ${embedding_dir} --seed ${seed} \
    --test-results-dir ${test_results})
echo $args

module purge
module load bluebear
module load Python/3.6.3-iomkl-2018a
pip install --user numpy pandas networkx scikit-learn scikit-multilearn

python evaluate_nc.py ${args}