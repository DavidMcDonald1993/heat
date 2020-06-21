#!/bin/bash

#SBATCH --job-name=NKLP
#SBATCH --output=NKLP_%A_%a.out
#SBATCH --error=NKLP_%A_%a.err
#SBATCH --array=0-599
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

datasets=(cora_ml citeseer ppi pubmed mit)
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

echo $dataset $dim $seed 

data_dir=datasets/${dataset}
edgelist=${data_dir}/edgelist.tsv.gz
embedding_dir=../poincare-embeddings/embeddings/${dataset}
removed_edges_dir=$(printf edgelists/${dataset}/seed=%03d/removed_edges ${seed})

test_results=$(printf "test_results/${dataset}/${exp}/nk/dim=%03d/" ${dim})
embedding_dir=$(printf "${embedding_dir}/dim=%02d/seed=%03d/${exp}" ${dim} ${seed})
echo $embedding_dir

args=$(echo --edgelist ${edgelist} --removed_edges_dir ${removed_edges_dir} \
    --dist_fn poincare \
    --embedding ${embedding_dir} --seed ${seed} \
    --test-results-dir ${test_results})
echo $args

if [ ! -f ${test_results}/${seed}.pkl ]
then
    module purge
    module load bluebear
    module load future/0.16.0-foss-2018b-Python-3.6.6

    python evaluate_lp.py ${args}
else 
    echo ${test_results}/${seed}.pkl already exists 
fi