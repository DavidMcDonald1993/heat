#!/bin/bash

#SBATCH --job-name=EUCLIDRECON
#SBATCH --output=EUCLIDRECON_%A_%a.out
#SBATCH --error=EUCLIDRECON_%A_%a.err
#SBATCH --array=0-3599
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

datasets=(cora_ml citeseer ppi pubmed mit)
dims=(5 10 25 50)
seeds=({0..29})
methods=(abrw attrpure deepwalk tadw aane sagegcn)
exp=reconstruction_experiment

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_methods=${#methods[@]}

# for ((SLURM_ARRAY_TASK_ID=0; SLURM_ARRAY_TASK_ID<3600; SLURM_ARRAY_TASK_ID++))
# do

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_methods * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_methods * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / num_methods % num_seeds))
method_id=$((SLURM_ARRAY_TASK_ID % num_methods))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
method=${methods[$method_id]}

echo $dataset $dim $seed $method

#      if [ $dataset == mit ] && [ $dim == 25 ] && [ $method == attrpure ] && [ $seed == 9 ]
#     then 
#         echo $SLURM_ARRAY_TASK_ID
#         break
#     fi

# done


data_dir=datasets/${dataset}
edgelist=${data_dir}/edgelist.tsv.gz 
embedding_dir=$(echo ../OpenANE/embeddings/${dataset}/nc_experiment/${dim}/${method}/${seed})

test_results=$(printf "test_results/${dataset}/${exp}/${method}/dim=%03d/" ${dim})
echo $embedding_dir

args=$(echo --edgelist ${edgelist} --dist_fn euclidean \
    --embedding ${embedding_dir} --seed ${seed} \
    --test-results-dir ${test_results})
echo $args

module purge
module load bluebear
module load apps/python3/3.5.2

python evaluate_reconstruction.py ${args}