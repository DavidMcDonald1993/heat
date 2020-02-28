#!/bin/bash

#SBATCH --job-name=HEATRECON
#SBATCH --output=HEATRECON_%A_%a.out
#SBATCH --error=HEATRECON_%A_%a.err
#SBATCH --array=0-3599
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

datasets=(cora_ml citeseer ppi pubmed mit)
dims=(5 10 25 50)
seeds=({0..29})
alphas=(00 05 10 20 50 100)
exp=reconstruction_experiment

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_alphas=${#alphas[@]}

# for ((SLURM_ARRAY_TASK_ID=0; SLURM_ARRAY_TASK_ID<3600; SLURM_ARRAY_TASK_ID++))
# do

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

#     if [ $dataset == pubmed ] && [ $dim == 50 ] && [ $alpha == "1.00" ] && [ $seed == 9 ]
#     then 
#         echo $SLURM_ARRAY_TASK_ID
#         break
#     fi

# done

data_dir=datasets/${dataset}
edgelist=${data_dir}/edgelist.tsv.gz
embedding_dir=embeddings/${dataset}/nc_experiment

test_results=$(printf "test_results/${dataset}/${exp}/alpha=${alpha}/dim=%03d/" ${dim})
embedding_dir=$(printf "${embedding_dir}/alpha=${alpha}/seed=%03d/dim=%03d/" ${seed} ${dim})
echo $embedding_dir

args=$(echo --edgelist ${edgelist} --dist_fn hyperboloid \
    --embedding ${embedding_dir} --seed ${seed} \
    --test-results-dir ${test_results})
echo $args

module purge
module load bluebear
module load apps/python3/3.5.2

python evaluate_reconstruction.py ${args}