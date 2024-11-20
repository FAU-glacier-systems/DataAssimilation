#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --job-name=igm
##SBATCH --gres=gpu:a100:1
#module load cudnn/8.9.6.50-11.x
module load python/3.9-anaconda
conda activate igm

#cd OGGM_shop/Rhone/
#igm_run --param_file params.json
#cd ../../

#cd Hugonnet/
#python download_observations.py --params Rhone/params.json
#cd ../

# 1. Inversion
#cd Inversion/Rhone/
#igm_run --param_file params.json
#cd ../../

#python data_assimilation.py --experiment Experiments/Rhone/hyperparams.json
# --inflation 2 --seed 21 --etkf True

# 3. Data Assimilation
for seed in 3 4 5; do
  for observation_noise_factor in 1 2 3; do
    python data_assimilation.py --experiment Experiments/Rhone/hyperparams.json --inflation 1 --seed $seed --observation_noise_factor $observation_noise_factor
  done
done




