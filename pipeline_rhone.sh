#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
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

# 3. Data Assimilation
for inflation in 1.5 1.7 2.0 3.0 ;  do
  for seed in 1 2 3 4 5 6 7 8 9; do
      python data_assimilation.py --experiment Experiments/Rhone/hyperparams.json --inflation $inflation --seed $seed
  done
done



