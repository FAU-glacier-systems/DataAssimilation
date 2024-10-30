#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --job-name=igm
##SBATCH --gres=gpu:a100:1
#module load cudnn/8.9.6.50-11.x
module load python/3.9-anaconda
conda activate igm

#cd OGGM_shop/Allalin/
#igm_run --param_file params.json
#cd ../../

#cd Hugonnet/
#python download_observations.py --params Allalin/params.json
#cd ../

# 1. Inversion
#cd Inversion/Allalin/
#igm_run --param_file params.json
#cd ../../


# 3. Data Assimilation
#python data_assimilation.py --experiment Experiments/Allalin/hyperparams.json

for inflation in 1.0 1.1 1.2 1.3;  do
  for seed in {1..9}; do
      python data_assimilation.py --experiment Experiments/Allalin/hyperparams.json --inflation $inflation --seed $seed
  done
done

#cd GLAMOS/
#python GLAMOS_data.py --p Allalin/params.json
#cd ../

