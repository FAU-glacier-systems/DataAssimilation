#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=igm
#SBATCH --gres=gpu:a100:1
module load cudnn/8.9.6.50-11.x
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
python data_assimilation.py --experiment Experiments/Rhone/hyperparams1.json
python data_assimilation.py --experiment Experiments/Rhone/hyperparams2.json
python data_assimilation.py --experiment Experiments/Rhone/hyperparams3.json
python data_assimilation.py --experiment Experiments/Rhone/hyperparams4.json
python data_assimilation.py --experiment Experiments/Rhone/hyperparams5.json