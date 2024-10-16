#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=igm
#SBATCH --gres=gpu:a100:1
module load cudnn/8.9.6.50-11.x
module load python/3.9-anaconda
conda activate igm


#cd OGGM_shop/Aletsch/
#igm_run --param_file params.json
#cd ../
#python down_scale.py Aletsch/input_saved.nc Aletsch/downscaled_file.nc --scale 0.5
#cd ../

# 1. Inversion
#cd Inversion/Aletsch/
#igm_run --param_file params.json
#cd ../../


cd Hugonnet/
python download_observations.py --params Aletsch/params.json
cd ../

# 3. Data Assimilation
#python data_assimilation.py --experiment Experiments/Aletsch/hyperparams.json
