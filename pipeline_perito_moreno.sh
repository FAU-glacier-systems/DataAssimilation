#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=igm

#module load cudnn/8.9.6.50-11.x
module load python/3.9-anaconda
conda activate igm

#cd OGGM_shop/Perito_Moreno/
#igm_run --param_file params.json
#cd ../
#python down_scale.py Perito_Moreno/input_saved.nc Perito_Moreno/downscaled_file.nc --scale 0.5
#cd ../

# 1. Inversion
cd Inversion/Perito_Moreno/
igm_run --param_file params.json
cd ../../

#cd ReferenceSimulation/Perito_Moreno/
#igm_run --param_file params.json
#cd ../../
#cd Hugonnet/
#python download_observations.py --params Perito_Moreno/params.json
#cd ../

# 3. Data Assimilation
python data_assimilation.py --experiment Experiments/Perito_Moreno/hyperparams.json

