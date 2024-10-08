#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=igm
#module load python
#conda activate igm_p3.11


# 0. Download data
cd OGGM_shop/Perito_Moreno/
igm_run --param_file params.json
cd ../
python down_scale.py
cd ../

# 1. Inversion
cd Inversion/Perito_Moreno/
igm_run --param_file params.json
cd ../../


# 3. Data Assimilation
python data_assimilation.py --experiment Experiments/Perito_Moreno/hyperparams.json

