#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=igm
#SBATCH --output=log.txt
#SBATCH --error=error.txt
module load python
conda activate igm_p3.11


# 0. Download data
#igm_run --param_file OGGM_shop/params.json

# 1. Inversion
#igm_run --param_file Inversion/params.json

# 2. Reference Run
#igm_run --param_file ReferenceSimulation/params.json

# 3. Data Assimilation
python data_assimilation.py

# 4. Evaluation
python evaulate.py



