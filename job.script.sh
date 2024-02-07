#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=igm
#SBATCH --output=log.txt
#SBATCH --error=error.txt
module load python
conda activate igm_p3.11
python data_assimilation.py