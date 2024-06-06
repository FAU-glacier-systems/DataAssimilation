#!/bin/bash

# Define the base remote directory
#remote_base_dir="gwgi026h@fritz:/home/saturn/gwgi/gwgi026h/DataAssimilation/Results/Results_initial_spread"
#remote_base_dir="gwgi026h@fritz:/home/saturn/gwgi/gwgi026h/DataAssimilation/Results/Results_observation_uncertainty"
#remote_base_dir="gwgi026h@fritz:/home/saturn/gwgi/gwgi026h/DataAssimilation/Results/Results_Area"
remote_base_dir="gwgi026h@fritz:/home/saturn/gwgi/gwgi026h/DataAssimilation/Results/Results_Ensemble_Size"
# Define the base local directory
#local_base_dir="Results/Results_initial_spread"
local_base_dir="Results/Results_Ensemble_Size"
#local_base_dir="Results/Results_observation_uncertainty"
mkdir -p "${local_base_dir}"
# Define the area numbers
#areas=(0.2  0.4  0.6  0.8  1.0  2.0)
#areas=(0 20 40 60 80 100)
#areas=(2 4 8 16 32 64)
areas=(5 10 20 30 40 50)
# Loop through each area and execute the scp command
for area in "${areas[@]}"; do
  scp -r -p "${remote_base_dir}/${area}/iter*" "${local_base_dir}/${area}"
done
