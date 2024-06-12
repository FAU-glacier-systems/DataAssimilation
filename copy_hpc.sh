#!/bin/bash

# Define the base remote directory
remote_base_dir="gwgi026h@fritz:/home/saturn/gwgi/gwgi026h/DataAssimilation/Results/Results_observation_uncertainty"
local_base_dir="Results/Results_observation_uncertainty"
values=(0.2  0.4  0.8  1.6  3.2)

#remote_base_dir="gwgi026h@fritz:/home/saturn/gwgi/gwgi026h/DataAssimilation/Results/Results_Area"
#local_base_dir="Results/Results_Area"
#values=(0.1 0.2  0.5  1  2  10)

#remote_base_dir="gwgi026h@fritz:/home/saturn/gwgi/gwgi026h/DataAssimilation/Results/Results_Ensemble_Size"
#local_base_dir="Results/Results_Ensemble_Size"
#values=(3 5 10 25 50 100)

#remote_base_dir="gwgi026h@fritz:/home/saturn/gwgi/gwgi026h/DataAssimilation/Results/Results_initial_offset"
#local_base_dir="Results/Results_initial_offset"
#values=(20 40 60 80 100)

mkdir -p "${local_base_dir}"

# Loop through each area and execute the scp command
for value in "${values[@]}"; do
  scp -r -p "${remote_base_dir}/${value}/result*" "${local_base_dir}/${value}"
  scp -r -p "${remote_base_dir}/${value}/iter*" "${local_base_dir}/${value}"
done
