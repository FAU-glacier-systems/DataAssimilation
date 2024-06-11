import os
import json
import numpy as np
from data_assimilation import DataAssimilation
import argparse
import shutil

np.random.seed(45)

def main(hyperparameter_range):
    with open('ReferenceSimulation/params.json') as f:
        params = json.load(f)
        smb = params['smb_simple_array']
        base_ela = smb[1][3]
        base_abl_grad = smb[1][1]
        base_acc_grad = smb[1][2]




    initial_offsets = np.random.randint(0, 100, size=30)

    for hyperparameter in hyperparameter_range.keys():
        print("Start Hyperparameter: ", hyperparameter)

        for value in hyperparameter_range[hyperparameter]:
            # default
            covered_area = 2
            ensemble_size = 25
            observation_uncertainty = 0.2
            seeds=np.random.randint(0, 1000, size=10)

            if hyperparameter == 'Area':
                covered_area = value

            elif hyperparameter == 'Ensemble_Size':
                ensemble_size = value

            elif hyperparameter == 'observation_uncertainty':
                observation_uncertainty = value

            number_of_experiments = 10

            for i in range(number_of_experiments):
                initial_offset = int(initial_offsets[i])
                initial_spread = int(initial_offsets[i])
                seeds = int(seeds[i])


                if hyperparameter == 'initial_offset':
                    initial_offset = value
                    initial_spread = value
                elif hyperparameter == 'initial_spread':
                    initial_spread = value
                    initial_offset = value

                print("covered_area:", covered_area)
                print("ensemble_size:", ensemble_size)
                print("observation_uncertainty:", observation_uncertainty)
                print("initial_offset:", initial_offset)
                print("initial_spread:", initial_spread)

                sign = np.random.choice([-1, 1], 3)

                initial_est = [(base_ela + 1000 * (initial_offset / 100) * sign[0]),
                               (base_abl_grad + 0.01 * (initial_offset / 100) * sign[1]),
                               (base_acc_grad + 0.01 * (initial_offset / 100) * sign[2])]
                output_dir = f"Results/Results_{hyperparameter}/{value}/"
                params = {"synthetic": True,
                          "RGI_ID": "RGI60-11.01238",
                          "smb_simple_array": smb,
                          "covered_area": covered_area,
                          "num_iterations": 5,
                          "ensemble_size": ensemble_size,
                          "time_interval": 20,
                          "initial_offset": initial_offset,
                          "initial_spread": initial_spread,
                          "initial_estimate": initial_est,
                          "process_noise": 0,
                          "observation_uncertainty": observation_uncertainty,
                          "observations_file": "ReferenceSimulation/output.nc",
                          "output_dir": output_dir,
                          "visualise": False,
                          "seed": seed,
                          }

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(output_dir + f"params_o_{initial_offset}_s_{initial_spread}_seed_{seed}.json", 'w') as f:
                    json.dump(params, f, indent=4, separators=(',', ': '))

                DA = DataAssimilation(params)

                ensemble = DA.initialize_ensemble()

                # Run loop of predictions and updates
                estimates = DA.run_iterations(ensemble, visualise=False)
                DA.save_results(estimates)
        shutil.rmtree(output_dir + "Ensemble/")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--hyperparameter_range",
                        type=str,
                        help="Path pointing to the parameter file",
                        required=True)
    arguments, _ = parser.parse_known_args()

    # Load the JSON file with parameters
    with open(arguments.hyperparameter_range, 'r') as f:
        params = json.load(f)

    main(params['hyperparameter_range'])
