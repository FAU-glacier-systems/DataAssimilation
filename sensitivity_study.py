import os
import json
import numpy as np
import argparse
from data_assimilation import DataAssimilation


def main(hyperparameter_range):
    # load surface mass balance parameters os reference run
    with open('ReferenceSimulation/params.json') as f:
        params = json.load(f)
        smb = params['smb_simple_array']
        base_ela = smb[1][3]
        base_abl_grad = smb[1][1]
        base_acc_grad = smb[1][2]

    # loop over the hyperparameters specified in the input
    for hyperparameter in hyperparameter_range.keys():
        print("Start Hyperparameter: ", hyperparameter)
        # default
        covered_area = 2
        ensemble_size = 25
        initial_offset = 60
        initial_spread = 60
        observation_uncertainty = 0.2
        seeds = [1, 2, 30, 20, 324, 34, 211, 323, 643, 987]

        number_of_experiments = 10

        # iterate over the values for the specified hyperparameter
        for value in hyperparameter_range[hyperparameter]:

            if hyperparameter == 'initial_offset':
                initial_offset = value
                initial_spread = value

            for i in range(number_of_experiments):
                seed = int(seeds[i])

                print("covered_area:", covered_area)
                print("ensemble_size:", ensemble_size)
                print("observation_uncertainty:", observation_uncertainty)
                print("initial_offset:", initial_offset)
                print("initial_spread:", initial_spread)

                sign = np.random.choice([-1, 1], 3)
                initial_est = [(base_ela + 1000 * (initial_offset / 100) * sign[0]),
                               (base_abl_grad + 0.01 * (initial_offset / 100) * sign[1]),
                               (base_acc_grad + 0.01 * (initial_offset / 100) * sign[2])]

                # Initialize prior (uncertainty) P
                initial_spread = [[initial_offset ** 2 * 100, 0, 0],
                                  [0, initial_offset ** 2 * 1e-8, 0],
                                  [0, 0, initial_offset ** 2 * 1e-8]]

                output_dir = f"Results/Results_{hyperparameter}/{value}/"
                params = dict(synthetic=True,
                              RGI_ID="RGI60-11.01238",
                              smb_simple_array=smb,
                              covered_area=covered_area,
                              ensemble_size=ensemble_size,
                              observation_uncertainty=observation_uncertainty,
                              initial_offset=initial_offset,
                              initial_estimate=initial_est,
                              initial_spread=initial_spread,
                              time_interval=20,
                              num_iterations=5,
                              process_noise=0,
                              observations_file="ReferenceSimulation/output.nc",
                              output_dir=output_dir,
                              visualise=False,
                              seed=seed)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(output_dir + f"params_seed_{seed}.json", 'w') as f:
                    json.dump(params, f, indent=4, separators=(',', ': '))

                ### START DATA ASSIMILATION ###
                DA = DataAssimilation(params)
                ensemble = DA.initialize_ensemble()
                estimates = DA.run_iterations(ensemble, visualise=False)
                DA.save_results(estimates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparameter_range",
                        type=str,
                        help="Path pointing to the parameter file",
                        required=True)
    arguments, _ = parser.parse_known_args()

    # Load the JSON file with parameters
    with open(arguments.hyperparameter_range, 'r') as f:
        params = json.load(f)

    main(params['hyperparameter_range'])
