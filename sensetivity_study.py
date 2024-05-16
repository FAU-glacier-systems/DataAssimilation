import os
import json
import numpy as np
from data_assimilation import DataAssimilation


def main():
    with open('ReferenceSimulation/params.json') as f:
        params = json.load(f)
        smb = params['smb_simple_array']
        base_ela = smb[1][3]
        base_abl_grad = smb[1][1]
        base_acc_grad = smb[1][2]

    hyperparameter_range = {
        "Area": [1, 2, 4, 8, 16, 32, 64],
        "Ensemble_Size": [5, 10, 20, 30, 40, 50],
        "observation_uncertainty": [0, 0.2, 0.4, 0.6, 0.8, 1.0],

        "initial_offset": [0, 20, 40, 60, 80, 100],
        "initial_spread": [0, 20, 40, 60, 80, 100],
    }
    initial_offsets = np.random.randint(0, 100, size=10)
    initial_spreads = np.random.randint(0, 100, size=10)

    for hyperparameter in hyperparameter_range.keys():
        print("Start Hyperparameter: ", hyperparameter)

        for value in hyperparameter_range[hyperparameter]:
            # default
            covered_area = 50
            ensemble_size = 10
            observation_uncertainty = 0.2

            if hyperparameter == 'Area':
                covered_area = value

            elif hyperparameter == 'Ensemble_Size':
                ensemble_size = value

            elif hyperparameter == 'observation_uncertainty':
                observation_uncertainty = value

            number_of_experiments = 10

            for i in range(number_of_experiments):
                initial_offset = int(initial_offsets[i])
                initial_spread = int(initial_spreads[i])

                if hyperparameter == 'initial_offset':
                    initial_offset = value
                elif hyperparameter == 'initial_spread':
                    initial_spread = value

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
                          }

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(output_dir + f"params_o_{initial_offset}_s_{initial_spread}.json", 'w') as f:
                    json.dump(params, f, indent=4, separators=(',', ': '))

                DA = DataAssimilation(params)

                ensemble = DA.initialize_ensemble()

                # Run loop of predictions and updates
                estimates = DA.run_iterations(ensemble, visualise=True)
                DA.save_results(estimates)


if __name__ == '__main__':
    main()
