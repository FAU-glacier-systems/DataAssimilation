import json
import numpy as np

def main():
    with open('ReferenceSimulation/params.json') as f:
        params = json.load(f)
        smb = params['smb_simple_array']
        base_ela = smb[1][3]
        base_abl_grad = smb[1][1]
        base_acc_grad = smb[1][2]

        # [samplepoints^1/2, ensemble members, inital state, inital varianc]

    hyperparameter_range = {
        # "Area": [1, 2, 4, 8, 16, 32, 64],
        # "Observation_Interval": [1, 2],
        # "Process_Noise": [0, 0.5, 2, 4],
        # "Ensemble_Size": [5, 10, 20, 30, 40, 50],
        # "initial_offset" : [0,20,40,60,80,100],
        # "initial_uncertainty": [100],
        # "bias": [0, 2, 4, 6, 8, 10],
        "specal_noise": [1]
    }
    initial_offsets = np.random.randint(0, 100, size=10)
    initial_uncertainties = np.random.randint(0, 100, size=10)
    biases = np.random.randint(-10, 10, size=10)
    specal_noises = np.random.randint(1, 3, size=10)

    for hyperparameter in hyperparameter_range.keys():
        print("Start Hyperparameter: ", hyperparameter)

        # if hyperparameter == 'external_parameter':
        #     l_bounds = [0, 0, 0, 1]
        #     u_bounds = [100, 100, 10, 3]
        #     sampler = qmc.LatinHypercube(d=4)
        #     number_of_experiments = hyperparameter_range['external_parameter']
        #     sample = sampler.integers(l_bounds=l_bounds, u_bounds=u_bounds, n=number_of_experiments)

        for value in hyperparameter_range[hyperparameter]:
            if hyperparameter == 'Area':
                covered_area = value
                dt = 4
                ensemble_size = 25
                process_noise = 1

            elif hyperparameter == 'Observation_Interval':
                covered_area = 16
                dt = value
                ensemble_size = 25
                process_noise = 1

            elif hyperparameter == 'Process_Noise':
                covered_area = 16
                dt = 4
                ensemble_size = 25
                process_noise = value

            elif hyperparameter == 'Ensemble_Size':
                covered_area = 16
                dt = 4
                ensemble_size = value
                process_noise = 1
            elif hyperparameter == 'external_parameter':
                covered_area = 16
                dt = 4
                ensemble_size = 25
                process_noise = 1
            else:
                covered_area = 16
                dt = 4
                ensemble_size = 25
                process_noise = 1

            number_of_experiments = 1

            for i in range(number_of_experiments):
                initial_offset = initial_offsets[i]
                initial_uncertainty = initial_uncertainties[i]
                bias = biases[i]
                specal_noise = specal_noises[i]

                if hyperparameter == 'initial_offset':
                    initial_offset = value
                elif hyperparameter == 'initial_uncertainty':
                    initial_uncertainty = value
                elif hyperparameter == 'bias':
                    bias = value
                elif hyperparameter == 'specal_noise':
                    specal_noise = value
                else:
                    pass

                print("initial_offset:", initial_offset)
                print("initial_uncertainty:", initial_uncertainty)
                print("bias:", bias)
                print("special_noise:", specal_noise)
                print("covered_area:", covered_area)
                print("process_noise:", process_noise)
                print("ensemble_size:", ensemble_size)
                print("dt:", dt)

                sign = np.random.choice([-1, 1], 3)

                initial_est = [(base_ela + 1000 * (initial_offset / 100)) * sign[0],
                               (base_abl_grad + 0.01 * (initial_offset / 100)) * sign[1],
                               (base_acc_grad + 0.01 * (initial_offset / 100)) * sign[2]]

                initial_est_var = [initial_uncertainty ** 2 * 100,
                                   initial_uncertainty ** 2 * 0.00000001,
                                   initial_uncertainty ** 2 * 0.00000001]

                DA = DataAssimilation(int(covered_area), int(ensemble_size), int(dt), initial_est,
                                      initial_est_var, initial_offset, initial_uncertainty, specal_noise, bias,
                                      process_noise, synthetic)
                results = DA.start_ensemble()

                if params["synthetic"]:
                    results_folder_path = f"Results/Results_{params['hyperparameter']}/{params['value']}/"
                    if not os.path.exists(results_folder_path):
                        os.makedirs(results_folder_path)

                    with open(f"{results_folder_path}\
                                _o_{params['initial_offset']}\
                                _u_{params['initial_uncertainty']}\
                                _b_{params['bias']}\
                                _s_{params['specal_noise']}.json", 'w') as f:
                        json.dump(results, f, indent=4, separators=(',', ': '))

if __name__ == '__main__':