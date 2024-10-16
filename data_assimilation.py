import argparse
import shutil

import time
import os
import json
import copy
import numpy as np
import oggm
from oggm import utils
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from ensemble_kalman_filter import EnsembleKalmanFilter as EnKF
from ensemble_member import EnsembleMember

from monitor import Monitor
from pathlib import Path

os.environ['PYTHONWARNINGS'] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'





class DataAssimilation:
    def __init__(self, params):
        # Save arguments
        self.params = params
        self.seed = params['seed']
        np.random.seed(self.seed)
        self.synthetic = params['synthetic']
        self.visualise = params['visualise']

        self.covered_area = params['covered_area']
        self.ensemble_size = params['ensemble_size']
        if self.synthetic:
            self.initial_offset = params['initial_offset']
            self.observation_uncertainty = params['observation_uncertainty']


        self.initial_estimate = params['initial_estimate']
        self.initial_spread = params['initial_spread']

        self.process_noise = params['process_noise']
        self.time_interval = params['time_interval']
        self.num_iterations = params['num_iterations']


        self.observations_file = params['observations_file']
        self.geology_file = params['geology_file']
        self.output_dir = Path(params['output_dir'])
        self.ensemble_dir = self.output_dir / "Ensemble/"


        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Change between synthetic and real observations ###

        # load observations
        self.observed_glacier = Dataset(self.observations_file)

        self.year_range = np.array(self.observed_glacier['time']).astype(int)[::self.time_interval]
        self.start_year = self.year_range[0]

        self.initialize_ensemble()


    def initialize_ensemble(self):
        # sample observation points

        if self.synthetic:
            observation_folder = Path(self.observations_file).parent
            observations_params = observation_folder / 'params_saved.json'
            with open(observations_params, 'r') as f:
                obs_params = json.load(f)
                hidden_smb = obs_params['smb_simple_array']

        else:
            try:
                reference_smb = self.params['reference_smb']

                hidden_smb = [['time', 'gradabl', 'gradacc', 'ela', 'accmax'],
                              reference_smb]
            except:
                hidden_smb = [['time', 'gradabl', 'gradacc', 'ela', 'accmax'],
                              [None, None, None, None, None],
                              [None, None, None, None, None]]



        # load geology
        geology_glacier = Dataset(self.geology_file)

        icemask = np.array(geology_glacier['icemask'])
        surface = np.array(geology_glacier['usurf'])
        thickness = geology_glacier['thk']

        gx, gy = np.where(icemask)
        glacier_points = np.array(list(zip(gx, gy)))
        num_sample_points = int((self.covered_area / 100) * np.sum(icemask))
        print('Number of points: {}'.format(num_sample_points))

        """
        # Mask the arrays to only consider glacier pixels
        masked_elevation = np.where(icemask, surface, np.nan)
        masked_thickness = np.where(icemask, thickness, np.nan)

        #  Define the bin edges based on the surface elevation range
        elevation_min = np.nanmin(masked_elevation)
        elevation_max = np.nanmax(masked_elevation)
        bins = np.linspace(elevation_min, elevation_max, num_sample_points + 1)

        # Assign each pixel to a bin based on the surface elevation
        bin_indices = np.digitize(masked_elevation, bins) - 1  # -1 because bins start at 1

        result = []
    
        for i in range(num_sample_points):
            # Get the mask for pixels in the current bin
            bin_mask = (bin_indices == i)
            print(i)
            if np.any(bin_mask):
                # Find the maximum thickness in this bin
                max_thickness = np.nanmax(masked_thickness[bin_mask])

                # Get the indices of pixels with this maximum thickness
                max_indices = np.where((masked_thickness == max_thickness) & bin_mask)

                result.append([max_indices[0][0], max_indices[1][0]])


        observation_points = list(result)
        """

        random_state = np.random.RandomState(seed=self.seed)
        observation_index = random_state.choice(len(glacier_points), num_sample_points, replace=False)
        observation_points = glacier_points[observation_index]


        def get_pixel_value(point):
            x, y = point
            return surface[x][y]

        sorted_observation_points = sorted(observation_points, key=get_pixel_value)
        self.observation_points = np.array(sorted_observation_points)

        self.ensemble_members = []

        ### PARALLIZE ###
        for i in range(self.ensemble_size):
            # create folder for every ensemble member

            dir_name = self.ensemble_dir / f"{i:03}/"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            # copy results of inversion as initial state for every ensemble member
            shutil.copy2(self.geology_file, dir_name / "geology-optimized.nc")
            # create folder for igm trained model
            if os.path.exists(dir_name / "iceflow-model"):
                shutil.rmtree(dir_name / "iceflow-model")
            # copy trained igm parameters
            Path(self.geology_file).parent
            shutil.copytree(Path(self.geology_file).parent / "iceflow-model/", dir_name / "iceflow-model/")

            self.ensemble_members.append(EnsembleMember(i, self.ensemble_dir, self.start_year))


        # Initial estimate
        state_x = np.array(self.initial_estimate).astype(float)
        prior_x = np.array(self.initial_spread).astype(float)

        dim_z = len(self.observation_points)

        # Create Ensemble Kalman Filter
        self.KalmanFilter = EnKF(x=state_x, P=prior_x, dim_z=dim_z, dt=self.time_interval, N=self.ensemble_size,
                                 start_year=self.start_year)

        # Update Process noise (Q)
        self.KalmanFilter.Q = np.zeros_like(prior_x)
        self.KalmanFilter.Q[0, 0] = 100 * self.process_noise
        self.KalmanFilter.Q[1, 1] = 1e-8 * self.process_noise
        self.KalmanFilter.Q[2, 2] = 1e-8 * self.process_noise

        # Compute Observation noise (R) of EnKF
        self.usurf = np.array(self.observed_glacier['usurf'])

        if self.synthetic:
            uncertainty_factor = np.arange(len(self.usurf)) * self.observation_uncertainty
            self.observation_uncertainty_field = np.array([icemask * f for f in uncertainty_factor])
        else:
            self.observation_uncertainty_field = np.array(self.observed_glacier['obs_error'])

        self.KalmanFilter.R = np.eye(dim_z) * self.observation_uncertainty_field[
            0, self.observation_points[:, 0], self.observation_points[:, 1]]


        self.monitor_instance = Monitor(self.params, self.observed_glacier, self.observation_uncertainty_field,
                                        self.observation_points, hidden_smb)


    def run_iterations(self, visualise=True):
        estimates = [copy.copy(self.KalmanFilter.sigmas)]

        #self.monitor_instance.plot_iterations(np.array(estimates), self.ensemble_members)

        for iteration in range(self.num_iterations):
            self.KalmanFilter.year = self.start_year

            # Initialize ensemble surface elevation and velocity
            for member in self.ensemble_members:
                member.reset(2000)

            if visualise:
                self.monitor_instance.plot(iteration, self.year_range[0], self.KalmanFilter.sigmas, self.ensemble_members)

            for year in self.year_range[:-1]:
                print("==== %i ====" % year)
                year_index = self.KalmanFilter.year - self.start_year
                # Predict
                start_time = time.time()
                ### PREDICT ####
                self.KalmanFilter.predict(self.ensemble_members)
                print("Prediction time: ", time.time() - start_time)
                self.KalmanFilter.year += self.time_interval

                # Plot predictions
                if visualise:
                    self.monitor_instance.plot(iteration, self.KalmanFilter.year, self.KalmanFilter.sigmas, self.ensemble_members)

                # Update
                year_index = int((self.KalmanFilter.year - self.start_year))

                # Sample random noise for each ensemble member to add to the difference
                random_factors = np.random.normal(0, 1, self.ensemble_size)
                uncertainty_field_year = self.observation_uncertainty_field[year_index]
                observation_noise_samples = np.array([uncertainty_field_year * rf for rf in random_factors])

                observed_usurf = self.usurf[year_index]
                sampled_observations = observed_usurf[self.observation_points[:, 0], self.observation_points[:, 1]]
                e_r = observation_noise_samples[:, self.observation_points[:, 0], self.observation_points[:, 1]]

                # Update hidden parameters
                R = np.eye(len(self.observation_points)) * self.observation_uncertainty_field[
                    year_index, self.observation_points[:, 0], self.observation_points[:, 1]]

                ### UPDATE ####
                self.KalmanFilter.update(sampled_observations, self.ensemble_members, self.observation_points, e_r, R)

                # Update the surface elevation
                self.ensemble_usurfs = np.array([copy.copy(observed_usurf) for _ in range(self.ensemble_size)])

                if visualise:
                    self.monitor_instance.plot(iteration, self.KalmanFilter.year, self.KalmanFilter.sigmas, self.ensemble_members)

            if visualise:
                self.monitor_instance.reset()

            estimates.append(copy.copy(self.KalmanFilter.sigmas))

            #if visualise:
            #self.monitor_instance.plot_iterations(estimates)
            self.monitor_instance.plot_iterations(np.array(estimates), self.ensemble_members)

        return estimates

    def save_results(self, estimates):

        self.params['final_ensemble'] = [list(sigma) for sigma in estimates[-1]]
        self.params['ensemble_history'] = [[list(s) for s in sigma] for sigma in estimates]
        ensemble_estimates = np.array([list(sigma) for sigma in estimates[-1]])
        self.params['final_mean_estimate']  = list(ensemble_estimates.mean(axis=0))
        self.params['final_std'] = list(ensemble_estimates.std(axis=0))
        with open(self.output_dir / f"result_seed_{self.seed}.json", 'w') as f:
            json.dump((self.params), f, indent=4, separators=(',', ': '))





def main():
    # load parameter file
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment",
                        type=str,
                        default='Experiments/Hugonnet.json',
                        help="Path pointing to the parameter file",
                        required=True)
    arguments, _ = parser.parse_known_args()

    # Load the JSON file with parameters
    with open(arguments.experiment, 'r') as f:
        params = json.load(f)

    ### START DATA ASSIMILATION ###
    DA = DataAssimilation(params)
    estimates = DA.run_iterations(params["visualise"])
    results = DA.save_results(estimates)

    return results


if __name__ == '__main__':
    main()
