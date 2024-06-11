import argparse
import shutil
import subprocess
import traceback
import time
import os
import json
import netCDF4 as nc
import copy
import numpy as np
import xarray as xr
from ensemble_kalman_filter import EnsembleKalmanFilter as EnKF
import matplotlib.pyplot as plt
import monitor
from pathlib import Path

os.environ['PYTHONWARNINGS'] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# np.random.seed(1233)
#np.random.seed(37)


class DataAssimilation:
    def __init__(self, params):
        # Save arguments
        self.params = params
        self.seed = params['seed']
        np.random.seed(self.seed)
        self.synthetic = params['synthetic']
        self.output_dir = params['output_dir']
        self.ensemble_dir = Path(self.output_dir) / "Ensemble/"

        self.covered_area = params['covered_area']
        self.ensemble_size = params['ensemble_size']
        self.process_noise = params['process_noise']
        self.time_interval = params['time_interval']
        self.num_iterations = params['num_iterations']
        self.initial_spread = params['initial_spread']

        if self.synthetic:
            self.initial_offset = params['initial_offset']
            self.observation_uncertainty = params['observation_uncertainty']

        self.initial_estimate = params['initial_estimate']

        # placeholders
        self.ensemble_usurfs = []
        self.ensemble_velo = []

        # Change between synthetic and real observations ###
        self.observed_glacier = nc.Dataset(params['observations_file'])
        self.hidden_smb = params['smb_simple_array']

        # Extract metadata from ground observations
        self.year_range = np.array(self.observed_glacier['time'])[::self.time_interval]
        self.start_year = self.year_range[0]
        self.icemask = np.array(self.observed_glacier['icemask'])[0]
        self.surface = np.array(self.observed_glacier['usurf'])[0]
        self.bedrock = self.observed_glacier['topg'][0]
        self.visualise = params['visualise']

        # sample observation points
        gx, gy = np.where(self.icemask)
        glacier_points = np.array(list(zip(gx, gy)))
        num_sample_points = int((self.covered_area / 100) * np.sum(self.icemask))
        print('Number of points: {}'.format(num_sample_points))
        observation_index = np.random.choice(len(glacier_points), num_sample_points, replace=False)
        observation_points = glacier_points[observation_index]

        def get_pixel_value(point):
            x, y = point
            return self.surface[x][y]

        sorted_observation_points = sorted(observation_points, key=get_pixel_value)
        self.observation_points = np.array(sorted_observation_points)

        ### PARALLIZE ###
        for i in range(self.ensemble_size):
            # create folder for every ensemble member

            dir_name =  self.ensemble_dir / f"{i:03}/"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            # copy results of inversion as initial state for every ensemble member
            shutil.copy2("Inversion/geology-optimized.nc", dir_name / "init_input.nc")
            # create folder for igm trained model
            if os.path.exists(dir_name / "iceflow-model"):
                shutil.rmtree(dir_name / "iceflow-model")
            # copy trained igm parameters
            shutil.copytree("Inversion/iceflow-model/", dir_name /"iceflow-model/")

        ### LOOP OVER YEAR RANGE ###
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initialize_ensemble(self):
        # Import necessary modules
        import monitor

        # Initial estimate
        state_x = np.array(self.initial_estimate).astype(float)

        # Initialize prior (uncertainty) P
        prior_x = np.zeros((len(state_x), len(state_x)))
        prior_x[0, 0] = self.initial_spread ** 2 * 100
        prior_x[1, 1] = self.initial_spread ** 2 * 0.00000001
        prior_x[2, 2] = self.initial_spread ** 2 * 0.00000001

        dim_z = len(self.observation_points)
        print(state_x)
        print([prior_x[0,0], prior_x[1,1], prior_x[2,2]])
        # Create Ensemble Kalman Filter
        ensemble = EnKF(x=state_x, P=prior_x, dim_z=dim_z, dt=self.time_interval, N=self.ensemble_size,
                        hx=self.generate_observation, fx=self.forward_model,
                        start_year=self.start_year)

        # Update Process noise (Q)
        ensemble.Q = np.zeros_like(prior_x)
        ensemble.Q[0, 0] = 100 * self.process_noise
        ensemble.Q[1, 1] = 0.00000001 * self.process_noise
        ensemble.Q[2, 2] = 0.00000001 * self.process_noise

        # Compute Observation noise (R) of EnKF
        self.usurf = np.array(self.observed_glacier['usurf'])
        if self.synthetic:
            uncertainty_factor = np.arange(len(self.usurf)) * self.observation_uncertainty
            observation_uncertainty_field = np.array([self.icemask * f for f in uncertainty_factor])
        else:
            observation_uncertainty_field = np.array(self.observed_glacier['obs_error'])

        self.observation_uncertainty_field = observation_uncertainty_field
        ensemble.R = np.eye(dim_z) * observation_uncertainty_field[
            0, self.observation_points[:, 0], self.observation_points[:, 1]]

        return ensemble

    def run_iterations(self, ensemble, visualise=True):
        estimates = [copy.copy(ensemble.sigmas)]


        for iteration in range(self.num_iterations):
            ensemble.year = self.start_year

            # Initialize ensemble surface elevation and velocity
            self.initialize_ensemble_fields()
            monitor_instance = monitor.Monitor(self.params, self.usurf, self.observation_uncertainty_field,
                                               self.observation_points)
            if visualise:
                monitor_instance.plot(self.year_range[0], ensemble.sigmas, self.ensemble_usurfs,
                                      self.ensemble_velo)

            for year in self.year_range[:-1]:
                print("==== %i ====" % year)
                year_index = int((ensemble.year - self.start_year))

                self.predict_and_update(ensemble, year_index, monitor_instance, visualise=visualise)

            estimates.append(copy.copy(ensemble.sigmas))

            #if visualise:
            monitor_instance.plot_iterations(estimates)

        return estimates

    def save_results(self, estimates):
        self.params['result'] = [list(sigma) for sigma in estimates[-1]]
        with open(self.output_dir + f"result_o_{self.initial_offset}_s_{self.initial_spread}_seed_{self.seed}.json", 'w') as f:
            json.dump((self.params), f, indent=4, separators=(',', ': '))

    def initialize_ensemble_fields(self):
        # Initialize ensemble surface elevation
        self.ensemble_usurfs = np.array([copy.copy(self.usurf[0]) for _ in range(self.ensemble_size)])
        self.ensemble_velo = np.array([np.zeros_like(self.usurf[0]) for _ in range(self.ensemble_size)])

    def predict_and_update(self, ensemble, year_index, monitor_instance, visualise=True):
        # Predict
        start_time = time.time()
        ensemble.predict()
        print("Prediction time: ", time.time() - start_time)
        ensemble.year += self.time_interval

        # Plot predictions
        if visualise:
            monitor_instance.plot(ensemble.year, ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)

        # Update
        year_index = int((ensemble.year - self.start_year))
        self.update_ensemble(ensemble, year_index)

        if visualise:
            monitor_instance.plot(ensemble.year, ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)

    def update_ensemble(self, ensemble, year_index):
        # Get noisy observation
        noisy_usurf = self.usurf[year_index]


        # Sample random noise for each ensemble member to add to the difference
        random_factors = np.random.normal(0, 1, self.ensemble_size)
        uncertainty_field_year = self.observation_uncertainty_field[year_index]
        observation_noise_samples = np.array([uncertainty_field_year * rf for rf in random_factors])

        sampled_observations = noisy_usurf[self.observation_points[:, 0], self.observation_points[:, 1]]
        e_r = observation_noise_samples[:, self.observation_points[:, 0], self.observation_points[:, 1]]

        # Update hidden parameters
        R = np.eye(len(self.observation_points)) * self.observation_uncertainty_field[
            year_index, self.observation_points[:, 0], self.observation_points[:, 1]]

        # update ensemble
        ensemble.update(sampled_observations, e_r, R)

        # Update the surface elevation
        self.ensemble_usurfs = np.array([copy.copy(noisy_usurf) for _ in range(self.ensemble_size)])


    def forward_model(self, state_x, dt, i, year):
        # create new params.json
        year_next = year + dt

        ela, grad_abl, grad_acc = state_x[[0, 1, 2]]

        data = {"modules_preproc": ["load_ncdf"],
                "modules_process": ["smb_simple", "iceflow", "time", "thk", ],
                "modules_postproc": ["write_ncdf", "print_info"],
                "smb_simple_array": [
                    ["time", "gradabl", "gradacc", "ela", "accmax"],
                    [year, grad_abl, grad_acc, ela, 100],
                    [year_next, grad_abl, grad_acc, ela, 100]],
                "iflo_emulator": "iceflow-model",
                "lncd_input_file": 'input.nc',
                "wncd_output_file": f'output_{year}.nc',
                "time_start": year,
                "time_end": year_next,
                "iflo_retrain_emulator_freq": 0,
                # "time_step_max": 0.2,
                }

        with open(self.ensemble_dir  /f"{i:03}/params.json", 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))

        # create new input.nc
        # TODO
        input_file = self.ensemble_dir / f"{i:03}/init_input.nc"
        try:
            with xr.open_dataset(input_file) as ds:
                # load usurf from ensemble
                usurf = self.ensemble_usurfs[i]
                ds['usurf'] = xr.DataArray(usurf, dims=('y', 'x'))

                thickness = usurf - self.bedrock
                thk_da = xr.DataArray(thickness, dims=('y', 'x'))
                thk_da = xr.DataArray(thickness, dims=('y', 'x'))
                ds['thk'] = thk_da

                # ds_drop = ds.drop_vars("thkinit")
                ds.to_netcdf(self.ensemble_dir / f'{i:03}/input.nc')

        except:
            traceback.print_exc()
            print("could not read input" + str(input_file))

        ### IGM RUN ###
        try:
            subprocess.run(["igm_run"], cwd=self.ensemble_dir / f'{i:03}', shell=True)
        except:
            print("could not run igm_run")
        # update state x and return
        try:
            with xr.open_dataset(self.ensemble_dir / f'{i:03}/output_{year}.nc') as new_ds:
                new_usurf = np.array(new_ds['usurf'][-1])
                new_velo = np.array(new_ds['velsurf_mag'][-1])
        except:
            print("could not read output")

        self.ensemble_usurfs[i] = new_usurf
        self.ensemble_velo[i] = new_velo

        return state_x

    def generate_observation(self, state_x, i):
        """
        h(x)
        :returns thickness map as 1D array
        """

        usurf = self.ensemble_usurfs[i]

        modelled_observations = usurf[self.observation_points[:, 0], self.observation_points[:, 1]]

        return modelled_observations


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

    ### START DATAASSIMILATION ###
    DA = DataAssimilation(params)

    ensemble = DA.initialize_ensemble()

    # Run loop of predictions and updates
    estimates = DA.run_iterations(ensemble)

    # Evaluate results
    results = DA.save_results(estimates)

    return results


if __name__ == '__main__':
    main()
