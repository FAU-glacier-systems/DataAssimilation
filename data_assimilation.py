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

os.environ['PYTHONWARNINGS'] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# np.random.seed(1233)
np.random.seed(37)


class DataAssimilation:
    def __init__(self, params):
        # Save arguments
        self.params = params
        self.synthetic = params['synthetic']
        self.output_dir = params['output_dir']

        self.covered_area = params['covered_area']
        self.ensemble_size = params['ensemble_size']
        self.process_noise = params['process_noise']
        self.time_interval = params['time_interval']

        self.initial_estimate = params['initial_estimate']
        self.initial_spread = params['initial_spread']
        self.initial_offset = params['initial_offset']

        if self.synthetic:
            self.specal_noise = params['specal_noise']
            self.elevation_bias = params['elevation_bias']

        # placeholders
        self.ensemble_usurfs = []
        self.ensemble_velo = []

        # Change between synthetic and real observations ###
        self.observed_glacier = nc.Dataset(params['observations_file'])
        self.smb = params['smb_simple_array']

        # Extract metadata from ground observations
        self.year_range = np.array(self.observed_glacier['time'])[::self.time_interval]
        self.start_year = self.year_range[0]
        self.icemask = np.array(self.observed_glacier['icemask'])[0]
        self.surface = np.array(self.observed_glacier['usurf'])[0]
        self.bedrock = self.observed_glacier['topg'][0]

        # sample observation points
        gx, gy = np.where(self.icemask)
        glacier_points = np.array(list(zip(gx, gy)))
        num_sample_points = int(self.covered_area / 100 * np.sum(self.icemask))
        observation_index = np.random.choice(len(glacier_points), num_sample_points, replace=False)
        observation_points = glacier_points[observation_index]

        def get_pixel_value(point):
            x, y = point
            return self.surface[x][y]

        sorted_observation_points = sorted(observation_points, key=get_pixel_value)
        self.observation_points = np.array(sorted_observation_points)

    def start_ensemble(self):
        import monitor
        ### INITIALIZE ###
        # Initial estimate
        state_x = np.array(self.initial_estimate).astype(float)

        # Initialise prior (uncertainty) P
        prior_x = np.zeros((len(state_x), len(state_x)))
        prior_x[0, 0] = self.initial_spread ** 2 * 100
        prior_x[1, 1] = self.initial_spread ** 2 * 0.00000001
        prior_x[2, 2] = self.initial_spread ** 2 * 0.00000001

        dim_z = len(self.observation_points)
        ### Create Ensemble Kalman Filter
        ensemble = EnKF(x=state_x, P=prior_x, dim_z=dim_z, dt=self.time_interval, N=self.ensemble_size,
                        hx=self.generate_observation, fx=self.forward_model,
                        start_year=self.start_year)

        # update Process noise (Q)
        ensemble.Q = np.zeros_like(prior_x)
        ensemble.Q[0, 0] = 1000 * self.process_noise * self.initial_spread
        ensemble.Q[1, 1] = 0.00000001 * self.process_noise * self.initial_spread
        ensemble.Q[2, 2] = 0.00000001 * self.process_noise * self.initial_spread

        # make observations noisy and compute Observation noise (R) of EnKF
        if self.synthetic:
            usurfs = np.array(self.observed_glacier['usurf'])
            elevation_2000 = usurfs[0]
            min, max = np.min(elevation_2000[self.icemask == 1]), np.max(elevation_2000[self.icemask == 1])
            elevation_bias_single = (elevation_2000 - min) / (max - min)
            observation_uncertainty_field = elevation_bias_single * self.bias + np.random.normal(0, self.specal_noise,
                                                                                     elevation_bias_single.shape)
            noisy_usurf = [usurf + observation_uncertainty_field * np.random.normal(0, 1) for usurf in usurfs]
            self.noisy_usurf = np.array(noisy_usurf)

        else:
            self.noisy_usurf = np.array(self.observed_glacier['usurf'])
            observation_uncertainty_field = np.array(self.observed_glacier['obs_error'][0])

        observation_uncertainty_field[self.icemask == 0] = 0
        ensemble.R = np.eye(dim_z) * observation_uncertainty_field[self.observation_points[:, 0], self.observation_points[:, 1]]

        ### PARALLIZE ###
        for i in range(self.ensemble_size):
            # make a copy of first usurf for every ensemble member

            observation_noise_samples = np.random.normal(0, 1) * observation_uncertainty_field

            self.ensemble_usurfs.append(copy.copy(self.noisy_usurf[0]))  # + observation_noise_samples)
            # make a velocity field for every ensemble member
            self.ensemble_velo.append(np.zeros_like(self.noisy_usurf[0]))
            # create folder for every ensemble member
            if not os.path.exists(f"Ensemble/{i}"):
                os.makedirs(f"Ensemble/{i}")
            # copy results of inversion as initial state for every ensemble member
            shutil.copy2("Inversion/geology-optimized.nc", f"Ensemble/{i}/init_input.nc")
            # create folder for igm trained model
            if os.path.exists(f"Ensemble/{i}/iceflow-model"):
                shutil.rmtree(f"Ensemble/{i}/iceflow-model")
            # copy trained igm parameters
            shutil.copytree("Inversion/iceflow-model/", f"Ensemble/{i}/iceflow-model/")

        # convert to numpy
        self.ensemble_usurfs = np.array(self.ensemble_usurfs)
        self.ensemble_velo = np.array(self.ensemble_velo)

        ### VISUALIZE ###
        # create a Monitor for visualisation
        monitor = monitor.Monitor(self.params, self.noisy_usurf, observation_uncertainty_field, self.observation_points)

        # draw plot of inital state
        monitor.plot(self.year_range[0], ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)
        ### LOOP OVER YEAR RANGE ###
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        ### START LOOP ###
        for year in self.year_range[:-1]:
            print("==== %i ====" % year)

            ### PREDICT ###
            start_time = time.time()
            ensemble.predict()
            print("Prediction time: ", time.time() - start_time)
            ensemble.year = year + self.time_interval

            # get observations
            # usurf = self.observed_glacier['usurf'][int((ensemble.year - self.start_year))]
            velo = self.observed_glacier['velsurf_mag'][int((ensemble.year - self.start_year))]

            monitor.plot(ensemble.year, ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)

            ### UPDATE ###

            # get the noisey observation
            print(int((ensemble.year - self.start_year)))
            noisy_usurf = self.noisy_usurf[int((ensemble.year - self.start_year))]
            sampled_observations = noisy_usurf[self.observation_points[:, 0], self.observation_points[:, 1]]

            # sample random noise for each ensemlbe memember to add to the difference
            # interpreation: every ensemble member gets a slightly different observation

            observation_noise_samples = np.array(
                [r_norm * observation_uncertainty_field for r_norm in np.random.normal(0, 1, self.ensemble_size)])
            e_r = observation_noise_samples[:, self.observation_points[:, 0], self.observation_points[:, 1]]

            # update the hidden parameters
            ensemble.update(sampled_observations, e_r)

            # update the surface elevation
            # self.ensemble_usurfs = np.array([noisy_usurf + noise for noise in observation_noise_samples])
            self.ensemble_usurfs = np.array([copy.copy(noisy_usurf) for i in range(self.ensemble_size)])

            # plot the update
            monitor.plot(ensemble.year, ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)


        ### EVALUATION ###
        if self.synthetic:
            true_x = [self.smb[-1][3], self.smb[-1][1], self.smb[-1][2]]
            results = dict(true_parameter=true_x,
                           esti_parameter=ensemble.x.tolist(),
                           esit_var=ensemble.P.tolist(),
                           ensemble_size=self.ensemble_size,
                           dt=int(self.time_interval),
                           initial_offset=int(self.initial_offset),
                           initial_uncertainity=int(self.initial_spread),
                           bias=int(self.elevation_bias),
                           specal_noise=int(self.specal_noise),
                           covered_area=self.covered_area,
                           initial_estimate=[int(i) for i in self.initial_estimate],
                           initial_estimate_var=[int(j) for j in self.initial_estimate_var],
                           process_noise=self.process_noise
                           )
        else:
            #TODO
            results=None

        return results

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
                "lncd_input_file": 'input_.nc',
                "wncd_output_file": f'output_{year}.nc',
                "time_start": year,
                "time_end": year_next,
                "iflo_retrain_emulator_freq": 0,
                # "time_step_max": 0.2,
                }

        with open(f'Ensemble/{i}/params.json', 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))

        # create new input.nc
        input_file = f"Ensemble/{i}/init_input.nc"
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
                ds.to_netcdf(f'Ensemble/{i}/input_.nc')

        except:
            traceback.print_exc()
            print("could not read input" + input_file)

        ### IGM RUN ###
        try:
            subprocess.run(["igm_run"], cwd=f'Ensemble/{i}', shell=True)
        except:
            print("could not run igm_run")
        # update state x and return
        try:
            with xr.open_dataset(f'Ensemble/{i}/output_{year}.nc') as new_ds:
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
    results = DA.start_ensemble()

    return results


if __name__ == '__main__':
    main()
