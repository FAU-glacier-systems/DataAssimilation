import copy
import shutil
import subprocess
import time
import os
import json

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import rasterio
import xarray as xr
from ensemble_kalman_filter import EnsembleKalmanFilter as EnKF
from scipy.stats import qmc

os.environ['PYTHONWARNINGS'] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(1233)


class DataAssimilation:
    def __init__(self, covered_area, ensemble_size, dt, initial_estimate, initial_estimate_var, initial_offset,
                 initial_uncertainity, specal_noise, bias, process_noise, synthetic):
        # load true glacier

        self.noisey_usruf = None
        self.covered_area = covered_area
        self.ensemble_size = ensemble_size
        self.dt = dt
        self.initial_estimate = initial_estimate
        self.initial_estimate_var = initial_estimate_var
        self.initial_offset = initial_offset
        self.initial_uncertainity = initial_uncertainity
        self.specal_noise = specal_noise
        self.bias = bias
        self.process_noise = process_noise

        ### Change between synthetic and real observations ###
        self.synthetic = synthetic
        if self.synthetic:
            self.true_glacier = netCDF4.Dataset('ReferenceSimulation/output.nc')
        else:
            self.true_glacier = netCDF4.Dataset('Hugonnet/merged_dataset.nc')

        # extract metadata from ground truth glacier
        self.year_range = np.array(self.true_glacier['time'])[::dt]

        self.start_year = int(self.year_range[0])
        self.end_year = int(self.year_range[-1])
        self.map_resolution = self.true_glacier['x'][1] - self.true_glacier['x'][0]
        self.map_shape_x = self.true_glacier.dimensions['x'].size
        self.map_shape_y = self.true_glacier.dimensions['y'].size
        self.icemask = np.array(self.true_glacier['icemask'])[0]
        self.surface = np.array(self.true_glacier['usurf'])[0]
        self.bedrock = self.true_glacier['topg'][0]

        # sample observation points

        gx, gy = np.where(self.icemask)
        glacier_points = np.array(list(zip(gx, gy)))
        num_sample_points = int(covered_area/100 * np.sum(self.icemask))
        observation_index = np.random.choice(len(glacier_points), num_sample_points, replace=False)
        observation_points = glacier_points[observation_index]

        def get_pixel_value(point):
            x, y = point
            return self.surface[x][y]

        sorted_observation_points = sorted(observation_points, key=get_pixel_value)
        self.observation_points = np.array(sorted_observation_points)

        # placeholder for ensemble surface elevation list
        self.ensemble_usurfs = []
        self.ensemble_velo = []

    def start_ensemble(self, hyperparameter):
        import monitor_small
        # initial surface elevation
        surf_x = self.true_glacier['usurf'][0].astype(float)
        velo = self.true_glacier['velsurf_mag'][0].astype(float)

        # initial guess
        state_x = np.array(self.initial_estimate).astype(float)

        # initialise prior (uncertainty) P
        prior_x = np.zeros((len(state_x), len(state_x)))

        prior_x[0, 0] = self.initial_estimate_var[0]
        prior_x[1, 1] = self.initial_estimate_var[1]
        prior_x[2, 2] = self.initial_estimate_var[2]

        # ensemble parameters
        # number of ensemble members
        dim_z = len(self.observation_points)

        # create Ensemble Kalman Filter
        ensemble = EnKF(x=state_x, P=prior_x, dim_z=dim_z, dt=self.dt, N=self.ensemble_size,
                        hx=self.generate_observation, fx=self.forward_model,
                        start_year=self.start_year)

        # update Process noise (Q) and Observation noise (R)
        ensemble.Q = np.zeros_like(prior_x)
        ensemble.Q[0, 0] = 1000 * self.process_noise * self.dt
        ensemble.Q[1, 1] = 0.00000001 * self.process_noise * self.dt
        ensemble.Q[2, 2] = 0.00000001 * self.process_noise * self.dt

      # high means high confidence in state and low confidence in observation

        usurfs = np.array(self.true_glacier['usurf'])
        self.noisey_usruf = []
        elevation_bias_2000 = None
        for i, usurf in enumerate(usurfs):
            min = np.min(usurf[self.icemask==1])
            max = np.max(usurf[self.icemask==1])
            elevation_bias = self.icemask * usurf
            elevation_bias[self.icemask==1] = (elevation_bias[self.icemask==1]-min)/(max-min)
            if i ==0:
                elevation_bias_2000 = elevation_bias
            white_noise = np.random.normal(0, self.specal_noise, size=usurf.shape)

            noisy_usurf =  usurf + elevation_bias * self.bias + white_noise

            self.noisey_usruf.append(noisy_usurf)


        self.noisey_usruf = np.array(self.noisey_usruf)

        ensemble.R = np.eye(
            dim_z) * specal_noise * elevation_bias_2000[self.observation_points[:, 0], self.observation_points[:, 1]] * 10

        # make copy for parallel ensemble forward step
        for i in range(self.ensemble_size):
            ensemble_noisey_usruf = self.noisey_usruf[0] + np.random.normal(0, specal_noise,
                                                                            size=self.noisey_usruf[0].shape)

            self.ensemble_usurfs.append(ensemble_noisey_usruf)
            self.ensemble_velo.append(np.zeros_like(surf_x))  # + np.random.normal(0, np.sqrt(ensemble.R[0,0])))
            if not os.path.exists(f"Experiments/{i}"):
                os.makedirs(f"Experiments/{i}")
            shutil.copy2("Inversion/geology-optimized.nc", f"Experiments/{i}/init_input.nc")
            if os.path.exists(f"Experiments/{i}/iceflow-model"):
                shutil.rmtree(f"Experiments/{i}/iceflow-model")
            shutil.copytree("Inversion/iceflow-model/", f"Experiments/{i}/iceflow-model")

        self.ensemble_usurfs = np.array(self.ensemble_usurfs)
        self.ensemble_velo = np.array(self.ensemble_velo)

        # create a Monitor for visualisation
        monitor = monitor_small.Monitor(self.ensemble_size, self.true_glacier, self.observation_points, self.dt,
                                        self.synthetic, self.initial_offset, self.initial_uncertainity,
                                        self.noisey_usruf)
        # draw plot of inital state
        monitor.plot(self.year_range[0], ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)

        for year in self.year_range[:-1]:
            print("==== %i ====" % year)

            ### PREDICT ###
            start_time = time.time()
            ensemble.predict()
            print("Prediction time: ", time.time() - start_time)
            ensemble.year = year + dt

            # get observations
            usurf = self.true_glacier['usurf'][int((ensemble.year - self.start_year))]
            velo = self.true_glacier['velsurf_mag'][int((ensemble.year - self.start_year))]

            noisey_usurf = self.noisey_usruf[int((ensemble.year - self.start_year))]
            sampled_observations = noisey_usurf[self.observation_points[:, 0], self.observation_points[:, 1]]

            monitor.plot(ensemble.year, ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)

            ### UPDATE ###

            observation_noise = np.random.normal(0, specal_noise, size=(ensemble_size,) + noisey_usurf.shape)

            e_r = observation_noise[:, self.observation_points[:, 0], self.observation_points[:, 1]]

            ensemble.update(sampled_observations, e_r)
            print(ensemble.P)
            ### UPDATE ###

            self.ensemble_usurfs = np.array([noisey_usurf + noise for noise in
                                             observation_noise])  # + np.random.normal(0, np.sqrt(ensemble.R[0, 0]))

            monitor.plot(ensemble.year, ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)

        ### EVALUATION ###k
        with open('ReferenceSimulation/params.json') as f:
            params = json.load(f)
        smb = params['smb_simple_array']
        true_x = [smb[-1][3], smb[-1][1], smb[-1][2]]
        results = dict(true_parameter=true_x,
                       esti_parameter=ensemble.x.tolist(),
                       esit_var=ensemble.P.tolist(),
                       ensemble_size=self.ensemble_size,
                       dt=int(dt),
                       initial_offset=int(self.initial_offset),
                       initial_uncertainity=int(self.initial_uncertainity),
                       bias= int(self.bias),
                       specal_noise = int(self.specal_noise),
                       map_resolution=int(self.map_resolution),
                       covered_area = self.covered_area,
                       initial_estimate=[int(i) for i in self.initial_estimate],
                       initial_estimate_var=[int(j) for j in self.initial_estimate_var]
                       )

        if not os.path.exists(f"Results_{hyperparameter}/"):
            os.makedirs(f"Results_{hyperparameter}/")
        with open(
                f"Results_{hyperparameter}/result_{self.initial_offset}_{self.initial_uncertainity}_{self.bias}_{self.specal_noise}.json",
                'w') as f:
            json.dump(results, f, indent=4, separators=(',', ': '))

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
                "time_step_max": 0.2,
                }

        with open(f'Experiments/{i}/params.json', 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))

        # create new input.nc
        input_file = f"Experiments/{i}/init_input.nc"
        with xr.open_dataset(input_file) as ds:
            # load usurf from ensemble
            usurf = self.ensemble_usurfs[i]
            ds['usurf'] = xr.DataArray(usurf, dims=('y', 'x'))

            thickness = usurf - self.bedrock
            thk_da = xr.DataArray(thickness, dims=('y', 'x'))
            thk_da = xr.DataArray(thickness, dims=('y', 'x'))
            ds['thk'] = thk_da

            # ds_drop = ds.drop_vars("thkinit")
            ds.to_netcdf(f'Experiments/{i}/input_.nc')

        ### IGM RUN ###
        subprocess.run(["igm_run"], cwd=f'Experiments/{i}', shell=True)

        # update state x and return
        with xr.open_dataset(f'Experiments/{i}/output_{year}.nc') as new_ds:
            new_usurf = np.array(new_ds['usurf'][-1])
            new_velo = np.array(new_ds['velsurf_mag'][-1])

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


if __name__ == '__main__':
    from scipy.stats import qmc

    synthetic = True

    with open('ReferenceSimulation/params.json') as f:
        params = json.load(f)
        base_ela = params['smb_simple_array'][1][3]
        base_abl_grad = params['smb_simple_array'][1][1]
        base_acc_grad = params['smb_simple_array'][1][2]

    # [samplepoints^1/2, ensemble members, inital state, inital varianc]
    hyperparameter_range = {
        "Area": [1, 2, 4, 8, 16, 32, 64],
        "Observation_Interval" : [1, 2, 4, 5, 10, 20],
        "Process_Noise": [0, 0.5, 1, 2, 4],
        "Ensemble_Size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    }
    for hyperparamter in hyperparameter_range.keys():

        for value in hyperparameter_range[hyperparamter]:
            if hyperparamter=='Area':
                covered_area = value
                dt = 2
                ensemble_size = 25
                process_noise = 1

            elif hyperparamter=='Observation_Interval':
                covered_area = 16
                dt = value
                ensemble_size = 25
                process_noise = 1

            elif hyperparamter == 'Process_Noise':
                covered_area = 16
                dt = 2
                ensemble_size = 25
                process_noise = value

            elif hyperparamter == 'Ensemble_Size':
                covered_area = 16
                dt = 2
                ensemble_size = value
                process_noise = 1

            number_of_experiments = 10
            l_bounds = [0, 0, 0, 1]
            u_bounds = [100, 100, 10, 3]
            sampler = qmc.LatinHypercube(d=4)
            sample = sampler.integers(l_bounds=l_bounds, u_bounds=u_bounds, n=number_of_experiments)
            """
            points = 10
            sizes = 20
            random_dt = [10]
            offsets = 50
            uncertainities = 84
            specal_noise = 0.1
            bias = 0
            process_noise = 0.5
            sample = [[points, sizes, offsets, uncertainities]]
            """

            # for num_sample_points, ensemble_size, dt, initial_offset, initial_uncertainity in zip(points, sizes, random_dt,
            # offsets, uncertainities):
            for initial_offset, initial_uncertainity, bias, specal_noise in sample:


                sign = np.random.choice([-1, 1], 3)

                initial_est = [(base_ela + 1000 * (initial_offset / 100) )* sign[0],
                               (base_abl_grad + 0.01 * (initial_offset / 100)) * sign[1],
                                (base_acc_grad + 0.01 * (initial_offset / 100)) * sign[2]]

                initial_est_var = [initial_uncertainity ** 2 * 100,
                                   initial_uncertainity ** 2 * 0.00000001,
                                   initial_uncertainity ** 2 * 0.00000001]

                DA = DataAssimilation(int(covered_area), int(ensemble_size), int(dt), initial_est,
                                      initial_est_var, initial_offset, initial_uncertainity, specal_noise, bias, process_noise, synthetic)
                DA.start_ensemble(hyperparamter)
