import random
import shutil
import subprocess
import time
import os
import json
import netCDF4
import copy
import numpy as np
import xarray as xr
from ensemble_kalman_filter import EnsembleKalmanFilter as EnKF
from filterpy.common import outer_product_sum

os.environ['PYTHONWARNINGS'] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(1233)


class DataAssimilation:
    def __init__(self, covered_area, ensemble_size, dt, initial_estimate, initial_estimate_var, initial_offset,
                 initial_uncertainty, specal_noise, bias, process_noise, synthetic):

        # Save arguments
        self.covered_area = covered_area
        self.ensemble_size = ensemble_size
        self.dt = dt
        self.initial_estimate = initial_estimate
        self.initial_estimate_var = initial_estimate_var
        self.initial_offset = initial_offset
        self.initial_uncertainity = initial_uncertainty
        self.specal_noise = specal_noise
        self.bias = bias
        self.process_noise = process_noise
        self.synthetic = synthetic

        # placeholders
        self.ensemble_usurfs = []
        self.ensemble_velo = []

        # Change between synthetic and real observations ###
        if self.synthetic:
            self.true_glacier = netCDF4.Dataset('ReferenceSimulation/output.nc')
        else:
            self.true_glacier = netCDF4.Dataset('Hugonnet/merged_dataset.nc')

        # Extract metadata from ground truth glacier
        self.year_range = np.array(self.true_glacier['time'])[::dt]
        self.start_year = self.year_range[0]
        self.icemask = np.array(self.true_glacier['icemask'])[0]
        self.surface = np.array(self.true_glacier['usurf'])[0]
        self.bedrock = self.true_glacier['topg'][0]

        # sample observation points
        gx, gy = np.where(self.icemask)
        glacier_points = np.array(list(zip(gx, gy)))
        num_sample_points = int(covered_area / 100 * np.sum(self.icemask))
        observation_index = np.random.choice(len(glacier_points), num_sample_points, replace=False)
        observation_points = glacier_points[observation_index]

        def get_pixel_value(point):
            x, y = point
            return self.surface[x][y]

        sorted_observation_points = sorted(observation_points, key=get_pixel_value)
        self.observation_points = np.array(sorted_observation_points)

    def start_ensemble(self, hyperparameter, value):
        import monitor_small
        ### INITIALIZE ###
        # Initial estimate
        state_x = np.array(self.initial_estimate).astype(float)

        # Initialise prior (uncertainty) P
        prior_x = np.zeros((len(state_x), len(state_x)))
        prior_x[0, 0] = self.initial_estimate_var[0]
        prior_x[1, 1] = self.initial_estimate_var[1]
        prior_x[2, 2] = self.initial_estimate_var[2]

        dim_z = len(self.observation_points)
        ### Create Ensemble Kalman Filter
        ensemble = EnKF(x=state_x, P=prior_x, dim_z=dim_z, dt=self.dt, N=self.ensemble_size,
                        hx=self.generate_observation, fx=self.forward_model,
                        start_year=self.start_year)

        # update Process noise (Q)
        ensemble.Q = np.zeros_like(prior_x)
        ensemble.Q[0, 0] = 1000 * self.process_noise * self.dt
        ensemble.Q[1, 1] = 0.00000001 * self.process_noise * self.dt
        ensemble.Q[2, 2] = 0.00000001 * self.process_noise * self.dt

        # make observations noisy adn compute Observation noise (R) of EnKF
        usurfs = np.array(self.true_glacier['usurf'])
        elevation_bias_2000 = None
        self.noisey_usurf = []
        for i, usurf in enumerate(usurfs):
            # get min max elevation of glacier area
            min, max = np.min(usurf[self.icemask == 1]), np.max(usurf[self.icemask == 1])
            # elevation only on glacier
            elevation_bias = (usurf - min) / (max - min)
            # save 2000 elevation
            if i == 0: elevation_bias_2000 = elevation_bias
            # compute  white noice with specal
            white_noise = np.random.normal(0, self.specal_noise, size=usurf.shape)
            # add noises to usurf
            noisy_usurf = usurf + elevation_bias * self.bias + white_noise
            # append to noisey usurfs
            self.noisey_usurf.append(noisy_usurf)

        self.noisey_usurf = np.array(self.noisey_usurf)

        # compute R with s**2 +b**2 where b is dependent on height * 10 the maximum high elevation bias
        ensemble.R = np.eye(
            dim_z) * (specal_noise ** 2 + (
                elevation_bias_2000[self.observation_points[:, 0], self.observation_points[:, 1]] * 10) ** 2)

        ### PARALLIZE ###
        for i in range(self.ensemble_size):
            # make a copy of first usurf for every ensemble member
            self.ensemble_usurfs.append(copy.copy(self.noisey_usurf[0]))
            # make a velocity field for every ensemble member
            self.ensemble_velo.append(np.zeros_like(self.noisey_usurf[0]))
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
        monitor = monitor_small.Monitor(self.ensemble_size, self.true_glacier, self.observation_points, self.dt,
                                        self.synthetic, self.initial_offset, self.initial_uncertainity,
                                        self.noisey_usurf, self.specal_noise, self.bias, hyperparameter, value)
        # draw plot of inital state
        monitor.plot(self.year_range[0], ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)

        ### LOOP OVER YEAR RANGE ###
        if not os.path.exists(f"Results_{hyperparameter}/"):
            os.makedirs(f"Results_{hyperparameter}/")
        ### STARY LOOP ###
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

            noisey_usurf = self.noisey_usurf[int((ensemble.year - self.start_year))]
            sampled_observations = noisey_usurf[self.observation_points[:, 0], self.observation_points[:, 1]]

            monitor.plot(ensemble.year, ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)

            ### UPDATE ###

            observation_noise = np.random.normal(0, 1, size=(ensemble_size,) + noisey_usurf.shape)

            e_r = observation_noise[:, self.observation_points[:, 0], self.observation_points[:, 1]]
            R_diag = ensemble.R.diagonal()
            e_r = e_r * np.sqrt(R_diag)

            try:
                ensemble.update(sampled_observations, e_r)
            except:
                print("ERROR")

                print(self.bias, self.specal_noise, self.dt, self.covered_area,self.bias, self.specal_noise)
                with open(f"Results_{hyperparameter}/{value}/debug_info{self.bias, self.specal_noise, self.dt, self.covered_area,self.bias, self.specal_noise}.txt", "w") as file:
                    # Write each piece of information to the file
                    file.write(f"bias: {self.bias}\n")
                    file.write(f"specal_noise: {self.specal_noise}\n")
                    file.write(f"dt: {self.dt}\n")
                    file.write(f"covered_area: {self.covered_area}\n")
                    file.write(f"ensemble.sigmas: {ensemble.sigmas}\n")
                    file.write(f"process_noise: {self.process_noise}\n")
                # transform sigma points into measurement space
                sigmas_h = [ensemble.hx(ensemble.sigmas[i], i) for i in range(self.ensemble_size)]
                z_mean = np.mean(sigmas_h, axis=0)
                P_zz = (outer_product_sum(sigmas_h - z_mean) / (ensemble.N - 1)) + ensemble.R
                # Check if any row is a multiple of another row


                np.save(f"Results_{hyperparameter}/{value}/P_zz{self.bias, self.specal_noise, self.dt, self.covered_area,self.bias, self.specal_noise}.npy", P_zz)
                return
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
                       bias=int(self.bias),
                       specal_noise=int(self.specal_noise),
                       covered_area=self.covered_area,
                       initial_estimate=[int(i) for i in self.initial_estimate],
                       initial_estimate_var=[int(j) for j in self.initial_estimate_var],
                       process_noise=self.process_noise
                       )

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
                "time_step_max": 0.2,
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
            print("could not read input")

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
        #"Area": [1, 2, 4, 8, 16, 32, 64],
        "Observation_Interval": [1, 2, 4, 5, 10, 20],
        "Process_Noise": [0, 0.5, 1, 2, 4],
        "Ensemble_Size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    }
    initial_offsets = np.random.randint(0, 100, size=10)
    initial_uncertainties = np.random.randint(0, 100, size=10)
    biases = np.random.randint(0, 10, size=10)
    specal_noises = np.random.randint(1, 3, size=10)

    for hyperparameter in hyperparameter_range.keys():
        print("Start Hyperparameter: ", hyperparameter)

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

            #l_bounds = [0, 0, 0, 1]
            #u_bounds = [100, 100, 10, 3]
            #sampler = qmc.LatinHypercube(d=4)
            #sample = sampler.integers(l_bounds=l_bounds, u_bounds=u_bounds, n=number_of_experiments)
            # for num_sample_points, ensemble_size, dt, initial_offset, initial_uncertainity in zip(points, sizes, random_dt,
            # offsets, uncertainities):

            if not os.path.exists(f"Results_{hyperparameter}/{value}"):
                os.makedirs(f"Results_{hyperparameter}/{value}")

            number_of_experiments = 10
            for i in range(number_of_experiments):
                initial_offset = initial_offsets[i]
                initial_uncertainty =initial_uncertainties[i]
                bias = biases[i]
                specal_noise = specal_noises[i]

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
                results = DA.start_ensemble(hyperparameter, value)

                with open(
                        f"Results_{hyperparameter}/{value}/result_o_{initial_offset}_u_{initial_uncertainty}_b_{bias}_s_{specal_noise}.json",
                        'w') as f:
                    json.dump(results, f, indent=4, separators=(',', ': '))
