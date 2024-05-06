import random
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
from filterpy.common import outer_product_sum
import matplotlib.pyplot as plt

os.environ['PYTHONWARNINGS'] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#np.random.seed(1233)
np.random.seed(37)

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
            self.true_glacier = nc.Dataset('ReferenceSimulation/output.nc')
        else:
            self.true_glacier = nc.Dataset('Hugonnet/merged_dataset.nc')

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

    def start_ensemble(self, hyperparameter, value, smb):
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
        if synthetic:
            usurfs = np.array(self.true_glacier['usurf'])
            elevation_2000 = usurfs[0]
            min, max = np.min(elevation_2000[self.icemask == 1]), np.max(elevation_2000[self.icemask == 1])
            elevation_bias_single = (elevation_2000 - min) / (max - min)
            observation_error = elevation_bias_single * self.bias + np.random.normal(0, self.specal_noise, elevation_bias_single.shape)
            noisy_usurf = [usurf+observation_error*np.random.normal(0,1) for usurf in usurfs]
            self.noisy_usurf = np.array(noisy_usurf)

        else:
            self.noisy_usurf = np.array(self.true_glacier['usurf'])
            observation_error = np.array(self.true_glacier['obs_error'][0])

        observation_error[self.icemask == 0] = 0
        ensemble.R = np.eye(dim_z) * observation_error[self.observation_points[:, 0], self.observation_points[:, 1]]**2


        ### PARALLIZE ###
        for i in range(self.ensemble_size):
            # make a copy of first usurf for every ensemble member

            observation_noise_samples = np.random.normal(0, 1) * observation_error

            self.ensemble_usurfs.append(copy.copy(self.noisy_usurf[0]) + observation_noise_samples)
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
        monitor = monitor_small.Monitor(self.ensemble_size, self.true_glacier, self.observation_points, self.dt,
                                        self.process_noise, observation_error,
                                        self.synthetic, self.initial_offset, self.initial_uncertainity, smb,
                                        self.noisy_usurf, self.specal_noise, self.bias, hyperparameter, value)
        # draw plot of inital state
        monitor.plot(self.year_range[0], ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)
        ### LOOP OVER YEAR RANGE ###
        if not os.path.exists("Results"):
            os.makedirs("Results")
        if not os.path.exists(f"Results/Results_{hyperparameter}/"):
            os.makedirs(f"Results/Results_{hyperparameter}/")
        ### START LOOP ###
        for year in self.year_range[:-1]:
            print("==== %i ====" % year)

            ### PREDICT ###
            start_time = time.time()
            ensemble.predict()
            print("Prediction time: ", time.time() - start_time)
            ensemble.year = year + dt

            # get observations
            #usurf = self.true_glacier['usurf'][int((ensemble.year - self.start_year))]
            velo = self.true_glacier['velsurf_mag'][int((ensemble.year - self.start_year))]

            monitor.plot(ensemble.year, ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)

            ### UPDATE ###

            # get the noisey observation
            print(int((ensemble.year - self.start_year)))
            noisey_usurf = self.noisy_usurf[int((ensemble.year - self.start_year))]
            sampled_observations = noisey_usurf[self.observation_points[:, 0], self.observation_points[:, 1]]

            # sample random noise for each ensemlbe memember to add to the difference
            # interpreation: every ensemble member gets a slightly different observation

            observation_noise_samples = np.array([r_norm * observation_error for r_norm in np.random.normal(0, 1, ensemble_size)])
            e_r = observation_noise_samples[:, self.observation_points[:, 0], self.observation_points[:, 1]]
            #e_r = np.ones_like(sampled_observations) * observation_noise_samples
            #R_diag = ensemble.R.diagonal()
            #e_r = e_r * np.sqrt(R_diag)

            # update the hidden parameters
            ensemble.update(sampled_observations, e_r)

            # update the surface elevation
            self.ensemble_usurfs = np.array([noisey_usurf + noise for noise in observation_noise_samples])
            #self.ensemble_usurfs = np.array([copy.copy(noisey_usurf) for i in  range(ensemble_size)])

            # plot the update
            monitor.plot(ensemble.year, ensemble.sigmas, self.ensemble_usurfs, self.ensemble_velo)

        ### EVALUATION ###


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
                #"time_step_max": 0.2,
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


if __name__ == '__main__':
    from scipy.stats import qmc

    synthetic = False

    if synthetic:
        with open('ReferenceSimulation/params.json') as f:
            params = json.load(f)
            smb = params['smb_simple_array']
            base_ela = smb[1][3]
            base_abl_grad = smb[1][1]
            base_acc_grad = smb[1][2]

        # [samplepoints^1/2, ensemble members, inital state, inital varianc]

        hyperparameter_range = {
            #"Area": [1, 2, 4, 8, 16, 32, 64],
            #"Observation_Interval": [1, 2],
            #"Process_Noise": [0, 0.5, 2, 4],
            #"Ensemble_Size": [5, 10, 20, 30, 40, 50],
            #"initial_offset" : [0,20,40,60,80,100],
            #"initial_uncertainty": [100],
            #"bias": [0, 2, 4, 6, 8, 10],
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


                if not os.path.exists(f"Results/Results_{hyperparameter}/{value}"):
                    os.makedirs(f"Results/Results_{hyperparameter}/{value}")


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
                    results = DA.start_ensemble(hyperparameter, value, smb)

                    with open(
                            f"Results/Results_{hyperparameter}/{value}/result_o_{initial_offset}_u_{initial_uncertainty}_b_{bias}_s_{specal_noise}.json",
                            'w') as f:
                        json.dump(results, f, indent=4, separators=(',', ': '))
    else:

        smb = [['time', 'gradabl', 'gradacc', 'ela', 'accmax'],
               [2000, 0.0082, 0.0016, 2966, 100],
               [2020, 0.0082, 0.0016, 2966, 100]]
        covered_area = 50
        ensemble_size = 35
        dt = 4
        initial_est = [2500, 0.009, 0.005]
        initial_uncertainty = 30
        initial_est_var = [initial_uncertainty ** 2 * 100,
                           initial_uncertainty ** 2 * 0.00000001,
                           initial_uncertainty ** 2 * 0.00000001]
        initial_offset = initial_est[0]
        specal_noise = 10
        bias = 0
        process_noise = 0
        hyperparameter = "real"

        value = True

        DA = DataAssimilation(int(covered_area), int(ensemble_size), int(dt), initial_est,
                              initial_est_var, initial_offset, initial_uncertainty, specal_noise, bias,
                              process_noise, synthetic)
        results = DA.start_ensemble(hyperparameter, value, smb)

        with open(f"Results/Results_{hyperparameter}/{value}/result_o_{initial_offset}_u_{initial_uncertainty}_b_{bias}_s_{specal_noise}.json",'w') as f:
            json.dump(results, f, indent=4, separators=(',', ': '))