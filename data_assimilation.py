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


# np.random.seed(32)


class DataAssimilation:
    def __init__(self, num_sample_points, ensemble_size, dt):
        # load true glacier

        self.num_sample_points = num_sample_points
        self.ensemble_size = ensemble_size
        self.dt = dt

        self.true_glacier = netCDF4.Dataset('ReferenceRun/output.nc')
        #self.true_glacier = netCDF4.Dataset('Hugonnet/merged_dataset.nc')
        self.synthetic = True
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
        observation_index = np.random.choice(len(glacier_points), num_sample_points, replace=False)
        observation_points = glacier_points[observation_index]

        def get_pixel_value(point):
            x, y = point
            return self.surface[x][y]

        sorted_observation_points = sorted(observation_points, key=get_pixel_value)
        self.observation_points = np.array(sorted_observation_points)

        # placeholder for ensemble surface elevation list
        self.ensemble_usurfs = []

    def start_ensemble(self):
        import monitor
        # initial surface elevation
        surf_x = self.true_glacier['usurf'][0].astype(float)

        # initial guess
        state_x = np.array([3000, 0.01, 0.002]).astype(float)

        # initialise prior (uncertainty) P
        prior_x = np.zeros((len(state_x), len(state_x)))

        prior_x[0, 0] = 1000
        prior_x[1, 1] = 0.00001
        prior_x[2, 2] = 0.00001

        # ensemble parameters
        # number of ensemble members
        # dt = int(self.true_glacier['time'][1] - self.true_glacier["time"][0])  # time step [years]
        dim_z = len(self.observation_points)

        # create Ensemble Kalman Filter
        ensemble = EnKF(x=state_x, P=prior_x, dim_z=dim_z, dt=self.dt, N=self.ensemble_size,
                        hx=self.generate_observation, fx=self.forward_model,
                        start_year=self.start_year)
        # update Process noise (Q) and Observation noise (R)
        ensemble.Q = np.zeros_like(prior_x)
        #ensemble.Q[0,0] = 100
        ensemble.R = np.eye(dim_z)  # high means high confidence in state and low confidence in observation

        for i in range(self.ensemble_size):
            self.ensemble_usurfs.append(copy.copy(surf_x))  # + np.random.normal(0, np.sqrt(ensemble.R[0,0])))
            if not os.path.exists(f"Experiments/{i}"):
                os.makedirs(f"Experiments/{i}")
            shutil.copy2("ReferenceRun/input_merged.nc", f"Experiments/{i}/init_input.nc")
            if os.path.exists(f"Experiments/{i}/iceflow-model"):
                shutil.rmtree(f"Experiments/{i}/iceflow-model")
            shutil.copytree("Inversion/iceflow-model/", f"Experiments/{i}/iceflow-model")

        # create a Monitor for visualisation
        monitor = monitor.Monitor(self.ensemble_size, self.true_glacier, self.observation_points, self.dt, sythetic=self.synthetic)
        # draw plot of inital state
        monitor.plot(self.year_range[0], ensemble.sigmas, self.ensemble_usurfs)

        for year in self.year_range[:-1]:
            print("==== %i ====" % year)

            ### PREDICT ###
            start_time = time.time()
            ensemble.predict()
            print("Prediction time: ", time.time() - start_time)
            ensemble.year = year + dt
            monitor.plot(ensemble.year, ensemble.sigmas, self.ensemble_usurfs)

            ### UPDATE ###
            usurf = self.true_glacier['usurf'][int((ensemble.year - self.start_year))]

            real_observations = usurf[self.observation_points[:, 0], self.observation_points[:, 1]]
            ensemble.update(real_observations)
            for i in range(self.ensemble_size):
                self.ensemble_usurfs[i] = copy.copy(usurf)  # + np.random.normal(0, np.sqrt(ensemble.R[0, 0]))

            monitor.plot(ensemble.year, ensemble.sigmas, self.ensemble_usurfs)

        ### EVALUATION ###
        with open('ReferenceRun/params.json') as f:
            params = json.load(f)
        smb = params['smb_simple_array']
        true_x = [smb[-1][3], smb[-1][1], smb[-1][2]]
        results = dict(true_parameter=true_x,
                       esti_parameter=ensemble.x.tolist(),
                       esit_var=ensemble.P.tolist(),
                       ensemble_size=self.ensemble_size,
                       dt=int(dt),
                       map_resolution=int(self.map_resolution),
                       num_sample_points=self.num_sample_points,
                       )

        with open(f"Experiments/result_{self.num_sample_points}_{dt}_{self.ensemble_size}.json", 'w') as f:
            json.dump(results, f, indent=4, separators=(',', ': '))

    def forward_model(self, state_x, dt, i, year):
        # create new params.json
        year_next = year + dt

        ela, grad_abl, grad_acc = state_x[[0, 1, 2]]

        data = {"modules_preproc": ["load_ncdf"],
                "modules_process": ["smb_simple", "iceflow", "time", "thk"],
                "modules_postproc": ["write_ncdf"],
                "smb_simple_array": [
                    ["time", "gradabl", "gradacc", "ela", "accmax"],
                    [year, grad_abl, grad_acc, ela, 99],
                    [year_next, grad_abl, grad_acc, ela, 99]],
                "iflo_emulator": "iceflow-model",
                "lncd_input_file": f'input_.nc',
                "wncd_output_file": f'output_{year}.nc',
                "time_start": year,
                "time_end": year_next,
                "iflo_retrain_emulator_freq": 0}

        with open(f'Experiments/{i}/params.json', 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))

        # create new input.nc
        input_file = f"Experiments/{i}/init_input.nc"
        ds = xr.open_dataset(input_file)

        # load usurf from ensemble
        usurf = self.ensemble_usurfs[i]
        ds['usurf'] = xr.DataArray(usurf, dims=('y', 'x'))

        thickness = usurf - self.bedrock
        thk_da = xr.DataArray(thickness, dims=('y', 'x'))
        ds['thk'] = thk_da

        # ds_drop = ds.drop_vars("thkinit")
        ds.to_netcdf(f'Experiments/{i}/input_.nc')
        ds.close()

        ### IGM RUN ###
        subprocess.run(["igm_run"], cwd=f'Experiments/{i}', shell=True)

        # update state x and return
        new_ds = xr.open_dataset(f'Experiments/{i}/output_{year}.nc')
        new_usurf = np.array(new_ds['usurf'][-1])

        self.ensemble_usurfs[i] = new_usurf

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

    number_of_experiments = 100
    l_bounds = [10, 4]
    u_bounds = [44, 30]
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.integers(l_bounds=l_bounds, u_bounds=u_bounds, n=number_of_experiments)
    random_dt = np.random.choice([1, 2, 4, 5, 10, 20], size=number_of_experiments)

    #sample = [[1000, 20]]
    #random_dt = [5]

    for (num_sample_points, ensemble_size), dt in zip(sample, random_dt):
        num_sample_points = num_sample_points**2
        print(num_sample_points, ensemble_size, int(dt))
        DA = DataAssimilation(int(num_sample_points), int(ensemble_size), int(dt))
        DA.start_ensemble()
