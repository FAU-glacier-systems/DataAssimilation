import copy

import matplotlib.pyplot as plt

import monitor
import os
import json
import netCDF4
import numpy as np
import xarray as xr
from ensemble_kalman_filter import EnsembleKalmanFilter as EnKF
import warnings
os.environ['PYTHONWARNINGS'] ="ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(1334)

# load true glacier
true_glacier = netCDF4.Dataset('ReferenceRun/output.nc')

# extract metadata from ground truth glacier
year_range = np.array(true_glacier['time'])
start_year = int(year_range[0])
end_year = int(year_range[-1])
map_resolution = true_glacier['x'][1] - true_glacier['x'][0]
map_shape_x = true_glacier.dimensions['x'].size
map_shape_y = true_glacier.dimensions['y'].size
icemask = np.array(true_glacier['icemask'])[0]
surface = np.array(true_glacier['usurf'])[0]
bedrock = true_glacier['topg'][0]

# sample observation points
sp = 1000
gx, gy = np.where(icemask)
glacier_points = np.array(list(zip(gx, gy)))
observation_index = np.random.choice(len(glacier_points), sp, replace=False)
observation_points = glacier_points[observation_index]

def get_pixel_value(point):
    x, y = point
    return surface[x][y]

sorted_observation_points = sorted(observation_points, key=get_pixel_value)
observation_points = np.array(sorted_observation_points)

# placeholder for ensemble surface elevation list
ensemble_usurfs = []


def forward_model(state_x, dt, i, year):
    # create new params.json
    year_next = year + dt

    ela, grad_abl, grad_acc = state_x[[0, 1, 2]]

    data = {"modules_preproc": ["load_ncdf"],
            "modules_process": ["smb_simple", "iceflow", "time", "thk"],
            "modules_postproc": ["write_ncdf"],
            "smb_simple_array": [
                ["time", "gradabl", "gradacc", "ela", "accmax"],
                [year, grad_abl, grad_acc, ela, 2.0],
                [year_next, grad_abl, grad_acc, ela, 2.0]],
            "iflo_emulator": "../Inversion/iceflow-model",
            "lncd_input_file": 'input.nc',
            "wncd_output_file": 'output_' + str(year) + '.nc',
            "time_start": year,
            "time_end": year_next,
            "iflo_retrain_emulator_freq": 0}

    with open('Experiments/params.json', 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))

    # create new input.nc
    input_file = "ReferenceRun/input_merged.nc"
    ds = xr.open_dataset(input_file)

    # load usurf from ensemble
    usurf = ensemble_usurfs[i]
    ds['usurf'] = xr.DataArray(usurf, dims=('y', 'x'))

    thickness = usurf - bedrock
    thk_da = xr.DataArray(thickness, dims=('y', 'x'))
    ds['thk'] = thk_da

    #ds_drop = ds.drop_vars("thkinit")
    ds.to_netcdf('Experiments/input.nc')
    ds.close()

    ### IGM RUN ###
    os.chdir("Experiments")
    os.system("igm_run")
    os.chdir("..")

    # update state x and return
    new_ds = xr.open_dataset('Experiments/output_' + str(year) + '.nc')
    new_usurf = np.array(new_ds['usurf'][-1])

    ensemble_usurfs[i] = new_usurf

    return state_x


def generate_observation(state_x, i):
    """
    h(x)
    :returns thickness map as 1D array
    """
    usurf = ensemble_usurfs[i]
    modelled_observations = usurf[observation_points[:, 0], observation_points[:, 1]]

    return modelled_observations


if __name__ == '__main__':

    # initial surface elevation
    surf_x = true_glacier['usurf'][0].astype(float)

    # initial guess
    state_x = np.array([2900, 0.011, 0.0039]).astype(float)

    # initialise prior (uncertainty) P
    prior_x = np.zeros((len(state_x), len(state_x)))

    prior_x[0, 0] = 100
    prior_x[1, 1] = 0.000001
    prior_x[2, 2] = 0.000001

    # ensemble parameters
    N = 20  # number of ensemble members
    dt = int(true_glacier['time'][1] - true_glacier["time"][0])  # time step [years]
    dim_z = len(observation_points)

    # create Ensemble Kalman Filter
    ensemble = EnKF(x=state_x, P=prior_x, dim_z=dim_z, dt=dt, N=N, hx=generate_observation, fx=forward_model, start_year=start_year)
    # update Process noise (Q) and Observation noise (R)
    ensemble.Q = np.zeros_like(prior_x)
    #ensemble.Q[0,0] = 100
    ensemble.R = np.eye(dim_z) # high means high confidence in state and low confidence in observation


    for i in range(N):
        ensemble_usurfs.append(copy.copy(surf_x)) #+ np.random.normal(0, np.sqrt(ensemble.R[0,0])))

    # create a Monitor for visualisation
    monitor = monitor.Monitor(N, true_glacier)
    # draw plot of inital state
    monitor.plot(year_range[0], ensemble.sigmas, ensemble_usurfs)

    for year in year_range[:-1]:
        print("==== %i ====" % year)

        ### PREDICT ###
        ensemble.predict()
        monitor.plot(year + dt, ensemble.sigmas, ensemble_usurfs)

        ### UPDATE ###
        ensemble.year = year + dt
        usurf = true_glacier['usurf'][int((year - start_year) / dt) + 1]
        real_observations = usurf[observation_points[:, 0], observation_points[:, 1]]
        ensemble.update(real_observations)
        for i in range(N):
            ensemble_usurfs[i] = copy.copy(usurf) #+ np.random.normal(0, np.sqrt(ensemble.R[0, 0]))

        monitor.plot(year + dt, ensemble.sigmas, ensemble_usurfs)


