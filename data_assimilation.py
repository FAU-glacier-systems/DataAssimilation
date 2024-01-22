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

np.random.seed(1234)

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
sp = 100
gx, gy = np.where(icemask)
glacier_points = np.array(list(zip(gx, gy)))
observation_index = np.random.choice(len(glacier_points), sp, replace=False)
observation_points = glacier_points[observation_index]
"""
fig, ax = plt.subplots(figsize=[5,10])
ax.imshow(surface, cmap='Blues_r')
ax.invert_yaxis()
ax.scatter(observation_points[:, 1], observation_points[:,0],
           marker='s', edgecolors='red', facecolors='none', s=10)
plt.savefig('observation_points.png')
"""



def forward_model(state_x, dt):
    # create new params.json
    year = round(state_x[0])
    year_next = year + dt

    ela, grad_abl, grad_acc = state_x[[1, 2, 3]]
    grad_abl = state_x[2]
    grad_acc = state_x[3]

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
            "time_end": year_next}

    with open('Experiments/params.json', 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))

    # create new input.nc
    input_file = "Inversion/input_saved.nc"
    ds = xr.open_dataset(input_file)

    usurf = state_x[4:].reshape((map_shape_y, map_shape_x))
    ds['usurf'] = xr.DataArray(usurf, dims=('y', 'x'))

    thickness = usurf - bedrock
    thk_da = xr.DataArray(thickness, dims=('y', 'x'))
    ds['thk'] = thk_da

    ds_drop = ds.drop_vars("thkinit")
    ds_drop.to_netcdf('Experiments/input.nc')
    ds_drop.close()

    ### IGM RUN ###
    os.chdir("Experiments")
    os.system("igm_run")
    os.chdir("..")

    # update state x and return
    new_ds = xr.open_dataset('Experiments/output_' + str(year) + '.nc')
    new_usurf = np.array(new_ds['usurf'][-1])

    state_x[0] = year_next
    state_x[4:] = new_usurf.flatten()

    return state_x


def generate_observation(state_x):
    """
    h(x)
    :returns thickness map as 1D array
    """
    surf_flat = state_x[4:]
    usurf = surf_flat.reshape((map_shape_y, map_shape_x))
    modelled_observations = usurf[observation_points[:, 0], observation_points[:, 1]]

    return modelled_observations


if __name__ == '__main__':

    # initial surface elevation
    surf_x = true_glacier['usurf'][0].astype(float)

    # initial guess
    smb_x = np.array([2950, 0.011, 0.0039]).astype(float)

    # initial state_x vector
    state_x = np.concatenate(([int(start_year)], smb_x, surf_x.flatten()))

    # initialise prior (uncertainty) P
    prior_x = np.zeros((len(state_x), len(state_x)))
    prior_x[0, 0] = 0.0
    prior_x[1, 1] = 1000
    prior_x[2, 2] = 0.000001
    prior_x[3, 3] = 0.000001

    # ensemble parameters
    N = 30  # number of ensemble members
    dt = int(true_glacier['time'][1] - true_glacier["time"][0])  # time step [years]
    dim_z = len(observation_points)

    # create Ensemble Kalman Filter
    ensemble = EnKF(x=state_x, P=prior_x, dim_z=dim_z, dt=dt, N=N, hx=generate_observation, fx=forward_model)
    # update Process noise (Q) and Observation noise (R)
    ensemble.Q = np.zeros_like(prior_x)
    ensemble.R = np.eye(dim_z)  # high means high confidence in state and low confidence in observation

    # create a Monitor for visualisation
    monitor = monitor.Monitor(N, true_glacier)
    # draw plot of inital state
    monitor.plot(year_range[0], ensemble.sigmas)

    for year in year_range[:-1]:
        print("==== %i ====" % year)

        ### PREDICT ###
        ensemble.predict()
        monitor.plot(year + dt, ensemble.sigmas)

        ### UPDATE ###
        usurf = true_glacier['usurf'][int((year - start_year) / dt) + 1]
        real_observations = usurf[observation_points[:, 0], observation_points[:, 1]]
        ensemble.update(real_observations)
        monitor.plot(year + dt, ensemble.sigmas)
