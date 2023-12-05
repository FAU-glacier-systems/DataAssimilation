import time

import netCDF4
import numpy as np
from ensemble_kalman_filter import EnsembleKalmanFilter as EnKF
import json
import os
import xarray as xr
import monitor
from skimage import measure
import matplotlib.pyplot as plt

np.random.seed(1234)

true_glacier = netCDF4.Dataset('data_synthetic_observations/default/output.nc')

# extract meta_data
map_resolution = 200
map_shape_x = true_glacier.dimensions['x'].size
map_shape_y = true_glacier.dimensions['y'].size
year_range = np.array(true_glacier['time'])
icemask = np.array(true_glacier['icemask'][0])==1
bedrock = true_glacier['topg'][0]
start_year = int(year_range[0])
end_year = int(year_range[-1])


def glacier_properties(usurf):
    """
    get real observation
    :returns area [km²] and volume [km³] of the ground truth in given year
             and thickness map
    """

    thk_map = usurf - bedrock
    volume = np.sum(thk_map) * map_resolution ** 2 / 1000 ** 3
    area = len(thk_map[thk_map > 0.5]) * map_resolution ** 2 / 1000 ** 2
    icemask = thk_map > 0.5
    contours = measure.find_contours(icemask, 0)
    outline_len = np.sum([len(line) for line in contours]) * map_resolution / 1000

    return area, volume, outline_len


def forward_model(state_x, dt):
    # create new params.json
    start_time = time.time()
    year = round(state_x[0])
    year_next = year + dt
    ela = state_x[1]
    grad_abl = state_x[2]
    grad_acc = state_x[3]

    data = {"modules_preproc": ["load_ncdf"],
            "modules_process": ["smb_simple", "iceflow", "time", "thk"],
            "modules_postproc": ["write_ncdf"],
            "smb_simple_array": [
                ["time", "gradabl", "gradacc", "ela", "accmax"],
                [year, grad_abl, grad_acc, ela, 2.0],
                [year_next, grad_abl, grad_acc, ela, 2.0]
            ],
            "lncd_input_file": 'input.nc',
            "wncd_output_file": 'output_' + str(year) + '.nc',
            "time_start": year,
            "time_end": year_next
            }

    with open('params.json', 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))

    # create new input.nc
    input_file = "data_synthetic_observations/default/input_saved.nc"
    ds = xr.open_dataset(input_file)

    usurf = state_x[4:].reshape((map_shape_y, map_shape_x))

    thickness = usurf - bedrock
    #thickness[icemask==0] = 0
    thk_da = xr.DataArray(thickness, dims=('y', 'x'))
    ds['thk'] = thk_da

    icemask = np.zeros_like(thickness)
    icemask[thickness > 0.01] = 1
    icemask_da = xr.DataArray(icemask, dims=('y', 'x'))
    ds['icemask'] = icemask_da

    ds['usurf'] = xr.DataArray(usurf, dims=('y', 'x'))
    ds_drop = ds.drop_vars("thkinit")
    ds_drop.to_netcdf('input.nc')
    ds_drop.close()

    print("write input files: %f" % (time.time() - start_time))
    ### IGM RUN ###
    start_time = time.time()
    os.system("igm_run")
    print("igm_run: %f" % (time.time() - start_time))
    # update state x and return
    start_time = time.time()
    new_ds = xr.open_dataset('output_' + str(year) + '.nc')
    new_usurf = np.array(new_ds['usurf'][-1])
    #new_usurf[icemask==0] = 0
    state_x[4:] = new_usurf.flatten()
    state_x[0] = year_next
    print("read output files: %f" % (time.time() - start_time))

    return state_x


def generate_observation(state_x):
    """
    h(x)
    :returns thickness map as 1D array
    """
    surf_flat = state_x[4:]
    return surf_flat


if __name__ == '__main__':
    # load true glacier

    # initialise state vector xa
    surf_x = true_glacier['usurf'][0].astype(float)
    smb_x = np.array([2950, 0.01, 0.004]).astype(float)
    state_x = np.concatenate(([int(start_year)], smb_x, surf_x.flatten()))

    # initialise prior (uncertainty)
    prior_x = np.zeros((len(state_x), len(state_x)))
    prior_x[0, 0] = 0.0
    prior_x[1, 1] = 1000
    prior_x[2, 2] = 0.000001
    prior_x[3, 3] = 0.000001

    # ensemble parameters
    N = 10  # number of ensemble members
    dt = int(true_glacier['time'][1] - true_glacier["time"][0]) # time step [years]
    dim_z = map_shape_x * map_shape_y

    ensemble = EnKF(x=state_x, P=prior_x, dim_z=dim_z, dt=dt, N=N, hx=generate_observation, fx=forward_model)
    ensemble.R = np.eye(dim_z)  # high means high confidence in state and low confidence in observation
    ensemble.Q = np.zeros_like(prior_x)

    hist_true_y = []
    hist_observ = []
    for year in year_range:
        usurf = true_glacier['usurf'][int((year - start_year) / dt)]#.astype(np.float16)
        area, volume, outline_len = glacier_properties(usurf)
        hist_true_y.append([area, volume, outline_len])

        #usurf_noise = usurf + np.random.normal(0, 1, usurf.shape)
        #area_obs, volume_obs, outline_len_obs = glacier_properties(usurf_noise)
        #hist_observ.append([area_obs, volume_obs, outline_len_obs])

    monitor = monitor.Monitor(N, year_range, true_glacier, hist_true_y, map_resolution, dt)

    ensemble_y = [glacier_properties(e[4:].reshape((map_shape_y, map_shape_x))) for e in ensemble.sigmas]
    monitor.plot(year_range[0], np.mean(ensemble.sigmas, axis=0), ensemble.sigmas, ensemble_y)

    for year in year_range[:-1]:
        print("==== %i ====" % year)
        ### PREDICT ###
        ensemble.predict()

        ensemble_y = [glacier_properties(e[4:].reshape((map_shape_y, map_shape_x))) for e in ensemble.sigmas]
        monitor.plot(year + dt, ensemble.x, ensemble.sigmas, ensemble_y)

        ### UPDATE ###
        usurf = true_glacier['usurf'][int((year - start_year) / dt) + 1].astype(np.float16)
        ensemble.update(np.asarray(usurf.flatten()))

        ensemble_y = [glacier_properties(e[4:].reshape((map_shape_y, map_shape_x))) for e in ensemble.sigmas]
        monitor.plot(year + dt, ensemble.x, ensemble.sigmas, ensemble_y)
