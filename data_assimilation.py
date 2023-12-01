import netCDF4
import numpy as np
from filterpy.kalman import EnsembleKalmanFilter as EnKF
import igm.igm_run
import json
import os
import xarray as xr
import monitor
from skimage import measure
import matplotlib.pyplot as plt

np.random.seed(1234)

true_glacier = netCDF4.Dataset('data_synthetic_observations/output.nc')

# extract meta_data
map_resolution = 200
map_shape_x = true_glacier.dimensions['x'].size
map_shape_y = true_glacier.dimensions['y'].size
year_range = np.array(true_glacier['time'])
start_year = int(year_range[0])
end_year = int(year_range[-1])


def glacier_properties(thk_map):
    """
    get real observation
    :returns area [km²] and volume [km³] of the ground truth in given year
             and thickness map
    """
    thk_mask = thk_map > 0.01
    area = len(thk_map[thk_mask]) * map_resolution ** 2 / 1000 ** 2
    volume = sum(thk_map[thk_mask]) * map_resolution ** 2 / 1000 ** 3
    contours = measure.find_contours(thk_mask, 0)
    outline_len = np.sum([len(line) for line in contours]) * map_resolution / 1000

    return area, volume, outline_len


def forward_model(state_x, dt):
    # create new params.json
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
    input_file = "data_synthetic_observations/input_cropped.nc"
    ds = xr.open_dataset(input_file)

    thickness = state_x[4:].reshape((map_shape_y, map_shape_x))

    bedrock = ds['usurf'] - ds['thk']
    thk_da = xr.DataArray(thickness, dims=('y', 'x'))
    ds['thk'] = thk_da

    icemask = np.zeros_like(thickness)
    icemask[thickness > 0.01] = 1
    icemask_da = xr.DataArray(icemask, dims=('y', 'x'))
    ds['icemask'] = icemask_da

    usurf = bedrock + thickness
    ds['usurf'] = usurf

    ds.to_netcdf('input.nc')
    ds.close()
    ### IGM RUN ###
    os.system("igm_run")

    # update state x and return
    new_ds = xr.open_dataset('output_' + str(year) + '.nc')
    new_thk = np.array(new_ds['thk'][-1])
    state_x[4:] = new_thk.flatten()
    state_x[0] = year_next
    return state_x


def generate_observation(state_x):
    """
    h(x)
    :returns thickness map as 1D array
    """
    thickness_flat = state_x[4:]
    thickness_flat[thickness_flat<0.01] = 0
    return thickness_flat


if __name__ == '__main__':
    # load true glacier

    # initialise state vector x
    thk_x = true_glacier['thk'][0].astype(float)
    smb_x = np.array([2950, 0.01, 0.004]).astype(float)
    state_x = np.concatenate(([int(start_year)], smb_x, thk_x.flatten()))

    # initialise prior (uncertainty)
    prior_x = np.zeros((len(state_x), len(state_x)))
    prior_x[0, 0] = 0.0
    prior_x[1, 1] = 1000
    prior_x[2, 2] = 0.000001
    prior_x[3, 3] = 0.000001

    # ensemble parameters
    N = 10  # number of ensemble members
    dt = 10  # time step [years]
    dim_z = map_shape_x * map_shape_y

    ensemble = EnKF(x=state_x, P=prior_x, dim_z=dim_z, dt=dt, N=N, hx=generate_observation, fx=forward_model)
    ensemble.R = np.eye(dim_z)  # high means high confidence in state and low confidence in observation
    ensemble.Q = np.zeros_like(prior_x)
    #ensemble.Q = prior_x

    hist_true_y = []

    for year in year_range:
        thk_map = true_glacier['thk'][int((year - start_year) / dt)].astype(np.float16)
        area, volume, outline_len = glacier_properties(thk_map)
        hist_true_y.append([area, volume, outline_len])

    monitor = monitor.Monitor(N, year_range, true_glacier, hist_true_y, map_resolution, dt)
    ensemble_y = [glacier_properties(e[4:].reshape((map_shape_y, map_shape_x))) for e in ensemble.sigmas]
    monitor.plot(year_range[0], state_x, ensemble.sigmas, ensemble_y)

    for year in year_range[:-1]:
        print("==== %i ====" % year)
        thk_map = true_glacier['thk'][int((year - start_year) / dt) + 1].astype(np.float16)


        ensemble.predict()
        ensemble.update(np.asarray(thk_map.flatten()))
        ensemble_y = [glacier_properties(e[4:].reshape((map_shape_y, map_shape_x))) for e in ensemble.sigmas]
        monitor.plot(year + dt, ensemble.x, ensemble.sigmas, ensemble_y)
