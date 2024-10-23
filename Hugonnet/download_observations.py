from oggm import utils
from netCDF4 import Dataset
import numpy as np
import argparse
import json
import numpy.ma as ma


def download_observations(params):
    # File paths
    oggm_shop_file = params['oggm_shop_file']

    # Open datasetsmer
    oggm_shop_ds = Dataset(oggm_shop_file, 'r')

    # Geodetic mass balance data
    geodetic_mb = utils.get_geodetic_mb_dataframe()
    geodetic_mb = geodetic_mb[geodetic_mb.index == params['RGI_ID']]
    geodetic_mb_2020 = geodetic_mb[geodetic_mb['period'] == '2000-01-01_2020-01-01']
    geodetic_mb_dmdtda_err = geodetic_mb_2020['err_dmdtda'].values[0]

    # Convert mass to volume
    geodetic_mb_dmdtda_err /= 0.91
    dhdt_err = np.array(oggm_shop_ds.variables['icemask'][:]) * geodetic_mb_dmdtda_err


    dhdt = np.array(oggm_shop_ds.variables['dhdt'][:])

    time_range = np.arange(2000, 2021).astype(float)
    usurf_2000 = oggm_shop_ds.variables['usurf'][:]
    thk_2000 = oggm_shop_ds.variables['thkinit'][:]
    usurf_change = []
    thk_change = []
    dhdt_errors = []
    bedrock = usurf_2000 - thk_2000

    for i, time in enumerate(time_range):
        usurf_i = usurf_2000 + dhdt * i
        usurf_i = np.maximum(bedrock, usurf_i)
        dhdt_err_i = dhdt_err * i
        usurf_change.append(usurf_i)
        thk_change.append(usurf_i - bedrock)
        dhdt_errors.append(dhdt_err_i)

    usurf_change = np.array(usurf_change)
    thk_change = np.array(thk_change)
    dhdt_errors = np.array(dhdt_errors)

    year_range = 21

    dhdt_data = np.array([dhdt] * year_range)
    error_variable = np.array(dhdt_errors)

    topg_data = np.array([bedrock] * year_range)
    icemask_data = np.array([oggm_shop_ds.variables['icemask'][:]] * year_range)

    smb = np.zeros_like(dhdt)
    smb_data = np.array([smb] * year_range)

    uvelo = oggm_shop_ds.variables['uvelsurfobs'][:]
    vvelo = oggm_shop_ds.variables['vvelsurfobs'][:]
    velo = ma.sqrt(uvelo ** 2 + vvelo ** 2)

    velo_data = np.array([velo] * year_range)

    # Create a new netCDF file
    with Dataset(params['output_file'], 'w') as merged_ds:
        # Create dimensions
        merged_ds.createDimension('time', len(time_range))
        merged_ds.createDimension('x', oggm_shop_ds.dimensions['x'].size)
        merged_ds.createDimension('y', oggm_shop_ds.dimensions['y'].size)

        # Create variables
        time_var = merged_ds.createVariable('time', 'f4', ('time',))
        x_var = merged_ds.createVariable('x', 'f4', ('x',))
        y_var = merged_ds.createVariable('y', 'f4', ('y',))
        thk_var = merged_ds.createVariable('thk', 'f4', ('time', 'y', 'x'))
        usurf_var = merged_ds.createVariable('usurf', 'f4', ('time', 'y', 'x'))
        topg_var = merged_ds.createVariable('topg', 'f4', ('time', 'y', 'x'))
        icemask_var = merged_ds.createVariable('icemask', 'f4', ('time', 'y', 'x'))
        obs_error_var = merged_ds.createVariable('obs_error', 'f4', ('time', 'y', 'x'))
        dhdt_var = merged_ds.createVariable('dhdt', 'f4', ('time', 'y', 'x'))
        smb_var = merged_ds.createVariable('smb', 'f4', ('time', 'y', 'x'))
        velsurf_mag_var = merged_ds.createVariable('velsurf_mag', 'f4', ('time', 'y', 'x'))

        # Assign data to variables
        time_var[:] = time_range
        x_var[:] = oggm_shop_ds.variables['x'][:]
        y_var[:] = oggm_shop_ds.variables['y'][:]
        thk_var[:] = thk_change
        usurf_var[:] = usurf_change
        topg_var[:] = topg_data
        icemask_var[:] = icemask_data
        obs_error_var[:] = error_variable
        dhdt_var[:] = dhdt_data
        smb_var[:] = smb_data
        velsurf_mag_var[:] = velo_data


def main():
    # load parameter file
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params",
                        type=str,
                        help="Path pointing to the parameter file",
                        required=True)
    arguments, _ = parser.parse_known_args()

    # Load the JSON file with parameters
    with open(arguments.params, 'r') as f:
        params = json.load(f)

    download_observations(params)


if __name__ == '__main__':
    main()