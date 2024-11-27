from netCDF4 import Dataset
import numpy as np
import argparse
import json
import numpy.ma as ma
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds



def crop_hugonnet_to_glacier(hugonnet_dataset, oggm_shop_ds):
    # Get bounds of the OGGM shop dataset area
    area_x = oggm_shop_ds['x'][:]
    area_y = oggm_shop_ds['y'][:]

    # Calculate the bounds from the min and max coordinates of oggm_shop_ds
    min_x, max_x = area_x.min(), area_x.max()
    min_y, max_y = area_y.min(), area_y.max()

    # Define the window to crop the hugonnet dataset using these bounds
    window = from_bounds(min_x, min_y, max_x, max_y, hugonnet_dataset.transform)

    # Read the data from the specified window (cropped area)
    cropped_map = hugonnet_dataset.read(1, window=window)
    filtered_map = np.where(cropped_map == -9999, np.nan, cropped_map)

    return filtered_map


def download_observations(params):


    # File paths
    oggm_shop_file = params['oggm_shop_file']

    time_period = params['time_period']
    time_range = np.arange(2000, 2021).astype(float)

    # Open datasetsmer
    oggm_shop_ds = Dataset(oggm_shop_file, 'r')

    with (rasterio.open('../Hugonnet/central_europe/11_rgi60_2000-01-01_2020-01-01'
                        '/dhdt_err/N46E008_2000-01-01_2020-01-01_dhdt_err.tif')
          as hugonnet_dataset):

        cropped_dhdt_err = crop_hugonnet_to_glacier(hugonnet_dataset, oggm_shop_ds)

    with (rasterio.open('../Hugonnet/central_europe/11_rgi60_2000-01-01_2020-01-01'
                        '/dhdt/N46E008_2000-01-01_2020-01-01_dhdt.tif')
          as hugonnet_dataset):

        cropped_dhdt = crop_hugonnet_to_glacier(hugonnet_dataset, oggm_shop_ds)

    dhdt = cropped_dhdt[::-1]
    dhdt_err = cropped_dhdt_err[::-1]


    icemask_2000 = oggm_shop_ds.variables['icemask'][:]
    usurf_2000 = oggm_shop_ds.variables['usurf'][:]
    thk_2000 = oggm_shop_ds.variables['thkinit'][:]
    usurf_change = []
    thk_change = []
    dhdt_err_time = []
    bedrock = usurf_2000 - thk_2000


    for i, time in enumerate(time_range):
        # compute glacier geometry based on dhdt
        usurf_i = np.maximum(bedrock, usurf_2000 + dhdt * i)
        # compute uncertainty overtime
        dhdt_err_i = dhdt_err * np.sqrt(i)

        usurf_change.append(usurf_i)
        thk_change.append(usurf_i - bedrock)
        dhdt_err_time.append(dhdt_err_i)

    # transform to numpy array
    usurf_change = np.array(usurf_change)
    thk_change = np.array(thk_change)
    dhdt_err_time = np.array(dhdt_err_time)

    # compute velocity magnitude
    uvelo = oggm_shop_ds.variables['uvelsurfobs'][:]
    vvelo = oggm_shop_ds.variables['vvelsurfobs'][:]
    velo = ma.sqrt(uvelo ** 2 + vvelo ** 2)

    # create placeholder smb
    smb = np.zeros_like(dhdt)

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
        dhdt_var = merged_ds.createVariable('dhdt', 'f4', ('time', 'y', 'x'))
        dhdt_err_var = merged_ds.createVariable('dhdt_err', 'f4', ('time', 'y','x'))
        smb_var = merged_ds.createVariable('smb', 'f4', ('time', 'y', 'x'))
        velsurf_mag_var = merged_ds.createVariable('velsurf_mag', 'f4', ('time', 'y', 'x'))

        # Assign data to variables
        time_var[:] = time_range
        x_var[:] = oggm_shop_ds.variables['x'][:]
        y_var[:] = oggm_shop_ds.variables['y'][:]
        thk_var[:] = thk_change
        usurf_var[:] = usurf_change
        topg_var[:] = bedrock
        icemask_var[:] = icemask_2000
        dhdt_var[:] = dhdt
        dhdt_err_var[:] = dhdt_err_time
        smb_var[:] = smb
        velsurf_mag_var[:] = velo


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