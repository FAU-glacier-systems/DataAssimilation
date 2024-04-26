import matplotlib.pyplot as plt
import rasterio
import rioxarray
import xarray as xr
import numpy as np

input_file = '../Inversion/geology-optimized.nc'
input_ds = xr.open_dataset(input_file)


dhdt_file_path = '../Hugonnet/dhdt/N46E008_2000-01-01_2020-01-01_dhdt.tif'
dhdt_err_file_path = '../Hugonnet/dhdt_err/N46E008_2000-01-01_2020-01-01_dhdt_err.tif'

# Define the source and target CRS
ulx, lrx = int(input_ds['x'].values[0]), int(input_ds['x'].values[-1]),
uly, lry = int(input_ds['y'].values[0]), int(input_ds['y'].values[-1])

dhdt_large = rioxarray.open_rasterio(dhdt_file_path)
dhdt_err_large = rioxarray.open_rasterio(dhdt_err_file_path)
dhdt = np.array(dhdt_large.rio.clip_box(ulx, uly, lrx, lry)[0])
dhdt_err = np.array(dhdt_err_large.rio.clip_box(ulx, uly, lrx, lry)[0])


time_range = np.arange(2000, 2021).astype(float)
usurf_2000 = input_ds['usurf']
thk_2000 = input_ds['thk']
usurf_change = []
thk_change = []
bedrock = usurf_2000 - thk_2000

for i,time in enumerate(time_range):
    usurf_i = usurf_2000 + dhdt * i
    usurf_i = np.maximum(bedrock, usurf_i)
    usurf_change.append(usurf_i)
    thk_change.append(usurf_i-bedrock)

usurf_variable = xr.DataArray(usurf_change, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])
thk_variable = xr.DataArray(thk_change, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])

year_range = 21

error_data = [dhdt_err]*year_range
error_variable = xr.DataArray(error_data, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])

topg_data = [bedrock]*year_range
topg_variable = xr.DataArray(topg_data, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])

icemask_data = [input_ds['icemask']]*year_range
icemask_variable = xr.DataArray(icemask_data, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])

smb = [np.zeros_like(dhdt)]*year_range
smb_variable = xr.DataArray(smb, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])

velo = [input_ds['velsurfobs_mag']]*year_range
velo_variable = xr.DataArray(velo, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])

merged_ds = input_ds.assign(thk=thk_variable,
                            usurf=usurf_variable,
                            topg=topg_variable,
                            icemask=icemask_variable,
                            obs_error=error_variable,
                            smb=smb_variable,
                            velsurf_mag=velo_variable)

merged_ds.to_netcdf('../Hugonnet/merged_dataset.nc')
print()
