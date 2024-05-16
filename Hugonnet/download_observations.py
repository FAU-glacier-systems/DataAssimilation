import copy
from oggm import utils
import matplotlib.pyplot as plt
import rasterio
import rioxarray
import xarray as xr
import numpy as np

# https://www.sedoo.fr/theia-publication-products/?uuid=c428c5b9-df8f-4f86-9b75-e04c778e29b9
inversion_file = '../Inversion/geology-optimized.nc'
oggm_file = '../OGGM_shop/input_saved.nc'
inversion_ds = xr.open_dataset(inversion_file)
oggm_ds = xr.open_dataset(oggm_file)

dhdt_file_path = 'dhdt/N46E008_2000-01-01_2020-01-01_dhdt.tif'

geodetic_mb = utils.get_geodetic_mb_dataframe()
geodetic_mb = geodetic_mb[geodetic_mb.index == 'RGI60-11.01238']

geodetic_mb_2020 = geodetic_mb[geodetic_mb['period'] == '2000-01-01_2020-01-01']
geodetic_mb_dmdtda_err = geodetic_mb_2020['err_dmdtda'].values[0]

# convert mass to volume
geodetic_mb_dmdtda_err/=0.9
dhdt_err = np.array(inversion_ds['icemask'])*geodetic_mb_dmdtda_err

# Define the source and target CRS
ulx, lrx = int(inversion_ds['x'].values[0]), int(inversion_ds['x'].values[-1]),
uly, lry = int(inversion_ds['y'].values[0]), int(inversion_ds['y'].values[-1])

dhdt_large = rioxarray.open_rasterio(dhdt_file_path)
dhdt = np.array(dhdt_large.rio.clip_box(ulx, uly, lrx, lry)[0])[::-1]

dhdt_rhone = copy.copy(dhdt[::-1])
icemask = inversion_ds['icemask']
dhdt_rhone[inversion_ds['icemask'] == 0] = 0


time_range = np.arange(2000, 2021).astype(float)
usurf_2000 = inversion_ds['usurf']
thk_2000 = inversion_ds['thk']
usurf_change = []
thk_change = []
dhdt_errors = []
bedrock = usurf_2000 - thk_2000

for i,time in enumerate(time_range):
    usurf_i = usurf_2000 + dhdt * i
    usurf_i = np.maximum(bedrock, usurf_i)
    dhdt_err_i = dhdt_err * i
    usurf_change.append(usurf_i)
    thk_change.append(usurf_i-bedrock)
    dhdt_errors.append(dhdt_err_i)

usurf_variable = xr.DataArray(usurf_change, coords={'time': time_range, 'x': inversion_ds['x'], 'y': inversion_ds['y']}, dims=['time', 'y', 'x'])
thk_variable = xr.DataArray(thk_change, coords={'time': time_range, 'x': inversion_ds['x'], 'y': inversion_ds['y']}, dims=['time', 'y', 'x'])

year_range = 21

dhdt_data = [dhdt]*year_range
dhdt_variables = xr.DataArray(dhdt_data, coords={'time': time_range, 'x': inversion_ds['x'], 'y': inversion_ds['y']}, dims=['time', 'y', 'x'])

error_variable = xr.DataArray(dhdt_errors, coords={'time': time_range, 'x': inversion_ds['x'], 'y': inversion_ds['y']}, dims=['time', 'y', 'x'])

topg_data = [bedrock]*year_range
topg_variable = xr.DataArray(topg_data, coords={'time': time_range, 'x': inversion_ds['x'], 'y': inversion_ds['y']}, dims=['time', 'y', 'x'])

icemask_data = [inversion_ds['icemask']] * year_range
icemask_variable = xr.DataArray(icemask_data, coords={'time': time_range, 'x': inversion_ds['x'], 'y': inversion_ds['y']}, dims=['time', 'y', 'x'])

smb = [np.zeros_like(dhdt)]*year_range
smb_variable = xr.DataArray(smb, coords={'time': time_range, 'x': inversion_ds['x'], 'y': inversion_ds['y']}, dims=['time', 'y', 'x'])

velo = [inversion_ds['velsurfobs_mag']] * year_range
velo_variable = xr.DataArray(velo, coords={'time': time_range, 'x': inversion_ds['x'], 'y': inversion_ds['y']}, dims=['time', 'y', 'x'])

merged_ds = inversion_ds.assign(thk=thk_variable,
                                usurf=usurf_variable,
                                topg=topg_variable,
                                icemask=icemask_variable,
                                obs_error=error_variable,
                                dhdt = dhdt_variables,
                                smb=smb_variable,
                                velsurf_mag=velo_variable)

merged_ds.to_netcdf('../Hugonnet/merged_dataset.nc')
print()
