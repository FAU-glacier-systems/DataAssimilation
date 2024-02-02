import matplotlib.pyplot as plt

import xarray as xr
import numpy as np

input_file = '../Inversion/geology-optimized.nc'
input_ds = xr.open_dataset(input_file)

"""
input_tif_path = '../Hugonnet/N46E008_2000-01-01_2020-01-01_dhdt.tif'
output_tif_path = '../Hugonnet/huggonnet20.tif'

# Define the source and target CRS
dst_crs = 'EPSG:4326'  # WGS 84
ulx, lrx = int(input_ds['x'].values[0]), int(input_ds['x'].values[-1]),
uly, lry = int(input_ds['y'].values[0]), int(input_ds['y'].values[-1])

# opening  lry, uly =and warping files to desired resolution, extent and coordinate reference system (crs)
elevation_change = gdal.Open(input_tif_path)
gdal.Warp(output_tif_path, elevation_change, dstSRS=dst_crs,
                           srcSRS=dst_crs,
                           resampleAlg='bilinear', xRes=50, yRes=50,
                           outputBounds=(ulx-25, uly-25, lrx+25, lry+25)
          )
with rasterio.open(output_tif_path) as dataset:
    # Read the raster data
    elevation_change_crop = dataset.read(1)
"""

time_range = np.arange(2000, 2021)
usurf_2000 = input_ds['usurf']
thk_2000 = input_ds['thk']
usurf_change = []
thk_change = []
bedrock = usurf_2000 - thk_2000
elevation_change_crop = input_ds['dhdt']
#elevation_change_crop[elevation_change_crop==-9999] = 0
#elevation_change_crop = np.flip(elevation_change_crop, axis=0)

for i,time in enumerate(time_range):
    print((i/20))
    usurf_change.append(usurf_2000 + elevation_change_crop*(i/20))
    thk_change.append(thk_2000 + elevation_change_crop*(i/20))

usurf_variable = xr.DataArray(usurf_change, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])
thk_variable = xr.DataArray(thk_change, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])


topg_data = [bedrock]*21
topg_variable = xr.DataArray(topg_data, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])

icemask_data = [input_ds['icemask']]*21
icemask_variable = xr.DataArray(icemask_data, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])

smb = [np.zeros_like(elevation_change_crop)]*21
smb_variable = xr.DataArray(smb, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])

velo = [input_ds['velsurfobs_mag']]*21
velo_variable = xr.DataArray(velo, coords={'time': time_range, 'x': input_ds['x'], 'y': input_ds['y']}, dims=['time', 'y', 'x'])

merged_ds = input_ds.assign(thk=thk_variable,
                            usurf=usurf_variable,
                            topg=topg_variable,
                            icemask=icemask_variable,
                            smb=smb_variable,
                            velsurf_mag=velo_variable)

merged_ds.to_netcdf('../Hugonnet/merged_dataset.nc')
print()
