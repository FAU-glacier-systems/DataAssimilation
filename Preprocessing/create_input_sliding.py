import xarray as xr

# Specify the input and output file paths
optimized_file = '../Inversion/v2/geology-optimized.nc'
input_file = '../ReferenceSimulation/default/input_saved.nc'
output_file = '../ReferenceSimulation/v2/input_merged.nc'

opti_ds = xr.open_dataset(optimized_file)
input_ds = xr.open_dataset(input_file)

merged_ds = input_ds.assign(thk=opti_ds["thk"], velsurf_mag=opti_ds['velsurf_mag'])

# Crop the dataset based on latitude and longitude ranges
#ds_cropped = merged_ds.sel(y=merged_ds.y[30:-30:2], x=merged_ds.x[30:-30:2])

# drop variables
ds_dropped= merged_ds.drop_vars(['thkinit'])

ds_dropped.to_netcdf(output_file)