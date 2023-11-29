import xarray as xr

# Specify the input and output file paths
optimized_file = '../inversion/geology-optimized.nc'
input_file = '../synthetic_observations/input_saved.nc'
output_file = '../synthetic_observations/input_merged.nc'

opti_ds = xr.open_dataset(optimized_file)
input_ds = xr.open_dataset(input_file)

arrhenius_field = opti_ds["arrhenius"]
merged_ds = input_ds.assign(arrhenius=opti_ds["arrhenius"], slidingco=opti_ds['slidingco'])

# Crop the dataset based on latitude and longitude ranges
#ds_cropped = merged_ds.sel(y=merged_ds.y[30:-30:2], x=merged_ds.x[30:-30:2])

# drop variables
ds_dropped= merged_ds.drop_vars(['usurfobs', 'thkobs', 'icemaskobs', 'uvelsurfobs', 'vvelsurfobs', 'thkinit'])

ds_dropped.to_netcdf(output_file)