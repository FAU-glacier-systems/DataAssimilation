import xarray as xr

# Specify the input and output file paths
optimized_file = '../data_inversion/v2/geology-optimized.nc'
input_file = '../data_synthetic_observations/default/input_saved.nc'
output_file = '../data_synthetic_observations/v2/input_merged.nc'

opti_ds = xr.open_dataset(optimized_file)
input_ds = xr.open_dataset(input_file)

arrhenius_field = opti_ds["arrhenius"]
merged_ds = input_ds.assign(arrhenius=opti_ds["arrhenius"], slidingco=opti_ds['slidingco'])

# Crop the dataset based on latitude and longitude ranges
#ds_cropped = merged_ds.sel(y=merged_ds.y[30:-30:2], x=merged_ds.x[30:-30:2])

# drop variables
ds_dropped= merged_ds.drop_vars(['usurfobs', 'thkobs', 'icemaskobs', 'uvelsurfobs', 'vvelsurfobs', 'thkinit'])

ds_dropped.to_netcdf(output_file)