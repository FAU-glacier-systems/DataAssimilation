import xarray as xr

# Specify the input and output file paths
optimized_file = '../Inversion/geology-optimized.nc'
input_file = '../Inversion/input_saved.nc'
output_file = '../ReferenceRun/input_merged.nc'

opti_ds = xr.open_dataset(optimized_file)
input_ds = xr.open_dataset(input_file)

merged_ds = input_ds.assign(thk=opti_ds["thk"], slidingco=opti_ds['slidingco'])

# drop variables
ds_dropped= merged_ds.drop_vars(['usurfobs', 'thkobs', 'icemaskobs', 'uvelsurfobs', 'vvelsurfobs', 'thkinit'])

ds_dropped.to_netcdf(output_file)