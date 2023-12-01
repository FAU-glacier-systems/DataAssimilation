import xarray as xr

# Specify the input and output file paths
input_file = '../data_synthetic_observations/input_saved.nc'
output_file = '../data_synthetic_observations/input_cropped.nc'

# Crop and rescale the NetCDF file
ds = xr.open_dataset(input_file)

# Crop the dataset based on latitude and longitude ranges
#ds_cropped = ds.sel(y=ds.y[30:-30:2], x=ds.x[30:-30:2])

# drop variables
ds_cropped = ds.drop_vars(['usurfobs', 'thkobs', 'icemaskobs', 'uvelsurfobs', 'vvelsurfobs', 'thkinit'])

# Save the cropped and rescaled dataset to a new NetCDF file
ds_cropped.to_netcdf(output_file)

# Close the datasets
ds.close()
ds_cropped.close()
print(f"Successfully cropped and rescaled data. Saved to {output_file}")
