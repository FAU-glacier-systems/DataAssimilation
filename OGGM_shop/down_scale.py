import xarray as xr
import numpy as np
from scipy.ndimage import zoom

# Load the NetCDF file
file_path = 'Rhone/input_saved.nc'
ds = xr.open_dataset(file_path)


# Define the downscaling function using interpolation
def downscale(data, scale_factor):
    """
    Downscale the data array by a given scale factor.

    Parameters:
    data (numpy.ndarray): The data array to downscale.
    scale_factor (float): The scale factor for downscaling.

    Returns:
    numpy.ndarray: The downscaled data array.
    """
    return zoom(data, zoom=scale_factor, order=0)  # using bilinear interpolation (order=1)


# Downscale each data variable (assuming 2D or 3D arrays)
scale_factor = 0.5  # Example: downscale by a factor of 2
downscaled_data_vars = {}
for var in ds.data_vars:
    data = ds[var].values
    if data.ndim == 2:  # For 2D data
        downscaled_data = downscale(data, scale_factor)
    elif data.ndim == 3:  # For 3D data
        downscaled_data = np.array([downscale(layer, scale_factor) for layer in data])
    else:
        raise ValueError(f"Unsupported data dimensions: {data.ndim}")
    if var == 'icemask':
        downscaled_data[downscaled_data >=0.5] = 1
        downscaled_data[downscaled_data < 0.5] = 0
    downscaled_data_vars[var] = (ds[var].dims, downscaled_data)

# Create a new xarray dataset with the downscaled data
downscaled_ds = xr.Dataset(
    downscaled_data_vars,
    coords={
        'y': np.linspace(ds.y.min(), ds.y.max(), downscaled_data.shape[-2]),
        'x': np.linspace(ds.x.min(), ds.x.max(), downscaled_data.shape[-1]),
        # Add other coordinates if needed
    }
)

# Save the downscaled dataset to a new NetCDF file
downscaled_file_path = 'Rhone/downscaled_file.nc'
downscaled_ds.to_netcdf(downscaled_file_path)