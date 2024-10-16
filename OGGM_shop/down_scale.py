import xarray as xr
import numpy as np
from scipy.ndimage import zoom
import argparse

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


def main(input_file, output_file, scale_factor):
    # Load the NetCDF file
    ds = xr.open_dataset(input_file)

    # Downscale each data variable (assuming 2D or 3D arrays)
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
            downscaled_data[downscaled_data >= 0.5] = 1
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
    downscaled_ds.to_netcdf(output_file)
    print(f"Downscaled data saved to {output_file}")


if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Downscale NetCDF data.")
    parser.add_argument("input_file", type=str, help="Path to the input NetCDF file.")
    parser.add_argument("output_file", type=str, help="Path to save the downscaled NetCDF file.")
    parser.add_argument("--scale", type=float, default=0.5, help="Scaling factor for downscaling (default: 0.5)")

    args = parser.parse_args()

    # Call the main function with arguments
    main(args.input_file, args.output_file, args.scale)
