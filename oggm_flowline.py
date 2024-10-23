import oggm
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# Example: Glacier directory for a glacier
gdir = oggm.GlacierDirectory('RGI60-11.00897')

# Step 1: Get flowline coordinates from OGGM (centerline or inversion flowlines)
flowlines = gdir.read_pickle('inversion_flowlines')  # Can also use 'centerlines'

# Extract the coordinates of the flowline
x_coords = flowlines[0].line.coords.xy[0]
y_coords = flowlines[0].line.coords.xy[1]

# Step 2: Read your raster (e.g., DEM or other georeferenced raster)
raster_path = '/path/to/your/raster.tif'
with rasterio.open(raster_path) as src:
    raster_data = src.read(1)  # Assuming the raster data is in the first band
    transform = src.transform

# Step 3: Create a plot of the raster and flowline
fig, ax = plt.subplots()

# Plot the raster using imshow and Rasterio's transform for correct geo-referencing
ax.imshow(raster_data, cmap='terrain', extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top))

# Overlay the flowline on top of the raster
ax.plot(x_coords, y_coords, color='red', linewidth=2, label='Flowline')

# Optional: Add labels, title, legend
ax.set_title('OGGM Flowline over Raster')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()

plt.show()