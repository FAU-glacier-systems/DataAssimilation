import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

output_file = 'data_synthetic_observations/v1/output.nc'
optimized_file = 'data_inversion/v1/geology-optimized.nc'
figure_path = "data_synthetic_observations/v1/sliding_arrhenius.png"

ds = xr.open_dataset(output_file)
ds_optimized = xr.open_dataset(optimized_file)

thickness = np.array(ds_optimized["thk"])
inversion = False
if inversion:
    arrhenius = np.array(ds_optimized["arrhenius"])
    slidingco = np.array(ds_optimized["slidingco"])
    velocity = np.array(ds_optimized["velsurf_mag"])
else:
    arrhenius = np.array(ds["arrhenius"])[0]
    slidingco = np.array(ds["slidingco"])[0]
    velocity = np.array(ds["velsurf_mag"])[-1]

velocity_obs = np.array(ds_optimized["velsurfobs_mag"])


fig, ax = plt.subplots(2, 2, figsize=(10, 10))


velocity_obs[thickness < 0.01] = None
vel_img = ax[0, 0].imshow(velocity_obs, vmin=0, vmax=70, cmap="magma")
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('surface velocity [m/yr]', rotation=90)
ax[0, 0].invert_yaxis()
ax[0, 0].set_title("Observed velocity [Millan22]")

velocity_iter = velocity
velocity_iter[thickness < 0.01] = None
vel_img = ax[0, 1].imshow(velocity_iter, vmin=0, vmax=70, cmap="magma")
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('surface velocity [m/yr]', rotation=90)
ax[0, 1].invert_yaxis()
ax[0, 1].set_title("Modelled velocity")


arrhenius_iter = arrhenius
arrhenius_iter[thickness<0.01] = None
img = ax[1,0].imshow(arrhenius_iter, vmin=0, vmax=78)
cbar = fig.colorbar(img)
cbar.ax.set_ylabel('arrhenius [$MPa^{-3}$]', rotation=90)
ax[1,0].invert_yaxis()
ax[1,0].set_title("Optimized arrhenius ")

slidingco_iter = slidingco
slidingco_iter[thickness < 0.01] = None
img = ax[1,1].imshow(slidingco_iter)
cbar = fig.colorbar(img)
cbar.ax.set_ylabel('slidingco $[km MPa^{-3}a^{-1}]$', rotation=90)
ax[1,1].invert_yaxis()
ax[1,1].set_title("Optimized slidingco")

plt.savefig(figure_path)
