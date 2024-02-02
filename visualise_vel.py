import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

output_file = 'ReferenceSimulation/v2/output.nc'
optimized_file = 'Inversion/v2/geology-optimized.nc'
figure_path = "Inversion/v2/sliding_arrhenius.png"

ds = xr.open_dataset(output_file)
ds_optimized = xr.open_dataset(optimized_file)

icemask = np.array(ds_optimized["icemask"])
inversion = True
if inversion:
    arrhenius = np.array(ds_optimized["arrhenius"])
    slidingco = np.array(ds_optimized["slidingco"])
    velocity = np.array(ds_optimized["velsurf_mag"])
    thickness = np.array(ds_optimized["thk"])
else:
    arrhenius = np.array(ds["arrhenius"])[0]
    slidingco = np.array(ds["slidingco"])[0]
    velocity = np.array(ds["velsurf_mag"])[0]
    thickness = np.array(ds["thk"])[0]

velocity_obs = np.array(ds_optimized["velsurfobs_mag"])


fig, ax = plt.subplots(2, 2, figsize=(10, 10))

velocity_obs[icemask < 0.01] = None
vel_img = ax[0, 0].imshow(velocity_obs, vmin=0, vmax=70, cmap="magma")
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('surface velocity [m/yr]', rotation=90)
ax[0, 0].invert_yaxis()
ax[0, 0].set_title("Observed velocity [Millan22]")

velocity_iter = velocity
velocity_iter[icemask < 0.01] = None
vel_img = ax[1, 0].imshow(velocity_iter, vmin=0, vmax=70, cmap="magma")
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('surface velocity [m/yr]', rotation=90)
ax[1, 0].invert_yaxis()
ax[1, 0].set_title("Modelled velocity")

thickness[icemask<0.01] = None
img = ax[0,1].imshow(thickness, cmap='Blues')
cbar = fig.colorbar(img)
cbar.ax.set_ylabel('thickness [$m$]', rotation=90)
ax[0,1].invert_yaxis()
ax[0,1].set_title("Optimized thickness ")

slidingco_iter = slidingco
slidingco_iter[icemask < 0.01] = None
img = ax[1,1].imshow(slidingco_iter)
cbar = fig.colorbar(img)
cbar.ax.set_ylabel('slidingco $[?]$', rotation=90)
ax[1,1].invert_yaxis()
ax[1,1].set_title("Optimized slidingco")
fig.suptitle("inversion", fontsize=32)
plt.savefig(figure_path)
