import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

inversion_file = 'optimize.nc'

inversion_ds = xr.open_dataset(inversion_file)

thickness = np.array(inversion_ds["thk"])[-1]
arrhenius = np.array(inversion_ds["arrhenius"])
slidingco = np.array(inversion_ds["slidingco"])
velocity = np.array(inversion_ds["velsurf_mag"])
velocity_obs = np.array(inversion_ds["velsurfobs_mag"])


fig, ax = plt.subplots(2, 2, figsize=(10, 10))


velocity_obs_iter = velocity_obs[-1]
velocity_obs_iter[thickness < 0.01] = None
ax[0,0].imshow(velocity_obs_iter, vmin=0, vmax=78, cmap="magma")
ax[0,0].invert_yaxis()
ax[0,0].set_title("Observed velocity [Millan 22]")


velocity_iter = velocity[-1]
velocity_iter[thickness < 0.01] = None
ax[0,1].imshow(velocity_iter, vmin=0, vmax=78, cmap="magma")
ax[0,1].invert_yaxis()
ax[0,1].set_title("Modeled velocity")

arrhenius_iter = arrhenius[-1]
arrhenius_iter[thickness<0.01] = None
ax[1,0].imshow(arrhenius_iter, vmin=0, vmax=78)
ax[1,0].invert_yaxis()
ax[1,0].set_title("Optimized arrhenius")

slidingco_iter = slidingco[-1]
slidingco_iter[thickness < 0.01] = None
ax[1,1].imshow(slidingco_iter, vmin=0, vmax=78)
ax[1,1].invert_yaxis()
ax[1,1].set_title("Optimized slidingco")
plt.savefig("sliding_arrhenius.png")
