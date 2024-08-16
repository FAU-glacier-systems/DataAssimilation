import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

#output_file = '../ReferenceSimulation/output.nc'
optimized_file = 'Rhone/geology-optimized.nc'
figure_path = "Rhone/inversion_result.pdf"

#ds = xr.open_dataset(output_file)
ds_optimized = xr.open_dataset(optimized_file)

icemask = np.array(ds_optimized["icemask"])
surface = np.array(ds_optimized["usurf"])
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

fig, ax = plt.subplots(1, 4, figsize=(12, 4))
plt.subplots_adjust(left=0, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)

for i in range(4):
    ax[i].imshow(surface, cmap='gray', vmin=1450, vmax=3600)
velocity_obs[icemask < 0.01] = None
vel_img = ax[0].imshow(velocity_obs, vmin=0, vmax=70, cmap="magma", zorder=2)
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('Surface Velocity (m a$^{-1}$)', rotation=90)
ax[0].set_title("Observed Velocity [Millan22]")

velocity_iter = velocity
velocity_iter[icemask < 0.01] = None
vel_img = ax[1].imshow(velocity_iter, vmin=0, vmax=70, cmap="magma", zorder=2)
cbar = fig.colorbar(vel_img)
cbar.ax.set_ylabel('Surface Velocity (m a$^{-1}$)', rotation=90)

ax[1].set_title("Modelled Velocity")

thickness[icemask < 0.01] = None
img = ax[2].imshow(thickness, cmap='Blues', zorder=2)
cbar = fig.colorbar(img)
cbar.ax.invert_yaxis()
cbar.ax.set_ylabel('Thickness (m)', rotation=90)

ax[2].set_title("Optimized Thickness ")

slidingco_iter = slidingco
slidingco_iter[icemask < 0.01] = None
img = ax[3].imshow(slidingco_iter, zorder=2)
cbar = fig.colorbar(img)
cbar.ax.set_ylabel('Sliding Co. (MPa a$^{3}$ m$^{-3}$)', rotation=90)

ax[3].set_title("Optimized Sliding Co.")


def formatter(x, pos):
    del pos
    return str(int(x * 100 / 1000))

for i in range(4):
    ax[i].invert_yaxis()
    ax[i].set_xlim(15, 65)
    ax[i].set_ylim(15, 105)
    ax[i].yaxis.set_ticks([20, 40, 60, 80, 100])
    ax[i].xaxis.set_ticks([ 20, 40, 60])
    ax[i].xaxis.set_major_formatter(formatter)
    ax[i].yaxis.set_major_formatter(formatter)
    ax[i].grid(axis="y", color="black", linestyle="--", zorder=0, alpha=.2)
    ax[i].grid(axis="x", color="black", linestyle="--", zorder=0, alpha=.2)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[i].spines[axis].set_linewidth(0)
    ax[i].set_xlabel('km', color='black')
    ax[i].tick_params(axis='x', colors='black')
    ax[i].tick_params(axis='y', colors='black')



#fig.suptitle("inversion", fontsize=32)
plt.tight_layout()
plt.savefig(figure_path, format='pdf')
