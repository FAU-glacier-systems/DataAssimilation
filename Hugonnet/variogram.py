import numpy as np
from netCDF4 import Dataset
import gstools as gs
import matplotlib.pyplot as plt

class Variogram_hugonnet(gs.CovModel):
    def cor(self, d: np.ndarray):
        """
        Spatial correlation of error in mean elevation change (Hugonnet et al., 2021).

        :param d: Distance between two glaciers (meters).

        :return: Spatial correlation function (input = distance in meters, output = correlation between 0 and 1).
        """

        # About 50% correlation of error until 150m, 80% until 2km, etc...
        ranges = [150, 2000, 5000, 20000, 50000, 500000]
        psills = [0.47741896, 0.34238422, 0.06662273, 0.06900394, 0.01602816, 0.02854199]

        # Spatial correlation at a given distance (using sum of exponential models)
        return 1 - np.sum((psills[i] * (1 - np.exp(- 3 * d / ranges[i])) for i in range(len(ranges))))

# Open dataset containing Hugonnet dhdt and dhdt_err
file_path = "Glaciers/Rhone/observations.nc"
oggm_shop_ds = Dataset(file_path, 'r')
dhdt = oggm_shop_ds['dhdt'][0][:]
dhdt_err = oggm_shop_ds['dhdt_err'][-1][:]
icemask = oggm_shop_ds['icemask'][0][:]
x = oggm_shop_ds['x'][:]
y = oggm_shop_ds['y'][:]

def plot_mean_uncerainty(oggm_shop_ds):
    dhdt_err_time = np.array(oggm_shop_ds['dhdt_err'])
    icemask = np.array(oggm_shop_ds['icemask'][0])
    time = []
    time_2 = []
    for i,dhdt_err_time_i in enumerate(dhdt_err_time):
        time.append(0.263 * np.sqrt(i))
    #for dhdt_err_time_i in dhdt_err_time:
        time_2.append(0.262 * i)
        #time2.append(np.mean(dhdt_err_time_i[icemask == 1]))

    plt.plot(time)
    plt.plot(time_2)
    plt.xlabel('Time')
    plt.ylabel('dhdt_err')
    plt.savefig('dhdt_err_time.png', dpi=300)

plot_mean_uncerainty(oggm_shop_ds)

# create spatial random field based on Hugonnet Variogram parameters
model = Variogram_hugonnet(dim=2)
srf = gs.SRF(model, mode_no=100)
srf.set_pos([y, x], "structured")
samples_srf = [srf(seed=i) for i in range(1000)]

dhdt_err_samples = samples_srf * dhdt_err
dhdt_samples = dhdt + dhdt_err_samples

fig_calc, axs_calc = plt.subplots(1, 4, figsize=(9, 3))  # 4x4 grid with appropriate
axs_calc[0].set_title('SRF Sample')
im = axs_calc[0].imshow(samples_srf[0], cmap='viridis', origin='lower',
                        interpolation='nearest', clim=(-5, 5))
axs_calc[1].set_title('dhdt_err')
im = axs_calc[1].imshow(dhdt_err, cmap='viridis', origin='lower',
                        interpolation='nearest', clim=(-5, 5))
axs_calc[2].set_title('dhdt')
im = axs_calc[2].imshow(dhdt, cmap='viridis', origin='lower',
                        interpolation='nearest', clim=(-5, 5))
axs_calc[3].set_title('dhdt_sample')
im = axs_calc[3].imshow(dhdt_samples[0], cmap='viridis', origin='lower',
                        interpolation='nearest', clim=(-5, 5))
for i in range(4):
    axs_calc[i].set_xticks([])  # Remove x-axis ticks
    axs_calc[i].set_yticks([])

cbar = fig_calc.colorbar(im, ax=axs_calc, orientation='vertical', fraction=0.05, pad=0.04)
cbar.set_label("Elevation Change (m yr$^{-1}$)", fontsize=12)

fig_calc.savefig('equation_parts.png', dpi=300, bbox_inches='tight')
specific_mb = [np.mean(dhdt_sampled[icemask == 1]) for dhdt_sampled in dhdt_samples]

print(np.mean(specific_mb))
print(np.std(specific_mb))

# Create a 4x4 grid of the 16 samples
fig, axs = plt.subplots(2, 2, figsize=(5, 6))  # 4x4 grid with appropriate
# figure size

# Flatten the axes array to easily iterate through it
axs_flat = axs.flatten()

# Plot each sample in the 4x4 grid
for i, dhdt_sampled in enumerate(dhdt_samples[:4]):
    im = axs_flat[i].imshow(dhdt_sampled, cmap='viridis', origin='lower', interpolation='nearest', clim=(-20, 20))
    axs_flat[i].set_title(f"Sample {i+1}")
    axs_flat[i].axis('off')  # Turn off axis for better visualization

# Add a colorbar to the entire figure
cbar = fig.colorbar(im, ax=axs[:, 1])
cbar.set_label("Elevation Change (m yr$^{-1}$)", fontsize=12)


# Save the figure with the 4x4 grid of samples
fig.savefig('4_samples_grid.png', bbox_inches='tight', dpi=300)