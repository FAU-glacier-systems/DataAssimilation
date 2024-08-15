import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from oggm import workflow, utils
import json
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def extract_ela_gradients(group_df):
    # Find the index of the row with the smallest absolute mass balance
    mb = group_df['annual mass balance']
    elevation = group_df['upper elevation of bin'] - 50
    min_idx = mb.abs().idxmin()
    min_mb = mb[min_idx]
    min_ela = elevation[min_idx]
    if min_mb > 0:
        min2_idx = min_idx - 1
    elif min_mb < 0:
        min2_idx = min_idx + 1
    else:
        min2_idx = min_idx
        print('zero')

    # linear interpolate the ela from 2 elevation bins
    min2_mb = mb[min2_idx]
    min2_ela = elevation[min2_idx]
    mb_diff = abs(min2_mb - min_mb)
    alpha = abs(min_mb) / mb_diff
    alpha2 = abs(min2_mb) / mb_diff
    ela = (min_ela * alpha2 + min2_ela * alpha)

    # compute gradients
    ablation = mb[mb < 0]
    accumulation = mb[mb > 0]

    gradient_ablation = np.polyfit(elevation[mb < 0], ablation, 1)[0]
    gradient_accumulation = np.polyfit(elevation[mb > 0], accumulation, 1)[0]

    # conversion to volume
    gradient_ablation = gradient_ablation / 1000
    gradient_accumulation = gradient_accumulation / 1000
    return ela, gradient_ablation, gradient_accumulation


def compute_specific_mass_balance(group_df):
    mb = np.array(group_df['annual mass balance']).astype(float)
    area = np.array(group_df['area of elevation bin']).astype(float)

    # mass balance in m/yr
    mass_balance_bin = mb / 1000 * area
    specific_mass_balance = np.sum(mass_balance_bin) / np.sum(area)

    return specific_mass_balance


def compute_specific_mass_balance_from_ela(ela, gradabl, gradacc, usurf, icemask):
    maxacc = 100
    smb = usurf - ela
    smb *= np.where(np.less(smb, 0), gradabl, gradacc)
    smb = np.clip(smb, -100, maxacc)

    smb = np.where((smb < 0) | (icemask > 0.5), smb, -10)
    mb = np.sum(smb[icemask == 1]) / np.sum(icemask)
    return mb


def moving_average(data, window_size):
    """Compute the moving average of data with a given window size."""
    smoothed_data = []
    for i in range(len(data)):
        # Compute the start and end indices for the window
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)

        # Compute the average of the values in the window
        window_values = data[start_idx:end_idx]
        smoothed_value = sum(window_values) / len(window_values)

        smoothed_data.append(smoothed_value)

    return smoothed_data


### GEODETIC ###
file_path_hugonnet = '../Hugonnet/Rhone/observations.nc'
#file_path_oggm = '../OGGM_shop/Rhone/input_saved.nc'
#oggm_nc = xr.open_dataset(file_path_oggm)
hugonnet_nc = xr.open_dataset(file_path_hugonnet)

geodetic_mb = utils.get_geodetic_mb_dataframe()
geodetic_mb = geodetic_mb[geodetic_mb.index == 'RGI60-11.01238']

geodetic_mb_2020 = geodetic_mb[geodetic_mb['period'] == '2000-01-01_2020-01-01']
geodetic_mb_dmdtda = geodetic_mb_2020['dmdtda'].values[0]
geodetic_mb_dmdtda_err = geodetic_mb_2020['err_dmdtda'].values[0]

geodetic_mb_2010 = geodetic_mb[geodetic_mb['period'] == '2000-01-01_2010-01-01']
geodetic_mb_dmdtda_2010 = geodetic_mb_2010['dmdtda'].values[0]
geodetic_mb_dmdtda_err_2010 = geodetic_mb_2010['err_dmdtda'].values[0]

geodetic_mb_2010_2020 = geodetic_mb[geodetic_mb['period'] == '2010-01-01_2020-01-01']
geodetic_mb_dmdtda_2010_2020 = geodetic_mb_2010_2020['dmdtda'].values[0]
geodetic_mb_dmdtda_err_2010_2020 = geodetic_mb_2010_2020['err_dmdtda'].values[0]


# get raster data
dhdt = np.array(hugonnet_nc['dhdt'])[0]
dhdt_error = np.array(hugonnet_nc['obs_error'])[0]
icemask = np.array(hugonnet_nc['icemask'])[0]
dhdt_oggm = np.array(hugonnet_nc['dhdt'])

# compute specific mass balance
dhdt[icemask == 0] = 0
dhdt_error[icemask == 0] = 0

dhdt_flat = dhdt.flatten()
dhdt_error_flat = dhdt_error.flatten()

sample = np.random.normal(0, dhdt_error_flat)

hugonnet_mass_balance = np.sum(dhdt) / np.sum(icemask)
# hugonnet_error = np.sum(dhdt_error)/np.sum(icemask)
oggm_mass_balance = np.sum(dhdt_oggm) / np.sum(icemask)

# volume to mass conversion
# oggm_mass_balance *= 0.85
hugonnet_mass_balance *= 0.91
# hugonnet_error *= 0.85

# ensembel results
results_file = '../Experiments/Rhone/result_seed_111.json'
with open(results_file, 'r') as f:
    results = json.load(f)

ensemble_results = np.array(results['final_ensemble'])
usurf = hugonnet_nc['usurf'][0]
usurf2020 = usurf - dhdt*10
icemask = hugonnet_nc['icemask'][0]

mbs = []
for ensemble_member in ensemble_results:
    ela = ensemble_member[0]
    gradabl = ensemble_member[1]*0.91
    grad_acc = ensemble_member[2]*0.91
    mbs.append(compute_specific_mass_balance_from_ela(ela, gradabl, grad_acc, usurf, icemask))

mean_mb = np.mean(mbs)
std_mb = np.std(mbs)

print(mean_mb, std_mb)
### GLACIOLOGICAL ###
file_path_glamos_bin = 'massbalance_observation_elevationbins.csv'
file_path_glamos = 'massbalance_observation.csv'

# Read the CSV file into a pandas DataFrame, skipping the first 6 lines
df_glamos_bin = pd.read_csv(file_path_glamos_bin, delimiter=';', skiprows=6)
df_glamos = pd.read_csv(file_path_glamos, delimiter=';', skiprows=6)

# Filter rows where the 'glacier name' column is 'Rhonegletscher'
rhone_glacier_df = df_glamos_bin[df_glamos_bin['glacier name'] == 'Rhonegletscher']
rhone_glacier_total = df_glamos[df_glamos['glacier name'] == 'Rhonegletscher']

# Convert the 'start date of observation' column to datetime
rhone_glacier_df['start date of observation'] = pd.to_datetime(rhone_glacier_df['start date of observation'])
rhone_glacier_df['end date of observation'] = pd.to_datetime(rhone_glacier_df['end date of observation'])
rhone_glacier_df['upper elevation of bin'] = pd.to_numeric(rhone_glacier_df['upper elevation of bin'])
rhone_glacier_df['annual mass balance'] = pd.to_numeric(rhone_glacier_df['annual mass balance'])
rhone_glacier_total['end date of observation'] = pd.to_datetime(rhone_glacier_total['end date of observation'])
rhone_glacier_total['annual mass balance'] = pd.to_numeric(rhone_glacier_total['annual mass balance'])

# Filter rows after the year 2000
rhone_glacier_2000_df = rhone_glacier_df[np.logical_and(rhone_glacier_df['end date of observation'].dt.year >= 2000,
                                                        rhone_glacier_df['end date of observation'].dt.year <= 2020)]
rhone_glacier_2000_total = rhone_glacier_total[
    np.logical_and(rhone_glacier_total['end date of observation'].dt.year >= 2000,
                   rhone_glacier_total['end date of observation'].dt.year <= 2020)]

rhone_glacier_total_mb = rhone_glacier_2000_total['annual mass balance'] / 1000

# Extract data for plotting
date = rhone_glacier_2000_df['end date of observation'].dt.year
elevation = rhone_glacier_2000_df['upper elevation of bin']
mass_balance = rhone_glacier_2000_df['annual mass balance']
mass_balance /= 1000

### calculate and plot ela ###
rhone_glacier_group = rhone_glacier_2000_df.groupby('end date of observation')

# Iterate over each group
ELA = []
grad_abl = []
grad_acc = []
specific_mass_balances = []
time = []
for start_date, group_df in rhone_glacier_group:
    ela, gradient_ablation, gradient_accumulation = extract_ela_gradients(group_df)
    specific_mass_balance = compute_specific_mass_balance(group_df)

    specific_mass_balances.append(specific_mass_balance)
    ELA.append(ela)
    grad_abl.append(gradient_ablation)
    grad_acc.append(gradient_accumulation)
    time.append(start_date.year)

avg_ela = sum(ELA) / len(ELA)
print(np.std(np.array(ELA)))
print(np.std(np.array(grad_abl)))
print(np.std(np.array(grad_acc)))
avg_grad_abl = sum(grad_abl) / len(grad_abl)
avg_grad_acc = sum(grad_acc) / len(grad_acc)

ELA_min = np.min(ELA)
grad_abl_min = np.min(grad_abl)
grad_acc_min = np.min(grad_acc)

ELA_max = np.max(ELA)
grad_abl_max = np.max(grad_abl)
grad_acc_max= np.max(grad_acc)
### plot left figure ###
# plot background elevation change bin
fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[6, 4])  # Adjust the width ratios as needed

a0 = fig.add_subplot(gs[0])
a1 = fig.add_subplot(gs[1])
# plot elevation bins
scatter_bins = a0.scatter(date, elevation - 50, c=mass_balance, cmap='seismic_r',
                          vmin=-6, vmax=6,
                          marker='s', s=300, zorder=2)
fig.colorbar(scatter_bins, ax=a0, label='Mass Balance (m w.e. a$^{-1}$)')

# plot ELA and mean ELA
a0.scatter(np.array(time) - 0.03, ELA, alpha=0.3, c='black', label="Equilibrium Line Altitude", marker='_', s=300,
           zorder=3)
# a0.plot([2006.5]+time+[2020.5], [avg_ela]*(len(time)+2), c='black', label="Mean Equilibrium Line Altitude")

# write mean values of ela and gradients


# write description
a0.set_xlabel('Year of Measurement')
a0.set_xticks(list(np.array(time)[1::2]) + [2023, 2026],
              list(np.array(time)[1::2]) + ['GLAMOS\n Mean', 'EnKF\n Mean'])

a0.set_ylabel('Elevation (m)')
a0.set_title('Elevation dependent Annual Mass Balance')
a0.legend(loc='upper left')

### plot right figure ###
# plot geodetic
time20 = list(np.arange(2000, 2007)) + time
a1.set_title('Specific Mass Balance')
hugonnet_loss = [hugonnet_mass_balance] * len(time20)
# a1.plot(time20, hugonnet_loss, label='Geodetic Mean (2000-2020) [Hugonnet website]', color='C0', zorder=10)
# a1.fill_between(time20,hugonnet_loss -hugonnet_error, hugonnet_loss + hugonnet_error, color='C0', alpha=0.1, zorder=0,
#                label='error')
# a1.text(time20[0], hugonnet_loss[-1] - 0.4, f'{hugonnet_mass_balance:.4f} $m \, w.e./yr$', color='C0', zorder=10)

# oggm_loss = [oggm_mass_balance]*len(time20)
# a1.plot(time20, oggm_loss, label='Geodetic rasta data[Hugonnet OGGM]', color='C2', zorder=7)
# a1.text(time20[7], oggm_loss[-1] - 0.4, f'{oggm_mass_balance:.4f} $m \, w.e./yr$', color='C2', zorder=10)

geodetic_mb_dmdtda_2010 = [geodetic_mb_dmdtda_2010] * 2
a1.plot([2000, 2010], geodetic_mb_dmdtda_2010, color='C0', zorder=10)
a1.fill_between([2000, 2010], geodetic_mb_dmdtda_2010 - geodetic_mb_dmdtda_err_2010, geodetic_mb_dmdtda_2010 + geodetic_mb_dmdtda_err_2010, color='C0',
                alpha=0.1, zorder=0,)
a1.text(2000, geodetic_mb_dmdtda_2010[-1] + 0.03, f'{geodetic_mb_dmdtda_2010[0]:.4f}', color='C0', zorder=10)

geodetic_mb_dmdtda_2010_2020 = [geodetic_mb_dmdtda_2010_2020] * 2
a1.plot([2010, 2020], geodetic_mb_dmdtda_2010_2020, color='C0', zorder=10)
a1.fill_between([2010, 2020], geodetic_mb_dmdtda_2010_2020 - geodetic_mb_dmdtda_err_2010_2020, geodetic_mb_dmdtda_2010_2020 + geodetic_mb_dmdtda_err_2010_2020, color='C0',
                alpha=0.1, zorder=0,)
a1.text(2010, geodetic_mb_dmdtda_2010_2020[-1] + 0.03, f'{geodetic_mb_dmdtda_2010_2020[0]:.4f}', color='C0', zorder=10)

oggm_dmdtda = [geodetic_mb_dmdtda] * 2
a1.plot([2032, 2038], [geodetic_mb_dmdtda, geodetic_mb_dmdtda], label='Geodetic Mean [Hugonnet21]', color='C0', zorder=10)
a1.fill_between([2032, 2038], oggm_dmdtda - geodetic_mb_dmdtda_err, oggm_dmdtda + geodetic_mb_dmdtda_err, color='C0',
                alpha=0.1, zorder=0,
                label='Geodetic Uncertainty [Hugonnet21]')
a1.text(2032, oggm_dmdtda[-1] + 0.03, f'{geodetic_mb_dmdtda:.4f}', color='C0', zorder=10)

# plot glaciological
avg_mass_loss = moving_average(specific_mass_balances, len(time) * 2)
glamos_loss = avg_mass_loss[:1] * len(time)

a1.plot(time, glamos_loss, label='Glaciological Mean [GLAMOS]', color='black')
a1.plot(time, specific_mass_balances, label='Glaciological Annually [GLAMOS]', color='black', alpha=0.3)
a1.text(time[0], glamos_loss[-1] + 0.03, f'{avg_mass_loss[0]:.4f} ', color='black')

ensemble_mean_list = [mean_mb] * len(time20)
ensemble_var0_list = [mean_mb - std_mb] * 2
ensemble_var1_list = [mean_mb + std_mb] * 2

a1.plot([2025, 2031], [mean_mb, mean_mb], color='orange')
a1.fill_between([2025, 2031], ensemble_var0_list, ensemble_var1_list, color='C1', alpha=0.1, zorder=0,)
a1.text(2025, mean_mb + 0.03, f'{mean_mb:.4f}', color='C1', zorder=10)

a1.set_ylabel('Mass Balance (m w.e. a$^{-1}$)')
a1.set_ylim(-2.1, 1.1)
a1.set_xlabel('Year')
a1.set_xticks(list(np.array(time20)[::5]) + [2028, 2035],
              list(np.array(time20)[::5]) + ['EnKF\n Mean', 'Geodetic\n Mean'])
a1.legend(loc='upper left')

ensemble_results[:,1] *= 0.91
ensemble_results[:,2] *= 0.55
ensemble_mean = np.mean(ensemble_results, axis=0)
ensemble_std = np.std(ensemble_results, axis=0)
ela = ensemble_mean[0]
gradabl = ensemble_mean[1]
gradacc = ensemble_mean[2]


def creat_elevation_bins(ela, gradabl, gradacc):
    elevation_bins = np.arange(2250, 3750, 100)
    elevation_mb = []
    for elevation in elevation_bins:
        if elevation < ela:
            elevation_mb.append(-abs(elevation - ela) * gradabl)
        else:
            elevation_mb.append(abs(elevation - ela) * gradacc)
    return elevation_bins, elevation_mb


elevation_bins, elevation_mb = creat_elevation_bins(avg_ela, avg_grad_abl, avg_grad_acc)
a0.scatter([2023] * len(elevation_mb), elevation_bins, c=elevation_mb, cmap='seismic_r',
           vmin=-6, vmax=6,
           marker='s', s=300, zorder=2)

a0.scatter([2023], avg_ela, alpha=0.3, c='black', label="Equilibrium Line Altitude", marker='_', s=300, zorder=3)

a0.text(2024, avg_ela - 100, f'$s_{{ELA}}$: {int(avg_ela)}', rotation=90, color='black')
a0.text(2024, avg_ela - 550, f'$\gamma_{{abl}}$: {avg_grad_abl:.4f}', rotation=90, color='red')

a0.text(2024, avg_ela + 250, f'$\gamma_{{acc}}$: {avg_grad_acc:.4f}', rotation=90,
        label='Mean Accumulation Gradient', color='blue')

elevation_bins, elevation_mb = creat_elevation_bins(ela, gradabl, gradacc)

a0.scatter([2026] * len(elevation_mb), elevation_bins, c=elevation_mb, cmap='seismic_r',
           vmin=-6, vmax=6,
           marker='s', s=300, zorder=2)

a0.scatter([2026], ela, alpha=0.3, c='black', label="Equilibrium Line Altitude", marker='_', s=300, zorder=3)

a0.text(2027, ela - 100, f'$s_{{ELA}}$: {int(ela)}', rotation=90, color='black')
a0.text(2027, ela - 550, f'$\gamma_{{abl}}$: {gradabl:.4f}', rotation=90,
        label='Mean Ablation Gradient', color='red')
a0.text(2027, ela + 250, f'$\gamma_{{acc}}$: {gradacc:.4f}', rotation=90,
        label='Mean Accumulation Gradient', color='blue')

# fig.colorbar(scatter_bins, cax=a0, label='m w.e./yr')

# a2.set_ylabel('Elevation [m]')
# a2.set_xticks([1, 2], ['Glaciological \n Mean', 'Ensemble \n Mean'])

for axi in [a0, a1]:
    axi.grid(axis="y", color="lightgray", linestyle="-", zorder=-1)
    axi.grid(axis="x", color="lightgray", linestyle="-", zorder=-1)
    axi.spines['top'].set_visible(False)
    axi.spines['right'].set_visible(False)
    axi.spines['bottom'].set_visible(False)
    axi.spines['left'].set_visible(False)
    axi.xaxis.set_tick_params(bottom=False)
    axi.yaxis.set_tick_params(left=False)
# dhdt_error[icemask==0] = np.nan
# a0.set_title('Error map provided by Hugonnet')
# im_error = a0.imshow(dhdt_error, vmin=0, vmax=10, cmap='C2s')
# fig.colorbar(im_error, ax=a0, label='error m w.e./yr')

plt.tight_layout()
plt.savefig('mass_loss.pdf', format='pdf')
plt.savefig('mass_loss.png', format='png', dpi=300)
