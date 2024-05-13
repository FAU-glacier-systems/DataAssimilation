import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from oggm import workflow, utils

def extract_ela_gradients(group_df):
    # Find the index of the row with the smallest absolute mass balance
    mb = group_df['annual mass balance']
    elevation = group_df['upper elevation of bin']-50
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
    accumulation = mb[mb>0]

    gradient_ablation = np.polyfit(elevation[mb<0], ablation, 1)[0]
    gradient_accumulation = np.polyfit(elevation[mb>0], accumulation, 1)[0]

    # conversion to volume
    gradient_ablation = gradient_ablation/850
    gradient_accumulation = gradient_accumulation/850
    return ela, gradient_ablation, gradient_accumulation

def compute_specific_mass_balance(group_df):
    mb = np.array(group_df['annual mass balance']).astype(float)
    area = np.array(group_df['area of elevation bin']).astype(float)

    # mass balance in m/yr
    mass_balance_bin = mb/1000 * area
    specific_mass_balance = np.sum(mass_balance_bin) / np.sum(area)

    return specific_mass_balance

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
file_path_hugonnet = '../Hugonnet/merged_dataset.nc'
file_path_oggm = '../OGGM_shop/input_saved.nc'
oggm_nc = xr.open_dataset(file_path_oggm)
hugonnet_nc = xr.open_dataset(file_path_hugonnet)

geodetic_mb = utils.get_geodetic_mb_dataframe()
geodetic_mb = geodetic_mb[geodetic_mb.index == 'RGI60-11.01238']

geodetic_mb_2020 = geodetic_mb[geodetic_mb['period'] == '2000-01-01_2020-01-01']
geodetic_mb_dmdtda = geodetic_mb_2020['dmdtda'].values[0]
geodetic_mb_dmdtda_err = geodetic_mb_2020['err_dmdtda'].values[0]

# get raster data

dhdt = np.array(hugonnet_nc['dhdt'])[0]
dhdt_error = np.array(hugonnet_nc['obs_error'])[0]
icemask = np.array(hugonnet_nc['icemask'])[0]
dhdt_oggm = np.array(oggm_nc['dhdt'])


# compute specific mass balance

dhdt[icemask == 0] = 0
dhdt_error[icemask == 0] = 0
hugonnet_mass_balance = np.sum(dhdt)/np.sum(icemask)
hugonnet_error = np.sum(dhdt_error)/np.sum(icemask)
oggm_mass_balance = np.sum(dhdt_oggm)/np.sum(icemask)

# volume to mass conversion
#oggm_mass_balance *= 0.85
#hugonnet_mass_balance *= 0.85
#hugonnet_error *= 0.85


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
rhone_glacier_2000_total = rhone_glacier_total[np.logical_and(rhone_glacier_total['end date of observation'].dt.year >= 2000,
                                            rhone_glacier_total['end date of observation'].dt.year <= 2020)]

rhone_glacier_total_mb = rhone_glacier_2000_total['annual mass balance']/1000

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
avg_grad_abl = sum(grad_abl) / len(grad_abl)
avg_grad_acc = sum(grad_acc) / len(grad_acc)

### plot left figure ###
# plot background elevation change bin
fig, (a1, a0) = plt.subplots(1, 2, figsize=(11, 5))

# plot elevation bins
scatter_bins = a0.scatter(date, elevation-50, c=mass_balance, cmap='seismic_r',
            vmin=-6, vmax=6,
            marker='s', s=270, )
fig.colorbar(scatter_bins, ax=a0, label='m w.e.')

# plot ELA and mean ELA
a0.scatter(np.array(time)-0.03, ELA, alpha=0.3, c='black', label="Equilibrium Line Altitude", marker='_', s=260)
a0.plot([2006.5]+time+[2020.5], [avg_ela]*(len(time)+2), c='black', label="Mean Equilibrium Line Altitude")

# write mean values of ela and gradients
a0.text(time[-1]+1.5, avg_ela-50, f'{int(avg_ela)}', rotation=90, color='black')
a0.text(time[-1]+1.5, avg_ela-600, f'{avg_grad_abl:.4f} m/yr/m', rotation=90,
         label='Mean Ablation Gradient', color='red')
a0.text(time[-1]+1.5, avg_ela+200, f'{avg_grad_acc:.4f} m/yr/m', rotation=90,
         label='Mean Accumulation Gradient', color='blue')

#write description
a0.set_xlabel('Year of Measurement')
a0.set_xticks(np.array(time)[1::2])
a0.set_ylabel('Elevation [m]')
a0.set_title('Elevation dependent Annual Mass Balance')
a0.legend(loc='upper left')

### plot right figure ###
# plot geodetic
time20 = list(np.arange(2000,2007)) + time
a1.set_title('Specific Mass Balance')
hugonnet_loss =[hugonnet_mass_balance]*len(time20)
a1.plot(time20, hugonnet_loss, label='Geodetic Mean (2000-2020) [Hugonnet website]', color='C0', zorder=10)
a1.fill_between(time20,hugonnet_loss -hugonnet_error, hugonnet_loss + hugonnet_error, color='C0', alpha=0.1, zorder=0,
                label='error')
a1.text(time20[0], hugonnet_loss[-1] - 0.4, f'{hugonnet_mass_balance:.4f} $m \, w.e./yr$', color='C0', zorder=10)

oggm_loss = [oggm_mass_balance]*len(time20)
a1.plot(time20, oggm_loss, label='Geodetic rasta data[Hugonnet OGGM]', color='C1', zorder=7)
a1.text(time20[7], oggm_loss[-1] - 0.4, f'{oggm_mass_balance:.4f} $m \, w.e./yr$', color='C1', zorder=10)

oggm_dmdtda = [geodetic_mb_dmdtda] * len(time20)
a1.plot(time20, oggm_dmdtda, label='Geodetic Mean single value[Hugonnet OGGM]', color='C2', zorder=10)
a1.fill_between(time20, oggm_dmdtda - geodetic_mb_dmdtda_err, oggm_dmdtda + geodetic_mb_dmdtda_err, color='C2', alpha=0.1, zorder=0,
                label='error')
a1.text(time20[13], oggm_dmdtda[-1] - 0.4, f'{geodetic_mb_dmdtda:.4f} $m \, w.e./yr$', color='C2', zorder=10)


# plot glaciological
avg_mass_loss = moving_average(specific_mass_balances, len(time)*2)
glamos_loss = avg_mass_loss[:1]*len(time)


a1.plot(time, glamos_loss, label='Glaciological Mean (2007-2020)[GLAMOS]', color='black')
a1.plot(time, specific_mass_balances, label='Glaciological [GLAMOS]', color='black', alpha=0.3)
a1.text(time[0], glamos_loss[-1] + 0.1, f'{avg_mass_loss[0]:.4f} $m \, w.e./yr$', color='black')

a1.set_ylabel('Annual Mass loss [$m \, w.e./yr$] water equivalent')
a1.set_ylim(-4, 4)
a1.set_xlabel('year')
a1.set_xticks(time20[::2])
a1.legend(loc='upper left')

# dhdt_error[icemask==0] = np.nan
# a0.set_title('Error map provided by Hugonnet')
# im_error = a0.imshow(dhdt_error, vmin=0, vmax=10, cmap='Blues')
# fig.colorbar(im_error, ax=a0, label='error m w.e./yr')

plt.tight_layout()
plt.savefig('mass_loss.png', dpi=300)


