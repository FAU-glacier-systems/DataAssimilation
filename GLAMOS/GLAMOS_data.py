import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


    gradient_ablation = gradient_ablation/910
    gradient_accumulation = gradient_accumulation/600
    print(gradient_ablation, gradient_accumulation)
    return ela, gradient_ablation, gradient_accumulation

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

# Specify the file path
file_path_glamos = 'massbalance_observation_elevationbins.csv'

# Read the CSV file into a pandas DataFrame, skipping the first 6 lines
df_glamos = pd.read_csv(file_path_glamos, delimiter=';', skiprows=6)

# Filter rows where the 'glacier name' column is 'Rhonegletscher'
rhone_glacier_df = df_glamos[df_glamos['glacier name'] == 'Rhonegletscher']

# Convert the 'start date of observation' column to datetime
rhone_glacier_df['start date of observation'] = pd.to_datetime(rhone_glacier_df['start date of observation'])
rhone_glacier_df['end date of observation'] = pd.to_datetime(rhone_glacier_df['end date of observation'])
rhone_glacier_df['upper elevation of bin'] = pd.to_numeric(rhone_glacier_df['upper elevation of bin'])
rhone_glacier_df['annual mass balance'] = pd.to_numeric(rhone_glacier_df['annual mass balance'])

# Filter rows after the year 2000
rhone_glacier_2000_df = rhone_glacier_df[np.logical_and(rhone_glacier_df['end date of observation'].dt.year >= 2000,
                                            rhone_glacier_df['end date of observation'].dt.year <= 2020)]

# Extract data for plotting
date = rhone_glacier_2000_df['end date of observation'].dt.year
elevation = rhone_glacier_2000_df['upper elevation of bin']
mass_balance = rhone_glacier_2000_df['annual mass balance'].values.astype(int)

### calculate and plot ela ###
rhone_glacier_group = rhone_glacier_2000_df.groupby('end date of observation')

# Iterate over each group
ELA = []
grad_abl = []
grad_acc = []

time = []
for start_date, group_df in rhone_glacier_group:
    ela, gradient_ablation, gradient_accumulation = extract_ela_gradients(group_df)
    ELA.append(ela)
    grad_abl.append(gradient_ablation)
    grad_acc.append(gradient_accumulation)
    time.append(start_date.year)

avg_ela =  moving_average(ELA, len(ELA) * 2, )
avg_grad_abl = sum(grad_abl) / len(grad_abl)
avg_grad_acc = sum(grad_acc) / len(grad_acc)

print(avg_ela[0], avg_grad_abl, avg_grad_acc)

# plot background elevation change bin
plt.scatter(date, elevation-50, c=mass_balance, cmap='seismic_r',
            vmin=-10000, vmax=10000,
            marker='s', s=200, )
plt.colorbar(label='mm w.e.')
plt.plot(time, ELA, c='lightgray', label="Equilibrium Line Altitude")
plt.plot(time, avg_ela, c='black', label="Mean Equilibrium Line Altitude")
plt.scatter([2021], avg_ela[0], c='white', marker='s', s=200, )

### WGMS data ###
# file_path_wgms = 'FoG_MB_473.csv'
# df_wgms = pd.read_csv(file_path_wgms, delimiter=',', skiprows=8, encoding='ISO-8859-1')
# df_wgms_2000 = df_wgms[df_wgms['SURVEY_YEAR']>=2000]
# wgms_ela = df_wgms_2000['ELA']
# wgms_time = df_wgms_2000['SURVEY_YEAR']
# plt.plot(wgms_time, wgms_ela, c='purple', label='WGMS ELA')

# Highlight the marker closest to 0 with a thicker black edge
plt.text(time[-1]+0.2, avg_ela[0] -20, f'{int(avg_ela[0])}')
plt.text(time[-1]+0.8, avg_ela[0] -600, f'{avg_grad_abl:.4f} m/yr/m', rotation=90,
         label='Mean Ablation Gradient', color='red')
plt.text(time[-1]+0.8, avg_ela[0] +200, f'{avg_grad_acc:.4f} m/yr/m', rotation=90,
         label='Mean Accumulation Gradient', color='blue')

plt.xlabel('Year of Measurement')
plt.ylabel('Elevation [m]')
plt.title('Annual Mass Balance of Rhone Glacier')
plt.legend()
plt.tight_layout()
plt.savefig('elevation_bins.png', dpi=300)