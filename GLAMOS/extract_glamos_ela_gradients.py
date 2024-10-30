import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GLAMOS_data
import matplotlib.cm as cm


file_path_glamos_bin = 'massbalance_observation_elevationbins.csv'
file_path_glamos = 'massbalance_observation.csv'

time = np.arange(2000, 2020, dtype=int)
# Read the CSV file into a pandas DataFrame, skipping the first 6 lines
df_glamos = pd.read_csv(file_path_glamos, delimiter=';', skiprows=6)[2:]
df_glamos_bin = pd.read_csv(file_path_glamos_bin, delimiter=';', skiprows=6)[2:]

df_glamos['end date of observation'] = pd.to_datetime(df_glamos['end date of observation'])
df_glamos['annual mass balance'] = pd.to_numeric(df_glamos['annual mass balance'])

df_glamos_bin['start date of observation'] = pd.to_datetime(df_glamos_bin['start date of observation'])
df_glamos_bin['end date of observation'] = pd.to_datetime(df_glamos_bin['end date of observation'])
df_glamos_bin['upper elevation of bin'] = pd.to_numeric(df_glamos_bin['upper elevation of bin'])
df_glamos_bin['annual mass balance'] = pd.to_numeric(df_glamos_bin['annual mass balance'])

df_glamos = df_glamos[np.logical_and(df_glamos['end date of observation'].dt.year >= 2000,
                                                   df_glamos['end date of observation'].dt.year <= 2019)]
df_glamos_bin = df_glamos_bin[np.logical_and(df_glamos_bin['end date of observation'].dt.year >= 2000,
                                                   df_glamos_bin['end date of observation'].dt.year <= 2019)]


# get all glaciers coverd by glamos
available_glacier_names = df_glamos['glacier name'].unique()
available_glacier_names_bin = df_glamos_bin['glacier name'].unique()

assert len(available_glacier_names) == len(available_glacier_names_bin)

# iterate over glacier names and collect data
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store data
mean_elas = []
mean_abl_gradients = []
mean_acc_gradients = []

# Create a figure with three subplots (1 row, 3 columns)
fig, (ax_ela, ax_abl_grad, ax_acc_grad) = plt.subplots(1, 3, figsize=(15, 5))
colors = cm.get_cmap('tab20', 40)  # A color map with 31 unique colors
line_styles = ['-', '--', '-.', ':']  # Different line styles

for idx, glacier_name in enumerate(available_glacier_names):
    print(glacier_name)
    if str(glacier_name) == 'nan':
        print('detected nan')
        continue

    # Get ELA and specific mass balance
    glacier_df = df_glamos[df_glamos['glacier name'] == glacier_name]
    elas, specific_mb = GLAMOS_data.get_ela_and_specific_mb(glacier_df, time)
    mean_ela = np.nanmean(elas)

    # Get ablation and accumulation gradients
    glacier_df_bin = df_glamos_bin[df_glamos_bin['glacier name'] == glacier_name]
    abl_gradients, acc_gradients = GLAMOS_data.extract_gradients(glacier_df_bin, elas, time)

    # Append data to lists
    mean_elas.append(mean_ela)
    mean_abl_gradients.append(np.nanmean(abl_gradients))
    mean_acc_gradients.append(np.nanmean(acc_gradients))

    color = colors(idx)  # Get unique color for each glacier
    line_style = line_styles[idx % len(line_styles)]  #
    # Plot ELA in the first subplot
    ax_ela.plot(time, elas, label=glacier_name, color=color, linestyle=line_style)

    # Plot ablation gradient in the second subplot
    ax_abl_grad.plot(time, abl_gradients, label=glacier_name, color=color, linestyle=line_style)

    # Plot accumulation gradient in the third subplot
    ax_acc_grad.plot(time, acc_gradients, label=glacier_name, color=color, linestyle=line_style)

import math
mean_acc_gradients = [v for v in mean_acc_gradients if not math.isnan(v)]

# ELA subplot settings
ax_ela.boxplot(mean_elas, positions=[2020], widths=0.8)
ax_ela.set_xticks(range(2000, 2021, 5))
ax_ela.set_xticklabels(range(2000, 2021, 5))
ax_ela.set_xlabel('Year')
ax_ela.set_ylabel('ELA')
ax_ela.set_title(f'Mean ELA: {np.mean(mean_elas):.0f} $\pm$ {np.std(mean_elas):.0f} m')

# Ablation gradient subplot settings
ax_abl_grad.boxplot(mean_abl_gradients, positions=[2020], widths=0.8)
ax_abl_grad.set_xticks(range(2000, 2021, 5))
ax_abl_grad.set_xticklabels(range(2000, 2021, 5))
ax_abl_grad.set_xlabel('Year')
ax_abl_grad.set_ylabel('Ablation Gradient')
ax_abl_grad.set_title(f'Mean Ablation gradient: {np.mean(mean_abl_gradients):.4f} $\pm$ {np.std(mean_abl_gradients):.4f} m/a/m')


# Accumulation gradient subplot settings
ax_acc_grad.boxplot(mean_acc_gradients, positions=[2020], widths=0.8)
ax_acc_grad.set_xticks(range(2000, 2021, 5))
ax_acc_grad.set_xticklabels(range(2000, 2021, 5))
ax_acc_grad.set_xlabel('Year')
ax_acc_grad.set_ylabel('Accumulation Gradient')
ax_acc_grad.set_title(f'Mean Acclation gradient: {np.mean(mean_acc_gradients):.4f} $\pm$ '
                      f'{np.std(mean_acc_gradients):.4f} m/a/m')


# Place legend outside to the right for the ELA plot

ax_acc_grad.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='8')
# Adjust layout and save figure
plt.tight_layout()
plt.savefig('../Plots/all_gradients.png', dpi=300)


