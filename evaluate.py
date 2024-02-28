import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import xarray as xr

# Specify the path to your JSON file
results = []
experiment_folder = 'HPC/'
for file in os.listdir(experiment_folder):
    if file.endswith('.json'):
        print(file)
        with open(experiment_folder + file, 'r') as f:
                results.append(json.load(f))


input_file = 'Inversion/geology-optimized.nc'
input_ds = xr.open_dataset(input_file)
icemask = input_ds['icemask']

ensemble_size = [exp['ensemble_size'] for exp in results]
dt = [exp['dt'] for exp in results]
offset = [exp['initial_offset'] for exp in results]
uncertainty = [exp['initial_uncertainity'] for exp in results]

#VAR = [exp['esit_var'][0][0] for exp in results]

area = np.sum(icemask) * 100 ** 2 / 1000 ** 2
num_sample_points = [exp['num_sample_points'] for exp in results]
area_ration_sample = [(num_p * 100 ** 2 / 1000 ** 2) / area for num_p in num_sample_points]

MAE = np.array([])
VAR = np.array([])
for i in range(3):
    MAE_para = np.array([abs(exp['true_parameter'][i] - exp['esti_parameter'][i]) for exp in results])
    MAE_para_norm = MAE_para / np.max(MAE_para)
    MAE = np.concatenate((MAE, MAE_para_norm))
    VAR_para = np.array([exp['esit_var'][0][0] for exp in results])
    VAR_para_norm = VAR_para / np.max(VAR_para)
    VAR = np.concatenate((VAR, VAR_para_norm))

df = pd.DataFrame({'MAE': MAE,
                   'area_ration_sample': area_ration_sample + area_ration_sample + area_ration_sample,
                   'VAR': VAR,
                   'ensemble_size': ensemble_size + ensemble_size + ensemble_size,
                   'dt': dt + dt + dt,
                   'initial_offset': offset + offset + offset,
                   'initial_uncertainty': uncertainty + uncertainty + uncertainty})

# df = df[df['MAE'] <= 100]
df_best = df[df['MAE'] <= 10]
print(len(df))
print(len(df_best))
colorscale = plt.get_cmap('tab20')
mean_color = colorscale(0)
median_color = colorscale(2)
var_color = colorscale(4)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), layout="tight")
for i, hyperparameter in enumerate(['ensemble_size', 'area_ration_sample', 'dt',]):

    # Create histogram
    if hyperparameter == 'dt':
        bin_centers = [1, 2, 4, 5, 10, 20]
        edges = [0, 1.5, 3, 4.5, 7.5, 15, 25]

    elif hyperparameter == 'area_ration_sample':
        edges = np.arange(10, 39)[::3] ** 2
        edges = np.array([(num_p * 100 ** 2 / 1000 ** 2) / area for num_p in edges])
        bin_centers = edges[:-1]

    else:
        bins = np.linspace(df[hyperparameter].min(), df[hyperparameter].max(), 10)
        counts, edges = np.histogram(df[hyperparameter], bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2

    # Calculate the average value for each bin
    bin_list = [df['MAE'][(df[hyperparameter] >= edges[i]) & (df[hyperparameter] < edges[i + 1])] for i in
                range(len(edges) - 1)]
    bin_means = np.array([bins.mean() for bins in bin_list])
    bin_medians = np.array([bins.median() for bins in bin_list])

    bin_std = np.array([bins.std() for bins in bin_list])

    bin_avgs_var = [df['VAR'][(df[hyperparameter] >= edges[i]) & (df[hyperparameter] < edges[i + 1])] for i in
                   range(len(edges) - 1)]
    bin_var_mean = np.array([bins.mean() for bins in bin_avgs_var])
    #bin_var_std = np.array([bins.std() for bins in bin_avgs_var])
    print(hyperparameter)
    print([len(bin) for bin in bin_list])
    ax[i].plot(np.arange(1, len(bin_centers) + 1), bin_var_mean, color=var_color, marker='v')
    ax[i].plot(np.arange(1,len(bin_centers)+1), bin_means,  color=mean_color)
    ax[i].plot(np.arange(1, len(bin_centers) + 1), bin_medians, color=median_color)
    ax[i].boxplot(bin_list, showmeans=True, showfliers = False,
                  meanprops=dict(marker='o', markeredgecolor='none', markersize=8,
                      markerfacecolor=mean_color),
                  medianprops=dict(linestyle='-', linewidth=4, color=median_color))


    if hyperparameter == 'area_ration_sample':
        ax[i].set_xticks(np.arange(1, len(bin_list)+1), ["%i" % (bin*100) for bin in bin_centers])
        ax[i].set_xlabel("Covered area [%]")
    elif hyperparameter == 'dt':
        ax[i].set_xticks(np.arange(1, len(bin_list)+1), bin_centers)
        ax[i].set_xlabel("Observation interval [years]")
    elif hyperparameter == 'ensemble_size':
        ax[i].set_xticks(np.arange(1, len(bin_list) + 1), ["%i" % (bin) for bin in bin_centers])
        ax[i].set_xlabel('Ensemble size')

    ax[i].set_ylabel('Normalized Mean Absolute Error')
    ax[i].set_ylim(0, 0.15)
    #ax[i].set_yscale('log')
    mean_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mean_color, markersize=10, label='Mean')
    median_legend = plt.Line2D([0], [0], color=median_color, lw=4, label='Median')
    var_legend = plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=var_color, markersize=10, label='Final Uncertainty')
    # Add legend

    ax[i].legend(handles=[mean_legend, median_legend, var_legend])
    #ax[i].set_xticks(bin_centers, ["0"] + ["{:.2f}".format(bin) for bin in bin_centers])

plt.savefig(f'Plots/evaluate.pdf', format="pdf")
plt.savefig(f'Plots/evaluate.png', format="png")


#fig, ax = plt.subplot_mosaic("AABBCC;DDDEEE")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),layout="tight")

for i, hyperparameter in enumerate(['initial_offset', 'initial_uncertainty']):
    # Create histogram
    if hyperparameter == 'dt':
        bin_centers = [1, 2, 4, 5, 10, 20.5]
        edges = [0, 1.5, 3, 4.5, 7.5, 15, 20.5]

    elif hyperparameter == 'area_ration_sample':
        edges = np.arange(10, 39)[::3] ** 2
        edges = np.array([(num_p * 100 ** 2 / 1000 ** 2) / area for num_p in edges])
        bin_centers = edges[:-1]

    else:
        bins = np.linspace(df[hyperparameter].min(), df[hyperparameter].max(), 10)
        counts, edges = np.histogram(df[hyperparameter], bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2

    # Calculate the average value for each bin
    bin_list = [df['MAE'][(df[hyperparameter] >= edges[i]) & (df[hyperparameter] < edges[i + 1])] for i in
                range(len(edges) - 1)]
    bin_means = np.array([bins.mean() for bins in bin_list])
    bin_medians = np.array([bins.median() for bins in bin_list])
    bin_std = np.array([bins.std() for bins in bin_list])

    bin_avgs_var = [df['VAR'][(df[hyperparameter] >= edges[i]) & (df[hyperparameter] < edges[i + 1])] for i in
                    range(len(edges) - 1)]
    bin_var_mean = np.array([bins.median() for bins in bin_avgs_var])
    bin_var_std = np.array([bins.std() for bins in bin_avgs_var])
    print(hyperparameter)
    print([len(bin) for bin in bin_list])

    ax[i].plot(np.arange(1, len(bin_centers) + 1), bin_var_mean, color=var_color, marker='v')
    ax[i].plot(np.arange(1,len(bin_centers)+1), bin_means,  color=mean_color)
    ax[i].plot(np.arange(1,len(bin_centers)+1), bin_medians, color = median_color)
    ax[i].boxplot(bin_list, showmeans=True, showfliers = False,
                  meanprops=dict(marker='o', markeredgecolor='none', markersize=8,
                      markerfacecolor=mean_color),
                  medianprops=dict(linestyle='-', linewidth=4, color=median_color))
    if hyperparameter == 'initial_offset':
        ax[i].set_xticks(np.arange(1, len(bin_list) + 1), ["%i" % bin for bin in bin_centers])
        ax[i].set_xlabel("Initial offset [%]")
    elif hyperparameter == 'initial_uncertainty':
        ax[i].set_xticks(np.arange(1, len(bin_list) + 1), ["%i" % bin for bin in bin_centers])
        ax[i].set_xlabel("Initial uncertainty [%]")

    ax[i].set_ylim(0, 0.15)
    ax[i].set_ylabel('Normalized Mean Absolute Error')
    mean_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mean_color, markersize=10, label='Mean')
    var_legend = plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=var_color, markersize=10, label='Final Uncertainty')

    median_legend = plt.Line2D([0], [0], color=median_color, lw=4, label='Median')

    # Add legend
    ax[i].legend(handles=[mean_legend, median_legend, var_legend])
    #ax[i].set_xticks(range(len(bin_centers) + 1), ["0"] + ["{:.2f}".format(bin) for bin in bin_centers])

plt.savefig(f'Plots/evaluate_init.pdf', format="pdf")
plt.savefig(f'Plots/evaluate_init.png', format="png")
