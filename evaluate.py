import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import xarray as xr

# Specify the path to your JSON file
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 3), layout="tight")

for fig_num, hyperparameter in enumerate(['ensemble_size', 'covered_area', 'dt', 'process_noise']):
#for fig_num, hyperparameter in enumerate([ 'covered_area', 'dt', 'process_noise']):

    results = []
    if hyperparameter == 'ensemble_size':
        experiment_folder = 'HPC/Results_Ensemble_Size/'
    elif hyperparameter == 'covered_area':
        experiment_folder = 'HPC/Results_Area/'
    elif hyperparameter == 'dt':
        experiment_folder = 'HPC/Results_Observation_Interval/'
    elif hyperparameter == 'process_noise':
        experiment_folder = 'HPC/Results_Process_Noise/'

    for file in os.listdir(experiment_folder):
        if file.endswith('.json'):
            #print(file)
            with open(experiment_folder + file, 'r') as f:
                results.append(json.load(f))

    hyper_results = [exp[hyperparameter] for exp in results]


    MAE = np.empty((0, len(hyper_results)))
    MAX = np.array([])
    for i in range(3):
        MAE_para = np.array([abs(exp['true_parameter'][i] - exp['esti_parameter'][i]) for exp in results])
        MAX_para = np.max(MAE_para)

        MAE = np.append(MAE, [MAE_para], axis=0)
        MAX = np.append(MAX, MAX_para)

    MAX = [5000, 0.05, 0.05]
    max_gradient = np.max(MAX[1:])
    MAE[0] = MAE[0] / MAX[0]
    MAE[1:] = MAE[1:] / max_gradient
    MAE = MAE.flatten()

    df = pd.DataFrame({'MAE': MAE,
                       hyperparameter: hyper_results + hyper_results + hyper_results,
                       # 'VAR': VAR,
                       })

    #df = df[df['MAE'] <= 100]
    #df_best = df[df['MAE'] <= 10]
    print(len(df))
    #print(len(df_best))
    colorscale = plt.get_cmap('tab20')
    mean_color = colorscale(6)
    median_color = colorscale(0)
    var_color = colorscale(2)

    # Create histogram
    if hyperparameter == 'dt':
        bin_centers = [1, 2, 4, 5, 10, 20]

    elif hyperparameter == 'covered_area':
        bin_centers = [1, 2, 4, 8, 16, 32, 64]

    elif hyperparameter=='ensemble_size':
        bin_centers = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    elif hyperparameter == 'process_noise':
        bin_centers = [0, 0.5, 1, 2, 4]



    # Calculate the average value for each bin
    #bin_list = [df['MAE'][(df[hyperparameter] >= edges[i]) & (df[hyperparameter] < edges[i + 1])] for i in
    #            range(len(edges) - 1)]
    bin_list = [df['MAE'][(df[hyperparameter] == center)] for center in bin_centers]
    bin_means = np.array([bins.mean() for bins in bin_list])
    bin_medians = np.array([bins.median() for bins in bin_list])

    bin_std = np.array([bins.std() for bins in bin_list])

    #bin_avgs_var = [df['VAR'][(df[hyperparameter] >= edges[i]) & (df[hyperparameter] < edges[i + 1])] for i in
    #               range(len(edges) - 1)]
    #bin_var_mean = np.array([bins.mean() for bins in bin_avgs_var])
    #bin_var_std = np.array([bins.std() for bins in bin_avgs_var])
    print(hyperparameter)
    print([len(bin) for bin in bin_list])
    bplot = ax[fig_num].boxplot(bin_list, showmeans=True, showfliers=False, patch_artist=True,
                  meanprops=dict(marker='o', markeredgecolor='none', markersize=8,
                                 markerfacecolor=mean_color),
                  medianprops=dict(linestyle='-', linewidth=4, color=median_color))
    #ax[i].plot(np.arange(1, len(bin_centers) + 1), bin_var_mean, color=var_color, marker='v')
    ax[fig_num].plot(np.arange(1,len(bin_centers)+1), bin_means,  color=mean_color)
    ax[fig_num].plot(np.arange(1, len(bin_centers) + 1), bin_medians, color=median_color)

    for patch in bplot['boxes']:
        patch.set_facecolor(colorscale(15))

    if hyperparameter == 'covered_area':
        # labels = sum([["%i" % (edge*100), ""] for edge in edges], [])[:-1]
        #ax[i].set_xticks(np.arange(0.5, len(bin_list) + 1, 0.5), labels)
        ax[fig_num].set_xlabel("Covered area ($a$) [%]")
    elif hyperparameter == 'dt':
        ax[fig_num].set_xticks(np.arange(1, len(bin_list)+1), bin_centers)
        ax[fig_num].set_xlabel("Observation interval ($dt$) [years]")
    elif hyperparameter == 'ensemble_size':
        #edges = [5,10,15,20,25,30,35,40,45,50]
        #labels = sum([["%i" % edge, ""] for edge in edges], [])[:-1]
        #ax[i].set_xticks(np.arange(0.5, len(bin_list) + 1, 0.5), labels)
        ax[fig_num].set_xlabel('Ensemble size ($N$)')

    grad_axis= ax[fig_num].secondary_yaxis('right')
    grad_axis.set_ylabel('Gradient Error [m/yr/m]')
    grad_axis.set_yticks(np.arange(0, 0.21, 0.1), ['%.3f'%e for e in [0, max_gradient/10, max_gradient/5]])
    ax[fig_num].set_ylabel('ELA Error [m]')
    #ax[fig_num].set_ylim(0, 0.2)
    ax[fig_num].set_yticks(np.arange(0, 0.21, 0.1),[0, int(MAX[0]/10), int(MAX[0]/5)])
    #ax[i].set_yscale('log')
    mean_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mean_color, markersize=10, label='Mean')
    median_legend = plt.Line2D([0], [0], color=median_color, lw=4, label='Median')
    var_legend = plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=var_color, markersize=10, label='Final Uncertainty')
    # Add legend

    ax[fig_num].legend(handles=[mean_legend, median_legend])
    #ax[i].set_xticks(bin_centers, ["0"] + ["{:.2f}".format(bin) for bin in bin_centers])

plt.savefig(f'Plots/evaluate.pdf', format="pdf")
plt.savefig(f'Plots/evaluate.png', format="png")
