import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import xarray as xr

# Specify the path to your JSON file

results = []
experiment_folder = 'Experiments/first_run/'
for file in os.listdir(experiment_folder):
    if file.endswith('.json'):
        with open(experiment_folder+file, 'r') as f:
            results.append(json.load(f))

input_file = 'Inversion/geology-optimized.nc'
input_ds = xr.open_dataset(input_file)
icemask = input_ds['icemask']

area = np.sum(icemask) * 50 ** 2 / 1000 ** 2

MAE = [abs(exp['true_parameter'][0]-exp['esti_parameter'][0]) for exp in results]
VAR = [exp['esit_var'][0][0] for exp in results]
ensemble_size = [exp['ensemble_size'] for exp in results]
dt = [exp['dt'] for exp in results]
num_sample_points = [exp['num_sample_points'] for exp in results]
area_ration_sample = np.array([(num_p * 50 ** 2 / 1000 ** 2)/area for num_p in num_sample_points])


df = pd.DataFrame({'MAE': MAE,
                   'area_ration_sample':area_ration_sample,
                   'VAR': VAR,
                   'ensemble_size': ensemble_size,
                   'dt': dt,})

df = df[df['MAE'] <= 10]
df = df[df['area_ration_sample']>0.02]
for hyperparameter in ['dt', 'area_ration_sample', 'ensemble_size', ]:
    # Create histogram
    if hyperparameter == 'dt':
        bin_centers = [1,2,4,5,10,20.5]
        edges = [0,1.5,3,4.5,7.5,15,20.5]


    else:
        bins = np.linspace(df[hyperparameter].min(), df[hyperparameter].max(), 6)
        counts, edges = np.histogram(df[hyperparameter], bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2



    # Calculate the average value for each bin

    bin_list = [df['MAE'][(df[hyperparameter] >= edges[i]) & (df[hyperparameter] < edges[i+1])] for i in range(len(edges)-1)]
    bin_means = np.array([bins.mean() for bins in bin_list])
    bin_std = np.array([bins.std() for bins in bin_list])

    bin_avgs_var = [df['VAR'][(df[hyperparameter] >= edges[i]) & (df[hyperparameter] < edges[i+1])] for i in range(len(edges)-1)]
    bin_var_mean = np.array([bins.mean() for bins in bin_avgs_var])
    bin_var_std = np.array([bins.std() for bins in bin_avgs_var])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.fill_between(bin_centers, bin_means-bin_std, bin_means+bin_std, alpha=0.3, color='C0')
    ax.plot(bin_centers, bin_means, label='Mean Absolute Error')
    ax.scatter(df[hyperparameter], df['MAE'], alpha=0.2)
    ax.fill_between(bin_centers, bin_var_mean-bin_var_std, bin_var_mean+bin_var_std, alpha=0.3, color='C1')
    ax.plot(bin_centers, bin_var_mean, label='Ensemble Variance')
    ax.scatter(df[hyperparameter], df['VAR'], alpha=0.2)
    if hyperparameter == 'area_ration_sample':
        ax.set_xlabel("Percentage of sampled area [%]")
    else:
        ax.set_xlabel(hyperparameter)
    ax.set_ylabel('Error of ELA estimate')
    ax.set_ylim(0, 10)
    ax.legend()
    plt.savefig(f'Plots/{hyperparameter}.png')
