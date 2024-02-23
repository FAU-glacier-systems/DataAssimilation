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
        with open(experiment_folder+file, 'r') as f:
            try:
                results.append(json.load(f))
            except:
                print("Error")


input_file = 'Inversion/geology-optimized.nc'
input_ds = xr.open_dataset(input_file)
icemask = input_ds['icemask']

area = np.sum(icemask) * 100 ** 2 / 1000 ** 2

MAE = [abs(exp['true_parameter'][0]-exp['esti_parameter'][0]) for exp in results]
VAR = [exp['esit_var'][0][0] for exp in results]
ensemble_size = [exp['ensemble_size'] for exp in results]
dt = [exp['dt'] for exp in results]
offset = [exp['initial_offset'] for exp in results]
uncertainty = [exp['initial_uncertainity'] for exp in results]

num_sample_points = [exp['num_sample_points'] for exp in results]
area_ration_sample = np.array([(num_p * 100 ** 2 / 1000 ** 2)/area for num_p in num_sample_points])


df = pd.DataFrame({'MAE': MAE,
                   'area_ration_sample':area_ration_sample,
                   'VAR': VAR,
                   'ensemble_size': ensemble_size,
                   'dt': dt,
                   'initial_offset':offset,
                   'initial_uncertainity':uncertainty})


#df = df[df['MAE'] <= 100]
#df = df[df['area_ration_sample']>0.02]
df_best = df[df['MAE'] <= 10]
print(len(df))
print(len(df_best))

for hyperparameter in ['dt', 'area_ration_sample', 'ensemble_size', 'initial_offset', 'initial_uncertainity']:
    # Create histogram
    if hyperparameter == 'dt':
        bin_centers = [1,2,4,5,10,20.5]
        edges = [0,1.5,3,4.5,7.5,15,20.5]


    elif hyperparameter == 'area_ration_sample':
        edges = np.arange(10, 39)[::3]**2
        edges = np.array([(num_p * 100 ** 2 / 1000 ** 2)/area for num_p in edges])
        bin_centers = edges[:-1]

    else:
        bins = np.linspace(df[hyperparameter].min(), df[hyperparameter].max(), 10)
        counts, edges = np.histogram(df[hyperparameter], bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2




    # Calculate the average value for each bin

    bin_list = [df['MAE'][(df[hyperparameter] >= edges[i]) & (df[hyperparameter] < edges[i+1])] for i in range(len(edges)-1)]
    bin_means = np.array([bins.median() for bins in bin_list])
    bin_std = np.array([bins.std() for bins in bin_list])

    bin_avgs_var = [df['VAR'][(df[hyperparameter] >= edges[i]) & (df[hyperparameter] < edges[i+1])] for i in range(len(edges)-1)]
    bin_var_mean = np.array([bins.median() for bins in bin_avgs_var])
    bin_var_std = np.array([bins.std() for bins in bin_avgs_var])
    print(hyperparameter)
    print([len(bin) for bin in bin_list])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.boxplot(bin_list)
    if hyperparameter == 'area_ration_sample':
        ax.set_xlabel("Percentage of sampled area [%]")
    else:
        ax.set_xlabel(hyperparameter)
    ax.set_ylabel('Error of ELA estimate')
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(bin_centers)+1), ["0"]+["{:.2f}".format(bin) for bin in bin_centers])
    ax.legend()
    plt.savefig(f'Plots/{hyperparameter}.png')
    plt.clf()
    #ax.fill_between(bin_centers, bin_means-bin_std, bin_means+bin_std, alpha=0.3, color='C0')
    ##ax.plot(bin_centers, bin_means, label='Mean Absolute Error')
    #ax.scatter(df[hyperparameter], df['MAE'], alpha=0.2)
    #ax.fill_between(bin_centers, bin_var_mean-bin_var_std, bin_var_mean+bin_var_std, alpha=0.3, color='C1')
    #ax.plot(bin_centers, bin_var_mean, label='Ensemble Variance')
    #ax.scatter(df[hyperparameter], df['VAR'], alpha=0.2)


    fig, ax = plt.subplots(figsize=(7, 7))

    if hyperparameter == 'area_ration_sample':
        edges = np.arange(10, 39)[::3] ** 2
        edges = np.array([(num_p * 100 ** 2 / 1000 ** 2) / area for num_p in edges])
        bin_centers = edges[:-1]
        ax.hist(df_best[hyperparameter], edges)
        ax.set_ylabel("Number of runs with ERROR < 100m")
        ax.set_xlabel("Percentage of sampled area [%]")
    else:
        ax.hist(df_best[hyperparameter])
        ax.set_ylabel("Number of runs with ERROR < 100m")
        ax.set_xlabel(hyperparameter)
    plt.savefig(f'Plots/{hyperparameter}_hist.png')
    plt.clf()



plt.scatter(df['initial_offset'], df['MAE'])
plt.ylim(0,50)
plt.savefig(f'Plots/MAE.png')


#fig = px.scatter_3d(df, x='ensemble_size', y='area_ration_sample', z='dt',
#              color='MAE', range_color=[0, 500])
#fig.show()