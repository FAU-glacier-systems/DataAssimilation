import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

glacier_name = "Aletsch"
ensemble_size = 100
experiment_dir = 'Experiments/' + glacier_name
result_files = os.listdir(experiment_dir)

inflation_data = {
    'inflation_factor': [],
    'seed':[],
    'final_mean_ela': [],
    'final_mean_grad_abl': [],
    'final_mean_grad_acc': [],
    'final_std_ela': [],
    'final_std_grad_abl': [],
    'final_std_grad_acc': [],

}
for file in result_files:
    print(file)
    if file.startswith('result') :
        result = json.load(open(os.path.join(experiment_dir, file), 'r'))
        file_split = file.split('_')

        inflation_data['seed'].append(float(file_split[2]))
        inflation_data['inflation_factor'].append(float(file_split[4].rstrip('.json')))
        inflation_data['final_mean_ela'].append(result['final_mean_estimate'][0])
        inflation_data['final_mean_grad_abl'].append(result['final_mean_estimate'][1])
        inflation_data['final_mean_grad_acc'].append(result['final_mean_estimate'][2])
        inflation_data['final_std_ela'].append(result['final_std'][0])
        inflation_data['final_std_grad_abl'].append(result['final_std'][1])
        inflation_data['final_std_grad_acc'].append(result['final_std'][2])

df = pd.DataFrame(inflation_data)

filter_values = {
    'Equilibrium_Line_Altitude': (2000, 4000, 'final_mean_ela', 'final_std_ela'),
    'Ablation_Gradient': (-1, 1, 'final_mean_grad_abl', 'final_std_grad_abl'),
    'Accumulation_Gradient': (-1, 1, 'final_mean_grad_acc', 'final_std_grad_acc')
}
filtered_df = df[
    (df['final_mean_ela'] >= filter_values['Equilibrium_Line_Altitude'][0]) &
    (df['final_mean_ela'] <= filter_values['Equilibrium_Line_Altitude'][1]) &
    (df['final_mean_grad_abl'] >= filter_values['Ablation_Gradient'][0]) &
    (df['final_mean_grad_abl'] <= filter_values['Ablation_Gradient'][1]) &
    (df['final_mean_grad_acc'] >= filter_values['Accumulation_Gradient'][0]) &
    (df['final_mean_grad_acc'] <= filter_values['Accumulation_Gradient'][1])
]
mean_ensemble_spread = filtered_df.groupby('inflation_factor').mean().reset_index()
std_ensemble_spread = filtered_df.groupby('inflation_factor').std().reset_index()

import matplotlib.pyplot as plt

# Create two figures, each with 3 subplots (one for each parameter)
fig_scatter, axes_scatter = plt.subplots(1, 3, figsize=(15, 5))
fig_spread, axes_spread = plt.subplots(1, 3, figsize=(15, 5))

for i, parameter in enumerate(filter_values.keys()):
    # Scatter plot for each parameter
    axes_scatter[i].scatter(filtered_df['inflation_factor'], filtered_df[filter_values[parameter][2]], marker='o')
    axes_scatter[i].set_xlabel("Inflation Factor")
    axes_scatter[i].set_ylabel(f"{parameter} Value")
    axes_scatter[i].set_title(f"Final Mean {parameter} of multiple runs\n {glacier_name}")

    # Line plot for mean and standard deviation for each parameter
    axes_spread[i].plot(mean_ensemble_spread['inflation_factor'], mean_ensemble_spread[filter_values[parameter][3]],
                        marker='o', linestyle='-', label='Mean std of individual calibration runs')
    axes_spread[i].plot(mean_ensemble_spread['inflation_factor'], std_ensemble_spread[filter_values[parameter][2]],
                        marker='o', linestyle='-', label='Std of mean of multiple calibration runs')
    axes_spread[i].set_xlabel("Inflation Factor")
    axes_spread[i].set_ylabel(f"{parameter} Standard Deviation")
    axes_spread[i].set_title(f"{parameter} spread of {glacier_name}")
    axes_spread[i].legend()

# Adjust layout and save both figures
fig_scatter.tight_layout()
fig_spread.tight_layout()
fig_scatter.savefig(f'Plots/{glacier_name}_{ensemble_size}_compare_inflation_scatter.png', dpi=300)
fig_spread.savefig(f'Plots/{glacier_name}_{ensemble_size}_compare_inflation_spread.png', dpi=300)



