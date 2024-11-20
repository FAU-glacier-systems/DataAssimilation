import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.data import experiment

experiment_dir = 'Experiments/Rhone/results_ETKF/'
result_files = os.listdir(experiment_dir)
df_dict = {
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
    if file.startswith('result'):
        result = json.load(open(os.path.join(experiment_dir, file), 'r'))
        file_split = file.split('_')

        df_dict['seed'].append(float(file_split[2]))
        df_dict['inflation_factor'].append(float(file_split[4].rstrip('.json')))
        df_dict['final_mean_ela'].append(result['final_mean_estimate'][0])
        df_dict['final_mean_grad_abl'].append(result['final_mean_estimate'][1])
        df_dict['final_mean_grad_acc'].append(result['final_mean_estimate'][2])
        df_dict['final_std_ela'].append(result['final_std'][0])
        df_dict['final_std_grad_abl'].append(result['final_std'][1])
        df_dict['final_std_grad_acc'].append(result['final_std'][2])

df = pd.DataFrame(df_dict)


filtered_df = df[df['inflation_factor'] == 1.0]

std_df = filtered_df.std().reset_index()
mean_df = filtered_df.mean().reset_index()
print(std_df['final_mean_ela'])