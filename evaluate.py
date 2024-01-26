import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Specify the path to your JSON file

results = []
experiment_folder = 'Experiments/'
for file in os.listdir(experiment_folder):
    if file.endswith('.json'):
        with open(experiment_folder+file, 'r') as f:
            results.append(json.load(f))


MAE = [abs(exp['true_parameter'][0]-exp['esti_parameter'][0]) for exp in results]
num_sample_points = [exp['num_sample_points'] for exp in results]
VAR = [exp['esit_var'][0][0] for exp in results]
ensemble_size = [exp['ensemble_size'] for exp in results]
dt = [exp['dt'] for exp in results]

df = pd.DataFrame({'MAE': MAE,
                   'num_sample_points':num_sample_points,
                   'VAR': VAR,
                   'ensemble_size': ensemble_size,
                   'dt': dt,})

df = df[df['MAE'] <= 10]
df_num_sample_points = df.sort_values(by='num_sample_points')
plt.figure(figsize=(5, 5))
plt.plot(df_num_sample_points['num_sample_points'],df_num_sample_points['MAE'], label='MAE')
plt.plot(df_num_sample_points['num_sample_points'], df_num_sample_points['VAR'], label='Mean ensemble variance')
plt.xlabel('Number of sample points')
plt.ylabel('Error of ELA estimate')
plt.legend()
plt.savefig('Plots/Num_sample_points.png')


# Group by 'dt' and calculate the mean for each group
df_grouped = df.groupby('dt').mean().reset_index()

# Plotting
plt.figure(figsize=(5, 5))

plt.plot(df_grouped['dt'], df_grouped['MAE'], label='Mean MAE')
plt.plot(df_grouped['dt'], df_grouped['VAR'], label='Mean ensemble variance')
plt.xlabel('Observation Interval')
plt.ylabel('Error of ELA estimate')
plt.legend()
plt.savefig('Plots/dt.png')


df_ensemble_size = df.sort_values(by='ensemble_size')
df_ensemble_size = df_ensemble_size.groupby('ensemble_size', ).mean().reset_index()

plt.figure(figsize=(5, 5))
plt.plot(df_ensemble_size['ensemble_size'],df_ensemble_size['MAE'], label='MAE')
plt.plot(df_ensemble_size['ensemble_size'], df_ensemble_size['VAR'], label='Mean ensemble variance')
plt.xlabel('ensemble_size')
plt.ylabel('Error of ELA estimate')
plt.legend()
plt.savefig('Plots/ensemble_size.png')


fig = px.scatter_3d(df, x='ensemble_size', y='num_sample_points', z='dt',
              color='MAE', range_color=(0,5))
fig.show()

# Now 'data' contains the content of your JSON file as a Python dictionary or list
print(results)