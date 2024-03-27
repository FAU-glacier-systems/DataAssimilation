import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_MAE():

    # global figure
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), layout="tight")

    # run for every hyperparameters
    for fignum, hyperparameter in enumerate(['ensemble_size', 'covered_area', 'dt', 'process_noise']):

        # translate name to folder name
        if hyperparameter == 'ensemble_size':
            experiment_folder = 'Results_Ensemble_Size/'
        elif hyperparameter == 'covered_area':
            experiment_folder = 'Results_Area/'
        elif hyperparameter == 'dt':
            experiment_folder = 'Results_Observation_Interval/'
        elif hyperparameter == 'process_noise':
            experiment_folder = 'Results_Process_Noise/'

        # load result json files
        results = []
        for folder in os.listdir(experiment_folder):
            for file in os.listdir(experiment_folder + folder):
                if file.endswith('.json'):
                    with open(experiment_folder + folder+'/'+file, 'r') as f:
                        content = json.load(f)
                        if content != None:
                            results.append(content)

        # get the hyperparameter values
        hyper_results = [exp[hyperparameter] for exp in results]

        # Compute the MAE and track the maximum MAE for normalisation
        MAE = np.empty((0, len(hyper_results)))
        MAX = np.array([])
        for x in range(3):
            MAE_para = np.array([abs(exp['true_parameter'][x] - exp['esti_parameter'][x]) for exp in results])
            MAX_para = np.max(MAE_para)
            MAE = np.append(MAE, [MAE_para], axis=0)
            MAX = np.append(MAX, MAX_para)

        # Normalise
        #MAX = [100, 0.05, 0.05]
        max_gradient = np.max(MAX[1:])
        MAE[0] = MAE[0] / MAX[0]
        MAE[1] = MAE[1] / max_gradient
        MAE[2] = MAE[2] / max_gradient
        #MAE = MAE.flatten()

        # create pandas data frame
        df = pd.DataFrame({'MAE0': MAE[0],
                           'MAE1': MAE[2],
                           'MAE2': MAE[1],
                           hyperparameter: hyper_results #+ hyper_results + hyper_results,
                           })
        # define colors
        print(len(df))
        colorscale = plt.get_cmap('tab20c')
        colormap = [colorscale(16),colorscale(18), colorscale(19),
                    colorscale(0), colorscale(2), colorscale(3),
                    colorscale(4), colorscale(6), colorscale(7)]

        mean_color = colorscale(6)
        median_color = colorscale(0)
        var_color = colorscale(4)
        # define bin centers
        if hyperparameter == 'dt':
            bin_centers = [1, 2, 4, 5, 10, 20]

        elif hyperparameter == 'covered_area':
            bin_centers = [1, 2, 4, 8, 16, 32]

        elif hyperparameter=='ensemble_size':
            bin_centers = [5, 10, 20, 30, 40, 50]

        elif hyperparameter == 'process_noise':
            bin_centers = [0, 0.5, 1, 2, 4]

        # group the MAE by bin_centers\\
        marker = ["o", "^", "v"]
        for x, label in enumerate(["ELA","grad_acc", "grad_abl"]):
            bin_list = [df['MAE'+str(x)][(df[hyperparameter] == center)] for center in bin_centers]
            bin_means = np.array([bins.mean() for bins in bin_list])
            bin_medians = np.array([bins.median() for bins in bin_list])

            print(hyperparameter)
            print([len(bin) for bin in bin_list])

            i = fignum % 2
            j = fignum // 2
            bplot = ax[i,j].boxplot(bin_list, positions=np.arange(x,len(bin_list)*3,3)-((x-1)*0.44),
                                    showmeans=True, showfliers=False, patch_artist=True,
                                    boxprops=dict(facecolor=colormap[x*3+2], color=colormap[x*3+1]),
                                    capprops=dict(color=colormap[x*3+1] ),
                                    whiskerprops=dict(color=colormap[x*3+1]),
                                    meanprops=dict(marker='o',
                                         markeredgecolor='none',
                                         markersize=8,
                                         markerfacecolor="none"),
                                    medianprops=dict(linestyle='-', linewidth=4, color="none"))
            # fill with colors

            #for patch in bplot['boxes']:
            #    patch.set_facecolor(colorscale(x*2+1))
            #ax[i].plot(np.arange(1, len(bin_centers) + 1), bin_var_mean, color=var_color, marker='v')
            ax[i,j].plot(np.arange(x,len(bin_list)*3,3)-((x-1)*0.44), bin_means,
                         color=colormap[x*3],marker=marker[x], label=label, zorder=10+(2-x))
        #ax[i,j].plot(np.arange(1, len(bin_centers) + 1), bin_medians, color=median_color)


        ax[i,j].set_xticks(np.arange(1,len(bin_list)*3,3), bin_centers)
        if hyperparameter == 'covered_area':
            ax[i,j].set_xlabel("Covered area ($a$) [%]")
        elif hyperparameter == 'dt':
            ax[i,j].set_xlabel("Observation interval ($dt$) [years]")
        elif hyperparameter == 'ensemble_size':
            ax[i,j].set_xlabel('Ensemble size ($N$)')
        elif hyperparameter == 'process_noise':
            ax[i,j].set_xlabel('Process noise ($Q$)')

        grad_axis= ax[i,j].secondary_yaxis('right')
        grad_axis.set_ylabel('Gradient Error [m/yr/m]')
        grad_axis.set_yticks([0, 0.5, 1], ['%.3f' % e for e in [0, max_gradient / 2, max_gradient]])
        ax[i,j].set_ylabel('ELA Error [m]')
        ax[i,j].set_ylim(0, 1)
        ax[i,j].set_yticks([0, 0.5, 1],[0, int(MAX[0]/2), int(MAX[0])])
        #ax[i].set_yscale('log')
        mean_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mean_color, markersize=10, label='Mean')
        median_legend = plt.Line2D([0], [0], color=median_color, lw=4, label='Median')
        var_legend = plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=var_color, markersize=10, label='Final Uncertainty')
        # Add legend

        ax[i,j].legend()#handles=[mean_legend, median_legend])
        #ax[i].set_xticks(bin_centers, ["0"] + ["{:.2f}".format(bin) for bin in bin_centers])

    plt.savefig(f'Plots/MAE.pdf', format="pdf")
    plt.savefig(f'Plots/MAE.png', format="png", dpi=300)

def plot_final_spread():
    # Specify the path to your JSON file
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), layout="tight")

    for fignum, hyperparameter in enumerate(['ensemble_size', 'covered_area', 'dt', 'process_noise']):
    #for fignum, hyperparameter in enumerate([ 'covered_area', 'dt',]):
        i = fignum % 2
        j = fignum // 2
        results = []
        if hyperparameter == 'ensemble_size':
            experiment_folder = 'Results_Ensemble_Size/'
        elif hyperparameter == 'covered_area':
            experiment_folder = 'Results_Area/'
        elif hyperparameter == 'dt':
            experiment_folder = 'Results_Observation_Interval/'
        elif hyperparameter == 'process_noise':
            experiment_folder = 'Results_Process_Noise/'

        for folder in os.listdir(experiment_folder):
            for file in os.listdir(experiment_folder + folder):
                if file.endswith('.json'):
                    # print(file)
                    with open(experiment_folder + folder +'/' + file, 'r') as f:
                        content = json.load(f)
                        if content != None:
                            results.append(content)

        hyper_results = [exp[hyperparameter] for exp in results]

        spread = np.empty((0, len(hyper_results)))
        MAX = np.array([])
        for x in range(3):
            spread_para = np.sqrt(np.array([exp['esit_var'][x][x] for exp in results]))
            MAX_spread = np.max(spread_para)
            spread = np.append(spread, [spread_para], axis=0)
            MAX = np.append(MAX, MAX_spread)

        #MAX = [5000, 0.05, 0.05]
        max_gradient = np.max(MAX[1:])
        spread[0] = spread[0] / MAX[0]
        spread[1:] = spread[1:] / max_gradient
        spread = spread.flatten()

        df = pd.DataFrame({'spread': spread,
                           hyperparameter: hyper_results + hyper_results + hyper_results,
                           })

        print(len(df))
        colorscale = plt.get_cmap('tab20')
        mean_color = colorscale(6)
        median_color = colorscale(0)
        var_color = colorscale(2)

        # Create histogram
        if hyperparameter == 'dt':
            bin_centers = [1, 2, 4, 5, 10, 20]

        elif hyperparameter == 'covered_area':
            bin_centers = [1, 2, 4, 8, 16, 32, 64]

        elif hyperparameter == 'ensemble_size':
            bin_centers = [5, 10, 20, 30, 40, 50]

        elif hyperparameter == 'process_noise':
            bin_centers = [0, 0.5, 1, 2, 4]

        bin_list = [df['spread'][(df[hyperparameter] == center)] for center in bin_centers]
        bin_means = np.array([bins.mean() for bins in bin_list])
        bin_medians = np.array([bins.median() for bins in bin_list])

        bin_std = np.array([bins.std() for bins in bin_list])
        print(hyperparameter)
        print([len(bin) for bin in bin_list])
        bplot = ax[i, j].boxplot(bin_list, showmeans=True, showfliers=False, patch_artist=True,
                                 meanprops=dict(marker='o', markeredgecolor='none', markersize=8,
                                                markerfacecolor=mean_color),
                                 medianprops=dict(linestyle='-', linewidth=4, color=median_color))
        # ax[i].plot(np.arange(1, len(bin_centers) + 1), bin_var_mean, color=var_color, marker='v')
        ax[i, j].plot(np.arange(1, len(bin_centers) + 1), bin_means, color=mean_color)
        ax[i, j].plot(np.arange(1, len(bin_centers) + 1), bin_medians, color=median_color)

        for patch in bplot['boxes']:
            patch.set_facecolor(colorscale(15))

        ax[i, j].set_xticks(np.arange(1, len(bin_centers) + 1), bin_centers)
        if hyperparameter == 'covered_area':
            ax[i, j].set_xlabel("Covered area ($a$) [%]")
        elif hyperparameter == 'dt':
            ax[i, j].set_xlabel("Observation interval ($dt$) [years]")
        elif hyperparameter == 'ensemble_size':
            ax[i, j].set_xlabel('Ensemble size ($N$)')
        elif hyperparameter == 'process_noise':
            ax[i, j].set_xlabel('Process noise ($Q$)')

        grad_axis = ax[i, j].secondary_yaxis('right')
        grad_axis.set_ylabel('Final spread of Gradient[m/yr/m]')
        grad_axis.set_yticks([0, 0.5, 1], ['%.3f' % e for e in [0, max_gradient / 2, max_gradient]])
        ax[i, j].set_ylabel('Final spread of ELA [m]')
        #ax[i, j].set_ylim(0, 0.2)
        ax[i, j].set_yticks([0, 0.5, 1], [0, int(MAX[0] / 2), int(MAX[0])])
        # ax[i].set_yscale('log')
        mean_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mean_color, markersize=10,
                                 label='Mean')
        median_legend = plt.Line2D([0], [0], color=median_color, lw=4, label='Median')
        var_legend = plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=var_color, markersize=10,
                                label='Final Uncertainty')
        # Add legend

        ax[i, j].legend(handles=[mean_legend, median_legend])
        # ax[i].set_xticks(bin_centers, ["0"] + ["{:.2f}".format(bin) for bin in bin_centers])

    plt.savefig(f'Plots/final_spread.pdf', format="pdf")
    plt.savefig(f'Plots/final_spread.png', format="png")

if __name__ == '__main__':
    plot_MAE()
    plot_final_spread()
