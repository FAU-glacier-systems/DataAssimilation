import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plt.style.use('seaborn-v0_8-whitegrid')

def plot_MAE():

    # global figure
    fig_para, ax_para = plt.subplots(nrows=2, ncols=2, figsize=(10, 5),) #layout="tight")
    fig_spread, ax_spread = plt.subplots(nrows=2, ncols=2, figsize=(10, 5),) #layout="tight")

    # run for every hyperparameters
    for fignum, hyperparameter in enumerate(['ensemble_size',  'process_noise', 'dt','covered_area']):

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
        MAX_para_total = np.array([])
        spread = np.empty((0, len(hyper_results)))
        MAX_spread_total = np.array([])
        for x in range(3):
            MAE_para = np.array([abs(exp['true_parameter'][x] - exp['esti_parameter'][x]) for exp in results])
            MAX_para = np.max(MAE_para)
            MAE = np.append(MAE, [MAE_para], axis=0)
            MAX_para_total = np.append(MAX_para_total, MAX_para)

            spread_para = np.sqrt(np.array([exp['esit_var'][x][x] for exp in results]))
            MAX_spread = np.max(spread_para)
            spread = np.append(spread, [spread_para], axis=0)
            MAX_spread_total = np.append(MAX_spread_total, MAX_spread)

        # Normalise
        #MAX = [100, 0.05, 0.05]
        print(int(MAX_para_total[0]))

        max_gradient = np.max(MAX_para_total[1:])
        MAE[0] = MAE[0] / MAX_para_total[0]
        MAE[1] = MAE[1] / max_gradient
        MAE[2] = MAE[2] / max_gradient
        # MAX = [5000, 0.05, 0.05]
        max_gradient_spread = np.max(MAX_spread_total[1:])
        spread[0] = spread[0] / MAX_spread_total[0]
        spread[1:] = spread[1:] / max_gradient_spread
        #MAE = MAE.flatten()

        # create pandas data frame
        df = pd.DataFrame({'MAE0': MAE[2],
                           'MAE1': MAE[0],
                           'MAE2': MAE[1],
                           'spread0': spread[2],
                           'spread1': spread[0],
                           'spread2': spread[1],
                           hyperparameter: hyper_results #+ hyper_results + hyper_results,
                           })
        # define colors
        #print(len(df))
        colorscale = plt.get_cmap('tab20c')
        colormap = [colorscale(0), colorscale(2), colorscale(3),
                    'black',colorscale(18), colorscale(19),

                    colorscale(4), colorscale(6), colorscale(7)]
        #csfont = {'fontname': 'Comic Sans'}

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
        marker = [ "^","o",  "v",]
        mean_max = 0
        for  x, label in enumerate(["Accumulation Gradient","Equilibrium Line Altitude", "Ablation Gradient"]):
            bin_list_para = [df['MAE'+str(x)][(df[hyperparameter] == center)] for center in bin_centers]
            bin_means_para = np.array([bins.mean() for bins in bin_list_para])
            if mean_max < max(bin_means_para):
                mean_max = max(bin_means_para)
            bin_list_spread = [df['spread' + str(x)][(df[hyperparameter] == center)] for center in bin_centers]
            bin_means_spread = np.array([bins.mean() for bins in bin_list_spread])
            if mean_max < max(bin_means_para):
                mean_max = max(bin_means_para)


            print(hyperparameter)
            print([len(bin) for bin in bin_list_para])

            i = fignum % 2
            j = fignum // 2
            bplot_para = ax_para[i,j].boxplot(bin_list_para, positions=np.arange(x,len(bin_list_para)*3,3)-((x-1)*0.44),
                                    showmeans=True, showfliers=True, patch_artist=True,
                                    boxprops=dict(facecolor=colormap[x*3+2], color=colormap[x*3+1]),
                                    capprops=dict(color=colormap[x*3+1] ),
                                    whiskerprops=dict(color=colormap[x*3+1]),
                                    flierprops=dict(markeredgecolor=colormap[x*3+2], marker=marker[x]),
                                    meanprops=dict(marker='o',
                                         markeredgecolor='none',
                                         markersize=8,
                                         markerfacecolor="none"),
                                    medianprops=dict(linestyle='-', linewidth=4, color="none"))

            bplot_spreadd = ax_spread[i, j].boxplot(bin_list_spread,
                                                    positions=np.arange(x, len(bin_list_spread) * 3, 3) - ((x - 1) * 0.44),
                                          showmeans=True, showfliers=True, patch_artist=True,
                                          boxprops=dict(facecolor=colormap[x * 3 + 2], color=colormap[x * 3 + 1]),
                                          capprops=dict(color=colormap[x * 3 + 1]),
                                          whiskerprops=dict(color=colormap[x * 3 + 1]),
                                          flierprops=dict(markeredgecolor=colormap[x * 3 + 2], marker=marker[x]),
                                          meanprops=dict(marker='o',
                                                         markeredgecolor='none',
                                                         markersize=8,
                                                         markerfacecolor="none"),
                                          medianprops=dict(linestyle='-', linewidth=4, color="none"))
            # fill with colors

            #for patch in bplot['boxes']:
            #    patch.set_facecolor(colorscale(x*2+1))
            #ax[i].plot(np.arange(1, len(bin_centers) + 1), bin_var_mean, color=var_color, marker='v')
            ax_para[i,j].plot(np.arange(x,len(bin_list_para)*3,3) - ((x-1)*0.44), bin_means_para,
                         color=colormap[x*3],marker=marker[x], label=label, zorder=10+(-abs(x-1)))
            ax_spread[i, j].plot(np.arange(x, len(bin_list_spread) * 3, 3) - ((x - 1) * 0.44), bin_means_spread,
                               color=colormap[x * 3], marker=marker[x], label=label, zorder=10 + (-abs(x-1)))
        #ax[i,j].plot(np.arange(1, len(bin_centers) + 1), bin_medians, color=median_color)
        grad_axis_para = ax_para[i, j].secondary_yaxis('right')
        grad_axis_para.set_ylabel('Gradient Error [m/yr/m]')
        ax_para[i, j].set_ylabel('ELA Error [m]')

        grad_axis_spread = ax_spread[i, j].secondary_yaxis('right')
        grad_axis_spread.set_ylabel('Gradient Spread [m/yr/m]')
        ax_spread[i, j].set_ylabel('ELA Spread [m]')
        ax_para[i, j].set_yticks([0, 1 / 3, 2 / 3, 1],
                            [0, int(MAX_para_total[0] / 3), int(MAX_para_total[0] * 2 / 3),
                             int(MAX_para_total[0])])
        grad_axis_para.set_yticks([0, 1 / 3, 2 / 3, 1], [0] + ['%.4f' % e for e in
                                                          [max_gradient / 3, max_gradient * 2 / 3,
                                                           max_gradient]])
        ax_spread[i, j].set_yticks([0, 1 / 3, 2 / 3, 1],
                                 [0, int(MAX_spread_total[0] / 3), int(MAX_spread_total[0] * 2 / 3),
                                  int(MAX_spread_total[0])])
        grad_axis_spread.set_yticks([0, 1 / 3, 2 / 3, 1], [0] + ['%.4f' % e for e in
                                                               [max_gradient_spread / 3, max_gradient_spread * 2 / 3,
                                                                max_gradient_spread]])

        for ax, grad_axis in [(ax_para, grad_axis_para), (ax_spread, grad_axis_spread)]:
            ax[i,j].set_xticks(np.arange(1,len(bin_list_para)*3,3), bin_centers)
            if hyperparameter == 'covered_area':
                ax[i,j].set_xlabel("Covered Area ($a$) [%]")
            elif hyperparameter == 'dt':
                ax[i,j].set_xlabel("Observation Interval ($dt$) [years]")
            elif hyperparameter == 'ensemble_size':
                ax[i,j].set_xlabel('Ensemble Size ($N$)')
            elif hyperparameter == 'process_noise':
                ax[i,j].set_xlabel('Process Noise ($Q$)')

            ax[i,j].spines['top'].set_visible(False)
            ax[i,j].spines['right'].set_visible(False)
            ax[i,j].spines['bottom'].set_visible(False)
            ax[i,j].spines['left'].set_visible(False)
            grad_axis.spines['right'].set_visible(False)

            ax[i,j].grid(axis="y", color="lightgray", linestyle="-")
            ax[i,j].grid(axis="x", color="lightgray", linestyle="-", which='minor')
            ax[i,j].set_ylim(-0.025,1.025)
            ax[i, j].set_xlim(-0.75, len(bin_list_para)*3-0.25)
            ax[i,j].set_xticks(np.arange(-0.5,len(bin_list_para)*3,3), minor=True)
            ax[i,j].yaxis.set_tick_params(left=False)
            #ax[i,j].xaxis.set_tick_params(bottom=True, which='minor',color="lightgray")
            ax[i, j].xaxis.set_tick_params(bottom=False, which='both',)



            grad_axis.yaxis.set_tick_params(right=False)
            handles, labels = ax[i, j].get_legend_handles_labels()


    fig_para.legend(handles, labels, loc='upper center', ncol=3)
    fig_para.tight_layout()
    fig_para.subplots_adjust(top=0.9, bottom=0.1)
    fig_para.savefig(f'Plots/MAE.pdf', format="pdf")
    fig_para.savefig(f'Plots/MAE.png', format="png", dpi=300)

    fig_spread.legend(handles, labels, loc='upper center', ncol=3)
    fig_spread.tight_layout()
    fig_spread.subplots_adjust(top=0.9, bottom=0.1)
    fig_spread.savefig(f'Plots/spread.pdf', format="pdf")
    fig_spread.savefig(f'Plots/spread.png', format="png", dpi=300)


if __name__ == '__main__':
    plot_MAE()

