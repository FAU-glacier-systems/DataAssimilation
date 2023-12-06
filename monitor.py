import copy
import os.path

import matplotlib.pyplot as plt
import numpy as np
import random
import json

class Monitor:
    def __init__(self, N, year_range, true_glacier, hist_true_y, res, dt):
        self.res = res
        self.dt = dt
        self.N = N
        self.year_range = year_range
        self.year_range_repeat = np.repeat(self.year_range, 2)[1:]
        self.start_year = year_range[0]
        self.true_glacier = true_glacier
        self.bedrock = true_glacier['topg'][0]
        self.bedrock = self.bedrock[::-1]

        self.hist_true_y = hist_true_y
        #self.hist_observ = hist_observ
        self.hist_state_x = []
        self.hist_ensemble_x = []
        self.hist_ensemble_y = []

        self.output_dir = 'plots/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open('data_synthetic_observations/default/params.json') as f:
            params = json.load(f)
            self.smb = params['smb_simple_array']


    def plot(self, year, state_x, ensemble_x, ensemble_y):
        self.hist_state_x.append(copy.copy(state_x))
        self.hist_ensemble_x.append(copy.copy(ensemble_x))
        self.hist_ensemble_y.append(copy.copy(ensemble_y))

        # plt.style.use('seaborn-v0_8')
        colorscale = plt.get_cmap('tab20')
        fig, ax = plt.subplots(4, 4, figsize=(20, 20))

        # plot ela
        for e in range(self.N):
            ax[1, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 1],
                          color='gold',
                          marker='x', markersize=10, markevery=[-1])
        ax[1, 1].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][3], self.smb[-1][3]], label='true', color=colorscale(8),
                      linewidth=3, linestyle='-.')
        ax[1, 1].plot(self.year_range_repeat[:len(self.hist_state_x)], np.array(self.hist_state_x)[:, 1], label='estimation',
                      color=colorscale(2),
                      marker='X', markersize=10, markevery=[-1], linewidth=2)
        ax[1, 1].set_title('Equilibrium line altitude [m]')
        ax[1, 1].set_ylim(2800,3000)
        ax[1, 1].set_xticks(range(2000, 2020 + 1, 5))
        ax[1, 1].legend()

        # plot gradable
        for e in range(self.N):
            ax[1, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 2],
                          color='gold',
                          marker='x', markersize=10, markevery=[-1])

        ax[1, 2].plot(self.year_range_repeat[:len(self.hist_state_x)], np.array(self.hist_state_x)[:, 2], label='estimation',
                      color=colorscale(2),
                      marker='X', markersize=10, markevery=[-1])

        ax[1, 2].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][1], self.smb[-1][1]], label='true',
                      color=colorscale(8),
                      linewidth=3, linestyle='-.')
        ax[1, 2].set_title('Ablation gradient [m/yr/m]')
        ax[1, 2].set_ylim(0.004, 0.014)
        ax[1, 2].set_xticks(range(2000, 2020 + 1, 5))
        ax[1, 2].legend()

        # plot gradacc
        for e in range(self.N):
            ax[1, 3].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 3],
                          color='gold',
                          marker='x', markersize=10, markevery=[-1])
        ax[1, 3].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][2], self.smb[-1][2]], label='true', color=colorscale(8),
                      linewidth=3,linestyle='-.')
        ax[1, 3].plot(self.year_range_repeat[:len(self.hist_state_x)], np.array(self.hist_state_x)[:, 3], label='estimation',
                      color=colorscale(2),
                      marker='X', markersize=10, markevery=[-1])
        ax[1, 3].set_title('Accumulation gradient [m/yr/m]')
        ax[1, 3].set_ylim(0, 0.01)
        ax[1, 3].set_xticks(range(2000, 2020 + 1, 5))
        ax[1, 3].legend()

        # plot volume
        for e in range(self.N):
            ax[0, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 1],
                          color=colorscale(5),
                          marker='o', markersize=10, markevery=[-1], )

        ax[0, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 1], axis=1),
                      label='estimation', color=colorscale(4), marker='o', markersize=10, markevery=[-1], linewidth=2)

        ax[0, 1].plot(self.year_range[:len(self.hist_true_y)], np.array(self.hist_true_y)[:, 1], label='true',
                      color=colorscale(0),
                      linewidth=0, marker='o', fillstyle='none', markersize=10)
        #ax[0, 1].plot(self.year_range[:len(self.hist_true_y)], np.array(self.hist_observ)[:, 1], label='observations',
        #              color=colorscale(1),
        #              linewidth=0, marker='o', fillstyle='none', markersize=10)


        ax[0, 1].set_title('Volume [$km^3$]')
        ax[0, 1].set_xticks(range(2000, 2020 + 1, 5))
        ax[0, 1].legend()

        # plot area
        for e in range(self.N):
            ax[0, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 0],
                          color=colorscale(5),
                          marker='o', markersize=10, markevery=[-1], )

        ax[0, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 0], axis=1),
                      label='estimation', color=colorscale(4), marker='o', markersize=10, markevery=[-1], linewidth=2)
        ax[0, 2].plot(self.year_range[:len(self.hist_true_y)], np.array(self.hist_true_y)[:, 0], label='true',
                      color=colorscale(0),
                      linewidth=0, marker='o', fillstyle='none', markersize=10)
        ax[0, 2].set_title('Area [$km^2$]')
        ax[0, 2].set_ylim(15.2, 16.6)
        ax[0, 2].set_xticks(range(2000, 2020 + 1, 5))
        ax[0, 2].legend()

        # plot outline
        for e in range(self.N):
            ax[0, 3].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 2],
                          color=colorscale(5),
                          marker='o', markersize=10, markevery=[-1], )
        ax[0, 3].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 2], axis=1),
                      label='estimation', color=colorscale(4), marker='o', markersize=10, markevery=[-1], linewidth=2)
        ax[0, 3].plot(self.year_range[:len(self.hist_true_y)], np.array(self.hist_true_y)[:, 2], label='true',
                      color=colorscale(0),
                      linewidth=0, marker='o', fillstyle='none', markersize=10)

        ax[0, 3].set_title('Outline length [$km$]')
        ax[0, 3].set_ylim(26, 30)
        ax[0, 3].set_xticks(range(2000, 2020+1, 5))
        ax[0, 3].legend()

        # plot thickness maps and smb
        true_usurf = self.true_glacier['usurf'][int((year - self.start_year) / self.dt)][::-1]
        true_smb = self.true_glacier['smb'][int((year - self.start_year) / self.dt)][::-1]
        true_bed = self.true_glacier['topg'][int((year - self.start_year) / self.dt)][::-1]
        true_icemask = self.true_glacier['icemask'][int((year - self.start_year) / self.dt)][::-1]

        random_id = random.sample(range(self.N), 4)
        for i, (id, glacier) in enumerate(zip(random_id, [self.hist_ensemble_x[-1][idx] for idx in random_id])):
            esti_usurf = glacier[4:].reshape(self.bedrock.shape)[::-1].astype(np.float32)

            ela, gradabl, gradacc = state_x[1:4]
            maxacc = 2.0

            smb = esti_usurf - ela

            smb *= np.where(np.less(smb, 0), gradabl, gradacc)
            smb = np.clip(smb, -100, maxacc)

            smb = np.where((smb < 0)|(true_icemask > 0.5), smb, -10)

            esti_smb = np.array(smb)


            pcm = ax[2, i].imshow(esti_usurf - true_usurf, cmap='seismic_r', vmin=-10, vmax=10)
            cbar = fig.colorbar(pcm, ax=ax[2, i ])
            # cbar.ax.invert_yaxis()
            ax[2, i].set(facecolor=colorscale(5))
            ax[2, i].set_title('Ensemble[%i]: surface elevation difference [m]' % id)
            pcm = ax[3, i ].imshow(esti_smb - true_smb, cmap='seismic_r', vmin=-10, vmax=10)
            cbar = fig.colorbar(pcm, ax=ax[3, i ])

            plt.setp(ax[2, i].spines.values(), color=colorscale(5))
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[2, i].spines[axis].set_linewidth(5)

            ax[3, i].set_title('Ensemble[%i]: smb difference [m/yr]' % id)

            plt.setp(ax[3, i].spines.values(), color='gold')
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[3, i].spines[axis].set_linewidth(5)

        def formatter(x, pos):
            del pos
            return str(x * self.res / 1000)

        #true_smb[true_thk <= 0.5] = None
        #true_thk[true_thk <= 0.5] = None


        pcm = ax[0, 0].imshow(true_usurf, cmap='Blues_r', vmin=1500, vmax=3500)
        # bed_img = ax[0, 0].imshow(self.bedrock, cmap='gray')
        #cbar = fig.colorbar(bed_img, ax=ax[0, 0], location='left')
        #cbar.ax.get_yaxis().labelpad = 10
        #cbar.ax.set_ylabel('bedrock height [m]', rotation=90)
        cbar = fig.colorbar(pcm, ax=ax[0, 0])
        #cbar.ax.invert_yaxis()

        ax[0, 0].set_title(' True surface elevation [m]')
        ax[0, 0].set(facecolor=colorscale(0))
        plt.setp(ax[0, 0].spines.values(), color=colorscale(0))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax[0, 0].spines[axis].set_linewidth(5)

        ax[0, 0].xaxis.set_major_formatter(formatter)
        ax[0, 0].yaxis.set_major_formatter(formatter)
        ax[0, 0].set_xlabel('[km]')

        pcm = ax[1, 0].imshow(true_smb, cmap='RdBu', vmin=-10, vmax=10)
        #bed_img = ax[1, 0].imshow(self.bedrock, cmap='gray')
        #cbar = fig.colorbar(bed_img, ax=ax[1, 0], location='left')
        #cbar.ax.get_yaxis().labelpad = 10
        #cbar.ax.set_ylabel('bedrock height [m]', rotation=90)
        fig.colorbar(pcm, ax=ax[1, 0])
        ax[1, 0].set_title('True surface mass balance [m/yr]')
        ax[1, 0].set(facecolor=colorscale(8))
        plt.setp(ax[1, 0].spines.values(), color=colorscale(8))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax[1, 0].spines[axis].set_linewidth(5)

        ax[1, 0].xaxis.set_major_formatter(formatter)
        ax[1, 0].yaxis.set_major_formatter(formatter)
        ax[1, 0].set_xlabel('[km]')
        # fig.delaxes(ax[1, 0])
        # fig.delaxes(ax[3, 0])

        fig.suptitle("%i Ensemble Kalman Filter, Ensemble Size: %i" % (year, self.N), fontsize=32)

        plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)
        if len(self.hist_state_x) % 2 == 1:
            plt.savefig(self.output_dir + 'report%i_update.png' % year)
        else:
            plt.savefig(self.output_dir + 'report%i_predict.png' % year)
