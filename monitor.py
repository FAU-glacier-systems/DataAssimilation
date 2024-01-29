import json
import copy
import random
import os.path
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import rasterio



class Monitor:
    def __init__(self, N, true_glacier, observation_points, dt):

        self.N = N
        self.true_glacier = true_glacier
        self.observation_points = observation_points

        self.year_range = np.array(true_glacier['time'])[::dt]
        self.dt = dt
        self.start_year = self.year_range[0]
        self.year_range_repeat = np.repeat(self.year_range, 2)[1:]

        self.res = true_glacier['x'][1] - true_glacier['x'][0]
        self.map_shape_x = true_glacier.dimensions['x'].size
        self.map_shape_y = true_glacier.dimensions['y'].size

        self.bedrock = true_glacier['topg'][0]
        #self.bedrock = self.bedrock[::-1]
        self.icemask = true_glacier['icemask'][0]
        #self.icemask = self.icemask[::-1]

        self.hist_state_x = []
        self.hist_ensemble_x = []
        self.hist_ensemble_y = []
        self.hist_true_y = []

        for year in self.year_range:
            usurf = true_glacier['usurf'][int((year - self.year_range[0]))]
            area, volume, outline_len = self.glacier_properties(usurf)
            self.hist_true_y.append([area, volume, outline_len])


        with open('ReferenceRun/params.json') as f:
            params = json.load(f)
            self.smb = params['smb_simple_array']

        self.output_dir = 'Plots/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def glacier_properties(self, usurf):
        """
        get real observation
        :returns area [km²] and volume [km³] of the ground truth in given year
                 and thickness map
        """
        thk_map = usurf - self.bedrock
        icemask = thk_map > 2
        volume = np.sum(thk_map) * self.res ** 2 / 1000 ** 3
        area = len(thk_map[icemask]) * self.res ** 2 / 1000 ** 2
        contours = measure.find_contours(icemask, 0)
        outline_len = np.sum([len(line) for line in contours]) * self.res / 1000
        if area > 40:
            print("area is to hight")
        return volume, area, outline_len

    def plot(self, year, ensemble_x, ensemble_usurfs):
        # compute mean
        dt = self.dt
        state_x = np.mean(ensemble_x, axis=0)
        # compute observations (y) form internal state (x)
        ensemble_y = [self.glacier_properties(usurf_y) for usurf_y in ensemble_usurfs]

        # append to history
        self.hist_state_x.append(copy.copy(state_x))
        self.hist_ensemble_x.append(copy.copy(ensemble_x))
        self.hist_ensemble_y.append(copy.copy(ensemble_y))

        def formatter(x, pos):
            del pos
            return str(x * self.res / 1000)

        # create canvas
        fig, ax = plt.subplots(4, 4, figsize=(20, 20))
        # define colorscale
        colorscale = plt.get_cmap('tab20')

        # get true usurf
        true_usurf = self.true_glacier['usurf'][int((year - self.start_year))]
        # get true smb
        true_smb = self.true_glacier['smb'][int((year - self.start_year))]

        # draw true surface elevation (usurf)/observation in ax[0,0]
        ax[0, 0].set_title(' True surface elevation [m]')
        usurf_im = ax[0, 0].imshow(true_usurf, cmap='Blues_r', vmin=1500, vmax=3500, origin='lower')
        usurf_ob = ax[0, 0].scatter(self.observation_points[:, 1], self.observation_points[:,0], edgecolors=colorscale(0), marker='s', c=None, facecolors='none')
        fig.colorbar(usurf_im, ax=ax[0, 0])
        plt.setp(ax[0, 0].spines.values(), color=colorscale(0))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax[0, 0].spines[axis].set_linewidth(5)
        ax[0, 0].xaxis.set_major_formatter(formatter)
        ax[0, 0].yaxis.set_major_formatter(formatter)
        ax[0, 0].set_xlabel('[km]')

        # draw true surface mass balance
        ax[1, 0].set_title('True surface mass balance [m/yr]')
        smb_im = ax[1, 0].imshow(true_smb, cmap='RdBu', vmin=-10, vmax=10, origin='lower')
        fig.colorbar(smb_im, ax=ax[1, 0])
        plt.setp(ax[1, 0].spines.values(), color=colorscale(8))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax[1, 0].spines[axis].set_linewidth(5)
        ax[1, 0].xaxis.set_major_formatter(formatter)
        ax[1, 0].yaxis.set_major_formatter(formatter)
        ax[1, 0].set_xlabel('[km]')

        # plot volume
        ax[0, 1].set_title('Volume [$km^3$]')
        for e in range(self.N):
            ax[0, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 0],
                          color=colorscale(5), marker='o', markersize=10, markevery=[-1])

        ax[0, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 0], axis=1),
                      label='estimation', color=colorscale(4), marker='o', markersize=10, markevery=[-1], linewidth=2)

        ax[0, 1].plot(self.year_range, np.array(self.hist_true_y)[:,0], label='true',
                      color=colorscale(0),  linewidth=0, marker='o', fillstyle='none', markersize=10)

        #ax[0, 1].set_ylim(1.43, 1.51)
        ax[0, 1].set_xticks(range(2000, 2020 + 1, 5))
        ax[0, 1].legend()

        # plot area
        ax[0, 2].set_title('Area [$km^2$]')
        for e in range(self.N):
            ax[0, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 1],
                          color=colorscale(5), marker='o', markersize=10, markevery=[-1], )

        ax[0, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 1], axis=1),
                      label='estimation', color=colorscale(4), marker='o', markersize=10, markevery=[-1], linewidth=2)

        ax[0, 2].plot(self.year_range, np.array(self.hist_true_y)[:, 1], label='true',
                      color=colorscale(0), linewidth=0, marker='o', fillstyle='none', markersize=10)

        #ax[0, 2].set_ylim(15.2, 16.6)
        ax[0, 2].set_xticks(range(2000, 2020 + 1, 5))
        ax[0, 2].legend()

        # plot outline
        ax[0, 3].set_title('Outline length [$km$]')
        for e in range(self.N):
            ax[0, 3].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 2],
                          color=colorscale(5), marker='o', markersize=10, markevery=[-1], )

        ax[0, 3].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 2], axis=1),
                      label='estimation', color=colorscale(4), marker='o', markersize=10, markevery=[-1], linewidth=2)

        ax[0, 3].plot(self.year_range, np.array(self.hist_true_y)[:, 2], label='true',
                      color=colorscale(0), linewidth=0, marker='o', fillstyle='none', markersize=10)

        #ax[0, 3].set_ylim(26, 30)
        ax[0, 3].set_xticks(range(2000, 2020 + 1, 5))
        ax[0, 3].legend()

        # plot ela
        ax[1, 1].set_title('Equilibrium line altitude [m]')
        for e in range(self.N):
            ax[1, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 0],
                          color='gold', marker='x', markersize=10, markevery=[-1])

        ax[1, 1].plot(self.year_range_repeat[:len(self.hist_state_x)],
                      np.array(self.hist_state_x)[:, 0], label='estimation',
                      color=colorscale(2), marker='X', markersize=10, markevery=[-1], linewidth=2)

        #ax[1, 1].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][3], self.smb[-1][3]], label='true', color=colorscale(8),
        #              linewidth=3, linestyle='-.')

        #ax[1, 1].set_ylim(2800,3100)
        ax[1, 1].set_xticks(range(2000, 2020 + 1, 5))
        ax[1, 1].legend()

        # plot gradable
        ax[1, 2].set_title('Ablation gradient [m/yr/m]')
        for e in range(self.N):
            ax[1, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 1],
                          color='gold',
                          marker='x', markersize=10, markevery=[-1])

        ax[1, 2].plot(self.year_range_repeat[:len(self.hist_state_x)], np.array(self.hist_state_x)[:, 1], label='estimation',
                      color=colorscale(2),
                      marker='X', markersize=10, markevery=[-1])

        #ax[1, 2].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][1], self.smb[-1][1]], label='true',
        #              color=colorscale(8),linewidth=3, linestyle='-.')

        #ax[1, 2].set_ylim(0.004, 0.014)
        ax[1, 2].set_xticks(range(2000, 2020 + 1, 5))
        ax[1, 2].legend()

        # plot gradacc
        ax[1, 3].set_title('Accumulation gradient [m/yr/m]')
        for e in range(self.N):
            ax[1, 3].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 2],
                          color='gold', marker='x', markersize=10, markevery=[-1])

        ax[1, 3].plot(self.year_range_repeat[:len(self.hist_state_x)], np.array(self.hist_state_x)[:, 2], label='estimation',
                      color=colorscale(2),marker='X', markersize=10, markevery=[-1])

        #ax[1, 3].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][2], self.smb[-1][2]], label='true', color=colorscale(8),
        #              linewidth=3,linestyle='-.')

        #ax[1, 3].set_ylim(0, 0.01)
        ax[1, 3].set_xticks(range(2000, 2020 + 1, 5))
        ax[1, 3].legend()

        # draw randomly selected members of the ensemble
        random_id = random.sample(range(self.N), 4)

        for i, (id, glacier) in enumerate(zip(random_id, [self.hist_ensemble_x[-1][idx] for idx in random_id])):

            # get surface elevation
            #esti_usurf = glacier[4:].reshape(self.bedrock.shape).astype(np.float32)
            esti_usurf = ensemble_usurfs[i]
            # generate SMB field
            ela, gradabl, gradacc = state_x[[0,1,2]]
            maxacc = 2.0

            smb = esti_usurf - ela
            smb *= np.where(np.less(smb, 0), gradabl, gradacc)
            smb = np.clip(smb, -100, maxacc)

            smb = np.where((smb < 0)|(self.icemask > 0.5), smb, -10)
            esti_smb = np.array(smb)

            ax[2, i].set_title('Ensemble[%i]: surface elevation difference [m]' % id)
            pcm = ax[2, i].imshow(esti_usurf - true_usurf, cmap='seismic_r'
                                  , vmin=-10, vmax=10
                                  , origin='lower')

            fig.colorbar(pcm, ax=ax[2, i ])
            plt.setp(ax[2, i].spines.values(), color=colorscale(5))
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[2, i].spines[axis].set_linewidth(5)
            ax[2, i].xaxis.set_major_formatter(formatter)
            ax[2, i].yaxis.set_major_formatter(formatter)

            ax[3, i].set_title('Ensemble[%i]: SMB [m/yr]' % id)
            pcm = ax[3, i ].imshow(esti_smb - true_smb, cmap='seismic_r', vmin=-10, vmax=10, origin='lower')
            fig.colorbar(pcm, ax=ax[3, i ])
            plt.setp(ax[3, i].spines.values(), color='gold')
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[3, i].spines[axis].set_linewidth(5)
            ax[3, i].xaxis.set_major_formatter(formatter)
            ax[3, i].yaxis.set_major_formatter(formatter)

        fig.suptitle(f"observation points: {len(self.observation_points)}, ensemble size: {self.N}, dt: {self.dt}", fontsize=32)

        plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)
        if len(self.hist_state_x) % 2 == 1:
            plt.savefig(self.output_dir + 'report%i_update.png' % year)
        else:
            plt.savefig(self.output_dir + 'report%i_predict.png' % year)
