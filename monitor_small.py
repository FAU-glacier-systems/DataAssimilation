import json
import copy
import random
import os.path
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import rasterio
import matplotlib.font_manager as fm




class Monitor:
    def __init__(self, N, true_glacier, num_sample_points, dt, synthetic, initial_offset, initial_uncertainty,
                 noise_observation, specal_noise, bias, hyperparameter):

        self.ensemble_size = N
        self.true_glacier = true_glacier
        self.num_sample_points = num_sample_points
        self.synthetic = synthetic
        self.initial_offset = initial_offset
        self.initial_uncertainty = initial_uncertainty

        self.year_range = np.array(true_glacier['time'])[::dt]
        self.dt = dt
        self.start_year = self.year_range[0]
        self.year_range_repeat = np.repeat(self.year_range, 2)[1:]

        self.res = true_glacier['x'][1] - true_glacier['x'][0]
        self.map_shape_x = true_glacier.dimensions['x'].size
        self.map_shape_y = true_glacier.dimensions['y'].size

        self.bedrock = true_glacier['topg'][0]
        # self.bedrock = self.bedrock[::-1]
        self.icemask = np.array(true_glacier['icemask'][0])
        # self.icemask = self.icemask[::-1]
        self.random_id = random.sample(range(self.ensemble_size), 4)

        self.hist_state_x = []
        self.hist_ensemble_x = []
        self.hist_ensemble_y = []
        self.hist_true_y = []
        self.low_point = self.num_sample_points[5]
        self.high_point = self.num_sample_points[-5]
        self.noise_observation = noise_observation
        self.specal_noise = specal_noise
        self.bias = bias
        self.hyperparameter = hyperparameter


        self.hist_true_y_noisy = []

        for year in self.year_range:
            usurf = true_glacier['usurf'][int((year - self.year_range[0]))]
            area, low_point, high_point = self.glacier_properties(usurf)
            area_n, low_point_n, high_point_n = self.glacier_properties(noise_observation[int((year - self.year_range[0]))])
            self.hist_true_y.append([area, low_point, high_point])
            self.hist_true_y_noisy.append([area_n, low_point_n, high_point_n])
        self.hist_true_y = np.array(self.hist_true_y)
        self.hist_true_y_noisy = np.array(self.hist_true_y_noisy)


        if synthetic:
            with open('ReferenceSimulation/params.json') as f:
                params = json.load(f)
                self.smb = params['smb_simple_array']
        else:
            self.smb = None

        self.output_dir = f"Results_{hyperparameter}/Plots/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def glacier_properties(self, usurf):
        """
        get real observation
        :returns area [km²] and volume [km³] of the ground truth in given year
                 and thickness map
        """
        thk_map = usurf - self.bedrock
        #icemask = np.logical_and(self.icemask,)
        volume = np.sum(thk_map[self.icemask==1]) * (self.res ** 2) / (1000 ** 3)
        low_sample = usurf[self.low_point[0], self.low_point[1]]
        high_sample = usurf[self.high_point[0], self.high_point[1]]

        return volume, low_sample, high_sample

    def plot(self, year, ensemble_x, ensemble_usurfs, ensemble_velo):
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
            return str(int(x * self.res / 1000))

        # create canvas
        fig, ax = plt.subplots(2, 4, figsize=(17, 8.5), layout="tight")
        #fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1)

        # define colorscale
        colorscale = plt.get_cmap('tab20')

        # get true usurf
        true_usurf = self.true_glacier['usurf'][int((year - self.start_year))]
        # get true smb
        true_smb = self.true_glacier['smb'][int((year - self.start_year))]
        # get true usurf
        true_vel = self.true_glacier['velsurf_mag'][int((year - self.start_year))]

        # draw true surface elevation (usurf)/observation in ax[0,0]
        ax[0, 3].set_title(f'Surface Elevation in {int(year)}')

        observations = self.noise_observation[int((year - self.start_year))]

        usurf_im = ax[0, 3].imshow(observations, cmap='Blues_r', vmin=1500, vmax=3500, origin='lower')
        usurf_ob = ax[0, 3].scatter(self.num_sample_points[:, 1], self.num_sample_points[:, 0],
                                    edgecolors='gray', marker='s', c=None, facecolors='none', s=10, label='Covered Area')

        ax[0, 3].scatter(self.high_point[1], self.high_point[0],
                         edgecolors='black', marker='^', c=None, facecolors='white', lw=2,  s=120, label='High Point')
        ax[0, 3].scatter(self.low_point[1], self.low_point[0],
                         edgecolors='black', marker='v', c=None, facecolors=colorscale(0),lw=2, s=120,label='Low Point')

        ax[0, 3].legend()
        cbar = fig.colorbar(usurf_im, ax=ax[0, 3], location='right')
        ax[0, 3].set_title('[$m$]', loc='right',x=1.15)

        plt.setp(ax[0, 3].spines.values(), color=colorscale(0))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax[0, 3].spines[axis].set_linewidth(5)
        ax[0, 3].xaxis.set_major_formatter(formatter)
        ax[0, 3].yaxis.set_major_formatter(formatter)
        ax[0, 3].set_xlabel('$km$')
        ax[0, 3].set_yticks([])

        # draw true surface mass balance
        ax[1, 3].set_title(f'Surface Mass Balance in {int(year)}')
        #true_smb[self.icemask==0] = None
        smb_im = ax[1, 3].imshow(true_smb, cmap='RdBu', vmin=-10, vmax=10, origin='lower')
        fig.colorbar(smb_im, ax=ax[1, 3], location='right', ticks=range(-10, 11, 5) )
        ax[1, 3].set_title('[$m/yr$]', loc='right', x=1.25)
        plt.setp(ax[1, 3].spines.values(), color=colorscale(8))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax[1, 3].spines[axis].set_linewidth(5)
        ax[1, 3].xaxis.set_major_formatter(formatter)
        ax[1, 3].yaxis.set_major_formatter(formatter)
        ax[1, 3].set_xlabel('$km$')
        ax[1, 3].set_yticks([])

        # draw true velocity
        """
        ax[0, 4].set_title(f'Velocity in {int(year)}')
        smb_im = ax[0, 4].imshow(true_vel, cmap='magma', vmin=0, vmax=70, origin='lower')
        fig.colorbar(smb_im, ax=ax[0, 4], location='right')
        ax[0, 4].set_title('[$m/yr$]', loc='right', x=1.15)
        ax[0, 4].xaxis.set_major_formatter(formatter)
        ax[0, 4].yaxis.set_major_formatter(formatter)
        ax[0, 4].set_xlabel('$km$')
        ax[0, 4].set_yticks([])

        # draw true velocity
        ax[1, 4].set_title(f'Mean Ensemble Velocity in {int(year)}')
        smb_im = ax[1, 4].imshow(np.mean(ensemble_velo, axis=0), cmap='magma', vmin=0, vmax=70, origin='lower')
        fig.colorbar(smb_im, ax=ax[1, 4], location='right')
        ax[1, 4].set_title('[$m/yr$]', loc='right', x=1.15)
        ax[1, 4].xaxis.set_major_formatter(formatter)
        ax[1, 4].yaxis.set_major_formatter(formatter)
        ax[1, 4].set_xlabel('$km$')
        ax[1, 4].set_yticks([])
        """
        # plot volume
        ax[0, 0].set_title('Volume')
        for e in range(self.ensemble_size):
            ax[0, 0].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 0],
                          color=colorscale(5), marker='o', markersize=10, markevery=[-1], zorder=1)

        ax[0, 0].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 0], axis=1),
                      label='Ensemble Kalman Filter', color=colorscale(4), marker='o', markersize=10,
                      markevery=[-1], linewidth=2,  zorder=2)

        #ax[0, 0].plot(self.year_range,self.hist_true_y[:, 0], label='True Variable',
        #              color=colorscale(1), linewidth=3, linestyle='-.', zorder=0)

        ax[0, 0].plot(self.year_range, self.hist_true_y_noisy[:,0], label="Observation",
                      color=colorscale(0), linewidth=0, marker='o', fillstyle='none', markersize=10, markeredgewidth=2,
                      zorder=3)

        # ax[0, 1].set_ylim(1.43, 1.51)
        ax[0, 0].set_xticks(range(2000, 2020 + 1, 4))
        ax[0, 0].set_title('[$km^3$]', loc='left')
        ax[0, 0].yaxis.set_label_position("right")
        #ax[0, 1].yaxis.tick_right()
        ax[0, 0].set_xlabel('$year$')
        ax[0, 0].legend()

        # plot area
        ax[0, 1].set_title('Elevation of Low Point')
        for e in range(self.ensemble_size):
            ax[0, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 1],
                          color=colorscale(5), marker='v', markersize=10, markevery=[-1], zorder=1)

        ax[0, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 1], axis=1),
                      label='Ensemble Kalman Filter', color=colorscale(4), marker='v', markersize=10, markevery=[-1],
                      linewidth=2, zorder=2)

        #ax[0, 1].plot(self.year_range, self.hist_true_y[:, 1], label='True Variable',
        #              color=colorscale(1), linewidth=3, linestyle='-.', zorder=0)

        ax[0, 1].plot(self.year_range, self.hist_true_y_noisy[:,1], label="Observation",
                      color=colorscale(0), linewidth=0, marker='v', fillstyle='none', markersize=10, markeredgewidth=2,
                      zorder=3)

        ax[0, 1].set_xticks(range(2000, 2020 + 1, 4))
        ax[0, 1].set_xlabel('$year$')
        ax[0, 1].set_title('[$m$]', loc='left')
        #ax[0, 2].yaxis.tick_right()
        ax[0, 1].legend()

        # plot outline
        ax[0, 2].set_title('Elevation of High Point')
        for e in range(self.ensemble_size):
            ax[0, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 2],
                          color=colorscale(5), marker='^', markersize=10, markevery=[-1], zorder=1)

        ax[0, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 2], axis=1),
                      label='Ensemble Kalman Filter', color=colorscale(4), marker='^', markersize=10, markevery=[-1],
                      linewidth=2, zorder=2)

        #ax[0, 2].plot(self.year_range, self.hist_true_y[:, 2], label='Observable Parameter',
        #              color=colorscale(1), linewidth=3, linestyle='-.', zorder=0)

        ax[0, 2].plot(self.year_range, self.hist_true_y_noisy[:,2], label="Observation",
                      color=colorscale(0), linewidth=0, marker='^', fillstyle='none', markersize=10, markeredgewidth=2,
                      zorder=3)

        ax[0, 2].set_xticks(range(2000, 2020 + 1, 4))
        ax[0, 2].set_xlabel('$year$')
        ax[0, 2].set_title('[$m$]', loc='left')
        #ax[0, 3].yaxis.tick_right()
        ax[0, 2].legend()

        # plot ela
        ax[1, 0].set_title('Equilibrium Line Altitude')
        for e in range(self.ensemble_size):
            ax[1, 0].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 0],
                          color='gold', marker='x', markersize=10, markevery=[-1])

        ax[1, 0].plot(self.year_range_repeat[:len(self.hist_state_x)],
                      np.array(self.hist_state_x)[:, 0], label='Ensemble Kalman Filter',
                      color=colorscale(2), marker='X', markersize=10, markevery=[-1], linewidth=2)

        if not self.smb == None:
            ax[1, 0].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][3], self.smb[-1][3]], label='Hidden Parameter',
                          color=colorscale(8),
                          linewidth=3, linestyle='-.')

        #ax[1, 1].set_ylim(2000, 4500)
        ax[1, 0].set_xlabel('$year$')
        ax[1, 0].set_xticks(range(2000, 2020 + 1, 4))
        #ax[1, 1].yaxis.set_label_position("right")
        ax[1, 0].set_title('[$m$]', loc='left')
        #ax[1, 1].yaxis.tick_right()
        ax[1, 0].legend()

        # plot gradable
        ax[1, 1].set_title('Ablation Gradient')
        for e in range(self.ensemble_size):
            ax[1, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 1],
                          color='gold',
                          marker='x', markersize=10, markevery=[-1])

        ax[1, 1].plot(self.year_range_repeat[:len(self.hist_state_x)], np.array(self.hist_state_x)[:, 1],
                      label='Ensemble Kalman Filter',
                      color=colorscale(2),
                      marker='X', markersize=10, markevery=[-1])

        if not self.smb == None:
            ax[1, 1].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][1], self.smb[-1][1]], label='Hidden Parameter',
                          color=colorscale(8), linewidth=3, linestyle='-.')

        #ax[1, 2].set_ylim(0, 0.03)
        ax[1, 1].set_xticks(range(2000, 2020 + 1, 4))
        ax[1, 1].set_xlabel('$year$')
        #ax[1, 2].yaxis.set_label_position("right")
        ax[1, 1].set_title('[$m/yr/m$]', loc='left')
        #ax[1, 2].yaxis.tick_right()
        ax[1, 1].legend()

        # plot gradacc
        ax[1, 2].set_title('Accumulation Gradient')
        for e in range(self.ensemble_size):
            ax[1, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 2],
                          color='gold', marker='x', markersize=10, markevery=[-1])

        ax[1, 2].plot(self.year_range_repeat[:len(self.hist_state_x)], np.array(self.hist_state_x)[:, 2],
                      label='Ensemble Kalman Filter',
                      color=colorscale(2), marker='X', markersize=10, markevery=[-1])

        if not self.smb == None:
            ax[1, 2].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][2], self.smb[-1][2]], label='Hidden Parameter',
                          color=colorscale(8),
                          linewidth=3, linestyle='-.')

        #ax[1, 3].set_ylim(0, 0.03)
        ax[1, 2].set_xticks(range(2000, 2020 + 1, 4))
        ax[1, 2].set_xlabel('$year$')
        #ax[1, 3].yaxis.set_label_position("right")
        ax[1, 2].set_title('[$m/yr/m$]', loc='left')
        #ax[1, 3].yaxis.tick_right()

        ax[1, 2].legend()

        # draw randomly selected members of the ensemble
        #plt.rcParams["font.family"] = "Open Sans"
        """
        fig.suptitle(
            f"observation points: {len(self.num_sample_points)}, ensemble size: {self.ensemble_size}, dt: {self.dt},\n initial_offset: {self.initial_offset}, initial_uncertainty: {self.initial_uncertainty}",
            fontsize=32)
        """
        plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)
        if len(self.hist_state_x) % 2 == 1:
            pass#plt.savefig(self.output_dir + 'report%i_update.png' % year, format='png')

        else:
            plt.savefig(self.output_dir + 'report%i_predict.png' % year, format='png')
            if year == self.year_range[-1]:
                plt.savefig(self.output_dir+f"result_{self.initial_offset}_{self.initial_uncertainty}_{self.bias}_{self.specal_noise}.pdf", format='pdf')
                plt.savefig(self.output_dir +
                    f"result_{self.initial_offset}_{self.initial_uncertainty}_{self.bias}_{self.specal_noise}.png",
                    format='png')

    plt.clf()
    plt.close()