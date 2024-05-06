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
    def __init__(self, N, true_glacier, observation_points, dt, process_noise, obs_uncertainty_field,
                 synthetic, initial_offset, initial_uncertainty, smb,
                 noise_observation, specal_noise, bias, hyperparameter, value):

        self.ensemble_size = N
        self.true_glacier = true_glacier
        self.observation_points = observation_points
        self.process_noise = process_noise
        self.synthetic = synthetic
        self.initial_offset = initial_offset
        self.initial_uncertainty = initial_uncertainty
        self.obs_uncertainty_field = obs_uncertainty_field

        self.smb = smb

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
        self.low_point = self.observation_points[0]
        self.high_point = self.observation_points[-1]
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


        self.volume_uncertainty = np.sum(self.obs_uncertainty_field[self.icemask==1])/(1000*100)


        self.output_dir = f"Results/Results_{hyperparameter}/{value}/Plots/"
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
        fig, ax = plt.subplots(3, 4, figsize=(20, 15), layout="tight")
        #fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1)

        # define colorscale
        colorscale = plt.get_cmap('tab20')



        # get true usurf
        true_vel = self.true_glacier['velsurf_mag'][int((year - self.start_year))]

        # draw true velocity
        ax_vel_true = ax[2, 3]
        ax_vel_true.set_title(f'Velocity in {int(year)}')
        vel_im = ax_vel_true.imshow(true_vel, cmap='magma', vmin=0, vmax=70, origin='lower')
        fig.colorbar(vel_im, ax=ax_vel_true, location='right')
        ax_vel_true.set_title('[$m/yr$]', loc='right', x=1.15)
        ax_vel_true.xaxis.set_major_formatter(formatter)
        ax_vel_true.yaxis.set_major_formatter(formatter)
        ax_vel_true.set_xlabel('$km$')
        ax_vel_true.set_xlim(20, 100)
        ax_vel_true.set_ylim(30, 130)
        #ax_vel_true.set_yticks([])

        # draw modeled velocity
        ax_vel_model = ax[2, 2]
        ax_vel_model.set_title(f'Mean Difference Surface Velocity')
        modeled_mean_vel = np.mean(ensemble_velo, axis=0)
        vel_im = ax_vel_model.imshow(true_vel-modeled_mean_vel, cmap='seismic_r', vmin=-30, vmax=30, origin='lower')
        fig.colorbar(vel_im, ax=ax_vel_model, location='right')
        ax_vel_model.set_title('[$m/yr$]', loc='right', x=1.15)
        ax_vel_model.xaxis.set_major_formatter(formatter)
        ax_vel_model.yaxis.set_major_formatter(formatter)
        ax_vel_model.set_xlabel('$km$')
        ax_vel_model.set_xlim(20, 100)
        ax_vel_model.set_ylim(30, 130)
        #ax_vel_model.set_yticks([])

        # plot volume
        ax[0, 0].set_title('Volume')
        ax[0, 0].plot(self.year_range, self.hist_true_y_noisy[:, 0], label="Noisy Observation",
                      color=colorscale(0), linewidth=0, marker='o', fillstyle='none', markersize=10, markeredgewidth=2,
                      zorder=5)



        for e in range(self.ensemble_size):
            ax[0, 0].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 0],
                          color=colorscale(5), marker='o', markersize=10, markevery=[-1], zorder=2)

        ax[0, 0].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 0], axis=1),
                      label='Ensemble Kalman Filter', color=colorscale(4), marker='o', markersize=10,
                      markevery=[-1], linewidth=2, zorder=4)

        if self.synthetic:
            ax[0, 0].plot(self.year_range, self.hist_true_y[:, 0], label='Hidden Truth',
                          color=colorscale(1), linewidth=3, linestyle='-.', zorder=3)
        else:
            ax[0, 0].fill_between(self.year_range,
                                  self.hist_true_y[:, 0]-self.volume_uncertainty,
                                  self.hist_true_y[:, 0]+self.volume_uncertainty,
                                  color=colorscale(1), alpha=0.2,label='Uncertainty of Observation',)

        ax[0, 0].set_xticks(range(2000, 2020 + 1, 4))
        ax[0, 0].set_title('[$km^3$]', loc='left')
        ax[0, 0].yaxis.set_label_position("right")
        ax[0, 0].set_xlabel('$year$')


        # plot lowest point
        ax[0, 1].set_title('Elevation of Lowest Point')
        ax[0, 1].plot(self.year_range, self.hist_true_y_noisy[:,1], label="Noisy Observation",
                      color=colorscale(0), linewidth=0, marker='v', fillstyle='none', markersize=10, markeredgewidth=2,
                     zorder=5)

        for e in range(self.ensemble_size):
            ax[0, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 1],
                          color=colorscale(5), marker='v', markersize=10, markevery=[-1], zorder=2)

        ax[0, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 1], axis=1),
                      label='Ensemble Kalman Filter', color=colorscale(4), marker='v', markersize=10, markevery=[-1],
                      linewidth=2, zorder=4)

        if self.synthetic:
            ax[0, 1].plot(self.year_range, self.hist_true_y[:, 1], label='Hidden Truth',
                          color=colorscale(1), linewidth=3, linestyle='-.', zorder=3)

        else:
            uncertainty_low = self.obs_uncertainty_field[self.low_point[0], self.low_point[1]]
            ax[0, 1].fill_between(self.year_range, self.hist_true_y_noisy[:, 1]-uncertainty_low,
                                  self.hist_true_y_noisy[:, 1]+uncertainty_low,
                                  color=colorscale(1), alpha=0.2,label='Uncertainty of Observation',)



        ax[0, 1].set_xticks(range(2000, 2020 + 1, 4))
        ax[0, 1].set_xlabel('$year$')
        ax[0, 1].set_title('[$m$]', loc='left')
        #ax[0, 2].yaxis.tick_right()


        # plot outline
        ax[0, 2].set_title('Elevation of Highest Point')
        ax[0, 2].plot(self.year_range, self.hist_true_y_noisy[:,2], label="Noisy Observation",
                      color=colorscale(0), linewidth=0, marker='^', fillstyle='none', markersize=10, markeredgewidth=2,
                     zorder=5)


        for e in range(self.ensemble_size):
            ax[0, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_y)], np.array(self.hist_ensemble_y)[:, e, 2],
                          color=colorscale(5), marker='^', markersize=10, markevery=[-1], zorder=2)

        ax[0, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_y)],
                      np.mean(np.array(self.hist_ensemble_y)[:, :, 2], axis=1),
                      label='Ensemble Kalman Filter', color=colorscale(4), marker='^', markersize=10, markevery=[-1],
                      linewidth=2, zorder=4)

        if self.synthetic:
            ax[0, 2].plot(self.year_range, self.hist_true_y[:, 2], label='Hidden Truth',
                          color=colorscale(1), linewidth=3, linestyle='-.', zorder=3)
        else:
            uncertainty_low = self.obs_uncertainty_field[self.high_point[0], self.high_point[1]]
            ax[0, 2].fill_between(self.year_range, self.hist_true_y_noisy[:, 2] - uncertainty_low,
                                  self.hist_true_y_noisy[:, 2] + uncertainty_low,
                                  color=colorscale(1), alpha=0.2, label='Observation Uncertainty')



        ax[0, 2].set_xticks(range(2000, 2020 + 1, 4))
        ax[0, 2].set_xlabel('$year$')
        ax[0, 2].set_title('[$m$]', loc='left')
        #ax[0, 3].yaxis.tick_right()


        # plot ela
        ax[1, 0].set_title('Equilibrium Line Altitude')


        for e in range(self.ensemble_size):
            ax[1, 0].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 0],
                          color='gold', marker='o', markersize=10, markevery=[-1],zorder=2)

        ax[1, 0].plot(self.year_range_repeat[:len(self.hist_state_x)],
                      np.array(self.hist_state_x)[:, 0], label='Ensemble Kalman Filter',
                      color=colorscale(2), marker='o', markersize=10, markevery=[-1], linewidth=2, zorder=4)

        ax[1, 0].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][3], self.smb[-1][3]], label='Hidden Parameter',
                      color=colorscale(9),
                      linewidth=3, linestyle='-.', zorder=3)




        #ax[1, 1].set_ylim(2000, 4500)
        ax[1, 0].set_xlabel('$year$')
        ax[1, 0].set_xticks(range(2000, 2020 + 1, 4))
        #ax[1, 1].yaxis.set_label_position("right")
        ax[1, 0].set_title('[$m$]', loc='left')
        #ax[1, 1].yaxis.tick_right()


        # plot gradable
        ax[1, 1].set_title('Ablation Gradient')

        for e in range(self.ensemble_size):
            ax[1, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 1],
                          color='gold',
                          marker='v', markersize=10, markevery=[-1],zorder=2)

        ax[1, 1].plot(self.year_range_repeat[:len(self.hist_state_x)], np.array(self.hist_state_x)[:, 1],
                      label='Ensemble Kalman Filter',
                      color=colorscale(2),
                      marker='v', markersize=10, markevery=[-1], zorder=4)

        ax[1, 1].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][1], self.smb[-1][1]], label='Hidden Parameter',
                          color=colorscale(9), linewidth=3, linestyle='-.', zorder=3)



        #ax[1, 2].set_ylim(0, 0.03)
        ax[1, 1].set_xticks(range(2000, 2020 + 1, 4))
        ax[1, 1].set_xlabel('$year$')
        #ax[1, 2].yaxis.set_label_position("right")
        ax[1, 1].set_title('[$m/yr/m$]', loc='left')
        #ax[1, 2].yaxis.tick_right()


        # plot gradacc
        ax[1, 2].set_title('Accumulation Gradient')


        for e in range(self.ensemble_size):
            ax[1, 2].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 2],
                          color='gold', marker='^', markersize=10, markevery=[-1], zorder=2)

        ax[1, 2].plot(self.year_range_repeat[:len(self.hist_state_x)], np.array(self.hist_state_x)[:, 2],
                      label='Ensemble Kalman Filter',
                      color=colorscale(2), marker='^', markersize=10, markevery=[-1], zorder=4)


        ax[1, 2].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][2], self.smb[-1][2]], label='Hidden Parameter',
                      color=colorscale(9),
                      linewidth=3, linestyle='-.', zorder=3)


        #ax[1, 3].set_ylim(0, 0.03)
        ax[1, 2].set_xticks(range(2000, 2020 + 1, 4))
        ax[1, 2].set_xlabel('$year$')
        #ax[1, 3].yaxis.set_label_position("right")
        ax[1, 2].set_title('[$m/yr/m$]', loc='left')
        #ax[1, 3].yaxis.tick_right()

        # get true usurf
        true_usurf = self.true_glacier['usurf'][int((year - self.start_year))]
        # get true smb


        # draw true surface elevation (usurf)/observation in ax[0,0]
        ax_obs_usurf = ax[0, 3]
        ax_obs_usurf.set_title(f'Surface Elevation in {int(year)}')

        observations = self.noise_observation[int((year - self.start_year))]


        usurf_im = ax_obs_usurf.imshow(observations, cmap='Blues_r', vmin=1450, vmax=3600, origin='lower')
        #observation_glacier = copy.copy(observations)
        #observation_glacier[self.icemask==0] = None
        #usurf_im = ax[0, 3].imshow(observation_glacier, cmap='Blues_r', vmin=2200, vmax=3600, origin='lower', zorder=2)
        #observatio_sample = np.full(observations.shape, np.nan)
        #observatio_sample[self.observation_points[:, 0], self.observation_points[:, 1]] = observations[self.observation_points[:, 0], self.observation_points[:, 1],]
        #usurf_im_samp = ax[0, 3].imshow(observatio_sample, cmap='Blues', vmin=1500, vmax=3500, origin='lower')
        ax_obs_usurf.scatter(self.high_point[1], self.high_point[0],
                        edgecolors='gray', marker='^', c=None,
                        facecolors='white', lw=2, s=120, label='Highest Point', zorder=10)

        ax_obs_usurf.scatter(self.observation_points[:, 1] - 0.5, self.observation_points[:, 0],
                                    edgecolors='gray', linewidths=0.8,
                                    marker='s', c=None, facecolors='None', s=8, label='Covered Area', zorder=5)

        blues = plt.cm.get_cmap('Blues_r')

        ax_obs_usurf.scatter(self.low_point[1], self.low_point[0],
                         edgecolors='gray', marker='v', c=None, facecolors=blues(0),
                         lw=2, s=120, label='Lowest Point', zorder=10)

        cbar = fig.colorbar(usurf_im, ax=ax_obs_usurf, location='right')
        ax_obs_usurf.set_title('[$m$]', loc='right', x=1.15)

        plt.setp(ax_obs_usurf.spines.values(), color=colorscale(0))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax_obs_usurf.spines[axis].set_linewidth(5)
        ax_obs_usurf.xaxis.set_major_formatter(formatter)
        ax_obs_usurf.yaxis.set_major_formatter(formatter)
        ax_obs_usurf.set_xlabel('$km$')
        ax_obs_usurf.set_xlim(20, 100)
        ax_obs_usurf.set_ylim(30, 130)

        legend = ax_obs_usurf.legend(loc='upper left', framealpha=.5)
        legend.legendHandles[1]._sizes = [100]
        legend.legendHandles[1]._linewidths = [2]
        legend.legendHandles[1]._facecolors = [colorscale(1)]

        # plot difference
        ax_obs_diff = ax[2,0]
        ax_obs_diff.set_title(f'Mean Difference Surface Elevation')
        surface_dif = np.mean(ensemble_usurfs, axis=0) - observations
        usurf_diff_im = ax_obs_diff.imshow(surface_dif, vmin=-50, vmax=50, cmap='seismic_r',  origin='lower')
        cbar = fig.colorbar(usurf_diff_im, ax=ax_obs_diff, location='right')
        ax_obs_diff.set_xlim(20, 100)
        ax_obs_diff.set_ylim(30, 130)
        ax_obs_diff.set_title('[$m$]', loc='right', x=1.15)

        ax_obs_diff.xaxis.set_major_formatter(formatter)
        ax_obs_diff.yaxis.set_major_formatter(formatter)
        ax_obs_diff.set_xlabel('$km$')


        ### SMB plot ##
        ela, gradabl, gradacc = state_x[[0, 1, 2]]
        maxacc = 100

        smb = true_usurf - ela
        smb *= np.where(np.less(smb, 0), gradabl, gradacc)
        smb = np.clip(smb, -100, maxacc)

        smb = np.where((smb < 0) | (self.icemask > 0.5), smb, -10)
        esti_smb = np.array(smb)

        esti_smb[self.icemask==0] = None

        if self.synthetic:
            true_smb = self.true_glacier['smb'][int((year - self.start_year))]
            true_smb[self.icemask == 0] = None
        else:

            [time,  gradabl, gradacc, ela, maxacc] = self.smb[1]

            smb = true_usurf - ela
            smb *= np.where(np.less(smb, 0), gradabl, gradacc)
            smb = np.clip(smb, -100, maxacc)

            smb = np.where((smb < 0) | (self.icemask > 0.5), smb, -10)
            smb[self.icemask==0] = np.nan
            true_smb = smb

        # draw true surface mass balance
        ax[1, 3].set_title(f'Surface Mass Balance in {int(year)}')
        background = ax[1, 3].imshow(observations, cmap='gray', vmin=1450, vmax=3600, origin='lower')

        smb_im = ax[1, 3].imshow(true_smb, cmap='RdBu', vmin=-8, vmax=8, origin='lower', zorder =5)
        fig.colorbar(smb_im, ax=ax[1, 3], location='right', ticks=range(-10, 11, 5))
        ax[1, 3].set_title('[$m/yr$]', loc='right', x=1.15)

        plt.setp(ax[1, 3].spines.values(), color=colorscale(8))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax[1, 3].spines[axis].set_linewidth(5)
        ax[1, 3].xaxis.set_major_formatter(formatter)
        ax[1, 3].yaxis.set_major_formatter(formatter)
        ax[1, 3].set_xlabel('$km$')
        #ax[1, 3].set_yticks([])

        ax[1, 3].set_xlim(20, 100)
        ax[1, 3].set_ylim(30, 130)

        # true_smb
        ax_true_smb = ax[2, 1]
        ax_true_smb.set_title(f'Mean Difference Surface Mass Balance')

        smb_im = ax_true_smb.imshow(esti_smb - true_smb, cmap='seismic_r',vmin=-3, vmax=3, origin='lower', zorder=5)
        fig.colorbar(smb_im, ax=ax_true_smb, location='right')
        ax_true_smb.set_title('[$m/yr$]', loc='right', x=1.15)

        ax_true_smb.xaxis.set_major_formatter(formatter)
        ax_true_smb.yaxis.set_major_formatter(formatter)
        ax_true_smb.set_xlabel('$km$')
        ax_true_smb.set_xlim(20, 100)
        ax_true_smb.set_ylim(30, 130)


        for axi in [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]:
            axi.spines['top'].set_visible(False)
            axi.spines['right'].set_visible(False)
            axi.spines['bottom'].set_visible(False)
            axi.spines['left'].set_visible(False)
            axi.grid(axis="y", color="lightgray", linestyle="-", zorder=0)
            axi.grid(axis="x", color="lightgray", linestyle="-", zorder=0)
            axi.xaxis.set_tick_params(bottom=False)
            axi.yaxis.set_tick_params(left=False)
            axi.legend(framealpha=1)
        for axi in [ax[0,3], ax[1,3]]:
            axi.grid(axis="y", color="black", linestyle="--", zorder=0, alpha=.2)
            axi.grid(axis="x", color="black", linestyle="--", zorder=0, alpha=.2)


        # draw randomly selected members of the ensemble
        #plt.rcParams["font.family"] = "Open Sans"

        #fig.suptitle(
        #    f"observation points: {len(self.observation_points)}, ensemble size: {self.ensemble_size}, dt: {self.dt}, process noise: {self.process_noise},\n "
        #    f"initial_offset: {self.initial_offset}, initial_uncertainty: {self.initial_uncertainty}, bias: {self.bias}, specal noise: {self.specal_noise}",
        #    fontsize=16)

        plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)
        if len(self.hist_state_x) % 2 == 1:
            plt.savefig(self.output_dir + 'report%i_update.png' % year, format='png', dpi=300)
            pass
        else:
            plt.savefig(self.output_dir + 'report%i_predict.png' % year, format='png', dpi=300)
            pass
            if year == self.year_range[-1]:
                plt.savefig(self.output_dir+f"result_o_{self.initial_offset}_u_{self.initial_uncertainty}_b_{self.bias}_s_{self.specal_noise}.pdf", format='pdf')
                plt.savefig(self.output_dir +
                    f"result_o_{self.initial_offset}_u_{self.initial_uncertainty}_b_{self.bias}_s_{self.specal_noise}.png",
                    format='png')

    plt.clf()
    plt.close()