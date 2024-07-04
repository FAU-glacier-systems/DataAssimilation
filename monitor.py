import copy
import os.path
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr



class Monitor:
    def __init__(self, params, observations, observation_uncertainty_field, observation_points):

        self.params = params
        self.synthetic = params['synthetic']
        self.output_dir = params['output_dir']
        self.seed = params['seed']
        self.monitor_dir = self.output_dir + 'Plot/'
        if not os.path.exists(self.monitor_dir):
            os.makedirs(self.monitor_dir)

        self.covered_area = params['covered_area']
        self.ensemble_size = params['ensemble_size']
        self.process_noise = params['process_noise']
        self.time_interval = params['time_interval']

        self.initial_estimate = params['initial_estimate']
        self.initial_spread = params['initial_spread']
        self.initial_offset = params['initial_offset']

        if self.synthetic:
            self.observation_uncertainty = params['observation_uncertainty']

        with xr.open_dataset(params['observations_file']) as ds:
            self.observed_glacier = ds

        self.observations = observations
        self.observation_uncertainty_field = observation_uncertainty_field
        self.observation_points = observation_points

        self.smb = params['smb_simple_array']

        self.year_range = np.array(self.observed_glacier['time'])[::self.time_interval]
        self.start_year = self.year_range[0]
        self.year_range_repeat = np.repeat(self.year_range, 2)[1:]

        self.res = self.observed_glacier['x'][1] - self.observed_glacier['x'][0]
        self.map_shape_x = self.observed_glacier.dims['x']
        self.map_shape_y = self.observed_glacier.dims['y']

        self.bedrock = self.observed_glacier['topg'][0]
        # self.bedrock = self.bedrock[::-1]
        self.icemask = np.array(self.observed_glacier['icemask'][0])
        # self.icemask = self.icemask[::-1]
        #self.random_id = random.sample(range(self.ensemble_size), 4)

        self.hist_state_x = []
        self.hist_ensemble_x = []
        self.hist_ensemble_y = []
        self.hist_true_y = []
        self.low_point = self.observation_points[0]
        self.high_point = self.observation_points[-1]
        self.hist_true_y_noisy = []
        self.hist_true_y_uncertainty = []

        for year in self.year_range:
            year_index = int(year - self.start_year)
            usurf = self.observed_glacier['usurf'][year_index]
            volume, low_point, high_point = self.glacier_properties(usurf)
            volume_n, low_point_n, high_point_n = self.glacier_properties(observations[year_index])
            self.hist_true_y.append([volume, low_point, high_point])
            self.hist_true_y_noisy.append([volume_n, low_point_n, high_point_n])

            volume_uncertainty = np.sum(self.observation_uncertainty_field[year_index, self.icemask==1])/(1000*100)
            low_sample = self.observation_uncertainty_field[year_index,self.low_point[0], self.low_point[1]]
            high_sample = self.observation_uncertainty_field[year_index,self.high_point[0], self.high_point[1]]

            self.hist_true_y_uncertainty.append([volume_uncertainty, low_sample, high_sample])

        self.hist_true_y = np.array(self.hist_true_y)
        self.hist_true_y_noisy = np.array(self.hist_true_y_noisy)
        self.hist_true_y_uncertainty = np.array(self.hist_true_y_uncertainty)
        print()


    def glacier_properties(self, usurf):
        """
        get real observation
        :returns area [km²] and volume [km³] of the ground truth in given year
                 and thickness map
        """
        thk_map = np.array(usurf - self.bedrock)
        # icemask = np.logical_and(self.icemask,)

        volume = np.sum(thk_map[self.icemask == 1]) * (self.res ** 2) / (1000 ** 3)
        low_sample = usurf[self.low_point[0], self.low_point[1]]
        high_sample = usurf[self.high_point[0], self.high_point[1]]

        return volume, low_sample, high_sample

    def plot(self, year, ensemble_x, ensemble_usurfs, ensemble_velo):
        # compute mean
        year_index = int(year - self.start_year)
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
        # fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1)

        # define colorscale
        colorscale = plt.get_cmap('tab20')

        # get true usurf
        true_vel = self.observed_glacier['velsurf_mag'][int((year - self.start_year))]

        # draw true velocity
        ax_vel_true = ax[2, 3]
        ax_vel_true.set_title(f'Velocity in 2000')
        vel_im = ax_vel_true.imshow(true_vel, cmap='magma', vmin=0, vmax=300, origin='lower')
        fig.colorbar(vel_im, ax=ax_vel_true, location='right')
        ax_vel_true.set_title('[$m/yr$]', loc='right', x=1.15)
        ax_vel_true.xaxis.set_major_formatter(formatter)
        ax_vel_true.yaxis.set_major_formatter(formatter)
        ax_vel_true.set_xlabel('$km$')
        #ax_vel_true.set_xlim(20, 100)
        #ax_vel_true.set_ylim(30, 130)
        # ax_vel_true.set_yticks([])

        # draw modeled velocity
        ax_vel_model = ax[2, 2]
        ax_vel_model.set_title(f'Mean Difference Surface Velocity')
        modeled_mean_vel = np.mean(ensemble_velo, axis=0)
        #vel_im = ax_vel_model.imshow(true_vel - modeled_mean_vel, cmap='bwr_r', vmin=-30, vmax=30, origin='lower')
        vel_im = ax_vel_model.imshow(modeled_mean_vel, cmap='magma', vmin=0, vmax=300, origin='lower')

        fig.colorbar(vel_im, ax=ax_vel_model, location='right')
        ax_vel_model.set_title('[$m/yr$]', loc='right', x=1.15)
        ax_vel_model.xaxis.set_major_formatter(formatter)
        ax_vel_model.yaxis.set_major_formatter(formatter)
        ax_vel_model.set_xlabel('$km$')
        #ax_vel_model.set_xlim(20, 100)
        #ax_vel_model.set_ylim(30, 130)
        # ax_vel_model.set_yticks([])

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
                                  self.hist_true_y[:, 0] - self.hist_true_y_uncertainty[:, 0],
                                  self.hist_true_y[:, 0] + self.hist_true_y_uncertainty[:, 0],
                                  color=colorscale(1), alpha=0.2, label='Uncertainty of Observation', )

        ax[0, 0].set_xticks(range(2000, 2020 + 1, 4))
        ax[0, 0].set_title('[$km^3$]', loc='left')
        ax[0, 0].yaxis.set_label_position("right")
        ax[0, 0].set_xlabel('$year$')

        # plot lowest point
        ax[0, 1].set_title('Elevation of Lowest Point')
        ax[0, 1].plot(self.year_range, self.hist_true_y_noisy[:, 1], label="Noisy Observation",
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
            ax[0, 1].fill_between(self.year_range, self.hist_true_y_noisy[:, 1] - self.hist_true_y_uncertainty[:, 1],
                                  self.hist_true_y_noisy[:, 1] + self.hist_true_y_uncertainty[:, 1],
                                  color=colorscale(1), alpha=0.2, label='Uncertainty of Observation', )

        ax[0, 1].set_xticks(range(2000, 2020 + 1, 4))
        ax[0, 1].set_xlabel('$year$')
        ax[0, 1].set_title('[$m$]', loc='left')
        # ax[0, 2].yaxis.tick_right()

        # plot outline
        ax[0, 2].set_title('Elevation of Highest Point')
        ax[0, 2].plot(self.year_range, self.hist_true_y_noisy[:, 2], label="Noisy Observation",
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
            ax[0, 2].fill_between(self.year_range, self.hist_true_y_noisy[:, 2] - self.hist_true_y_uncertainty[:, 2],
                                  self.hist_true_y_noisy[:, 2] + self.hist_true_y_uncertainty[:, 2],
                                  color=colorscale(1), alpha=0.2, label='Observation Uncertainty')

        ax[0, 2].set_xticks(range(2000, 2020 + 1, 4))
        ax[0, 2].set_xlabel('$year$')
        ax[0, 2].set_title('[$m$]', loc='left')
        # ax[0, 3].yaxis.tick_right()

        # plot ela
        ax[1, 0].set_title('Equilibrium Line Altitude')

        for e in range(self.ensemble_size):
            ax[1, 0].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 0],
                          color='gold', marker='o', markersize=10, markevery=[-1], zorder=2)

        ax[1, 0].plot(self.year_range_repeat[:len(self.hist_state_x)],
                      np.array(self.hist_state_x)[:, 0], label='Ensemble Kalman Filter',
                      color=colorscale(2), marker='o', markersize=10, markevery=[-1], linewidth=2, zorder=4)

        ax[1, 0].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][3], self.smb[-1][3]], label='Hidden Parameter',
                      color=colorscale(9),
                      linewidth=3, linestyle='-.', zorder=3)

        # ax[1, 1].set_ylim(2000, 4500)
        ax[1, 0].set_xlabel('$year$')
        ax[1, 0].set_xticks(range(2000, 2020 + 1, 4))
        # ax[1, 1].yaxis.set_label_position("right")
        ax[1, 0].set_title('[$m$]', loc='left')
        # ax[1, 1].yaxis.tick_right()

        # plot gradable
        ax[1, 1].set_title('Ablation Gradient')

        for e in range(self.ensemble_size):
            ax[1, 1].plot(self.year_range_repeat[:len(self.hist_ensemble_x)], np.array(self.hist_ensemble_x)[:, e, 1],
                          color='gold',
                          marker='v', markersize=10, markevery=[-1], zorder=2)

        ax[1, 1].plot(self.year_range_repeat[:len(self.hist_state_x)], np.array(self.hist_state_x)[:, 1],
                      label='Ensemble Kalman Filter',
                      color=colorscale(2),
                      marker='v', markersize=10, markevery=[-1], zorder=4)

        ax[1, 1].plot([self.smb[1][0], self.smb[-1][0]], [self.smb[1][1], self.smb[-1][1]], label='Hidden Parameter',
                      color=colorscale(9), linewidth=3, linestyle='-.', zorder=3)

        # ax[1, 2].set_ylim(0, 0.03)
        ax[1, 1].set_xticks(range(2000, 2020 + 1, 4))
        ax[1, 1].set_xlabel('$year$')
        # ax[1, 2].yaxis.set_label_position("right")
        ax[1, 1].set_title('[$m/yr/m$]', loc='left')
        # ax[1, 2].yaxis.tick_right()

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

        # ax[1, 3].set_ylim(0, 0.03)
        ax[1, 2].set_xticks(range(2000, 2020 + 1, 4))
        ax[1, 2].set_xlabel('$year$')
        # ax[1, 3].yaxis.set_label_position("right")
        ax[1, 2].set_title('[$m/yr/m$]', loc='left')
        # ax[1, 3].yaxis.tick_right()

        # get true usurf
        true_usurf = self.observed_glacier['usurf'][int((year - self.start_year))]
        # get true hidden_smb

        # draw true surface elevation (usurf)/observation in ax[0,0]
        ax_obs_usurf = ax[0, 3]
        ax_obs_usurf.set_title(f'Surface Elevation in {int(year)}')

        observations = self.observations[year_index]

        usurf_im = ax_obs_usurf.imshow(observations, cmap='Blues_r',  origin='lower')

        # observation_glacier = copy.copy(observations)
        # observation_glacier[self.icemask==0] = None
        # usurf_im = ax[0, 3].imshow(observation_glacier, cmap='Blues_r', vmin=2200, vmax=3600, origin='lower', zorder=2)
        # observatio_sample = np.full(observations.shape, np.nan)
        # observatio_sample[self.observation_points[:, 0], self.observation_points[:, 1]] = observations[self.observation_points[:, 0], self.observation_points[:, 1],]
        # usurf_im_samp = ax[0, 3].imshow(observatio_sample, cmap='Blues', vmin=1500, vmax=3500, origin='lower')
        ax_obs_usurf.scatter(self.high_point[1], self.high_point[0],
                             edgecolors='gray', marker='^', c=None,
                             facecolors='white', lw=2, s=120, label='Highest Point', zorder=10)

        ax_obs_usurf.scatter(self.observation_points[:, 1] - 0.5, self.observation_points[:, 0],
                             edgecolors='gray', linewidths=0.8,
                             marker='s', c=None, facecolors='None', s=8, label='Covered Area', zorder=5)

        import matplotlib as mpl
        blues = mpl.colormaps['Blues_r']

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
        #ax_obs_usurf.set_xlim(20, 100)
        #ax_obs_usurf.set_ylim(30, 130)

        legend = ax_obs_usurf.legend(loc='upper left', framealpha=.5)
        legend.legend_handles[1]._sizes = [100]
        legend.legend_handles[1]._linewidths = [2]
        legend.legend_handles[1]._facecolors = [colorscale(1)]

        # plot difference
        ax_obs_diff = ax[2, 0]
        ax_obs_diff.set_title(f'Mean Difference Surface Elevation')
        surface_dif = np.mean(ensemble_usurfs, axis=0) - observations
        usurf_diff_im = ax_obs_diff.imshow(surface_dif, vmin=-50, vmax=50, cmap='bwr_r', origin='lower')
        cbar = fig.colorbar(usurf_diff_im, ax=ax_obs_diff, location='right')
        #ax_obs_diff.set_xlim(20, 100)
        #ax_obs_diff.set_ylim(30, 130)
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

        esti_smb[self.icemask == 0] = None

        if self.synthetic:
            hidden_smb = self.observed_glacier['smb'][int((year - self.start_year))]
            hidden_smb[self.icemask == 0] = None
        else:
            [time, gradabl, gradacc, ela, maxacc] = self.smb[1]
            print("############## SMB ############")
            print(gradabl, gradacc, ela, maxacc)
            smb = true_usurf - ela
            smb *= np.where(np.less(smb, 0), gradabl, gradacc)
            smb = np.clip(smb, -100, maxacc)

            smb = np.where((smb < 0) | (self.icemask > 0.5), smb, -10)
            smb[self.icemask == 0] = np.nan
            hidden_smb = smb

        # draw true surface mass balance
        ax_smb = ax[1, 3]
        ax_smb.set_title(f'Surface Mass Balance in {int(year)}')
        background = ax_smb.imshow(observations, cmap='gray',  origin='lower')

        smb_im = ax_smb.imshow(esti_smb, cmap='RdBu', vmin=-10, vmax=10, origin='lower', zorder=5)
        fig.colorbar(smb_im, ax=ax_smb, location='right', ticks=range(-10, 11, 5))
        text_y, text_x = esti_smb.shape
        mb = np.sum(esti_smb[self.icemask == 1]) / np.sum(self.icemask)
        ax_smb.text(text_x / 2 - 7, text_y / 2, f'{mb:.4f} \n m/yr', zorder=10, size=20)
        ax_smb.set_title('[$m/yr$]', loc='right', x=1.15)

        plt.setp(ax_smb.spines.values(), color=colorscale(2))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax_smb.spines[axis].set_linewidth(5)
        ax_smb.xaxis.set_major_formatter(formatter)
        ax_smb.yaxis.set_major_formatter(formatter)
        ax_smb.set_xlabel('$km$')
        # ax[1, 3].set_yticks([])

        #ax_smb.set_xlim(20, 100)
        #ax_smb.set_ylim(30, 130)

        # true_smb
        ax_hidden_smb = ax[2, 1]
        ax_hidden_smb.set_title(f'Mean Difference Surface Mass Balance')

        smb_im = ax_hidden_smb.imshow(esti_smb - hidden_smb, cmap='bwr_r', vmin=-3, vmax=3, origin='lower', zorder=5)
        fig.colorbar(smb_im, ax=ax_hidden_smb, location='right')
        mb_glamos = np.sum(hidden_smb[self.icemask == 1]) / np.sum(self.icemask)
        ax_hidden_smb.text(20, 30, f'{mb_glamos:.4f} \n m/yr', zorder=10, size=20)
        ax_hidden_smb.set_title('[$m/yr$]', loc='right', x=1.15)

        ax_hidden_smb.xaxis.set_major_formatter(formatter)
        ax_hidden_smb.yaxis.set_major_formatter(formatter)
        ax_hidden_smb.set_xlabel('$km$')
        #ax_hidden_smb.set_xlim(20, 100)
        #ax_hidden_smb.set_ylim(30, 130)

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
        for axi in [ax[0, 3], ax[1, 3]]:
            axi.grid(axis="y", color="black", linestyle="--", zorder=0, alpha=.2)
            axi.grid(axis="x", color="black", linestyle="--", zorder=0, alpha=.2)

        # draw randomly selected members of the ensemble
        # plt.rcParams["font.family"] = "Open Sans"

        # fig.suptitle(
        #    f"observation points: {len(self.observation_points)}, ensemble size: {self.ensemble_size}, dt: {self.dt}, process noise: {self.process_noise},\n "
        #    f"initial_offset: {self.initial_offset}, initial_uncertainty: {self.initial_uncertainty}, bias: {self.bias}, specal noise: {self.specal_noise}",
        #    fontsize=16)

        plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)
        if len(self.hist_state_x) % 2 == 1:
            plt.savefig(self.monitor_dir + 'report%i_update.png' % year, format='png', dpi=300)
            pass
        else:
            plt.savefig(self.monitor_dir + 'report%i_predict.png' % year, format='png', dpi=300)

            if year == self.year_range[-1]:

                if self.synthetic:
                    file_name = f"result_o_{self.initial_offset}_u_{self.initial_spread}"
                else:
                    file_name = f"result_o_{self.initial_offset}_u_{self.initial_spread}"

                plt.savefig(self.output_dir + file_name + ".pdf", format='pdf')
                plt.savefig(self.output_dir + file_name + ".png", format='png')

    plt.clf()
    plt.close()

    def plot_iterations(self, estimates):
        colorscale = plt.get_cmap('tab20')
        estimates = np.array(estimates)
        estimates[:,:,1:3] *= 0.91
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax_ela = ax[0]
        #ax_ela.set_title('Equilibrium Line Altitude')
        ax_ela.plot(range(len(estimates)), [self.smb[-1][3]]*len(estimates), color=colorscale(9), linewidth=3,
                    linestyle='-.', label='Glaciological Mean [GLAMOS]', zorder=5)
        #ax_ela.fill_between(range(len(estimates)), [2845]*len(estimates),
        #                    [3155]*len(estimates), color=colorscale(9), alpha=0.2)
        ax_ela.plot(range(len(estimates)), np.array(estimates)[:, :, 0], color='gold', zorder=1)
        ax_ela.plot(range(len(estimates)), np.mean(np.array(estimates)[:, :, 0], axis=1), color=colorscale(2),
                    zorder=10)
        ax_ela.set_ylabel('Equilibrium Line Altitude [$m$]')
        ax_ela.set_xlabel('Iterations')

        ax_abl = ax[1]
        #ax_abl.set_title('Ablation Gradient')
        ax_abl.plot(range(len(estimates)), [self.smb[-1][1]] * len(estimates), color=colorscale(9), linewidth=3,
                    linestyle='-.', label='Glaciological Mean [GLAMOS]', zorder=5)
        #ax_abl.fill_between(range(len(estimates)), [0.00556]*len(estimates),
        #                    [0.01141]*len(estimates), color=colorscale(9), alpha=0.2)

        ax_abl.plot(range(len(estimates)), np.array(estimates)[:, :, 1], color='gold', zorder=1)
        ax_abl.plot(range(len(estimates)), np.mean(np.array(estimates)[:, :, 1], axis=1), color=colorscale(2),
                    label= 'ensemble mean', zorder=10)
        ax_abl.set_ylabel('Ablation Gradient [$m~w.e.~a^{-1}~m^{-1}$]')
        ax_abl.set_xlabel('Iterations')

        ax_acc = ax[2]
        #ax_acc.set_title('Accumulation Gradient')
        ax_acc.plot(range(len(estimates)), np.mean(np.array(estimates)[:, :, 2], axis=1), color=colorscale(2),
                    label= 'Ensemble Mean', zorder=10)


        ax_acc.plot(range(len(estimates)), np.array(estimates)[:, :, 2], color='gold', label='Ensemble Member', zorder=1)
        ax_acc.plot(range(len(estimates)), [self.smb[-1][2]] * len(estimates), color=colorscale(9), linewidth=3,
                    linestyle='-.', label='Glaciological Mean [GLAMOS]', zorder=5)
        #ax_acc.fill_between(range(len(estimates)), [0.000216] * len(estimates),
        #                    [0.00244] * len(estimates), color=colorscale(9), alpha=0.2, label='Extremes [GLAMOS]')
        ax_acc.set_ylabel('Accumulation Gradient [$m~w.e.~a^{-1}~m^{-1}$]')
        ax_acc.set_xlabel('Iterations')
        for axi in [ax[0], ax[1], ax[2]]:
            axi.spines['top'].set_visible(False)
            axi.spines['right'].set_visible(False)
            axi.spines['bottom'].set_visible(False)
            axi.spines['left'].set_visible(False)
            axi.grid(axis="y", color="lightgray", linestyle="-", zorder=0)
            axi.grid(axis="x", color="lightgray", linestyle="-", zorder=0)
            axi.xaxis.set_tick_params(bottom=False)
            axi.yaxis.set_tick_params(left=False)
            # axi.legend(framealpha=1)

        plt.tight_layout()
        handles, labels = axi.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)
        fig.subplots_adjust(top=0.9, bottom=0.15)
        plt.savefig(self.output_dir+f'iterations_o_{self.initial_offset}_s_{self.initial_spread}_seed_{self.seed}.png', dpi=300)
