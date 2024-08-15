import json
import numpy as np
import matplotlib.pyplot as plt


def plot_smb_ensemble(ensemble_json):
    with open(ensemble_json) as f:
        data = json.load(f)

    ensemble_history = np.array(data['ensemble_history'])
    for i, ensemble in enumerate(ensemble_history):
        for e in ensemble:
            ela = e[0]
            grad_abl = e[1]
            grad_acc = e[2]
            line = [grad_abl*(2200-ela), 0, grad_acc*(3500-ela)]
            coord = [2200, ela, 3500]


            plt.plot(line, coord)

        plt.ylabel('Elevation [m]')
        plt.ylim([2200, 3500])
        plt.xlabel('Surface Mass Balance [$m~a^{-1}$]')
        plt.plot([])
        plt.savefig(f'Plots/SMB_LINE/SMB_iterations_{i}.png')

if __name__ == '__main__':
    plot_smb_ensemble('Experiments/Rhone/result_seed_420.json')