import json
import shutil
from netCDF4 import Dataset
import numpy as np
import subprocess
import traceback
import xarray as xr
import os
import h5py


class EnsembleMember:
    def __init__(self, member_number, ensemble_dir,  start_year):
        self.member_number = member_number
        self.member_dir = ensemble_dir/ f"{self.member_number:03}/"
        self.geology = Dataset(self.member_dir / "geology-optimized.nc", "r")
        self.usurf = np.array(self.geology['usurf'])
        self.bedrock = np.array(self.geology['topg'])
        self.velo = np.zeros_like(self.usurf)
        self.year = start_year

    def forward(self, state_x, dt):
        # create new params.json
        year_next = self.year + dt

        ela, grad_abl, grad_acc = state_x[[0, 1, 2]]

        data = {"modules_preproc": ["load_ncdf"],
                "modules_process": ["smb_simple", "iceflow", "time", "thk"],
                "modules_postproc": ["write_ncdf", "print_info"],
                "smb_simple_array": [
                    ["time", "gradabl", "gradacc", "ela", "accmax"],
                    [self.year, grad_abl, grad_acc, ela, 100],
                    [year_next, grad_abl, grad_acc, ela, 100]],
                "iflo_emulator": "iceflow-model",
                "lncd_input_file": f'input.nc',
                "wncd_output_file": f'output.nc',
                "time_start": self.year,
                "time_end": year_next,
                "iflo_retrain_emulator_freq": 0,
                # "time_step_max": 0.2,
                }

        with open(self.member_dir/"params.json", 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))

        # create new input.nc
        input_file = self.member_dir / "input.nc"
        shutil.copy2(self.member_dir / "geology-optimized.nc", input_file)
        try:
            with Dataset(input_file, 'r') as ds:
                ds.variables['usurf'] =self.usurf

                thickness = self.usurf - self.bedrock
                ds.variables['thk'] = thickness

        except:
            traceback.print_exc()
            print("could not read input" + str(input_file))
            exit()

        ### IGM RUN ###
        try:
            subprocess.run(["igm_run"], cwd=self.member_dir, shell=True)
        except:
            print("could not run igm_run")
        # update state x and return
        output_file = self.member_dir /'output.nc'
        try:
            with Dataset(output_file) as new_ds:
                new_usurf = np.array(new_ds['usurf'][-1])
                new_velo = np.array(new_ds['velsurf_mag'][-1])
                new_thk = np.array(new_ds['thk'][-1])
        except:
            print("could not read output")
            print(output_file)

        #os.remove(input_file)
        #os.remove(output_file)


        self.usurf = new_usurf
        self.velo = new_velo
        self.year = year_next
        return state_x

    def reset(self, year):
        self.year = year
        self.usurf = self.geology['usurf']
        self.velo = np.zeros_like(self.usurf)

    def observe(self, observation_points):

        modelled_observations = self.usurf[observation_points[:, 0], observation_points[:, 1]]

        return modelled_observations
