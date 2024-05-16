import os
from PIL import Image
import numpy as np

# get file names in correct order
#dir = 'Results_real/True/Plots/'
#dir = 'Results_specal_noise/1/Plots/'
#dir = 'Plots/3D/'
dir = 'Results_real/True/Plots/'
files = os.listdir(dir)
files.sort()

# read all frames
frames = []
for image in files:
    if image.startswith('report'):
        frames.append(Image.open(dir+image))


#frames = frames + [frames[-1]]*5 + list(reversed(frames)) + [frames[0]]*5
frame_one = Image.open(dir+'report2000_update.png')
#frame_one = Image.open(dir+'glacier_surface_2000.png')

frame_one.save("Plots/EnKF_hugonnet.gif", format="GIF", append_images=frames, save_all=True, duration=1000, loop=0)

