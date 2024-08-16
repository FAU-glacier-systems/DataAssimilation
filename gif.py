import os
from PIL import Image
import numpy as np

# get file names in correct order
dir = 'Experiments/Rhone/Plot/'
files = os.listdir(dir)
files.sort()

# read all frames
frames = []
for image in files:
    if image.startswith('iterations_seed_111'):
        frames.append(Image.open(dir+image))


#frames = frames + [frames[-1]]*5 + list(reversed(frames)) + [frames[0]]*5
frame_one = Image.open(dir+'iterations_seed_111_2.png')
#frame_one = Image.open(dir+'glacier_surface_2000.png')

frame_one.save("Plots/EnKF_iterations_Rhone_real.gif", format="GIF", append_images=frames, save_all=True, duration=1000, loop=0)

