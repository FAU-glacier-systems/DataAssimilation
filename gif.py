import os
from PIL import Image

# get file names in correct order
dir = 'Plots/'
files = os.listdir(dir)
files.sort()

# read all frames
frames = []
for image in files:
    if image.endswith('predict.png'):
        frames.append(Image.open(dir+image))

# append all frames to a gif
if not os.path.exists('plots'):
    os.mkdir('plots')

frame_one = Image.open(dir+'report2000_update.png')
frame_one.save("Plots/EnKF.gif", format="GIF", append_images=frames, save_all=True, duration=100, loop=0)