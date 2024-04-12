import os
from PIL import Image

# get file names in correct order
dir = 'Results_real/True/Plots/'
#dir = 'Results_specal_noise/1/Plots/'
files = os.listdir(dir)
files.sort()

# read all frames
frames = []
for image in files:
    if image.startswith('report'):
        frames.append(Image.open(dir+image))


frame_one = Image.open(dir+'report2004_predict.png')
frame_one.save("Plots/EnKF.gif", format="GIF", append_images=frames, save_all=True, duration=1000, loop=0)