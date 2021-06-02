import h5py
import numpy as np
import sys

import matplotlib as mpl
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt

data = h5py.File(sys.argv[1], 'r', libver='latest', swmr=True)['all_events']
# NCHW

shape = data['images'].shape[1:]
channel, height, width= shape

dat = data['images']

x = []
y = []
val = []

for i in range(width):
    for j in range(height):
        x.append(i)
        y.append(j)
        val.append(dat[0][0][i][j])

xlim = [0,width] # x range
ylim=[0,height] # y range


H, xedges, yedges = np.histogram2d(x,y,weights = val, bins = [width,height], range = [xlim, ylim])
H =H.T

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111, title='')

cmap = mpl.colors.ListedColormap(["navy", "blue", "royalblue", "lightseagreen","yellowgreen","limegreen", "gold"])
cmap.set_under("w")

X,Y = np.meshgrid(xedges,yedges)
mesh = ax.pcolormesh(X, Y, H, cmap=cmap, vmin = 0.1)

cb = plt.colorbar(mesh , ax=ax)
cb.set_label('E [GeV]')

plt.savefig("img.png")
plt.show()
