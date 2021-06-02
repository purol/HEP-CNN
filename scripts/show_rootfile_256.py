import uproot
from glob import glob
import numpy as np
import sys

import matplotlib as mpl
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt

f = uproot.open(sys.argv[1])
tree = f["data"]
test_array = tree["Energy_S_first"].array()
print(len(test_array[0]))



x = []
y = []
val = []

for i in range(256):
    for j in range(256):
        x.append(i)
        y.append(j)
        val.append(test_array[0][i][j])

xlim = [0,256] # x range
ylim=[0,256] # y range

height = 256 # bin num at y axis
width = 256 # bin num at x axis

H, xedges, yedges = np.histogram2d(x,y,weights = val, bins = [width,height], range = [xlim, ylim])
H =H.T

fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111, title='')

cmap = mpl.colors.ListedColormap(["navy", "blue", "royalblue", "lightseagreen","yellowgreen","limegreen", "gold"])
cmap.set_under("w")

X,Y = np.meshgrid(xedges,yedges)
mesh = ax.pcolormesh(X, Y, H, cmap=cmap, vmin = 0.0001)

cb = plt.colorbar(mesh , ax=ax)
cb.set_label('E [GeV]')

plt.savefig("img.png")
