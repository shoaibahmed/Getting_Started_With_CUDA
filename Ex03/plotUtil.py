import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter

# Load the file
blocks = {}

performanceLogFilePath = os.path.join(os.getcwd(), "Output-task01.txt")
with open(performanceLogFilePath, 'r') as performanceLogFile:
	for line in performanceLogFile:
		linePortions = line.split(',')
		if int(linePortions[0]) not in blocks:
			blocks[int(linePortions[0])] = {}
			blocks[int(linePortions[0])]["threads"] = []
			blocks[int(linePortions[0])]["kernelA"] = []
			blocks[int(linePortions[0])]["kernelM"] = []

		blocks[int(linePortions[0])]["threads"].append(int(linePortions[1]))
		blocks[int(linePortions[0])]["kernelA"].append(float(linePortions[2]))
		blocks[int(linePortions[0])]["kernelM"].append(float(linePortions[3]))

fig = plt.figure()
ax = fig.gca(projection='3d')

def cc(arg):
    return colorConverter.to_rgba(arg, alpha=0.6)

verts = []
blockIds = list(blocks.keys())
blockIds.sort()
zs = range(len(blockIds))
for blockId in blockIds:
    verts.append(list(zip(blocks[blockId]["threads"], blocks[blockId]["kernelA"])))

poly = LineCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'), cc('y')])
# poly = LineCollection(verts)
poly.set_alpha(0.75)
ax.add_collection3d(poly, zs=zs, zdir='x')

ax.set_xlabel('Blocks per SM')
ax.set_xlim3d(0, 8)
ax.set_ylabel('Threads per block')
ax.set_ylim3d(32, 1024)
ax.set_zlabel('Occupancy')
ax.set_zlim3d(0, 1)


# plt.savefig('./kernel_A_occupancy.png', dpi=300)
plt.show()
plt.close('all')
