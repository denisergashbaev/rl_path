import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation



# https://stackoverflow.com/a/21712361/256002
from pytsp.tsp_computer import TSPComputer

fig, ax = plt.subplots()

#imshow portion
layer = np.load(os.path.join('data', '7_17.npy'))

tsp = TSPComputer(layer)
for (x, y), value in np.ndenumerate(layer):
        if (x,y) in tsp.back_coords:
            ax.text(y, x, tsp.back_coords[(x, y)], va='center', ha='center')

#set tick marks for grid
min_val_y, max_val_y, diff = 0, layer.shape[0], 1
min_val_x, max_val_x, diff = 0, layer.shape[1], 1
ax.set_xticks(np.arange(min_val_x-diff/2, max_val_x-diff/2))
ax.set_yticks(np.arange(min_val_y-diff/2, max_val_y-diff/2))
ax.set_xticklabels([])
ax.set_yticklabels([])
#ax.set_xlim(min_val-diff/2, max_val-diff/2)
#ax.set_ylim(min_val-diff/2, max_val-diff/2)
ax.grid()
im = ax.imshow(layer, interpolation=None, animated=True)

tsp_path = [0, 1, 2, 3, 4, 11, 12, 5, 6, 13, 14, 19, 23, 22, 18, 17, 21, 26, 27, 32, 31, 36, 35, 34, 38, 39, 37, 33, 28, 29, 30, 25, 24, 20, 16, 15, 10, 9, 8, 7]
print('tsp_cost:', tsp.tsp_cost((2, 1)))
print('tsp_check_cost:', tsp.idx_cost(tsp_path))

def update(i):
    r = tsp_path[i]
    if r in tsp.coords:
        for t in ax.texts:
            c = tsp.coords[r]
            if (t._x, t._y) == (c[1], c[0]):
                t._text = 'X'

ani = animation.FuncAnimation(fig, update, frames=len(tsp_path), interval=500)
plt.show()
