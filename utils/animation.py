import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation



# https://stackoverflow.com/a/21712361/256002
from pytsp.tsp_computer import TSPComputer

fig, ax = plt.subplots()

#imshow portion
layer = np.load(os.path.join('data', '7_17.npy'))
#layer = np.array([[0.1,0.2],[1.1,1.2]])
#layer = layer[0:3][:]


#text portion
#ind_array = np.arange(min_val, max_val, diff)
#x, y = np.meshgrid(ind_array, ind_array)

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


def update(i):
    for (x, y), value in np.ndenumerate(layer):
        if (x, y) in tsp.back_coords:
            v = tsp.back_coords[(x, y)]
            if v == i:
                for t in ax.texts:
                    if t._x == y and t._y == x:
                        t._text = 'X'


ani = animation.FuncAnimation(fig, update, frames=19, interval=1000)
plt.show()