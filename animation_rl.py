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

#number 0
rl_path_0_r0_1 = [(1, 3), (2, 3), (2, 4), (1, 4), (1, 5), (2, 5), (2, 6), (2, 7), (3, 7), (3, 6), (3, 5), (3, 4), (3, 3), (3, 2), (2, 2), (4, 2), (4, 3), (4, 4), (5, 4), (6, 4), (6, 3), (7, 4), (8, 4), (8, 3), (7, 3), (7, 2), (6, 2), (5, 2), (5, 3), (4, 6), (4, 7), (4, 8), (5, 8), (5, 7), (5, 6), (6, 6), (6, 7), (6, 8), (7, 7), (8, 7), (8, 6), (9, 6), (9, 5), (9, 4), (8, 5), (7, 5), (7, 6), (3, 8), (1, 7), (1, 6)]
rl_path_0_r0_5 = [(1, 3), (1, 4), (2, 4), (2, 3), (2, 2), (3, 2), (3, 3), (4, 3), (4, 2), (5, 2), (5, 3), (5, 4), (4, 4), (3, 4), (3, 5), (2, 5), (1, 5), (1, 6), (1, 7), (2, 7), (2, 6), (3, 6), (4, 6), (4, 7), (3, 7), (3, 8), (4, 8), (5, 8), (5, 7), (5, 6), (6, 6), (6, 7), (7, 7), (7, 6), (8, 6), (9, 6), (9, 5), (9, 4), (8, 4), (8, 5), (7, 5), (7, 4), (7, 3), (8, 3), (8, 2), (7, 2), (6, 2), (6, 3), (6, 4)]
rl_path_0_r0_09 = [(1, 3), (1, 4), (2, 4), (2, 3), (2, 2), (3, 2), (3, 3), (4, 3), (4, 2), (5, 2), (6, 2), (6, 3), (5, 3), (5, 4), (4, 4), (3, 4), (3, 5), (2, 5), (1, 5), (1, 6), (1, 7), (2, 7), (2, 6), (3, 6), (4, 6), (4, 7), (3, 7), (5, 7), (5, 6), (6, 6), (6, 7), (7, 7), (7, 6), (7, 5), (7, 4), (6, 4), (7, 3), (7, 2), (8, 2), (8, 3), (8, 4), (9, 4), (9, 5), (8, 5), (8, 6), (9, 6), (8, 7), (6, 8), (5, 8), (4, 8), (3, 8)]
#number 7
rl_path_7_r0_1 = [(2, 1), (3, 1), (3, 2), (3, 3), (2, 2), (2, 3), (2, 4), (2, 5), (3, 5), (3, 4), (4, 4), (4, 5), (4, 6), (3, 6), (2, 6), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (6, 6), (5, 6), (5, 5), (6, 5), (6, 4), (7, 4), (8, 4), (8, 5), (7, 5), (7, 6), (8, 6), (9, 5), (9, 4), (9, 3), (8, 3), (7, 3), (7, 7), (5, 8), (4, 8), (3, 8)]
rl_path_7_r0_5 = [(2, 1), (3, 1), (3, 2), (2, 2), (2, 3), (3, 3), (3, 4), (2, 4), (2, 5), (3, 5), (3, 6), (2, 6), (2, 7), (3, 7), (3, 8), (4, 8), (5, 8), (5, 7), (4, 7), (4, 6), (5, 6), (5, 5), (4, 5), (4, 4), (6, 4), (6, 5), (6, 6), (6, 7), (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (8, 3), (9, 3), (9, 4), (8, 4), (9, 5), (8, 5), (8, 6)]

rl = rl_path_7_r0_5
print('rl_cost:', tsp.rl_cost(rl))

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
    r = rl[i]
    if r in tsp.back_coords:
        for t in ax.texts:
            if (t._x, t._y) == (r[1], r[0]):
                t._text = 'X'

ani = animation.FuncAnimation(fig, update, frames=len(rl), interval=500)
plt.show()
