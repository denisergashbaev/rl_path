from scipy.spatial.distance import cdist
from pytsp import run, dumps_matrix
from collections import OrderedDict
import numpy as np


class TSPComputer:

    def __init__(self, layer_1):
        self.coords = OrderedDict()
        self.back_coords = {}
        for (x, y), value in np.ndenumerate(layer_1):
            if value > 0:
                i = len(self.coords.keys())
                c = (x, y)
                self.coords[i] = c
                self.back_coords[c] = i
        coords_arr = [v for v in self.coords.values()]
        self.dist_matrix = cdist(coords_arr, coords_arr, metric='cityblock')

    def rl_cost(self, coords):
        cost = 0
        idx = [self.back_coords[c] for c in coords]
        for p0 in idx:
            p1 = p0 + 1 if p0 < len(idx) - 1 else 0
            cost += self.dist_matrix[p0][p1]
        return cost

    def tsp_cost(self, start_coord):
        out_f = "./tsp_dist.tsp"
        with open(out_f, 'w') as dest:
            dest.write(dumps_matrix(self.dist_matrix, name="TSP_Route"))
        tour = run(out_f, start=self.back_coords[start_coord], solver="lkh")
        print("tour", tour)
        return tour['solution']