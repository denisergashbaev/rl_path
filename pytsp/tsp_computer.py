from scipy.spatial.distance import cdist
from pytsp import run, dumps_matrix
from collections import OrderedDict
import numpy as np
import logging

log = logging.getLogger(__name__)


class TSPComputer:
    def __init__(self, layer):
        self.coords = OrderedDict()
        self.back_coords = {}
        for (x, y), value in np.ndenumerate(layer):
            if value > 0:
                i = len(self.coords.keys())
                c = (x, y)
                self.coords[i] = c
                self.back_coords[c] = i
        coords_arr = [v for v in self.coords.values()]
        self.dist_matrix = cdist(coords_arr, coords_arr, metric='cityblock')

    def rl_cost(self, coords):
        return self.idx_cost([self.back_coords[c] for c in coords])

    def tsp_cost(self, start_coord):
        return self.tsp(start_coord)['solution']

    def tsp_path(self, start_coord):
        return self.tsp(start_coord)['tour']

    def tsp(self, start_coord):
        out_f = "./tsp_dist.tsp"
        with open(out_f, 'w') as dest:
            dest.write(dumps_matrix(self.dist_matrix, name="TSP_Route"))
        tour = run(out_f, start=self.back_coords[start_coord], solver="lkh")
        return tour

    #[0, 1, 2, 3, 4, 10, 16, 17, 23, 29, 35, 34, 41, 47, 46, 50, 49, 48, 44, 45, 39, 40, 33, 27, 28, 22,
    #                  21, 15, 9, 8, 14, 13, 20, 26, 32, 38, 37, 43, 42, 36, 30, 31, 25, 24, 18, 19, 12, 11, 5, 7, 6]
    def idx_cost(self, idx):
        cost = 0
        for i, p0 in enumerate(idx):
            p1 = idx[i +1] if i < len(idx) - 1 else 0
            cost += self.dist_matrix[p0][p1]
        return cost