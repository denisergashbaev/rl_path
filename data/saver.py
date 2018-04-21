import numpy as np
import os

layer = np.load(os.path.join('5_15.npy'))
#layer *= 255
#np.save('5_15.npy', layer)
print(layer)