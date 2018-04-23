import numpy as np
import os

#np.save('2x2.npy', np.array([[255, 0], [255, 255]]))
layer = np.load(os.path.join('2x2.npy'))
#layer *= 255
#np.save('5_15.npy', layer)
print(layer)

