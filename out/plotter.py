import os
import numpy as np
import matplotlib.pyplot as plt

mylist = np.load(os.path.join('data_file=0_13.npy,step_reward=-0.09,fast_fail=True,reuse_weights=True,test=False', 'episode_reward.npy'))
N = 3
cumsum, moving_aves = [0], []

for i, x in enumerate(mylist, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)

moving_aves = moving_aves[0::200]
plt.plot(moving_aves)
plt.show()