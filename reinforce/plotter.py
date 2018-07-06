import os
import numpy as np
import matplotlib.pyplot as plt

mylist = np.load('reinforce_total_reward.npy')
N = 3
cumsum, moving_aves = [0], []

for i, x in enumerate(mylist, 1):
    cumsum.append(cumsum[i-1] + x)
    if i >= N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)

step = 100
moving_aves = moving_aves[0::step]
plt.xticks(range(0, len(moving_aves)), range(1, len(moving_aves * step), step))
plt.title('CartPole simulation with REINFORCE algorithm')
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.plot(moving_aves)
plt.show()
