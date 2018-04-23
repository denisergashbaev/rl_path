import numpy as np
import os

a = np.load(os.path.join('data_file=0_13.npy,step_reward=-0.01,fast_fail=True,reuse_weights=True,test=False', 'episode_reward.npy'))

print(len(a))
for i in a:
    print(i)

