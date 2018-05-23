# Finding shortest path on a colored image with reinforcement learning 

Largely work in progress. The file [animation_rl.py](https://github.com/denisergashbaev/rl_path/blob/master/animation_rl.py)
 visualizes a path that a trained DQN-agent has found to fill out number O. 
![Number 0](https://github.com/denisergashbaev/rl_path/blob/master/number0_path.gif "Number 0"). 

The goal is to cover only yellow cells with (double visits prohibited) in the most efficient way (as defined by Manhattan distance)

The original DQN implementation is based on [rl_args_implementation](https://github.com/MaximilienLC/rl_algs_implementation) 
