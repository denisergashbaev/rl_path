import numpy as np
import logging

log = logging.getLogger(__name__)


class DqnEnv:

    def __init__(self, c):
        self.steps = []
        self.c = c

    # Color code

    # First layer

    # 0 - Blank Cell
    # 255 - Color Cell

    # Second layer

    # 0 - No agent
    # 255 - Agent

    def step(self, s_t, a_t):

        _, x_dim, y_dim = s_t.shape

        x_lower_boundary = 0
        y_lower_boundary = 0
        x_upper_boundary = x_dim - 1
        y_upper_boundary = y_dim - 1

        agent_position = tuple(np.argwhere(s_t[1] > 1)[0])

        s_tp1 = np.copy(s_t)

        r_tp1 = 0

        step_reward = self.c.step_reward
        done = False

        if a_t == 'r':

            r_tp1 = step_reward
            if agent_position[1] < y_upper_boundary:

                new_agent_position = tuple(np.array(agent_position) + (0, +1))

                s_tp1[1][agent_position] -= 255
                s_tp1[1][new_agent_position] += 255
            elif self.c.test:
                log.debug('out of boundary')

        elif a_t == 'l':

            r_tp1 = step_reward
            if agent_position[1] > y_lower_boundary:

                new_agent_position = tuple(np.array(agent_position) + (0, -1))
                s_tp1[1][agent_position] -= 255
                s_tp1[1][new_agent_position] += 255
            elif self.c.test:
                log.debug('out of boundary')

        elif a_t == 'd':

            r_tp1 = step_reward
            if agent_position[0] < x_upper_boundary:

                new_agent_position = tuple(np.array(agent_position) + (+1, 0))

                s_tp1[1][agent_position] -= 255
                s_tp1[1][new_agent_position] += 255
            elif self.c.test:
                log.debug('out of boundary')

        elif a_t == 'u':

            r_tp1 = step_reward
            if agent_position[0] > x_lower_boundary:

                new_agent_position = tuple(np.array(agent_position) + (-1, 0))

                s_tp1[1][agent_position] -= 255
                s_tp1[1][new_agent_position] += 255
            elif self.c.test:
                log.debug('out of boundary')

        elif a_t == 's':

            # If current cell is : Color Cell + Agent
            if s_tp1[0][agent_position] == 255:
                r_tp1 = 1
                # Current cell becomes : Blank Cell + Agent
                s_tp1[0][agent_position] = 0
                self.steps.append(agent_position)

            # Else, penalize the agent for trying to stitch in a wrong position
            else:
                r_tp1 = -1
                done = self.c.fast_fail or self.c.test
                if self.c.test:
                    log.debug('double stitch')


        # The agent is done stitching if there aren't any Color Cells or Color + Agent Cells
        if not np.any(s_tp1[0] == 255):
            done = True

        return s_tp1, r_tp1, done