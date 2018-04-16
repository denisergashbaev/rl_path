import numpy as np


class DqnEnv:

    def __init__(self):
        self.steps = []

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

        done = False

        if a_t == 'r':

            if agent_position[1] < y_upper_boundary:

                new_agent_position = tuple(np.array(agent_position) + (0, +1))

                s_tp1[1][agent_position] -= 255
                s_tp1[1][new_agent_position] += 255
                r_tp1 = -0.1

        elif a_t == 'l':

            if agent_position[1] > y_lower_boundary:

                new_agent_position = tuple(np.array(agent_position) + (0, -1))
                s_tp1[1][agent_position] -= 255
                s_tp1[1][new_agent_position] += 255
                r_tp1 = -0.1

        elif a_t == 'd':

            if agent_position[0] < x_upper_boundary:

                new_agent_position = tuple(np.array(agent_position) + (+1, 0))

                s_tp1[1][agent_position] -= 255
                s_tp1[1][new_agent_position] += 255
                r_tp1 = -0.1

        elif a_t == 'u':

            if agent_position[0] > x_lower_boundary:

                new_agent_position = tuple(np.array(agent_position) + (-1, 0))

                s_tp1[1][agent_position] -= 255
                s_tp1[1][new_agent_position] += 255
                r_tp1 = -0.1

        elif a_t == 's':

            # If current cell is : Color Cell + Agent
            if s_tp1[0][agent_position] == 255:

                # Current cell becomes : Blank Cell + Agent
                s_tp1[0][agent_position] = 0
                r_tp1 = 1

            # Else, penalize the agent for trying to stitch in a wrong position
            else:
                r_tp1 = -1
                done = True

            self.steps.append(agent_position)

        # The agent is done stitching if there aren't any Color Cells or Color + Agent Cells
        if not np.any(s_tp1[0] == 255):
            done = True

        return s_tp1, r_tp1, done