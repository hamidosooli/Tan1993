import numpy as np


class Agent:
    def __init__(self, agent_id, role, vfd, max_vfd, speed, curr_pos, num_actions, num_rows, num_cols):
        self.Role = role  # can be 'r': rescuer, 's': scout, 'rs': rescuer and scout, 'v': victim
        self.id = agent_id  # an identification for the agent
        self.VisualField = vfd
        self.max_VisualField = max_vfd

        self.curr_Sensation = [np.nan, np.nan]
        self.old_Sensation = self.curr_Sensation

        self.curr_Index = None
        self.old_Index = self.curr_Index

        self.CanSeeIt = False
        self.Finish = False
        self.wereHere = np.ones((num_rows, num_cols))
        self.Speed = speed  # is the number of cells the agent can move in one time-step

        self.curr_Pos = curr_pos
        self.old_Pos = self.curr_Pos

        self.Traj = []  # Trajectory of the agent locations
        self.RewHist = []  # Keeps track of the rewards in each step
        self.RewHist_seen = []  # Keeps track of the rewards after receiving first data
        self.Steps = []  # Keeps track of the steps in each step
        self.Steps_seen = []  # Keeps track of the steps after receiving first data

        self.num_actions = num_actions
        self.t_step_seen = 0
        self.action = None
        self.reward = None
        self.probs = np.nan
        self.Q = np.zeros(((2 * self.max_VisualField + 1) ** 2 + 1, self.num_actions))

    def reset(self, pos):
        self.old_Pos = pos
        self.curr_Pos = pos
        self.old_Sensation = [np.nan, np.nan]
        self.curr_Sensation = [np.nan, np.nan]
        self.CanSeeIt = False
        self.Finish = False
        self.t_step_seen = 0
        self.wereHere = np.ones_like(self.wereHere)

    # def receive_info(self, their_sensation, our_distance):
    #     if 'r' in self.Role:
    #         self.their_Sensation = their_sensation
    #         self.our_Distance = our_distance
    #     else:
    #         self.their_Sensation = [np.nan, np.nan]
    #         self.our_Distance = [np.nan, np.nan]
    #
    # def send_info(self):
    #     if 's' in self.Role:
    #         return self.curr_Sensation
    #     else:
    #         return [np.nan, np.nan]

    def cell_marker(self, pos):
        self.wereHere[pos[0], pos[1]] = 0

    def smart_move(self, idx):

        if idx == (2 * self.max_VisualField + 1) ** 2:
            if len(np.argwhere(self.wereHere)) > 0:
                for loc in np.argwhere(self.wereHere):
                    if np.sqrt((loc[0] - self.old_Pos[0]) ** 2 + (loc[1] - self.old_Pos[1]) ** 2) == 1:
                        self.curr_Pos = loc
                        break
                    else:
                        continue

        return self.curr_Pos

    def update_sensation(self, raw_sensation, sensation_evaluate, pos2pos, net_adj_mat, adj_mat):

        next_sensation = [np.nan, np.nan]
        self.CanSeeIt = False

        if any(sensation_evaluate[self.id, :]):
            first_victim = np.argwhere(sensation_evaluate[self.id, :])[0][0]
            next_sensation = raw_sensation[self.id, first_victim, :]
            self.CanSeeIt = True

        elif not all(np.isnan(net_adj_mat[self.id, :])):
            num_scouts = np.sum(adj_mat[self.id, :])
            for ns in range(int(num_scouts)):
                curr_scout = np.argwhere(adj_mat[self.id, :])[ns]

                if any(sensation_evaluate[curr_scout, :][0].tolist()):
                    first_victim = np.argwhere(sensation_evaluate[curr_scout, :][0])[0]

                    next_sensation[0] = (pos2pos[curr_scout, self.id][0][0] +
                                         raw_sensation[curr_scout, first_victim, :][0][0])
                    next_sensation[1] = (pos2pos[curr_scout, self.id][0][1] +
                                         raw_sensation[curr_scout, first_victim, :][0][1])

                    self.CanSeeIt = True
                    break

        return next_sensation

    def sensation2index(self, sensation, max_vfd):
        if self.CanSeeIt:
            index = ((sensation[0] + max_vfd) * (2 * max_vfd + 1) + (sensation[1] + max_vfd))
        else:
            index = (2 * max_vfd + 1) ** 2

        return int(index)

    def rescue_accomplished(self):

        if self.old_Sensation[0] == 0 and self.old_Sensation[0] == 0:
            self.Finish = True

        else:
            self.Finish = False

        return self.Finish

    def victim_rescued(self, agents_pos_list):

        for pos in agents_pos_list:

            if pos[0] == self.old_Pos[0] and pos[1] == self.old_Pos[1]:

                self.Finish = True

            else:

                self.Finish = False


