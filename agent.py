import numpy as np


class Agent:
    def __init__(self, agent_id, role, vfd, max_vfd, speed, init_pos, num_actions, num_rows, num_cols):
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
        self.Convergence = False
        self.First = True
        self.wereHere = np.ones((num_rows, num_cols))
        self.Speed = speed  # is the number of cells the agent can move in one time-step

        self.init_pos = init_pos
        self.curr_Pos = self.init_pos
        self.old_Pos = self.curr_Pos

        self.Traj = []  # Trajectory of the agent locations
        self.RewHist = []
        self.RewHist_seen = []
        self.RewSum = []  # Keeps track of the rewards in each step
        self.RewSum_seen = []  # Keeps track of the rewards after receiving first data
        self.Steps = []  # Keeps track of the steps in each step
        self.Steps_seen = []  # Keeps track of the steps after receiving first data

        self.num_actions = num_actions
        self.t_step_seen = 0
        self.action = None
        self.reward = None
        self.probs = np.nan
        self.Q = np.zeros(((2 * self.max_VisualField + 1) ** 2 + 1, self.num_actions))
        self.Q_hist = self.Q

    def reset(self):
        self.old_Pos = self.init_pos
        self.curr_Pos = self.init_pos
        self.old_Sensation = [np.nan, np.nan]
        self.curr_Sensation = [np.nan, np.nan]
        self.CanSeeIt = False
        self.Finish = False
        self.Convergence = False
        self.First = True
        self.t_step_seen = 0
        self.RewHist = []
        self.RewHist_seen = []
        self.Traj = []
        self.wereHere = np.ones_like(self.wereHere)

    def cell_marker(self, pos):
        self.wereHere[pos[0], pos[1]] = 0

    def smart_move(self, idx, wereHere):

        if idx == (2 * self.max_VisualField + 1) ** 2:
            if len(np.argwhere(wereHere)) > 0:
                for loc in np.argwhere(wereHere):
                    if np.sqrt((loc[0] - self.old_Pos[0]) ** 2 + (loc[1] - self.old_Pos[1]) ** 2) == 1:
                        self.curr_Pos = loc
                        break
                    else:
                        continue

        return self.curr_Pos

    def update_sensation(self, index, raw_sensation, sensation_evaluate, pos2pos, net_adj_mat, adj_mat):

        next_sensation = [np.nan, np.nan]
        self.CanSeeIt = False

        if any(sensation_evaluate[index, :]):
            first_victim = np.argwhere(sensation_evaluate[index, :])[0][0]
            next_sensation = raw_sensation[index, first_victim, :]
            self.CanSeeIt = True

        elif not all(np.isnan(net_adj_mat[index, :])):
            num_scouts = np.sum(adj_mat[index, :])
            for ns in range(int(num_scouts)):
                curr_scout = np.argwhere(adj_mat[index, :])[ns]

                if any(sensation_evaluate[curr_scout, :][0].tolist()):
                    first_victim = np.argwhere(sensation_evaluate[curr_scout, :][0])[0]

                    next_sensation[0] = (pos2pos[curr_scout, index][0][0] +
                                         raw_sensation[curr_scout, first_victim, :][0][0])
                    next_sensation[1] = (pos2pos[curr_scout, index][0][1] +
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

    def rescue_accomplished(self, rescue_team_Hist, agent, adj_mat):
        if self.old_Sensation[0] == 0 and self.old_Sensation[1] == 0:
            self.Finish = True

            adj_mat = np.delete(adj_mat, rescue_team_Hist.index(agent), 0)
            adj_mat = np.delete(adj_mat, rescue_team_Hist.index(agent), 1)

            rescue_team_Hist.remove(agent)
        if not self.Finish:
            self.old_Pos = self.curr_Pos

        return rescue_team_Hist, adj_mat

    def victim_rescued(self, rescuers_pos_list, victim, victims_Hist, victims_memory):

        for rescuer_pos in rescuers_pos_list:
            if ((rescuer_pos[0] == self.old_Pos[0] and rescuer_pos[1] == self.old_Pos[1]) and
               rescuer_pos not in victims_memory):
                self.Finish = True
                victims_Hist.remove(victim)
                who_rescued = rescuer_pos.tolist()
        if not self.Finish:
            self.old_Pos = self.curr_Pos
            who_rescued = []
        return victims_Hist, who_rescued

    def convergence_check(self, accuracy):

        if (np.abs(np.sum(self.Q - self.Q_hist) /
                   (np.shape(self.Q)[0] * np.shape(self.Q)[1])) <= accuracy):
            self.Convergence = True

        return self.Convergence
