import numpy as np


class Network:
    def __init__(self, adj_mat, num_agents, num_victims):
        self.adj_mat = adj_mat.copy()
        self.adj_mat[self.adj_mat == 0] = np.nan
        self.num_agents = num_agents
        self.num_victims = num_victims

    def pos2pos(self, pos_list):

        pos_array = np.empty((self.num_agents, self.num_agents, 2))
        pos_array[:, :, 0] = np.tile(pos_list[:, 0].reshape(self.num_agents, 1), (1, self.num_agents))
        pos_array[:, :, 1] = np.tile(pos_list[:, 1].reshape(self.num_agents, 1), (1, self.num_agents))

        pos2pos = np.subtract(pos_array, np.transpose(pos_array, (1, 0, 2)))
        return pos2pos

    def sensed_pos(self, victim_pos_list, rs_pos_list):
        num_victims = len(victim_pos_list)
        num_agents = len(rs_pos_list)
        rs_pos_array = np.empty((num_agents, num_victims, 2))
        rs_pos_array[:, :, 0] = np.tile(rs_pos_list[:, 0].reshape(num_agents, 1), (1, num_victims))
        rs_pos_array[:, :, 1] = np.tile(rs_pos_list[:, 1].reshape(num_agents, 1), (1, num_victims))

        victim_pos_array = np.empty((num_agents, num_victims, 2))
        victim_pos_array[:, :, 0] = np.tile(victim_pos_list[:, 0].reshape(1, num_victims), (num_agents, 1))
        victim_pos_array[:, :, 1] = np.tile(victim_pos_list[:, 1].reshape(1, num_victims), (num_agents, 1))

        return np.subtract(victim_pos_array, rs_pos_array)

    def is_seen(self, vfd_list, raw_sensation):
        vfd_mat = np.tile(vfd_list.reshape(self.num_agents, 1), (1, self.num_victims))
        condition = np.zeros_like(raw_sensation)
        condition[:, :, 0] = condition[:, :, 1] = vfd_mat
        tuple_cond = np.abs(raw_sensation) <= condition
        return np.logical_and(tuple_cond[:, :, 0], tuple_cond[:, :, 1])

    def wall_sensor(self, rescuer2wall, rescuer2victim, victim2wall, agent_id, is_seen):
        num_walls = np.shape(rescuer2wall)[1]
        num_victims = np.shape(rescuer2victim)[1]
        for wall in range(num_walls):
            for victim in range(num_victims):
                # Use triangle inequality to prevent seeing behind the walls
                if ((np.linalg.norm(rescuer2wall[agent_id, wall, :]) + np.linalg.norm(victim2wall[victim, wall, :])) <=
                        np.linalg.norm(rescuer2victim[agent_id, victim, :])):
                    is_seen[agent_id, victim] = False
        return is_seen

