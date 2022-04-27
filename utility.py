import numpy as np

# Actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
ACTIONS = [FORWARD, BACKWARD, RIGHT, LEFT]

#
#
# class Agent:
#     def __init__(self, agent_type, obs_size, pos):
#         self.agent_type = agent_type
#         self.obs_size = obs_size
#         self.pos = pos
#
#     def select_action(self):
#         action = np.random.choice(ACTIONS)
#         return action
#
#     def is_goal_in_sensation(self, goal_pos):
#         dist = np.subtract(goal_pos, self.pos)
#         if abs(dist[0]) <= self.obs_size and abs(dist[1]) <= self.obs_size:
#             return True
#         return False
#
#     def get_goal_location(self):
#         # return goal location to your teammate