import numpy as np
from agent import Agent
import h5py


# Actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
ACTIONS = [FORWARD, BACKWARD, RIGHT, LEFT]

#  Environment dimensions
Row_num = 10
Col_num = 10
row_lim = 9
col_lim = 9

hunter = Agent(2)
prey = Agent(2)


class QLearning:
    def __init__(self):
        self.t = .4  # Boltzmann temperature
        self.beta = 0.8  # Learning rate
        self.gamma = .9  # Discount factor
        self.episodes = 500  # Number of episodes

        self.steps = []  # Keeps track of the steps to finish each episode
        self.rewards = []  # Keeps track of the sum of rewards in each episode
        self.see_steps = []  # Keeps track of the steps to rescue the victim after receiving first data in each episode
        self.see_rewards = []  # Keeps track of the sum of rewards after receiving first data in each episode

        self.t_step = 0
        self.see_t_step = 0

    def boltzmann(self, q):
        return np.exp(q / self.t) / np.sum(np.exp(q / self.t))

    def rl_algorithm(self, q_dim):  # Q Learning
        q_dim.append(len(ACTIONS))
        q = np.zeros(q_dim)

        for eps in range(self.episodes):
            hunter.reset()
            prey.reset()
        while True:
            self.t_step += 1

            hunter.Traj.append(hunter.Pos)
            prey.Traj.append(prey.Pos)

            hunter_sensation_step1 = np.subtract(prey.Pos, hunter.Pos)
            hunter.update_sensation(hunter_sensation_step1)
            old_sensation = hunter.Sensation

            re = hunter.reward()
            hunter.Rew.append(re)

            idx = hunter.sensation2index(old_sensation)
            hunter_probs = self.boltzmann(q[idx, :])

            hunter_action = np.random.choice(ACTIONS, p=hunter_probs)
            prey_action = np.random.choice(ACTIONS)

            hunter.Pos = hunter.movement(hunter_action)
            prey.Pos = prey.movement(prey_action)

            hunter_sensation_prime_step1 = np.subtract(prey.Pos, hunter.Pos)
            hunter.update_sensation(hunter_sensation_prime_step1)
            new_sensation = hunter.Sensation

            hunter.Acts.append(hunter_action)
            prey.Acts.append(prey_action)

            if hunter.CanSeeIt:
                hunter.Rew_seen.append(re)
                self.see_t_step += 1

            idx_prime = hunter.sensation2index(new_sensation)
            q[idx, hunter_action] += self.beta * (re + self.gamma * np.max(q[idx_prime, :]) - q[idx, hunter_action])

            if hunter.Sensation == [0, 0]:

                self.steps.append(self.t_step+1)
                self.see_steps.append(self.see_t_step+1)

                self.rewards.append(sum(hunter.Rew))
                self.see_rewards.append(sum(hunter.Rew_seen))

                print(f'In episode {eps + 1} of {self.episodes}, the prey was captured in {self.t_step + 1} steps')

                break

        return hunter.Traj, prey.Traj, hunter.Acts, prey.Acts, self.rewards, self.steps, self.see_rewards, self.see_steps, q


rl_agent = QLearning()
dim = [(2 * hunter.VisualField + 1) ** 2 + 1]
T_hunter, T_prey, A_hunter, A_prey, rewards, steps, see_rewards, see_steps, Q = rl_agent.rl_algorithm(dim)

with h5py.File(f'Tan1993_case1.hdf5', "w") as f:
    f.create_dataset('T_hunter', data=T_hunter)
    f.create_dataset('T_prey', data=T_prey)

    f.create_dataset('A_hunter', data=A_hunter)
    f.create_dataset('A_prey', data=A_prey)

    f.create_dataset('rewards', data=rewards)
    f.create_dataset('steps', data=steps)
    f.create_dataset('see_rewards', data=see_rewards)
    f.create_dataset('see_steps', data=see_steps)

    f.create_dataset('Q', data=Q)