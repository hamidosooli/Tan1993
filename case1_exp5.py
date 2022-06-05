import numpy as np
import h5py
from action_selection import eps_greedy

# Actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
ACTIONS = [FORWARD, BACKWARD, RIGHT, LEFT]
nA = len(ACTIONS)

NUM_EPISODES = 500
NUM_RUNS = 100
Hunter_VFD = 2  # Hunter's visual field depth
Scout_VFD = 2
gamma = .9

Row_num = 10
Col_num = 10
row_lim = 9
column_lim = 9

can_see_it = False
can_see_it_scout = False
default_sensation = [np.nan, np.nan]


def Boltzmann(q, t=0.4):
    return np.exp(q / t) / np.sum(np.exp(q / t))


def movement(position, action):
    row = position[0]
    column = position[1]

    if action == 0:  # up
        next_position = [max(row - 1, 0), column]
    elif action == 1:  # down
        next_position = [min(row + 1, row_lim), column]
    elif action == 2:  # right
        next_position = [row, min(column + 1, column_lim)]
    elif action == 3:  # left
        next_position = [row, max(column - 1, 0)]

    return next_position


def update_sensation_scout(scout_sensation):
    global can_see_it_scout
    global default_sensation
    if abs(scout_sensation[0]) <= Scout_VFD and abs(scout_sensation[1]) <= Scout_VFD:
        row = scout_sensation[0]
        column = scout_sensation[1]
        can_see_it_scout = True

    else:  # if there is no prey in sight, a unique default sensation is used.
        row, column = [np.nan, np.nan]
        can_see_it_scout = False

    scout_sensation_prime = [row, column]

    return scout_sensation_prime


def update_sensation(hunter_sensation, scout_sensation, scout2hunter):
    global can_see_it
    global default_sensation
    if abs(hunter_sensation[0]) <= Hunter_VFD and abs(hunter_sensation[1]) <= Hunter_VFD:
        row = hunter_sensation[0]
        column = hunter_sensation[1]
        can_see_it = True
        default_sensation = [row, column]
    elif abs(scout_sensation[0]) <= Scout_VFD and abs(scout_sensation[1]) <= Scout_VFD:
        row = scout2hunter[0] + scout_sensation[0]
        column = scout2hunter[1] + scout_sensation[1]
        can_see_it = True
        default_sensation = [row, column]
    else:  # if there is no prey in sight, a unique default sensation is used.
        row, column = default_sensation
        can_see_it = False

    hunter_sensation_prime = [row, column]

    return hunter_sensation_prime


def reward(hunter_sensation_prime):
    if hunter_sensation_prime == [0, 0]:
        re = 1
    else:
        re = -.1
    return re


def sensation2index(sensation, VFD, can_see_it_local):
    if can_see_it_local:
        index = (sensation[0] + VFD) * (2 * VFD + 1) + (sensation[1] + VFD)
    else:
        index = (2 * VFD + 1) ** 2
    return index


def rl_agent(beta=0.8):
    global can_see_it
    global default_sensation
    Q = np.zeros(((2 * Row_num + 1) ** 2 + 1, nA))
    Q_scout = np.zeros(((2 * Scout_VFD + 1) ** 2 + 1, nA))

    steps = []
    rewards = []
    rewards_scout = []
    see_steps = []
    see_rewards = []
    see_steps_scout = []
    see_rewards_scout = []
    for eps in range(NUM_EPISODES):
        can_see_it = False
        can_see_it_scout = False
        default_sensation = [np.nan, np.nan]
        hunter_pos = [0, 9]  # [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]
        scout_pos = [6, 3]  # [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]
        prey_pos = [9, 0]  # [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]

        T_hunter = []
        T_scout = []
        T_prey = []

        R = []
        R_prime = []

        R_scout = []
        R_prime_scout = []

        A_hunter = []
        A_scout = []
        A_prey = []

        t_step = 0
        see_t_step = 0
        see_t_step_scout = 0
        while True:
            t_step += 1

            T_hunter.append(hunter_pos)
            T_scout.append(scout_pos)
            T_prey.append(prey_pos)

            scout2hunter = np.subtract(scout_pos, hunter_pos)

            scout_sensation_step1 = np.subtract(prey_pos, scout_pos)
            scout_sensation = update_sensation_scout(scout_sensation_step1)

            hunter_sensation_step1 = np.subtract(prey_pos, hunter_pos)
            hunter_sensation = update_sensation(hunter_sensation_step1, scout_sensation, scout2hunter)

            idx = sensation2index(hunter_sensation, Row_num, can_see_it)
            # hunter_probs = Boltzmann(Q[idx, :])
            hunter_action = eps_greedy(Q[idx, :], ACTIONS)#np.random.choice(ACTIONS, p=hunter_probs)

            idx_scout = sensation2index(scout_sensation, Scout_VFD, can_see_it_scout)
            # scout_probs = Boltzmann(Q_scout[idx_scout, :])
            scout_action = eps_greedy(Q_scout[idx_scout, :], ACTIONS)#np.random.choice(ACTIONS, p=scout_probs)

            prey_action = np.random.choice(ACTIONS)

            hunter_pos_prime = movement(hunter_pos, hunter_action)
            scout_pos_prime = movement(scout_pos, scout_action)
            prey_pos_prime = movement(prey_pos, prey_action)

            scout2hunter_prime = np.subtract(scout_pos_prime, hunter_pos_prime)
            scout_sensation_prime_step1 = np.subtract(prey_pos, scout_pos_prime)
            scout_sensation_prime = update_sensation_scout(scout_sensation_prime_step1)
            hunter_sensation_prime_step1 = np.subtract(prey_pos, hunter_pos_prime)
            hunter_sensation_prime = update_sensation(hunter_sensation_prime_step1, scout_sensation_prime, scout2hunter_prime)

            re = reward(hunter_sensation)
            re_scout = reward(scout_sensation)
            R.append(re)
            R_scout.append(re_scout)

            A_hunter.append(hunter_action)
            A_scout.append(scout_action)
            A_prey.append(prey_action)

            if can_see_it:
                R_prime.append(re)
                see_t_step += 1
            if can_see_it_scout:
                R_prime_scout.append(re_scout)
                see_t_step_scout += 1

            idx_prime = sensation2index(hunter_sensation_prime, Row_num, can_see_it)
            Q[idx, hunter_action] += beta * (re + gamma * np.max(Q[idx_prime, :]) - Q[idx, hunter_action])


            idx_scout_prime = sensation2index(scout_sensation_prime, Scout_VFD, can_see_it_scout)
            Q_scout[idx_scout, scout_action] += beta * (re_scout + gamma * np.max(Q_scout[idx_scout_prime, :]) -
                                                        Q_scout[idx_scout, scout_action])

            hunter_pos = hunter_pos_prime
            scout_pos = scout_pos_prime

            if t_step % 2 == 0:
                prey_pos = prey_pos_prime

            if hunter_sensation == [0, 0]:
                steps.append(t_step+1)
                see_steps.append(see_t_step+1)
                see_steps_scout.append(see_t_step+1)

                rewards.append(sum(R))
                rewards_scout.append(sum(R_scout))
                see_rewards.append(sum(R_prime))
                see_rewards_scout.append(sum(R_prime_scout))
                print(f'In episode {eps + 1} of {NUM_EPISODES}, the prey was captured in {t_step + 1} steps')

                break

    return rewards, rewards_scout, steps, see_rewards, see_steps, see_rewards_scout, see_steps_scout

rewards_hunter_runs = []
rewards_scout_runs = []
steps_runs = []
see_rewards_runs = []
see_steps_runs = []
see_rewards_scout_runs = []
see_steps_scout_runs = []

for run in range(NUM_RUNS):
    rewards, rewards_scout, steps, see_rewards, see_steps, see_rewards_scout, see_steps_scout = rl_agent(beta=0.8)

    rewards_hunter_runs.append(rewards)
    rewards_scout_runs.append(rewards_scout)
    steps_runs.append(steps)
    see_rewards_runs.append(see_rewards)
    see_steps_runs.append(see_steps)
    see_rewards_scout_runs.append(see_rewards_scout)
    see_steps_scout_runs.append(see_steps_scout)


with h5py.File(f'Tan1993_case1_with_learning_scout_2&2_100runs_epsGreedy.hdf5', "w") as f:

    f.create_dataset('rewards', data=rewards_hunter_runs)
    f.create_dataset('rewards_scout', data=rewards_scout_runs)
    f.create_dataset('steps', data=steps_runs)
    f.create_dataset('see_rewards', data=see_rewards_runs)
    f.create_dataset('see_steps', data=see_steps_runs)
    f.create_dataset('see_rewards_scout', data=see_rewards_scout_runs)
    f.create_dataset('see_steps_scout', data=see_steps_scout_runs)
