import numpy as np
import h5py


# Actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
ACTIONS = [FORWARD, BACKWARD, RIGHT, LEFT]
nA = len(ACTIONS)

NUM_EPISODES = 500
Hunter_VFD = 2  # Hunter's visual field depth
gamma = .9

Row_num = 10
Col_num = 10
row_lim = 9
column_lim = 9

default_sensation = [np.nan, np.nan]
can_see_it = False


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


def update_sensation(hunter_sensation):
    global default_sensation
    global can_see_it
    if abs(hunter_sensation[0]) <= Hunter_VFD and abs(hunter_sensation[1]) <= Hunter_VFD:
        row = hunter_sensation[0]
        column = hunter_sensation[1]
        default_sensation = [row, column]
        can_see_it = True

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


def sensation2index(sensation, VFD):
    global can_see_it
    if can_see_it:
        index = (sensation[0] + VFD) * (2 * VFD + 1) + (sensation[1] + VFD)
    else:
        index = (2 * VFD + 1) ** 2
    return index


def smart_move(were_here, pos, pos_prime):
    if len(np.argwhere(were_here)) > 0:
        for loc in np.argwhere(were_here):
            if np.sqrt((loc[0] - pos[0]) ** 2 + (loc[1] - pos[1]) ** 2) == 1:
                pos_prime = loc
                break
            else:
                continue
    return pos_prime


def rl_agent(beta=0.8):
    global default_sensation
    global can_see_it
    Q = np.zeros(((2 * Hunter_VFD + 1) ** 2 + 1, nA))

    steps = []
    rewards = []
    see_steps = []
    see_rewards = []

    for eps in range(NUM_EPISODES):
        can_see_it = False
        hunter_pos = [0, 9]  # [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]
        prey_pos = [9, 0]  # [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]

        T_hunter = []
        T_prey = []

        R = []
        R_prime = []

        t_step = 0
        see_t_step = 0
        default_sensation = [np.nan, np.nan]
        wereHere_hunter = np.ones((Row_num, Col_num))
        while True:
            t_step += 1

            T_hunter.append(hunter_pos)
            T_prey.append(prey_pos)
            wereHere_hunter[int(hunter_pos[0]), int(hunter_pos[1])] = 0

            hunter_sensation_step1 = np.subtract(prey_pos, hunter_pos)
            hunter_sensation = update_sensation(hunter_sensation_step1)

            idx = sensation2index(hunter_sensation, Hunter_VFD)
            hunter_probs = Boltzmann(Q[idx, :])

            hunter_action = np.random.choice(ACTIONS, p=hunter_probs)
            hunter_pos_prime = movement(hunter_pos, hunter_action)

            if idx == (2 * Hunter_VFD + 1) ** 2:
                hunter_pos_prime = smart_move(wereHere_hunter, hunter_pos, hunter_pos_prime)

            hunter_sensation_prime_step1 = np.subtract(prey_pos, hunter_pos_prime)
            hunter_sensation_prime = update_sensation(hunter_sensation_prime_step1)

            re = reward(hunter_sensation_prime)
            R.append(re)

            if can_see_it:
                R_prime.append(re)
                see_t_step += 1

            idx_prime = sensation2index(hunter_sensation_prime, Hunter_VFD)
            Q[idx, hunter_action] += beta * (re + gamma * np.max(Q[idx_prime, :]) - Q[idx, hunter_action])

            if hunter_sensation == [0, 0] or hunter_sensation_prime == [0, 0]:
                steps.append(t_step+1)
                see_steps.append(see_t_step+1)

                rewards.append(sum(R))
                see_rewards.append(sum(R_prime))

                print(f'In episode {eps + 1} of {NUM_EPISODES}, the prey was captured in {t_step + 1} steps')

                break

            prey_action = np.random.choice(ACTIONS)
            prey_pos_prime = movement(prey_pos, prey_action)

            hunter_pos = hunter_pos_prime
            prey_pos = prey_pos_prime

    return T_hunter, T_prey, rewards, steps, see_rewards, see_steps, Q


T_hunter, T_prey, rewards, steps, see_rewards, see_steps, Q = rl_agent(beta=0.8)

with h5py.File(f'Tan1993_case1_no_scout.hdf5', "w") as f:
    f.create_dataset('T_hunter', data=T_hunter)
    f.create_dataset('T_prey', data=T_prey)

    f.create_dataset('rewards', data=rewards)
    f.create_dataset('steps', data=steps)
    f.create_dataset('see_rewards', data=see_rewards)
    f.create_dataset('see_steps', data=see_steps)

    f.create_dataset('Q', data=Q)
