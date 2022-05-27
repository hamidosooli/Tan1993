import numpy as np
import h5py

# Actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
ACTIONS = [FORWARD, BACKWARD, RIGHT, LEFT]
num_Acts = len(ACTIONS)

NUM_EPISODES = 500

rescuers_VFD = [2, 2, 2, 3]  # Hunter's visual field depth
scouts_VFD = [2, 3, 4]  # Visual field depth for each scout

num_rescuers = len(rescuers_VFD)
num_scouts = len(scouts_VFD)
adj_mat = np.zeros((num_rescuers, num_scouts))
adj_mat[0, :1] = 1
adj_mat[1, 0] = 1
adj_mat[2, 2] = 1
adj_mat[-1, :] = 1
#                                     #     _  S1  S2  S3 _
# adj_mat = np.array([[1, 1, 0],      # H1 |   Y   Y   N   |
#                     [1, 0, 0],      # H2 |   Y   N   N   |
#                     [0, 0, 1],      # H3 |   N   N   Y   |
#                     [1, 1, 1]])     # H4 |_  Y   Y   Y  _|

adj_mat_loc = np.zeros((adj_mat.shape[0], adj_mat.shape[1], 2))
adj_mat_loc[:, :, 0] = adj_mat_loc[:, :, 1] = adj_mat

num_scouts4rescuers = adj_mat.sum(axis=1)

idx_scouts4rescuers = np.where(adj_mat == 1)[1].tolist()
idx_scouts4rescuers_organized = []
begin = 0
for k in range(num_rescuers):
    idx_scouts4rescuers_organized.append([])
    for num in range(int(begin), int(begin+num_scouts4rescuers[k])):
        idx_scouts4rescuers_organized[k].append(idx_scouts4rescuers[num])
    begin += num_scouts4rescuers[k]
gamma = .9

Row_num = 20
Col_num = 20
row_lim = Row_num - 1
col_lim = Col_num - 1

can_see_it_rescuers = np.full((num_rescuers, ), False)
can_see_it_scout = np.full((num_scouts, ), False)
default_sensation = np.tile([np.nan, np.nan], (num_rescuers, 1))


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
        next_position = [row, min(column + 1, col_lim)]
    elif action == 3:  # left
        next_position = [row, max(column - 1, 0)]

    return next_position


def update_sensation_scout(scouts_sensation, scouts_num):
    global can_see_it_scout
    global default_sensation
    if abs(scouts_sensation[0]) <= scouts_VFD[scouts_num] and abs(scouts_sensation[1]) <= scouts_VFD[scouts_num]:
        row = scouts_sensation[0]
        column = scouts_sensation[1]
        can_see_it_scout[scouts_num] = True

    else:  # if there is no prey in sight, a unique default sensation is used.
        row, column = [np.nan, np.nan]
        can_see_it_scout[scouts_num] = False

    scouts_sensation_prime = [row, column]

    return scouts_sensation_prime


def update_sensation_rescuer(rescuer_sensation, scouts_sensation, scouts2rescuer, rescuer_num):
    global can_see_it_rescuers
    can_see_it_rescuers[rescuer_num] = False
    global default_sensation
    global num_scouts4rescuers
    if abs(rescuer_sensation[0]) <= rescuers_VFD[rescuer_num] and abs(rescuer_sensation[1]) <= rescuers_VFD[rescuer_num]:
        row = rescuer_sensation[0]
        column = rescuer_sensation[1]
        can_see_it_rescuers[rescuer_num] = True
        default_sensation[rescuer_num] = [row, column]
    else:
        for i in range(int(num_scouts4rescuers[rescuer_num])):
            idx_scout = idx_scouts4rescuers_organized[rescuer_num][i]
            if abs(scouts_sensation[idx_scout, 0]) <= scouts_VFD[idx_scout] and abs(scouts_sensation[idx_scout, 1]) <= scouts_VFD[idx_scout]:
                row = scouts2rescuer[idx_scout, 0] + scouts_sensation[idx_scout, 0]
                column = scouts2rescuer[idx_scout, 1] + scouts_sensation[idx_scout, 1]
                can_see_it_rescuers[rescuer_num] = True
                default_sensation[rescuer_num] = [row, column]
    if not can_see_it_rescuers[rescuer_num]:  # if there is no prey in sight, a unique default sensation is used.
        row, column = default_sensation[rescuer_num]
    rescuer_sensation_prime = [row, column]

    return rescuer_sensation_prime


def reward(rescuer_sensation_prime):
    if rescuer_sensation_prime[0] == 0 and rescuer_sensation_prime[1] == 0:
        re = 1
    else:
        re = -.1
    return re


def sensation2index(sensation, VFD, can_see_it_local):
    if can_see_it_local:
        index = (sensation[0] + VFD) * (2 * VFD + 1) + (sensation[1] + VFD)
    else:
        index = (2 * VFD + 1) ** 2
    return int(index)


def rl_agent(beta=0.8):
    global can_see_it_rescuers
    global default_sensation
    global can_see_it_scout

    steps = []
    rewards = []
    see_steps = []
    see_rewards = []

    Q_rescuers = []
    for k in range(num_rescuers):
        Q_rescuers.append(np.zeros(((2 * Row_num + 1) ** 2 + 1, num_Acts)))
        steps.append([])
        rewards.append([])
        see_steps.append([])
        see_rewards.append([])

    Q_scouts = []
    for i in range(num_scouts):
        Q_scouts.append(np.zeros(((2 * scouts_VFD[i] + 1) ** 2 + 1, num_Acts)))

    for eps in range(NUM_EPISODES):
        can_see_it_rescuers = np.full((num_rescuers, ), False)
        can_see_it_scout = np.full((num_scouts, ), False)
        default_sensation = np.tile([np.nan, np.nan], (num_rescuers, 1))

        rescuers_pos = np.tile([row_lim, col_lim], (num_rescuers, 1))
        rescuers_pos_prime = rescuers_pos.copy()

        scouts_pos = np.array([[0, col_lim],
                               [row_lim, 0],
                               [row_lim, col_lim]])  # np.tile([row_lim, col_lim], (num_scouts, 1))
        scouts_pos_prime = scouts_pos.copy()

        prey_pos = [Row_num/2, Col_num/2]  # [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]

        T_rescuers = []
        R_rescuers = []
        for k in range(num_rescuers):
            T_rescuers.append([])
            R_rescuers.append([])

        T_scouts = []
        R_scouts = []
        for i in range(num_scouts):
            T_scouts.append([])
            R_scouts.append([])

        rescuers_sensation_step1 = np.zeros((num_rescuers, 2))
        rescuers_sensation = np.zeros((num_rescuers, 2))

        scouts_sensation_step1 = np.zeros((num_scouts, 2))
        scouts_sensation = np.zeros((num_scouts, 2))

        T_prey = []

        t_step = 0
        see_t_step = np.zeros((num_scouts,))
        while True:
            t_step += 1

            T_prey.append(prey_pos)

            scouts_pos_step1 = np.empty((num_rescuers, num_scouts, 2))
            scouts_pos_step1[:, :, 0] = np.tile(scouts_pos[:, 0], (num_rescuers, 1))
            scouts_pos_step1[:, :, 1] = np.tile(scouts_pos[:, 1], (num_rescuers, 1))

            rescuers_pos_step1 = np.empty((num_rescuers, num_scouts, 2))
            rescuers_pos_step1[:, :, 0] = np.tile(rescuers_pos[:, 0].reshape(num_rescuers, 1), (1, num_scouts))
            rescuers_pos_step1[:, :, 1] = np.tile(rescuers_pos[:, 1].reshape(num_rescuers, 1), (1, num_scouts))

            scouts2rescuers = np.subtract(scouts_pos_step1, rescuers_pos_step1)

            for i in range(num_scouts):
                T_scouts[i].append(scouts_pos[i, :])
                scouts_sensation_step1[i, :] = np.subtract(prey_pos, scouts_pos[i, :])
                scouts_sensation[i, :] = update_sensation_scout(scouts_sensation_step1[i, :], i)

            for k in range(num_rescuers):
                T_rescuers[k].append(rescuers_pos[k, :])
                rescuers_sensation_step1[k, :] = np.subtract(prey_pos, rescuers_pos[k, :])
                rescuers_sensation[k, :] = update_sensation_rescuer(rescuers_sensation_step1[k, :],
                                                                    scouts_sensation,
                                                                    scouts2rescuers[k, :], k)

            T_check = (eps+1) / NUM_EPISODES
            if T_check < 0.25:
                T = 0.4
            elif 0.25 < T_check < 0.5:
                T = 0.3
            elif 0.5 < T_check < 0.75:
                T = 0.2
            else:
                T = 0.1

            idx_rescuers = np.zeros((num_rescuers,), dtype=int)
            rescuer_probs = np.zeros((num_rescuers, num_Acts))
            rescuer_action = np.empty((num_rescuers, ))
            for k in range(num_rescuers):
                idx_rescuers[k] = sensation2index(rescuers_sensation[k, :], rescuers_VFD[k], can_see_it_rescuers[k])
                rescuer_probs[k, :] = Boltzmann(Q_rescuers[k][idx_rescuers[k], :], t=T)
                rescuer_action[k] = np.random.choice(ACTIONS, p=rescuer_probs[k])
                rescuers_pos_prime[k, :] = movement(rescuers_pos[k, :], rescuer_action[k])

            idx_scout = np.zeros((num_scouts, ), dtype=int)
            scouts_probs = np.zeros((num_scouts, num_Acts))
            scouts_action = np.empty((num_scouts, ))
            for i in range(num_scouts):
                idx_scout[i] = sensation2index(scouts_sensation[i, :], scouts_VFD[i], can_see_it_scout[i])
                scouts_probs[i, :] = Boltzmann(Q_scouts[i][idx_scout[i], :], t=T)
                scouts_action[i] = np.random.choice(ACTIONS, p=scouts_probs[i])
                scouts_pos_prime[i, :] = movement(scouts_pos[i, :], scouts_action[i])

            prey_action = np.random.choice(ACTIONS)
            prey_pos_prime = movement(prey_pos, prey_action)

            scouts_sensation_prime_step1 = np.zeros((num_scouts, 2))
            scouts_sensation_prime = np.zeros((num_scouts, 2))

            scouts_pos_prime_step1 = np.empty((num_rescuers, num_scouts, 2))
            scouts_pos_prime_step1[:, :, 0] = np.tile(scouts_pos_prime[:, 0],
                                                      (num_rescuers, 1))
            scouts_pos_prime_step1[:, :, 1] = np.tile(scouts_pos_prime[:, 1],
                                                      (num_rescuers, 1))

            rescuers_pos_prime_step1 = np.empty((num_rescuers, num_scouts, 2))
            rescuers_pos_prime_step1[:, :, 0] = np.tile(rescuers_pos_prime[:, 0].reshape(num_rescuers, 1),
                                                        (1, num_scouts))
            rescuers_pos_prime_step1[:, :, 1] = np.tile(rescuers_pos_prime[:, 1].reshape(num_rescuers, 1),
                                                        (1, num_scouts))

            scouts2rescuers_prime = np.subtract(scouts_pos_prime_step1, rescuers_pos_prime_step1)

            for i in range(num_scouts):
                scouts_sensation_prime_step1[i, :] = np.subtract(prey_pos, scouts_pos_prime[i, :])
                scouts_sensation_prime[i, :] = update_sensation_scout(scouts_sensation_prime_step1[i, :], i)

            rescuer_sensation_prime_step1 = np.zeros((num_rescuers, 2))
            rescuer_sensation_prime = np.zeros((num_rescuers, 2))
            for k in range(num_rescuers):
                rescuer_sensation_prime_step1[k, :] = np.subtract(prey_pos, rescuers_pos_prime[k, :])
                rescuer_sensation_prime[k, :] = update_sensation_rescuer(rescuer_sensation_prime_step1[k, :],
                                                                         scouts_sensation_prime,
                                                                         scouts2rescuers_prime[k, :], k)

            re_rescuers = np.zeros((num_rescuers,))
            for k in range(num_rescuers):
                re_rescuers[k] = reward(rescuers_sensation[k, :])
                
            re_scout = np.zeros((num_scouts,))
            for i in range(num_scouts):
                re_scout[i] = reward(scouts_sensation[i, :])
                
            for k in range(num_rescuers):
                R_rescuers[k].append(re_rescuers[k])

            for k in range(num_rescuers):
                for i in range(num_scouts):
                    if can_see_it_rescuers[k]:
                        R_scouts[i].append(re_scout[i])
                        see_t_step[i] += 1

            idx_rescuers_prime = np.zeros((num_rescuers,), dtype=int)
            for k in range(num_rescuers):
                idx_rescuers_prime[k] = sensation2index(rescuer_sensation_prime[k], rescuers_VFD[k],
                                                        can_see_it_rescuers[k])
                Q_rescuers[k][idx_rescuers[k],
                              int(rescuer_action[k])] += beta * (re_rescuers[k] +
                                                                 gamma * np.max(Q_rescuers[k][idx_rescuers_prime[k], :])
                                                                 - Q_rescuers[k][idx_rescuers[k],
                                                                                 int(rescuer_action[k])])

            idx_scouts_prime = np.zeros((num_scouts,), dtype=int)
            for i in range(num_scouts):
                idx_scouts_prime[i] = sensation2index(scouts_sensation_prime[i], scouts_VFD[i], can_see_it_scout[i])
                Q_scouts[i][idx_scout[i], 
                            int(scouts_action[i])] += beta * (re_scout[i] +
                                                              gamma * np.max(Q_scouts[i][idx_scouts_prime[i], :]) -
                                                              Q_scouts[i][idx_scout[i], int(scouts_action[i])])

            rescuers_pos = rescuers_pos_prime.copy()
            prey_pos = prey_pos_prime
            scouts_pos = scouts_pos_prime.copy()
            if np.array_equal(rescuers_sensation, np.tile([0, 0], (num_rescuers, 1))):

                steps.append(t_step+1)

                for i in range(num_scouts):
                    see_steps[i].append(see_t_step+1)
                    see_rewards[i].append(sum(R_scouts[i]))

                for k in range(num_rescuers):
                    rewards.append(sum(R_rescuers[k]))

                print(f'In episode {eps + 1} of {NUM_EPISODES}, the prey was captured in {t_step + 1} steps')

                break

    return T_rescuers, T_scouts, T_prey, rewards, steps, see_rewards, see_steps, Q


T_rescuers, T_scouts, T_prey, rewards, steps, see_rewards, see_steps, Q = rl_agent(beta=0.8)

with h5py.File(f'Tan1993_multi_rescuers_with_multi_learning_scout.hdf5', "w") as f:
    f.create_dataset('T_rescuers', data=T_rescuers)
    f.create_dataset('T_scouts', data=T_scouts)
    f.create_dataset('T_prey', data=T_prey)

    f.create_dataset('rewards', data=rewards)
    f.create_dataset('steps', data=steps)
    f.create_dataset('see_rewards', data=see_rewards)
    f.create_dataset('see_steps', data=see_steps)

    f.create_dataset('Q', data=Q)
