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
Scouts_VFD = [2, 2, 2]  # Visual field depth for each scout
num_Scouts = len(Scouts_VFD)
gamma = .9

Row_num = 30
Col_num = 30
row_lim = Row_num - 1
column_lim = Col_num - 1

can_see_it = False
can_see_it_scout = np.full((num_Scouts, ), False)
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


def update_sensation_scout(scout_sensation, scout_num):
    global can_see_it_scout
    global default_sensation
    if abs(scout_sensation[0]) <= Scouts_VFD[scout_num] and abs(scout_sensation[1]) <= Scouts_VFD[scout_num]:
        row = scout_sensation[0]
        column = scout_sensation[1]
        can_see_it_scout[scout_num] = True

    else:  # if there is no prey in sight, a unique default sensation is used.
        row, column = [np.nan, np.nan]
        can_see_it_scout[scout_num] = False

    scout_sensation_prime = [row, column]

    return scout_sensation_prime


def update_sensation(hunter_sensation, scouts_sensation, scouts2hunter):
    global can_see_it
    global default_sensation
    global NoOne
    NoOne = True
    if abs(hunter_sensation[0]) <= Hunter_VFD and abs(hunter_sensation[1]) <= Hunter_VFD:
        row = hunter_sensation[0]
        column = hunter_sensation[1]
        can_see_it = True
        default_sensation = [row, column]
    for i in range(num_Scouts):
        if abs(scouts_sensation[i, 0]) <= Scouts_VFD[i] and abs(scouts_sensation[i, 1]) <= Scouts_VFD[i] and not can_see_it:
            row = scouts2hunter[i, 0] + scouts_sensation[i, 0]
            column = scouts2hunter[i, 1] + scouts_sensation[i, 1]
            can_see_it = True
            default_sensation = [row, column]
            NoOne = False
    if NoOne:  # if there is no prey in sight, a unique default sensation is used.
        row, column = default_sensation
        can_see_it = False

    hunter_sensation_prime = [row, column]

    return hunter_sensation_prime


def reward(hunter_sensation_prime):
    if hunter_sensation_prime[0] == 0 and hunter_sensation_prime[1] == 0:
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
    global can_see_it
    global default_sensation
    global can_see_it_scout
    Q = np.zeros(((2 * Row_num + 1) ** 2 + 1, nA))
    Q_scouts = []
    for i in range(num_Scouts):
        Q_scouts.append(np.zeros(((2 * Scouts_VFD[i] + 1) ** 2 + 1, nA)))

    steps = []
    rewards = []
    see_steps = []
    see_rewards = []

    for eps in range(NUM_EPISODES):
        can_see_it = False
        can_see_it_scout = np.full((num_Scouts, ), False)
        default_sensation = [np.nan, np.nan]

        hunter_pos = [0, 0]  # [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]
        scouts_pos = np.array([[0, column_lim],
                               [row_lim, 0],
                               [row_lim, column_lim]])  # np.tile([row_lim, column_lim], (num_Scouts, 1))
        scouts_pos_prime = scouts_pos.copy()
        prey_pos = [Row_num/2, Col_num/2]  # [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]

        T_hunter = []
        T_scouts = []
        scout2hunter = np.zeros((num_Scouts, 2))
        scouts_sensation_step1 = np.zeros((num_Scouts, 2))
        scouts_sensation = np.zeros((num_Scouts, 2))

        for i in range(num_Scouts):
            T_scouts.append([])
        T_prey = []

        R = []
        R_prime = []

        A_hunter = []
        A_scout = []
        A_prey = []

        t_step = 0
        see_t_step = 0
        while True:
            t_step += 1

            T_hunter.append(hunter_pos)
            T_prey.append(prey_pos)
            for i in range(num_Scouts):
                T_scouts[i].append(scouts_pos[i, :])
                scout2hunter[i, :] = np.subtract(scouts_pos[i, :], hunter_pos)
                scouts_sensation_step1[i, :] = np.subtract(prey_pos, scouts_pos[i, :])
                scouts_sensation[i, :] = update_sensation_scout(scouts_sensation_step1[i, :], i)

            hunter_sensation_step1 = np.subtract(prey_pos, hunter_pos)

            hunter_sensation = update_sensation(hunter_sensation_step1, scouts_sensation, scout2hunter)

            idx = sensation2index(hunter_sensation, Row_num, can_see_it)
            T_check = (eps+1)/NUM_EPISODES
            if T_check < 0.25:
                T = 0.4
            elif 0.25 < T_check < 0.5:
                T = 0.3
            elif 0.5 < T_check < 0.75:
                T = 0.2
            else:
                T= 0.1
            hunter_probs = Boltzmann(Q[idx, :], t=T)
            hunter_action = np.random.choice(ACTIONS, p=hunter_probs)

            idx_scout = np.zeros((num_Scouts, ), dtype=int)
            scout_probs = np.zeros((num_Scouts, nA))
            scout_action = np.empty((num_Scouts, ))
            for i in range(num_Scouts):
                idx_scout[i] = sensation2index(scouts_sensation[i, :], Scouts_VFD[i], can_see_it_scout[i])
                scout_probs[i, :] = Boltzmann(Q_scouts[i][idx_scout[i], :])
                scout_action[i] = np.random.choice(ACTIONS, p=scout_probs[i])
                scouts_pos_prime[i, :] = movement(scouts_pos[i, :], scout_action[i])

            prey_action = np.random.choice(ACTIONS)

            hunter_pos_prime = movement(hunter_pos, hunter_action)
            prey_pos_prime = movement(prey_pos, prey_action)

            scout2hunter_prime = np.zeros((num_Scouts, 2))
            scout_sensation_prime_step1 = np.zeros((num_Scouts, 2))
            scout_sensation_prime = np.zeros((num_Scouts, 2))
            for i in range(num_Scouts):
                scout2hunter_prime[i, :] = np.subtract(scouts_pos_prime[i, :], hunter_pos_prime)
                scout_sensation_prime_step1[i, :] = np.subtract(prey_pos, scouts_pos_prime[i, :])
                scout_sensation_prime[i, :] = update_sensation_scout(scout_sensation_prime_step1[i, :], i)

            hunter_sensation_prime_step1 = np.subtract(prey_pos, hunter_pos_prime)
            hunter_sensation_prime = update_sensation(hunter_sensation_prime_step1, scout_sensation_prime, scout2hunter_prime)
            re = reward(hunter_sensation)
            re_scout = np.zeros((num_Scouts,))
            for i in range(num_Scouts):
                re_scout[i] = reward(scouts_sensation[i, :])
            R.append(re)

            A_hunter.append(hunter_action)
            A_scout.append(scout_action)
            A_prey.append(prey_action)

            if can_see_it:
                R_prime.append(re)
                see_t_step += 1

            idx_prime = sensation2index(hunter_sensation_prime, Row_num, can_see_it)
            Q[idx, hunter_action] += beta * (re + gamma * np.max(Q[idx_prime, :]) - Q[idx, hunter_action])

            idx_scouts_prime = np.zeros((num_Scouts,), dtype=int)
            for i in range(num_Scouts):
                idx_scouts_prime[i] = sensation2index(scout_sensation_prime[i], Scouts_VFD[i], can_see_it_scout[i])
                Q_scouts[i][idx_scout[i], int(scout_action[i])] += beta * (re_scout[i] +
                                                                        gamma * np.max(Q_scouts[i][idx_scouts_prime[i], :]) -
                                                                        Q_scouts[i][idx_scout[i], int(scout_action[i])])


            hunter_pos = hunter_pos_prime
            prey_pos = prey_pos_prime
            scouts_pos = scouts_pos_prime.copy()
            if hunter_sensation == [0, 0]:

                steps.append(t_step+1)
                see_steps.append(see_t_step+1)

                rewards.append(sum(R))
                see_rewards.append(sum(R_prime))

                print(f'In episode {eps + 1} of {NUM_EPISODES}, the prey was captured in {t_step + 1} steps')

                break

    return T_hunter, T_scouts, T_prey, A_hunter, A_scout, A_prey, rewards, steps, see_rewards, see_steps, Q


T_hunter, T_scouts, T_prey, A_hunter, A_scout, A_prey, rewards, steps, see_rewards, see_steps, Q = rl_agent(beta=0.8)

with h5py.File(f'Tan1993_case1_with_learning_scout.hdf5', "w") as f:
    f.create_dataset('T_hunter', data=T_hunter)
    f.create_dataset('T_scouts', data=T_scouts)
    f.create_dataset('T_prey', data=T_prey)

    f.create_dataset('A_hunter', data=A_hunter)
    f.create_dataset('A_scout', data=A_scout)
    f.create_dataset('A_prey', data=A_prey)

    f.create_dataset('rewards', data=rewards)
    f.create_dataset('steps', data=steps)
    f.create_dataset('see_rewards', data=see_rewards)
    f.create_dataset('see_steps', data=see_steps)

    f.create_dataset('Q', data=Q)
