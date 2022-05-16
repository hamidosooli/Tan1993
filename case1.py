import numpy as np
import h5py


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
NUM_RUNS = 100
NUM_EPISODES = 5000
Hunter_VFD = 2  # Hunter's visual field depth
Scout_VFD = 2  # Scout's visual field depth
max_step = 1000
# Actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
NORTHEAST = 4
NORTHWEST = 5
SOUTHEAST = 6
SOUTHWEST = 7
ACTIONS = [FORWARD, BACKWARD, RIGHT, LEFT, NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST]
nA = len(ACTIONS)
gamma = .9
Row_num = 10
Col_num = 10
row_lim = 9
column_lim = 9

default_sensation = [row_lim, column_lim]


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
    elif action == 4:  # northeast
        next_position = [max(row - 1, 0), min(column + 1, column_lim)]
    elif action == 5:  # northwest
        next_position = [max(row - 1, 0), max(column - 1, 0)]
    elif action == 6:  # southeast
        next_position = [min(row + 1, row_lim), min(column + 1, column_lim)]
    elif action == 7:  # southwest
        next_position = [min(row + 1, row_lim), max(column - 1, 0)]

    return next_position


def transition(hunter_sensation, scout_sensation, scout2hunter):
    global default_sensation
    if abs(hunter_sensation[0]) <= Hunter_VFD and abs(hunter_sensation[1]) <= Hunter_VFD:
        row = hunter_sensation[0]
        column = hunter_sensation[1]
        default_sensation = [row, column]
    elif abs(scout_sensation[0]) <= Scout_VFD and abs(scout_sensation[1]) <= Scout_VFD:
        row = scout2hunter[0] + scout_sensation[0]
        column = scout2hunter[1] + scout_sensation[1]
        default_sensation = [row, column]
    else:  # if there is no prey in sight, a unique default sensation is used.
        row, column = default_sensation

    hunter_sensation_prime = [row, column]

    return hunter_sensation_prime


def reward(hunter_sensation_prime):
    if hunter_sensation_prime == [0, 0]:
        re = 1
    else:
        re = -.1
    return re


def rl_agent(beta=0.8):
    global default_sensation
    Q = np.zeros((2*Row_num-1, 2*Col_num-1, nA))

    steps = []
    rewards = []
    see_steps = []
    see_rewards = []

    for eps in range(NUM_EPISODES):
        hunter_pos = [9, 0]  # [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]
        scout_pos = [9, 9]  # [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]
        prey_pos = [5, 9]  # [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]

        T_hunter = []
        T_scout = []
        T_prey = []

        R = []
        R_prime = []

        A_hunter = []
        A_scout = []
        A_prey = []

        counter = 0
        t_step = 0
        default_sensation = [row_lim, column_lim]
        while True:
            t_step += 1

            T_hunter.append(hunter_pos)
            T_scout.append(scout_pos)
            T_prey.append(prey_pos)

            scout2hunter = np.subtract(scout_pos, hunter_pos)
            scout_sensation = np.subtract(prey_pos, scout_pos)
            hunter_sensation_step1 = np.subtract(prey_pos, hunter_pos)
            hunter_sensation = transition(hunter_sensation_step1, scout_sensation, scout2hunter)

            hunter_probs = Boltzmann(Q[row_lim-hunter_sensation[0], column_lim-hunter_sensation[1], :])
            hunter_action = np.random.choice(ACTIONS, p=hunter_probs)
            scout_action = np.random.choice(ACTIONS)
            prey_action = np.random.choice(ACTIONS)

            hunter_pos_prime = movement(hunter_pos, hunter_action)
            scout_pos_prime = movement(scout_pos, scout_action)
            prey_pos_prime = movement(prey_pos, prey_action)

            scout2hunter_prime = np.subtract(scout_pos_prime, hunter_pos_prime)
            scout_sensation_prime = np.subtract(prey_pos_prime, scout_pos_prime)
            hunter_sensation_prime_step1 = np.subtract(prey_pos_prime, hunter_pos_prime)
            hunter_sensation_prime = transition(hunter_sensation_prime_step1, scout_sensation_prime, scout2hunter_prime)

            re = reward(hunter_sensation_prime)

            A_hunter.append(hunter_action)
            A_scout.append(scout_action)
            A_prey.append(prey_action)
            R.append(re)
            if default_sensation != [row_lim, column_lim]:
                R_prime.append(re)
                counter += 1
            if counter == 1:
                s = t_step
            Q[row_lim-hunter_sensation[0],
              column_lim-hunter_sensation[1], hunter_action] += beta * (re +
                                                 gamma * np.max(Q[row_lim-hunter_sensation_prime[0],
                                                                  column_lim-hunter_sensation_prime[1], :]) -
                                                 Q[row_lim-hunter_sensation[0],
                                                   column_lim-hunter_sensation[1], hunter_action])
            hunter_pos = hunter_pos_prime
            scout_pos = scout_pos_prime
            prey_pos = prey_pos_prime
            if hunter_sensation_prime == [0, 0]:
                T_hunter.append(hunter_pos)
                T_scout.append(scout_pos)
                T_prey.append(prey_pos)
                steps.append(t_step)
                see_steps.append(s)
                rewards.append(sum(R))
                see_rewards.append(sum(R_prime))
                print(f'In episode {eps + 1} of {NUM_EPISODES}, the prey was captured in {t_step + 1} steps')
                break

    return T_hunter, T_scout, T_prey, A_hunter, A_scout, A_prey, rewards, steps, see_rewards, see_steps, Q


T_hunter_runs = []
T_scout_runs = []
T_prey_runs = []
A_hunter_runs = []
A_scout_runs = []
A_prey_runs = []
rewards_runs = []
steps_runs = []
see_rewards_runs = []
see_steps_runs = []
Q_runs = []
for run in range(NUM_RUNS):
    print(bcolors.WARNING + 'Run ' + str(run + 1) + ' of ' + str(NUM_RUNS) + bcolors.ENDC)
    T_hunter, T_scout, T_prey, A_hunter, A_scout, A_prey, rewards, steps, see_rewards, see_steps, Q = rl_agent(beta=0.8)

    T_hunter_runs.append(T_hunter)
    T_scout_runs.append(T_scout)
    T_prey_runs.append(T_prey)
    A_hunter_runs.append(A_hunter)
    A_scout_runs.append(A_scout)
    A_prey_runs.append(A_prey)
    rewards_runs.append(rewards)
    steps_runs.append(steps)
    see_rewards_runs.append(see_rewards)
    see_steps_runs.append(see_steps)
    Q_runs.append(Q)


# T_hunter_runs = np.asarray(T_hunter_runs)
# T_scout_runs = np.asarray(T_scout_runs)
# T_prey_runs = np.asarray(T_prey_runs)
# A_hunter_runs = np.asarray(A_hunter_runs)
# A_scout_runs = np.asarray(A_scout_runs)
# A_prey_runs = np.asarray(A_prey_runs)
rewards_runs = np.asarray(rewards_runs)
steps_runs = np.asarray(steps_runs)
see_rewards_runs = np.asarray(see_rewards_runs)
see_steps_runs = np.asarray(see_steps_runs)
# Q_runs = np.asarray(Q_runs)


with h5py.File(f'Tan1993_case1_runs_exp1.hdf5', "w") as f:

    # f.create_dataset('T_hunter', data=T_hunter_runs)
    # f.create_dataset('T_scout', data=T_scout_runs)
    # f.create_dataset('T_prey', data=T_prey_runs)
    #
    # f.create_dataset('A_hunter', data=A_hunter_runs)
    # f.create_dataset('A_scout', data=A_scout_runs)
    # f.create_dataset('A_prey', data=A_prey_runs)

    f.create_dataset('rewards', data=rewards_runs)
    f.create_dataset('steps', data=steps_runs)
    f.create_dataset('see_rewards', data=see_rewards_runs)
    f.create_dataset('see_steps', data=see_steps_runs)

    # f.create_dataset('Q', data=Q_runs)
