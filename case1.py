import numpy as np
import h5py

NUM_EPISODES = 10000
Hunter_VFD = 2  # Hunter's visual field depth
Scout_VFD = 2  # Scout's visual field depth

# Actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
ACTIONS = [FORWARD, BACKWARD, RIGHT, LEFT]
nA = len(ACTIONS)
gamma = .9
Row_num = 10
Col_num = 10


def Boltzmann(q, t=0.4):
    return np.exp(q / t) / np.sum(np.exp(q / t))


def movement(position, action):
    row = position[0]
    column = position[1]

    row_lim = 9
    column_lim = 9
    if action == 0:  # up
        next_position = [max(row - 1, 0), column]
    elif action == 1:  # down
        next_position = [min(abs(row + 1), row_lim), column]
    elif action == 2:  # right
        next_position = [row, min(column + 1, column_lim)]
    elif action == 3:  # left
        next_position = [row, max(column - 1, 0)]

    return next_position


def transition(hunter_sensation, hunter_action, scout_sensation, scout_action, scout2hunter):
    row_lim = 9
    column_lim = 9
    scout_row = scout2hunter[0]
    scout_column = scout2hunter[1]

    if scout_action == 0:  # up
        scout2hunter_prime = [max(scout_row - 1, 0), scout_column]
    elif scout_action == 1:  # down
        scout2hunter_prime = [min(abs(scout_row + 1), row_lim), scout_column]
    elif scout_action == 2:  # right
        scout2hunter_prime = [scout_row, min(scout_column + 1, column_lim)]
    elif scout_action == 3:  # left
        scout2hunter_prime = [scout_row, max(scout_column - 1, 0)]

    global Hunter_VFD
    global Scout_VFD
    if abs(hunter_sensation[0]) < Hunter_VFD and abs(hunter_sensation[1]) < Hunter_VFD:
        row = hunter_sensation[0]
        column = hunter_sensation[1]
    elif abs(scout_sensation[0]) < Scout_VFD and abs(scout_sensation[1]) < Scout_VFD:
        row = min(scout2hunter_prime[0] + scout_sensation[0], row_lim)
        column = min(scout2hunter_prime[1] + scout_sensation[1], column_lim)
    else:  # if there is no prey in sight, a unique default sensation is used.
        row = row_lim
        column = column_lim

    if hunter_action == 0:  # up
        hunter_sensation_prime = [max(row - 1, 0), column]
    elif hunter_action == 1:  # down
        hunter_sensation_prime = [min(abs(row + 1), row_lim), column]
    elif hunter_action == 2:  # right
        hunter_sensation_prime = [row, min(column + 1, column_lim)]
    elif hunter_action == 3:  # left
        hunter_sensation_prime = [row, max(column - 1, 0)]

    return hunter_sensation_prime


def reward(hunter_sensation_prime):
    if hunter_sensation_prime == [0, 0]:
        re = 1
    else:
        re = -.1
    return re


def rl_agent(beta=0.8):
    Q = np.zeros((Row_num, Col_num, nA))

    steps = []
    rewards = []

    for eps in range(NUM_EPISODES):
        hunter_pose = [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]
        scout_pose = [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]
        prey_pose = [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]

        T_hunter = []
        T_scout = []
        T_prey = []

        R = []

        A_hunter = []
        A_scout = []
        A_prey = []

        t_step = 0
        while True:
            t_step += 1

            T_hunter.append(hunter_pose)
            T_scout.append(scout_pose)
            T_prey.append(prey_pose)

            scout2hunter = np.subtract(scout_pose, hunter_pose)
            hunter_sensation = np.subtract(prey_pose, hunter_pose)
            scout_sensation = np.subtract(prey_pose, scout_pose)

            hunter_probs = Boltzmann(Q[hunter_sensation[0], hunter_sensation[1], :])
            hunter_action = np.random.choice(ACTIONS, p=hunter_probs)
            scout_action = np.random.choice(ACTIONS)
            prey_action = np.random.choice(ACTIONS)

            hunter_sensation_prime = transition(hunter_sensation, hunter_action,
                                                scout_sensation, scout_action, scout2hunter)

            re = reward(hunter_sensation_prime)

            A_hunter.append(hunter_action)
            A_scout.append(scout_action)
            A_prey.append(prey_action)

            R.append(re)
            Q[int(hunter_sensation[0]),
              int(hunter_sensation[1]), hunter_action] += beta * (re +
                                                 gamma * np.max(Q[int(hunter_sensation_prime[0]),
                                                                  int(hunter_sensation_prime[1]), :]) -
                                                 Q[int(hunter_sensation[0]), int(hunter_sensation[1]), hunter_action])

            hunter_pose = movement(hunter_pose, hunter_action)
            scout_pose = movement(scout_pose, scout_action)
            prey_pose = movement(prey_pose, prey_action)

            if hunter_sensation_prime == [0, 0]:
                # Q[hunter_sensation_prime[0], hunter_sensation_prime[1], :] = 0
                steps.append(t_step)
                rewards.append(sum(R))
                print(f'In episode {eps + 1} of {NUM_EPISODES}, the prey was captured in {t_step + 1} steps')
                break

    return T_hunter, T_scout, T_prey, A_hunter, A_scout, A_prey, rewards, steps, Q


T_hunter, T_scout, T_prey, A_hunter, A_scout, A_prey, rewards, steps, Q = rl_agent(beta=0.8)


with h5py.File(f'Tan1993_case1.hdf5', "w") as f:

    f.create_dataset('T_hunter', data=T_hunter)
    f.create_dataset('T_scout', data=T_scout)
    f.create_dataset('T_prey', data=T_prey)

    f.create_dataset('A_hunter', data=A_hunter)
    f.create_dataset('A_scout', data=A_scout)
    f.create_dataset('A_prey', data=A_prey)

    f.create_dataset('rewards', data=rewards)
    f.create_dataset('steps', data=steps)

    f.create_dataset('Q', data=Q)
