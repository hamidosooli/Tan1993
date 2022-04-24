import numpy as np
import h5py


NUM_EPISODES = 10000
Hunter_VFD = 2  # Hunter's visual field depth

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


def transition(hunter_sensation, hunter_action):
    row_lim = 9
    column_lim = 9

    global Hunter_VFD
    if abs(hunter_sensation[0]) < Hunter_VFD and abs(hunter_sensation[1]) < Hunter_VFD:
        row = hunter_sensation[0]
        column = hunter_sensation[1]
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
    Q1 = np.zeros((Row_num, Col_num, nA))
    Q2 = np.zeros((Row_num, Col_num, nA))

    steps = []
    rewards1 = []
    rewards2 = []

    for eps in range(NUM_EPISODES):
        hunter1_pose = [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]
        hunter2_pose = [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]
        prey_pose = [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))]

        T_hunter1 = []
        T_hunter2 = []
        T_prey = []

        R1 = []
        R2 = []

        A_hunter1 = []
        A_hunter2 = []
        A_prey = []

        t_step = 0
        while True:
            t_step += 1

            T_hunter1.append(hunter1_pose)
            T_hunter2.append(hunter2_pose)
            T_prey.append(prey_pose)

            hunter1_sensation = np.subtract(prey_pose, hunter1_pose)
            hunter2_sensation = np.subtract(prey_pose, hunter2_pose)

            hunter1_probs = Boltzmann(Q1[hunter1_sensation[0], hunter1_sensation[1], :])
            hunter1_action = np.random.choice(ACTIONS, p=hunter1_probs)
            hunter2_probs = Boltzmann(Q2[hunter2_sensation[0], hunter2_sensation[1], :])
            hunter2_action = np.random.choice(ACTIONS, p=hunter2_probs)
            prey_action = np.random.choice(ACTIONS)

            hunter1_sensation_prime = transition(hunter1_sensation, hunter1_action)
            hunter2_sensation_prime = transition(hunter2_sensation, hunter2_action)

            re1 = reward(hunter1_sensation_prime)
            re2 = reward(hunter2_sensation_prime)

            A_hunter1.append(hunter1_action)
            A_hunter2.append(hunter2_action)
            A_prey.append(prey_action)

            R1.append(re1)
            R2.append(re2)

            Q1[int(hunter1_sensation[0]),
               int(hunter1_sensation[1]), hunter1_action] += beta * (re1 +
                                                 gamma * np.max(Q1[int(hunter1_sensation_prime[0]),
                                                                   int(hunter1_sensation_prime[1]), :]) -
                                                 Q1[int(hunter1_sensation[0]), int(hunter1_sensation[1]), hunter1_action])

            Q2[int(hunter2_sensation[0]),
               int(hunter2_sensation[1]), hunter2_action] += beta * (re1 +
                                                 gamma * np.max(Q2[int(hunter2_sensation_prime[0]),
                                                                   int(hunter2_sensation_prime[1]), :]) -
                                                 Q2[int(hunter2_sensation[0]), int(hunter2_sensation[1]), hunter2_action])

            hunter1_pose = movement(hunter1_pose, hunter1_action)
            hunter2_pose = movement(hunter2_pose, hunter2_action)
            prey_pose = movement(prey_pose, prey_action)

            if hunter1_sensation_prime == [0, 0] or hunter2_sensation_prime == [0, 0]:
                # Q[hunter_sensation_prime[0], hunter_sensation_prime[1], :] = 0
                steps.append(t_step)
                rewards1.append(sum(R1))
                rewards2.append(sum(R2))
                print(f'In episode {eps + 1} of {NUM_EPISODES}, the prey was captured in {t_step + 1} steps')
                break

    return T_hunter1, T_hunter2, T_prey, A_hunter1, A_hunter2, A_prey, rewards1, rewards2, steps, Q1, Q2


T_hunter1, T_hunter2, T_prey, A_hunter1, A_hunter2, A_prey, rewards1, rewards2, steps, Q1, Q2 = rl_agent(beta=0.8)


with h5py.File(f'Tan1993_case2.hdf5', "w") as f:

    f.create_dataset('T_hunter1', data=T_hunter1)
    f.create_dataset('T_hunter2', data=T_hunter2)
    f.create_dataset('T_prey', data=T_prey)

    f.create_dataset('A_hunter1', data=A_hunter1)
    f.create_dataset('A_hunter2', data=A_hunter2)
    f.create_dataset('A_prey', data=A_prey)

    f.create_dataset('rewards1', data=rewards1)
    f.create_dataset('rewards2', data=rewards2)
    f.create_dataset('steps', data=steps)

    f.create_dataset('Q1', data=Q1)
    f.create_dataset('Q2', data=Q2)
