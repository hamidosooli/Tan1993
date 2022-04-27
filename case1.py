import numpy as np
import h5py

NUM_EPISODES = 10000
Hunter_VFD = 2  # Hunter's visual field depth
Scout_VFD = 4  # Scout's visual field depth

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
row_lim = 9
column_lim = 9

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


def transition(hunter_sensation, scout_sensation, scout2hunter):
    row_lim = 9
    column_lim = 9

    if abs(hunter_sensation[0]) <= Hunter_VFD and abs(hunter_sensation[1]) <= Hunter_VFD:
        row = hunter_sensation[0]
        column = hunter_sensation[1]
    elif abs(scout_sensation[0]) <= Scout_VFD and abs(scout_sensation[1]) <= Scout_VFD:
        row = scout2hunter[0] + scout_sensation[0]
        column = scout2hunter[1] + scout_sensation[1]
    else:  # if there is no prey in sight, a unique default sensation is used.
        row = row_lim
        column = column_lim

    hunter_sensation_prime = [row, column]

    return hunter_sensation_prime


def reward(hunter_sensation_prime):
    if hunter_sensation_prime == [0, 0]:
        re = 1
    else:
        re = -.1
    return re


def rl_agent(beta=0.8):
    Q = np.zeros((2*Row_num-1, 2*Col_num-1, nA))

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

            hunter_sensation = np.subtract(prey_pose, hunter_pose)

            hunter_probs = Boltzmann(Q[hunter_sensation[0], hunter_sensation[1], :])
            hunter_action = np.random.choice(ACTIONS, p=hunter_probs)
            scout_action = np.random.choice(ACTIONS)
            prey_action = np.random.choice(ACTIONS)

            hunter_pose_prime = movement(hunter_pose, hunter_action)
            scout_pose_prime = movement(scout_pose, scout_action)
            prey_pose_prime = movement(prey_pose, prey_action)

            scout2hunter = np.subtract(scout_pose_prime, hunter_pose_prime)
            scout_sensation = np.subtract(prey_pose_prime, scout_pose_prime)
            hunter_sensation_prime_step1 = np.subtract(prey_pose_prime, hunter_pose_prime)
            hunter_sensation_prime = transition(hunter_sensation_prime_step1, scout_sensation, scout2hunter)

            re = reward(hunter_sensation_prime)

            A_hunter.append(hunter_action)
            A_scout.append(scout_action)
            A_prey.append(prey_action)

            R.append(re)

            Q[row_lim-hunter_sensation[0],
              row_lim-hunter_sensation[1], hunter_action] += beta * (re +
                                                 gamma * np.max(Q[row_lim-hunter_sensation_prime[0],
                                                                  row_lim-hunter_sensation_prime[1], :]) -
                                                 Q[row_lim-hunter_sensation[0],
                                                   row_lim-hunter_sensation[1], hunter_action])

            hunter_pose = hunter_pose_prime
            scout_pose = scout_pose_prime
            prey_pose = prey_pose_prime
            if hunter_sensation_prime == [0, 0]:
                # Q[hunter_sensation_prime[0], hunter_sensation_prime[1], :] = 0
                steps.append(t_step)
                rewards.append(sum(R))
                print(f'In episode {eps + 1} of {NUM_EPISODES}, the prey was captured in {t_step + 1} steps')
                break

    return T_hunter, T_scout, T_prey, A_hunter, A_scout, A_prey, rewards, steps, Q

# for i in range(50):
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
