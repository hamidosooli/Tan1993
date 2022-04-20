import numpy as np
import h5py

NUM_EPISODES = 100000

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


def transition(state, action):
    row = state[0]
    column = state[1]

    row_lim = 9
    column_lim = 9
    if action == 0:  # up
        next_state = (max(row - 1, 0), column)
    elif action == 1:  # down
        next_state = (min(abs(row + 1), row_lim), column)
    elif action == 2:  # right
        next_state = (max(row, 0), min(column + 1, column_lim))
    elif action == 3:  # left
        next_state = (max(row, 0), max(column - 1, 0))

    return next_state


def reward(hunter_vfd, prey_pose, hunter_pose):  # hunter visual field depth, prey position, hunter position
    if (np.abs(prey_pose[0] - hunter_pose[0]) < hunter_vfd and
            np.abs(prey_pose[1] - hunter_pose[1]) < hunter_vfd):

        if ((np.abs(prey_pose[0] - hunter_pose[0]) < 1 and np.abs(prey_pose[1] - hunter_pose[1]) < 1) or
                (prey_pose[0] == hunter_pose[0] and prey_pose[1] == hunter_pose[1])):
            rew = 1
        else:
            rew = -.1
    else:
        rew = -.1
    return rew


def rl_agent(num_preys=1, num_hunters=1, beta=0.8):
    Q = np.zeros((num_hunters, Row_num, Col_num, nA))

    re = np.zeros((num_hunters, 1))
    a = np.zeros((num_hunters, 1))
    steps = []
    rewards = []

    preys = np.zeros((num_preys, 2))
    hunters_init = np.zeros((num_hunters, 2))

    for eps in range(NUM_EPISODES):
        flag = False
        for hnum in range(num_hunters):
            hunters_init[hnum, :] = np.random.choice(Row_num), np.random.choice(Col_num)
        for pnum in range(num_preys):
            preys[pnum, :] = np.random.choice(Row_num), np.random.choice(Col_num)

        s = hunters_init
        sp = s.copy()

        T = [s]
        R = []
        A = []

        t_step = 0
        while True:
            t_step += 1

            for ind in range(num_hunters):
                probs = Boltzmann(Q[ind, int(s[ind, 0]), int(s[ind, 1]), :])

                a[ind] = np.random.choice(ACTIONS, p=probs)
                sp[ind, :] = transition(s[ind, :], a[ind])
                re[ind] = reward(2, preys[0], sp[ind, :])
                for pnum in range(1, num_preys):
                    if re[ind] < reward(2, preys[pnum], sp[ind, :]):
                        re[ind] = reward(2, preys[pnum], sp[ind, :])

                Q[ind, int(s[ind, 0]), int(s[ind, 1]), int(a[ind])] += beta * (
                            re[ind] + gamma * np.max(Q[ind, int(sp[ind, 0]), int(sp[ind, 1]), :]) -
                            Q[ind, int(s[ind, 0]), int(s[ind, 1]), int(a[ind])])

            A.append(a)
            T.append(sp)
            R.append(re)

            s = sp
            for rewrd in re:
                if rewrd == 1:
                    flag = True
            if flag:
                steps.append(t_step)
                rewards.append(np.sum(np.asarray(R), axis=1))
                print('Episode ' + str(eps + 1) + ' of ' + str(NUM_EPISODES) +
                      ' finished in ' + str(t_step + 1) + ' steps')
                break

            for pnum in range(num_preys):
                prey_act = np.random.choice(ACTIONS)
                preys[pnum, :] = transition(preys[pnum, :], prey_act)
    return T, rewards, A, steps, Q


Trajectory, Reward, ActionHistory, steps, Q_star = rl_agent(num_preys=2, num_hunters=2, beta=0.8)

with h5py.File(f'gridworld_Tan1993.hdf5', "w") as f:
    f.create_dataset('T', data=Trajectory)
    f.create_dataset('rewards', data=Reward)
    f.create_dataset('A', data=ActionHistory)
    f.create_dataset('steps', data=steps)
    f.create_dataset('Q', data=Q_star)
