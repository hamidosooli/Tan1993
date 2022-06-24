import numpy as np
import h5py
from gridworld_ma_1 import animate

f = h5py.File(f'Tan1993_case1_no_scout.hdf5', "r")

Q_opt = np.asarray(f['Q'])

Row_num = 10
Col_num = 10
row_lim = 9
column_lim = 9

# Actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
ACTIONS = [FORWARD, BACKWARD, RIGHT, LEFT]

Hunter_VFD = [2]  # Hunter's visual field depth
default_sensation = [np.nan, np.nan]


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
    if abs(hunter_sensation[0]) <= Hunter_VFD[0] and abs(hunter_sensation[1]) <= Hunter_VFD[0]:
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


def play(hunter_init_loc, prey_init_loc):
    R = []

    hunter_pos = hunter_init_loc
    prey_pos = prey_init_loc

    T_hunter = [hunter_pos]
    T_prey = [prey_pos]
    wereHere_hunter = np.ones((Row_num, Col_num))
    while True:
        T_hunter.append(hunter_pos)
        T_prey.append(prey_pos)
        wereHere_hunter[int(hunter_pos[0]), int(hunter_pos[1])] = 0
        hunter_sensation_step1 = np.subtract(prey_pos, hunter_pos)
        hunter_sensation = update_sensation(hunter_sensation_step1)

        idx = sensation2index(hunter_sensation, Hunter_VFD[0])
        hunter_action = np.argmax(Q_opt[idx, :])
        hunter_pos_prime = movement(hunter_pos, hunter_action)
        if idx == (2 * Hunter_VFD[0] + 1) ** 2:
            hunter_pos_prime = smart_move(wereHere_hunter, hunter_pos, hunter_pos_prime)
        hunter_sensation_prime_step1 = np.subtract(prey_pos, hunter_pos_prime)
        hunter_sensation_prime = update_sensation(hunter_sensation_prime_step1)

        re = reward(hunter_sensation_prime)
        R.append(re)

        if hunter_sensation == [0, 0] or hunter_sensation_prime == [0, 0]:
            T_hunter.append(hunter_pos_prime)
            break

        prey_action = np.random.choice(ACTIONS)
        prey_pos_prime = movement(prey_pos, prey_action)

        hunter_pos = hunter_pos_prime
        prey_pos = prey_pos_prime

    return R, T_hunter, T_prey


R, T_hunter, T_prey = play([5, 5], [9, 0])
animate(T_hunter, [], T_prey,
        Hunter_VFD[0], [2], wait_time=.5, have_scout=False)
f.close()
