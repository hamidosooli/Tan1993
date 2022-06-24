import numpy as np
import h5py
from gridworld_ma_1 import animate

# f = h5py.File(f'Tan1993_case1_with_learning_scout_normal_speed_multi.hdf5', "r")
# f = h5py.File(f'Tan1993_case1_with_learning_scout_fixed_multi.hdf5', "r")
f = h5py.File(f'Tan1993_case1_with_learning_scout_slow_multi.hdf5', "r")


Q_opt = np.asarray(f['Q'])
Q_opt_scout = np.asarray(f['Q_scout'])

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

Hunter_VFD = 2  # Hunter's visual field depth
Scout_VFD = 2

can_see_it = False
can_see_it_scout = False
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


def smart_move(were_here, pos, pos_prime):
    if len(np.argwhere(were_here)) > 0:
        for loc in np.argwhere(were_here):
            if np.sqrt((loc[0] - pos[0]) ** 2 + (loc[1] - pos[1]) ** 2) == 1:
                pos_prime = loc
                break
            else:
                continue
    return pos_prime


def play(hunter_init_loc, scout_init_loc, prey_init_loc):
    R = []
    R_scout = []
    global can_see_it
    global can_see_it_scout
    global default_sensation
    hunter_pos = hunter_init_loc
    scout_pos = scout_init_loc
    prey_pos = prey_init_loc

    T_hunter = [hunter_pos]
    T_scout = [scout_pos]
    T_prey = [prey_pos]

    wereHere_hunter = np.ones((Row_num, Col_num))
    wereHere_scout = np.ones((Row_num, Col_num))

    can_see_it = False
    can_see_it_scout = False
    t_step = 0
    while True:
        t_step += 1
        T_hunter.append(hunter_pos)
        T_scout.append(scout_pos)
        T_prey.append(prey_pos)

        wereHere_hunter[int(hunter_pos[0]), int(hunter_pos[1])] = 0
        wereHere_scout[int(scout_pos[0]), int(scout_pos[1])] = 0

        scout2hunter = np.subtract(scout_pos, hunter_pos)

        scout_sensation_step1 = np.subtract(prey_pos, scout_pos)
        scout_sensation = update_sensation_scout(scout_sensation_step1)

        hunter_sensation_step1 = np.subtract(prey_pos, hunter_pos)
        hunter_sensation = update_sensation(hunter_sensation_step1, scout_sensation, scout2hunter)

        idx = sensation2index(hunter_sensation, Row_num, can_see_it)
        hunter_action = np.argmax(Q_opt[idx, :])

        idx_scout = sensation2index(scout_sensation, Scout_VFD, can_see_it_scout)
        scout_action = np.argmax(Q_opt_scout[idx_scout, :])

        hunter_pos_prime = movement(hunter_pos, hunter_action)
        scout_pos_prime = movement(scout_pos, scout_action)

        if idx == (2 * Row_num + 1) ** 2:
            hunter_pos_prime = smart_move(wereHere_hunter, hunter_pos, hunter_pos_prime)

        if idx_scout == (2 * Scout_VFD + 1) ** 2:
            scout_pos_prime = smart_move(wereHere_scout, scout_pos, scout_pos_prime)

        scout2hunter_prime = np.subtract(scout_pos_prime, hunter_pos_prime)
        prey_action = np.random.choice(ACTIONS)
        prey_pos_prime = movement(prey_pos, prey_action)
        scout_sensation_prime_step1 = np.subtract(prey_pos_prime, scout_pos_prime)
        scout_sensation_prime = update_sensation_scout(scout_sensation_prime_step1)

        hunter_sensation_prime_step1 = np.subtract(prey_pos_prime, hunter_pos_prime)
        hunter_sensation_prime = update_sensation(hunter_sensation_prime_step1,
                                                  scout_sensation_prime,
                                                  scout2hunter_prime)

        re = reward(hunter_sensation_prime)
        re_scout = reward(scout_sensation_prime)

        R.append(re)
        R_scout.append(re_scout)

        if hunter_sensation_prime == [0, 0]:
            T_hunter.append(hunter_pos_prime)
            break



        hunter_pos = hunter_pos_prime
        scout_pos = scout_pos_prime
        # if t_step % 2 == 0:
        prey_pos = prey_pos_prime

    return R, T_hunter, T_scout, T_prey


R, T_hunter, T_scout, T_prey = play([0, 9], [6, 3], [9, 0])
animate(T_hunter, T_scout, T_prey, [Hunter_VFD], [Scout_VFD], wait_time=.5, have_scout=True)
# animate(f['T_hunter'], f['T_scout'], f['T_prey'], [Hunter_VFD], [Scout_VFD], wait_time=.5, have_scout=True)
f.close()
