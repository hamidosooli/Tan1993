from gridworld_ma import animate
import numpy as np
import matplotlib.pyplot as plt
import h5py

hunter_vfd = 2
scout_vfd = 2

T_hunter = []
T_scout = []
T_prey = []

A_hunter = []
A_scout = []
A_prey = []

rewards = []
steps = []
Q = []
with h5py.File('Tan1993_case1.hdf5', 'r') as gw_ma:
    T_hunter.append(np.asarray(gw_ma['T_hunter']))
    T_scout.append(np.asarray(gw_ma['T_scout']))
    T_prey.append(np.asarray(gw_ma['T_prey']))

    A_hunter.append(np.asarray(gw_ma['A_hunter']))
    A_scout.append(np.asarray(gw_ma['A_scout']))
    A_prey.append(np.asarray(gw_ma['A_prey']))

    rewards.append(np.asarray(gw_ma['rewards']))
    steps.append(np.asarray(gw_ma['steps']))
    Q.append(np.asarray(gw_ma['Q']))

    animate(T_hunter[0], T_scout[0], T_prey[0],
            A_hunter[0], A_scout[0], A_prey[0],
            hunter_vfd, scout_vfd, wait_time=0.5)

    # plt.figure('rewards')
    # plt.xlabel('Episodes')
    # plt.ylabel('Sum of Rewards during each Episode')
    # plt.plot(rewards[0])
    # plt.figure('steps')
    # plt.xlabel('Episodes')
    # plt.ylabel('Number of Steps on each Episode')
    # plt.plot(steps[0])
    # plt.show()
