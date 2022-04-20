from gridworld_ma import animate
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
T1 = []
T1_prime = []

T2 = []
T2_prime = []

A1 = []
A2 = []
Q1 = []
Q2 = []
rewards = []
steps = []
Q_star = []
# for i in range(1, 6):
with h5py.File('gridworld_new_agent.hdf5', 'r') as gw_ma:
    T1.append(np.asarray(gw_ma['T1']))
    # T1_prime.append(np.asarray(gw_ma['T1_prime']))
    T2.append(np.asarray(gw_ma['T2']))
    # T2_prime.append(np.asarray(gw_ma['T2_prime']))
    rewards.append(np.asarray(gw_ma['rewards']))
    A1.append(np.asarray(gw_ma['A1']))
    A2.append(np.asarray(gw_ma['A2']))
    steps.append(np.asarray(gw_ma['steps']))
    Q1.append(np.asarray(gw_ma['Q']))
    # Q2.append(np.asarray(gw_ma['Q2']))
    # time.sleep(5)
    # animate(T1, T1_prime, A1[0], A2[0], wait_time=5)
    plt.figure('rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards during each Episode')
    plt.plot(rewards[0][:,0])
    # plt.plot(np.mean(np.asarray(rewards), axis=0))
    plt.figure('steps')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps on each Episode')
    plt.plot(steps[0][:])
    # plt.plot(np.mean(np.asarray(steps), axis=0))
    plt.show()
