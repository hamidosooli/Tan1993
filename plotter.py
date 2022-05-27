from gridworld_ma import animate
import numpy as np
import matplotlib.pyplot as plt
import h5py

hunter_vfd = 2
scout_vfd = [2,3]
# mode = 'Multi Runs'
mode = 'Single Run'
# exp = 'No Scout'
# exp = 'With Random Scout'
exp = 'With Leaning Scout'

if exp == 'No Scout':
    filename = 'Tan1993_case1_no_scout.hdf5'
elif exp == 'With Random Scout':
    filename = 'Tan1993_case1_with_random_scout.hdf5'
elif exp == 'With Leaning Scout':
    # filename = 'Tan1993_case1_with_learning_scout.hdf5'
    filename = 'Tan1993_case1_with_several_learning_scout.hdf5'

T_hunter = []
T_scout = []
T_prey = []

A_hunter = []
A_scout = []
A_prey = []

rewards = []
steps = []
see_rewards = []
see_steps = []
if mode == 'Multi Runs' and exp == 'With Leaning Scout':
    rewards_scout = []
    see_rewards_scout = []
    see_steps_scout = []
    filename = 'Tan1993_case1_with_learning_scout_2&4_100runs.hdf5'
Q = []
with h5py.File(filename, 'r') as gw_ma:
    T_hunter.append(np.asarray(gw_ma['T_hunter']))
    T_prey.append(np.asarray(gw_ma['T_prey']))
    A_hunter.append(np.asarray(gw_ma['A_hunter']))
    A_prey.append(np.asarray(gw_ma['A_prey']))
    rewards.append(np.asarray(gw_ma['rewards']))
    steps.append(np.asarray(gw_ma['steps']))
    see_rewards.append(np.asarray(gw_ma['see_rewards']))
    see_steps.append(np.asarray(gw_ma['see_steps']))
    Q.append(np.asarray(gw_ma['Q']))
    if mode == 'Multi Runs' and exp == 'With Leaning Scout':
        rewards_scout.append(np.asarray(gw_ma['rewards_scout']))
        see_rewards_scout.append(np.asarray(gw_ma['see_rewards_scout']))
        see_steps_scout.append(np.asarray(gw_ma['see_steps_scout']))
    if exp != 'No Scout':
        T_scout.append(np.asarray(gw_ma['T_scouts']))
        A_scout.append(np.asarray(gw_ma['A_scout']))
        animate(T_hunter[0], T_scout, T_prey[0],
                A_hunter[0], A_scout, A_prey[0],
                hunter_vfd, scout_vfd, wait_time=.5, have_scout=True)
    else:
        animate(T_hunter[0], [], T_prey[0],
                A_hunter[0], [], A_prey[0],
                hunter_vfd, scout_vfd, wait_time=.5, have_scout=False)

plt.figure('Rewards')
plt.xlabel('Episodes')
plt.ylabel('Sum of Rewards during each Episode')
if mode == 'Multi Runs':
    plt.plot(np.mean(np.asarray(rewards[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards[0]), axis=0))
else:
    plt.plot(rewards[0])
    plt.plot(see_rewards[0])
plt.legend(['Total rewards for hunter', 'When we have data'])

plt.figure('Steps')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps on each Episode')
if mode == 'Multi Runs':
    plt.plot(np.mean(np.asarray(steps[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_steps[0]), axis=0))
else:
    plt.plot(steps[0])
    plt.plot(see_steps[0])
plt.legend(['Total steps', 'When we have data'])

if mode == 'Multi Runs' and exp == 'With Leaning Scout':
    plt.figure('Rewards_scout')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards during each Episode')
    plt.plot(np.mean(np.asarray(rewards_scout[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards_scout[0]), axis=0))
    plt.legend(['Total rewards for hunter', 'When we have data'])

    plt.figure('Steps_scout')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps on each Episode')
    plt.plot(np.mean(np.asarray(see_steps_scout[0]), axis=0))
    plt.legend(['Total steps', 'When we have data'])

plt.show()