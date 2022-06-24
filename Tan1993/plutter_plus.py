from gridworld_ma import animate
import numpy as np
import matplotlib.pyplot as plt
import h5py

hunter_vfd = 2
scout_vfd = 2
mode = 'Multi Runs'
# mode = 'Single Run'
# exp = 'No Scout'
exp = 'With Random Scout'
# exp = 'With Leaning Scout'

if exp == 'No Scout':
    filename = 'Tan1993_case1_no_scout.hdf5'
elif exp == 'With Random Scout':
    filename = 'Tan1993_case1_with_random_scout.hdf5'
elif exp == 'With Leaning Scout':
    filename = 'Tan1993_case1_with_learning_scout.hdf5'

T_hunter = []
T_scout = []
T_prey = []

A_hunter = []
A_scout = []
A_prey = []

rewards22 = []
steps22 = []
see_rewards22 = []
see_steps22 = []

rewards23 = []
steps23 = []
see_rewards23 = []
see_steps23 = []

rewards24 = []
steps24 = []
see_rewards24 = []
see_steps24 = []
if mode == 'Multi Runs' and exp == 'With Leaning Scout':
    rewards_scout22 = []
    see_rewards_scout22 = []
    see_steps_scout22 = []

    rewards_scout23 = []
    see_rewards_scout23 = []
    see_steps_scout23 = []

    rewards_scout24 = []
    see_rewards_scout24 = []
    see_steps_scout24 = []
filename22 = 'Tan1993_case1_with_random_scout_2&2_100runs.hdf5'
filename23 = 'Tan1993_case1_with_random_scout_2&3_100runs.hdf5'
filename24 = 'Tan1993_case1_with_random_scout_2&4_100runs.hdf5'
Q = []
with h5py.File(filename22, 'r') as gw_ma22:
    with h5py.File(filename23, 'r') as gw_ma23:
        with h5py.File(filename24, 'r') as gw_ma24:
    # T_hunter.append(np.asarray(gw_ma['T_hunter']))
    # T_prey.append(np.asarray(gw_ma['T_prey']))
    # A_hunter.append(np.asarray(gw_ma['A_hunter']))
    # A_prey.append(np.asarray(gw_ma['A_prey']))
            rewards22.append(np.asarray(gw_ma22['rewards']))
            steps22.append(np.asarray(gw_ma22['steps']))
            see_rewards22.append(np.asarray(gw_ma22['see_rewards']))
            see_steps22.append(np.asarray(gw_ma22['see_steps']))

            rewards23.append(np.asarray(gw_ma23['rewards']))
            steps23.append(np.asarray(gw_ma23['steps']))
            see_rewards23.append(np.asarray(gw_ma23['see_rewards']))
            see_steps23.append(np.asarray(gw_ma23['see_steps']))

            rewards24.append(np.asarray(gw_ma24['rewards']))
            steps24.append(np.asarray(gw_ma24['steps']))
            see_rewards24.append(np.asarray(gw_ma24['see_rewards']))
            see_steps24.append(np.asarray(gw_ma24['see_steps']))
            # Q.append(np.asarray(gw_ma['Q']))
            if mode == 'Multi Runs' and exp == 'With Leaning Scout':
                rewards_scout22.append(np.asarray(gw_ma22['rewards_scout']))
                see_rewards_scout22.append(np.asarray(gw_ma22['see_rewards_scout']))
                see_steps_scout22.append(np.asarray(gw_ma22['see_steps_scout']))

                rewards_scout23.append(np.asarray(gw_ma23['rewards_scout']))
                see_rewards_scout23.append(np.asarray(gw_ma23['see_rewards_scout']))
                see_steps_scout23.append(np.asarray(gw_ma23['see_steps_scout']))

                rewards_scout24.append(np.asarray(gw_ma24['rewards_scout']))
                see_rewards_scout24.append(np.asarray(gw_ma24['see_rewards_scout']))
                see_steps_scout24.append(np.asarray(gw_ma24['see_steps_scout']))
    # if exp != 'No Scout':
    #     T_scout.append(np.asarray(gw_ma['T_scout']))
    #     A_scout.append(np.asarray(gw_ma['A_scout']))
    #     animate(T_hunter[0], T_scout[0], T_prey[0],
    #             A_hunter[0], A_scout[0], A_prey[0],
    #             hunter_vfd, scout_vfd, wait_time=.5, have_scout=True)
    # else:
    #     animate(T_hunter[0], [], T_prey[0],
    #             A_hunter[0], [], A_prey[0],
    #             hunter_vfd, scout_vfd, wait_time=.5, have_scout=False)

plt.figure('Rewards')
plt.xlabel('Episodes')
plt.ylabel('Sum of Rewards for the Hunter during each Episode')
if mode == 'Multi Runs':
    plt.plot(np.mean(np.asarray(rewards22[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards22[0]), axis=0))

    plt.plot(np.mean(np.asarray(rewards23[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards23[0]), axis=0))

    plt.plot(np.mean(np.asarray(rewards24[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards24[0]), axis=0))
# else:
#     plt.plot(rewards[0])
#     plt.plot(see_rewards[0])
plt.legend(['Total rewards for hunter, hunter VFD=2 & scout VFD=2', 'When we have data, hunter VFD=2 & scout VFD=2',
            'Total rewards for hunter, hunter VFD=2 & scout VFD=3', 'When we have data, hunter VFD=2 & scout VFD=3',
            'Total rewards for hunter, hunter VFD=2 & scout VFD=4', 'When we have data, hunter VFD=2 & scout VFD=4'])

plt.figure('Steps')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps for the Hunter on each Episode')
if mode == 'Multi Runs':
    plt.plot(np.mean(np.asarray(steps22[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_steps22[0]), axis=0))

    plt.plot(np.mean(np.asarray(steps23[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_steps23[0]), axis=0))

    plt.plot(np.mean(np.asarray(steps24[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_steps24[0]), axis=0))
# else:
#     plt.plot(steps[0])
#     plt.plot(see_steps[0])
plt.legend(['Total steps, hunter VFD=2 & scout VFD=2', 'When we have data, hunter VFD=2 & scout VFD=2',
            'Total steps, hunter VFD=2 & scout VFD=3', 'When we have data, hunter VFD=2 & scout VFD=3',
            'Total steps, hunter VFD=2 & scout VFD=4', 'When we have data, hunter VFD=2 & scout VFD=4'])

if mode == 'Multi Runs' and exp == 'With Leaning Scout':
    plt.figure('Rewards_scout')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards for the Scout during each Episode')
    plt.plot(np.mean(np.asarray(rewards_scout22[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards_scout22[0]), axis=0))

    plt.plot(np.mean(np.asarray(rewards_scout23[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards_scout23[0]), axis=0))

    plt.plot(np.mean(np.asarray(rewards_scout24[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards_scout24[0]), axis=0))
    plt.legend(['Total rewards for scout, hunter VFD=2 & scout VFD=2', 'When we have data, hunter VFD=2 & scout VFD=2',
                'Total rewards for scout, hunter VFD=2 & scout VFD=3', 'When we have data, hunter VFD=2 & scout VFD=3',
                'Total rewards for scout, hunter VFD=2 & scout VFD=4', 'When we have data, hunter VFD=2 & scout VFD=4'])

    plt.figure('Steps_scout')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps for the Scout on each Episode')
    plt.plot(np.mean(np.asarray(see_steps_scout22[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_steps_scout23[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_steps_scout24[0]), axis=0))
    plt.legend(['Total steps for scout, hunter VFD=2 & scout VFD=2',
                'Total steps for scout, hunter VFD=2 & scout VFD=3',
                'Total steps for scout, hunter VFD=2 & scout VFD=4'])

plt.show()
