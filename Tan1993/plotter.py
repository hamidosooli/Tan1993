from gridworld_ma_1 import animate
import numpy as np
import matplotlib.pyplot as plt
import h5py


plt.rcParams.update({'font.size': 22})
hunter_vfd = [2]
scout_vfd = [2]
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
    # filename1 = 'Tan1993_case1_with_learning_scout.hdf5'
    # filename = 'Tan1993_case1_with_several_learning_scout.hdf5'
    # filename = 'Tan1993_case1_with_learning_scout_slow_prey.hdf5'
    # filename = 'Tan1993_case1_with_learning_scout_eps_greedy.hdf5'
    # filename = 'Tan1993_case1_with_learning_scout_prioritized_moves.hdf5'
    # filename = 'Tan1993_case1_with_learning_scout_2&2_100runs_epsGreedy.hdf5'
    filename = 'Tan1993_case1_with_learning_scout_normal_speed.hdf5'
    # filename = 'Tan1993_case1_with_learning_scout_fixed.hdf5'
    # filename = 'Tan1993_case1_with_learning_scout_slow.hdf5'
T_hunter = []
T_scout = []
T_prey = []

A_hunter = []
A_scout = []
A_prey = []

rewards = []
rewards_scout = []
steps = []
see_rewards = []
see_rewards_scout = []
see_steps = []

rewards1 = []
steps1 = []
see_rewards1 = []
see_steps1 = []
if mode == 'Multi Runs' and exp == 'With Leaning Scout':
    rewards_scout = []
    see_rewards_scout = []
    see_steps_scout = []

    rewards_scout1 = []
    see_rewards_scout1 = []
    see_steps_scout1 = []
    filename = 'Tan1993_case1_with_learning_scout_2&2_100runs.hdf5'
    filename1 = 'Tan1993_case1_with_learning_scout_2&2_100runs_slow.hdf5'
Q = []
with h5py.File(filename, 'r') as gw_ma:
    T_hunter.append(np.asarray(gw_ma['T_hunter']))
    T_prey.append(np.asarray(gw_ma['T_prey']))
    # A_hunter.append(np.asarray(gw_ma['A_hunter']))
    # A_prey.append(np.asarray(gw_ma['A_prey']))
    rewards.append(np.asarray(gw_ma['rewards']))
    rewards_scout.append(np.asarray(gw_ma['rewards_scout']))
    steps.append(np.asarray(gw_ma['steps']))
    see_rewards.append(np.asarray(gw_ma['see_rewards']))
    see_rewards_scout.append(np.asarray(gw_ma['see_rewards_scout']))
    see_steps.append(np.asarray(gw_ma['see_steps']))
    # Q.append(np.asarray(gw_ma['Q']))
    if mode == 'Multi Runs' and exp == 'With Leaning Scout':
        rewards_scout.append(np.asarray(gw_ma['rewards_scout']))
        see_rewards_scout.append(np.asarray(gw_ma['see_rewards_scout']))
        see_steps_scout.append(np.asarray(gw_ma['see_steps_scout']))
        with h5py.File(filename1, 'r') as gw_ma1:
            T_hunter.append(np.asarray(gw_ma['T_hunter']))
            T_prey.append(np.asarray(gw_ma['T_prey']))
            # A_hunter.append(np.asarray(gw_ma['A_hunter']))
            # A_prey.append(np.asarray(gw_ma['A_prey']))
            rewards1.append(np.asarray(gw_ma1['rewards']))
            steps1.append(np.asarray(gw_ma1['steps']))
            see_rewards1.append(np.asarray(gw_ma1['see_rewards']))
            see_steps1.append(np.asarray(gw_ma1['see_steps']))
            # Q.append(np.asarray(gw_ma['Q']))
            if mode == 'Multi Runs' and exp == 'With Leaning Scout':
                rewards_scout1.append(np.asarray(gw_ma1['rewards_scout']))
                see_rewards_scout1.append(np.asarray(gw_ma1['see_rewards_scout']))
                see_steps_scout1.append(np.asarray(gw_ma1['see_steps_scout']))
    if exp != 'No Scout':
        # T_scout.append(np.asarray(gw_ma['T_scout']))
        animate(T_hunter, T_scout, T_prey,
                hunter_vfd, scout_vfd, wait_time=.5, have_scout=True)
    else:
        animate(T_hunter[0], [], T_prey[0],
                hunter_vfd, scout_vfd, wait_time=.5, have_scout=False)

plt.figure('Rewards')
plt.xlabel('Episodes')
plt.ylabel('Sum of Rewards during each Episode')
if mode == 'Multi Runs':
    plt.plot(np.mean(np.asarray(rewards[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards[0]), axis=0))

    plt.plot(np.mean(np.asarray(rewards1[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards1[0]), axis=0))
else:
    plt.plot(rewards[0])
    plt.plot(see_rewards[0])
plt.legend(['Total rewards for rescuer with normal victim',
            'When we have data for with normal victim',
            'Total rewards for scout with slow victim',
            'When we have data for the scout with slow victim'])

plt.figure('Steps')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps on each Episode')
if mode == 'Multi Runs':
    plt.plot(np.mean(np.asarray(steps[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_steps[0]), axis=0))

    plt.plot(np.mean(np.asarray(steps1[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_steps1[0]), axis=0))
else:
    plt.plot(steps[0])
    plt.plot(see_steps[0])
plt.legend(['Total steps for the scout with normal victim',
            'When we have data for the scout with normal victim',
            'Total steps for the scout with slow victim',
            'When we have data for the scout with slow victim'
            ])

if mode == 'Multi Runs' and exp == 'With Leaning Scout':
    plt.figure('Rewards_scout')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards during each Episode')
    plt.plot(np.mean(np.asarray(rewards_scout[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards_scout[0]), axis=0))

    plt.plot(np.mean(np.asarray(rewards_scout1[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_rewards_scout1[0]), axis=0))
    plt.legend(['Total rewards for the scout with normal victim',
                'When we have data for the scout with normal victim',
                'Total rewards for the scout with slow victim',
                'When we have data for the scout with slow victim'])

    plt.figure('Steps_scout')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps on each Episode')
    plt.plot(np.mean(np.asarray(see_steps_scout[0]), axis=0))
    plt.plot(np.mean(np.asarray(see_steps_scout1[0]), axis=0))
    plt.legend(['Total steps for the scout with normal victim',
                'Total steps for the scout with slow victim'])

plt.show()
