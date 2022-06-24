import h5py
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 12})


nn = 100
length = 2000
rescue_team_legends = []

plt.figure('reward', dpi=300)
for num_s in range(1, 11):
    exp_name = '1R_' + str(num_s) + 'S_1V'
    file_name = f'../h5py data/multi_agent_Q_learning_{exp_name}_100Runs.hdf5'
    f = h5py.File(file_name, 'r')
    rescue_team_legends.append(f'{exp_name}')

    plt.plot(np.asarray(f[f'RS0_reward'])[0:length:nn], linewidth=5)

    plt.xlabel('Number of episodes')
    plt.ylabel('Rescuer Total Rewards')
    plt.legend(rescue_team_legends)
    plt.title('Average Over 100 Runs')
    f.close()
plt.savefig('reward.eps')

plt.figure('reward_seen', dpi=300)
for num_s in range(1, 11):
    exp_name = '1R_' + str(num_s) + 'S_1V'
    file_name = f'../h5py data/multi_agent_Q_learning_{exp_name}_100Runs.hdf5'
    f = h5py.File(file_name, 'r')
    rescue_team_legends.append(f'{exp_name}')

    plt.plot(np.asarray(f[f'RS0_reward_seen'])[0:length:nn], linewidth=5)

    plt.xlabel('Number of Episodes')
    plt.ylabel('Rescuer Rewards During Victim Visit')
    plt.legend(rescue_team_legends)
    plt.title('Average Over 100 Runs')
    f.close()
plt.savefig('reward_seen.eps')

plt.figure('steps', dpi=300)
for num_s in range(1, 11):
    exp_name = '1R_' + str(num_s) + 'S_1V'
    file_name = f'../h5py data/multi_agent_Q_learning_{exp_name}_100Runs.hdf5'
    f = h5py.File(file_name, 'r')
    rescue_team_legends.append(f'{exp_name}')

    plt.plot(np.asarray(f[f'RS0_steps'])[0:length:nn], linewidth=5)

    plt.xlabel('Number of Episodes')
    plt.ylabel('Rescuer Total Steps')
    plt.legend(rescue_team_legends)
    plt.title('Average Over 100 Runs')
    f.close()
plt.savefig('steps.eps')

plt.figure('steps_seen', dpi=300)
for num_s in range(1, 11):
    exp_name = '1R_' + str(num_s) + 'S_1V'
    file_name = f'../h5py data/multi_agent_Q_learning_{exp_name}_100Runs.hdf5'
    f = h5py.File(file_name, 'r')
    rescue_team_legends.append(f'{exp_name}')

    plt.plot(np.asarray(f[f'RS0_steps_seen'])[0:length:nn], linewidth=5)

    plt.xlabel('Number of Episodes')
    plt.ylabel('Rescuer Steps During Victim Visit')
    plt.legend(rescue_team_legends)
    plt.title('Average Over 100 Runs')
    f.close()
plt.savefig('steps_seen.eps')
plt.show()