from gridworld_multi_agent1_1 import animate
import numpy as np
import matplotlib.pyplot as plt
import h5py


plt.rcParams.update({'font.size': 22})
file_name = 'multi_agent_Q_learning_2r_3s_3rs.hdf5'

run_animate = True

rescue_team_Traj = []
rescue_team_RewSum = []
rescue_team_Steps = []
rescue_team_RewSum_seen = []
rescue_team_Steps_seen = []
rescue_team_Q = []
victims_Traj = []

with h5py.File('multi_agent_Q_learning.hdf5', 'r') as f:
    for idx in range(len(f['RS_VFD'])):
        rescue_team_Traj.append(f[f'RS{idx}_trajectory'])
        rescue_team_RewSum.append(f[f'RS{idx}_reward'])
        rescue_team_Steps.append(f[f'RS{idx}_steps'])
        rescue_team_RewSum_seen.append(f[f'RS{idx}_reward_seen'])
        rescue_team_Steps_seen.append(f[f'RS{idx}_steps_seen'])
        rescue_team_Q.append(f[f'RS{idx}_Q'])
    for idx in range(f['victims_num'][0]):
        victims_Traj.append(f[f'victim{idx}_trajectory'])

    if run_animate:
        animate(np.asarray(rescue_team_Traj), np.asarray(victims_Traj), np.asarray(f['RS_VFD']), wait_time=0.5)

    rescue_team_legends = []

    plt.figure('reward')
    for idx in range(len(f['RS_VFD'])):
        plt.plot(np.asarray(rescue_team_RewSum[idx]))
        rescue_team_legends.append(f'Agent {idx+1}')
    plt.xlabel('Number of episodes')
    plt.ylabel('Sum of the rewards during the whole rescue time')
    plt.legend(rescue_team_legends)

    plt.figure('reward_seen')
    for idx in range(len(f['RS_VFD'])):
        plt.plot(np.asarray(rescue_team_RewSum_seen[idx]))
    plt.xlabel('Number of episodes')
    plt.ylabel('Sum of the rewards after seeing the victim for the first time')
    plt.legend(rescue_team_legends)

    plt.figure('steps')
    for idx in range(len(f['RS_VFD'])):
        plt.plot(np.asarray(rescue_team_Steps[idx]))
    plt.xlabel('Number of episodes')
    plt.ylabel('Number of steps to finish the rescue mission')
    plt.legend(rescue_team_legends)

    plt.figure('steps_seen')
    for idx in range(len(f['RS_VFD'])):
        plt.plot(np.asarray(rescue_team_Steps_seen[idx]))
    plt.xlabel('Number of episodes')
    plt.ylabel('Number of steps from seeing the victim for the first time to the end')
    plt.legend(rescue_team_legends)

    plt.show()
