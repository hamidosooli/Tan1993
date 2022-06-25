from gridworld_multi_agent1_1 import animate
import numpy as np
import matplotlib.pyplot as plt
import h5py

exp_name1 = '2R_NS'
exp_name2 = '2R_2S'
exp_name3 = '2R_2S_A2A'
# exp_name = '5R_5S'
# exp_name = '5R_5RS_5S'

plt.rcParams.update({'font.size': 12})
file_name1 = f'../h5py data/multi_agent_Q_learning_{exp_name1}_100Runs.hdf5'
file_name2 = f'../h5py data/multi_agent_Q_learning_{exp_name2}_100Runs.hdf5'
file_name3 = f'../h5py data/multi_agent_Q_learning_{exp_name3}_100Runs.hdf5'
num_scouts = 2
run_animate = False

rescue_team_Traj = []
rescue_team_RewSum = []
rescue_team_Steps = []
rescue_team_RewSum_seen = []
rescue_team_Steps_seen = []
rescue_team_Q = []
victims_Traj = []

f1 = h5py.File(file_name1, 'r')
f2 = h5py.File(file_name2, 'r')
f3 = h5py.File(file_name3, 'r')

for idx in range(len(f1['RS_VFD'])):
    # rescue_team_Traj.append(f[f'RS{idx}_trajectory'])
    rescue_team_RewSum.append(f1[f'RS{idx}_reward'])
    rescue_team_Steps.append(f1[f'RS{idx}_steps'])
    rescue_team_RewSum_seen.append(f1[f'RS{idx}_reward_seen'])
    rescue_team_Steps_seen.append(f1[f'RS{idx}_steps_seen'])
for idx in range(len(f2['RS_VFD']) - num_scouts):
    # rescue_team_Traj.append(f[f'RS{idx}_trajectory'])
    rescue_team_RewSum.append(f2[f'RS{idx}_reward'])
    rescue_team_Steps.append(f2[f'RS{idx}_steps'])
    rescue_team_RewSum_seen.append(f2[f'RS{idx}_reward_seen'])
    rescue_team_Steps_seen.append(f2[f'RS{idx}_steps_seen'])
for idx in range(len(f3['RS_VFD'])-num_scouts):
    # rescue_team_Traj.append(f[f'RS{idx}_trajectory'])
    rescue_team_RewSum.append(f3[f'RS{idx}_reward'])
    rescue_team_Steps.append(f3[f'RS{idx}_steps'])
    rescue_team_RewSum_seen.append(f3[f'RS{idx}_reward_seen'])
    rescue_team_Steps_seen.append(f3[f'RS{idx}_steps_seen'])
        # rescue_team_Q.append(f[f'RS{idx}_Q'])
    # for idx in range(f['victims_num'][0]):
    #     victims_Traj.append(f[f'victim{idx}_trajectory'])

    # if run_animate:
    #     animate(np.asarray(rescue_team_Traj), np.asarray(victims_Traj),
    #             np.asarray(f['RS_VFD']), f['RS_ROLES'], wait_time=0.5)

rescue_team_legends = []

plt.figure('reward', dpi=300)
nn = 100
length = 2000
for idx in range(len(f1['RS_VFD'])):
    rescue_team_legends.append(f'Agent {idx+1} {exp_name1}')
for idx in range(len(f2['RS_VFD'])-num_scouts):
    rescue_team_legends.append(f'Agent {idx+1} {exp_name2}')
for idx in range(len(f3['RS_VFD'])-num_scouts):
    rescue_team_legends.append(f'Agent {idx+1} {exp_name3}')

for idx in range(len(f1['RS_VFD'])):
    re = rescue_team_RewSum[idx]
    plt.plot(np.asarray(re[0:length:nn]), linewidth=5)
for idx in range(2+len(f2['RS_VFD'])-num_scouts):
    re1 = rescue_team_RewSum[idx]
    plt.plot(np.asarray(re1[0:length:nn]), linewidth=5)
for idx in range(4+len(f3['RS_VFD'])-num_scouts):
    re2 = rescue_team_RewSum[idx]
    plt.plot(np.asarray(re2[0:length:nn]), linewidth=5)

for idx in range(len(f1['RS_VFD'])):
    re = rescue_team_RewSum[idx]
    plt.plot(np.asarray(re[0:length:nn]), linewidth=5)
for idx in range(2+len(f2['RS_VFD'])-num_scouts):
    re1 = rescue_team_RewSum[idx]
    plt.plot(np.asarray(re1[0:length:nn]), linewidth=5)
for idx in range(4+len(f3['RS_VFD'])-num_scouts):
    re2 = rescue_team_RewSum[idx]
    plt.plot(np.asarray(re2[0:length:nn]), linewidth=5)

plt.xlabel('Number of Episodes Divided by 100')
plt.ylabel('Rescue Team Total Rewards')
plt.legend(rescue_team_legends)
plt.title('Average Over 100 Runs')
plt.savefig('rewards.png')

plt.figure('reward_seen', dpi=300)

for idx in range(len(f1['RS_VFD'])):
    re3 = rescue_team_RewSum_seen[idx]
    plt.plot(np.asarray(re3[0:length:nn]), linewidth=5)
for idx in range(2+len(f2['RS_VFD'])-num_scouts):
    re4 = rescue_team_RewSum_seen[idx]
    plt.plot(np.asarray(re4[0:length:nn]), linewidth=5)
for idx in range(4+len(f3['RS_VFD'])-num_scouts):
    re5 = rescue_team_RewSum_seen[idx]
    plt.plot(np.asarray(re5[0:length:nn]), linewidth=5)

plt.xlabel('Number of Episodes Divided by 100')
plt.ylabel('Rescue Team Rewards During Victim Visit')
plt.legend(rescue_team_legends)
plt.title('Average Over 100 Runs')
plt.savefig('rewards_seen.png')

plt.figure('steps', dpi=300)

for idx in range(len(f1['RS_VFD'])):
    stp1 = rescue_team_Steps[idx]
    plt.plot(np.asarray(stp1[0:length:nn]), linewidth=5)
for idx in range(2+len(f2['RS_VFD'])-num_scouts):
    stp2 = rescue_team_Steps[idx]
    plt.plot(np.asarray(stp2[0:length:nn]), linewidth=5)
for idx in range(4+len(f3['RS_VFD'])-num_scouts):
    stp3 = rescue_team_Steps[idx]
    plt.plot(np.asarray(stp3[0:length:nn]), linewidth=5)

plt.xlabel('Number of Episodes Divided by 100')
plt.ylabel('Rescue Team Total Steps')
plt.legend(rescue_team_legends)
plt.title('Average Over 100 Runs')
plt.savefig('steps.png')

plt.figure('steps_seen', dpi=300)

for idx in range(len(f1['RS_VFD'])):
    stp4 = rescue_team_Steps_seen[idx]
    plt.plot(np.asarray(stp4[0:length:nn]), linewidth=5)
for idx in range(2+len(f2['RS_VFD'])-num_scouts):
    stp5 = rescue_team_Steps_seen[idx]
    plt.plot(np.asarray(stp5[0:length:nn]), linewidth=5)
for idx in range(4+len(f3['RS_VFD'])-num_scouts):
    stp6 = rescue_team_Steps_seen[idx]
    plt.plot(np.asarray(stp6[0:length:nn]), linewidth=5)

plt.xlabel('Number of Episodes Divided by 100')
plt.ylabel('Rescue Team Steps During Victim Visit')
plt.legend(rescue_team_legends)
plt.title('Average Over 100 Runs')
plt.savefig('steps_seen.png')
plt.show()
