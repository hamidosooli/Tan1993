import numpy as np
import h5py
import matplotlib.pyplot as plt

f1 = h5py.File(f'Tan1993_case1_with_learning_scout_normal_speed_multi.hdf5', "r")
f2 = h5py.File(f'Tan1993_case1_with_learning_scout_fixed_multi.hdf5', "r")
f3 = h5py.File(f'Tan1993_case1_with_learning_scout_slow_multi.hdf5', "r")

plt.rcParams.update({'font.size': 18})

plt.figure('Rewards')
plt.subplot(1, 2, 1)
plt.xlabel('Number of Episodes')
plt.ylabel('Sum of the Rewards (Average Over 100 Runs)')

plt.plot(np.mean(np.asarray(f1['rewards']), axis=0))

plt.plot(np.mean(np.asarray(f2['rewards']), axis=0))

plt.plot(np.mean(np.asarray(f3['rewards']), axis=0))

plt.title('Total Rescuer Rewards')
plt.legend(['Normal Speed Victim', 'Fixed Victim', 'Slow Victim'])

plt.subplot(1, 2, 2)
plt.xlabel('Number of Episodes')
plt.ylabel('Sum of the Rewards (Average Over 100 Runs)')

plt.plot(np.mean(np.asarray(f1['see_rewards']), axis=0))

plt.plot(np.mean(np.asarray(f2['see_rewards']), axis=0))

plt.plot(np.mean(np.asarray(f3['see_rewards']), axis=0))

plt.title('Rescuer Rewards During Victim Visit')
plt.legend(['Normal Speed Victim', 'Fixed Victim', 'Slow Victim'])

plt.figure('Rewards_scout')
plt.subplot(1, 2, 1)
plt.xlabel('Number of Episodes')
plt.ylabel('Sum of the Rewards (Average Over 100 Runs)')

plt.plot(np.mean(np.asarray(f1['rewards_scout']), axis=0))

plt.plot(np.mean(np.asarray(f2['rewards_scout']), axis=0))

plt.plot(np.mean(np.asarray(f3['rewards_scout']), axis=0))

plt.title('Total Scout Rewards')
plt.legend(['Normal Speed Victim', 'Fixed Victim', 'Slow Victim'])

plt.subplot(1, 2, 2)
plt.xlabel('Number of Episodes')
plt.ylabel('Sum of the Rewards (Average Over 100 Runs)')

plt.plot(np.mean(np.asarray(f1['see_rewards_scout']), axis=0))

plt.plot(np.mean(np.asarray(f2['see_rewards_scout']), axis=0))

plt.plot(np.mean(np.asarray(f3['see_rewards_scout']), axis=0))

plt.title('Scout Rewards During Victim Visit')
plt.legend(['Normal Speed Victim', 'Fixed Victim', 'Slow Victim'])

plt.figure('Steps')
plt.subplot(1, 2, 1)
plt.xlabel('Number of Episodes')
plt.ylabel('Sum of the Steps (Average Over 100 Runs)')

plt.plot(np.mean(np.asarray(f1['steps']), axis=0))

plt.plot(np.mean(np.asarray(f2['steps']), axis=0))

plt.plot(np.mean(np.asarray(f3['steps']), axis=0))

plt.title('Total Rescuer Steps')
plt.legend(['Normal Speed Victim', 'Fixed Victim', 'Slow Victim'])

plt.subplot(1, 2, 2)
plt.xlabel('Number of Episodes')
plt.ylabel('Sum of the Steps (Average Over 100 Runs)')

plt.plot(np.mean(np.asarray(f1['see_steps']), axis=0))

plt.plot(np.mean(np.asarray(f2['see_steps']), axis=0))

plt.plot(np.mean(np.asarray(f3['see_steps']), axis=0))

plt.title('Rescuer Steps During Victim Visit')
plt.legend(['Normal Speed Victim', 'Fixed Victim', 'Slow Victim'])
plt.show()

f1.close()
f2.close()
f3.close()
