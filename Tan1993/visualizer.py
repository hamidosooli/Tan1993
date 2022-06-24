from gridworld_multi_agent import animate
import numpy as np
import matplotlib.pyplot as plt
import h5py


plt.rcParams.update({'font.size': 22})
file_name = 'multi_agent_Q_learning_2r_3s_3rs.hdf5'

run_animate = True

rescuers_vfd = [2, 3]
scouts_vfd = [3, 2, 3]
rescuers_scouts_vfd = [2, 3, 4]

rescuers_traj = []
scouts_traj = []
rescuers_scouts_traj = []
victims_traj = []

with h5py.File(file_name, 'r') as f:

    rescuers_traj.append(f['r1_trajectory'])
    rescuers_traj.append(f['r2_trajectory'])
    rescuers_traj = np.asarray(rescuers_traj)

    scouts_traj.append(f['s1_trajectory'])
    scouts_traj.append(f['s2_trajectory'])
    scouts_traj.append(f['s3_trajectory'])
    scouts_traj = np.asarray(scouts_traj)

    rescuers_scouts_traj.append(f['rs1_trajectory'])
    rescuers_scouts_traj.append(f['rs2_trajectory'])
    rescuers_scouts_traj.append(f['rs3_trajectory'])
    rescuers_scouts_traj = np.asarray(rescuers_scouts_traj)

    victims_traj.append(f['v1_trajectory'])
    victims_traj.append(f['v2_trajectory'])
    victims_traj.append(f['v3_trajectory'])
    victims_traj.append(f['v4_trajectory'])
    victims_traj.append(f['v5_trajectory'])
    victims_traj = np.asarray(victims_traj)

    if run_animate:
        animate(rescuers_traj, rescuers_scouts_traj, scouts_traj, victims_traj,
                rescuers_vfd, scouts_vfd, rescuers_scouts_vfd, wait_time=0.5)

    plt.figure('reward')
    plt.plot(np.asarray(f['r1_reward']))
    plt.plot(np.asarray(f['r2_reward']))

    plt.plot(np.asarray(f['s1_reward']))
    plt.plot(np.asarray(f['s2_reward']))
    plt.plot(np.asarray(f['s3_reward']))

    plt.plot(np.asarray(f['rs1_reward']))
    plt.plot(np.asarray(f['rs2_reward']))
    plt.plot(np.asarray(f['rs3_reward']))

    plt.xlabel('Number of episodes')
    plt.ylabel('Sum of the rewards during the whole rescue time')
    plt.legend(['Rescuer 1', 'Rescuer 2',
                'Scout 1', 'Scout 2', 'Scout 3',
                'Rescuer/Scout 1', 'Rescuer/Scout 2', 'Rescuer/Scout 3'])

    plt.figure('reward_seen')
    plt.plot(np.asarray(f['r1_reward_seen']))
    plt.plot(np.asarray(f['r2_reward_seen']))

    plt.plot(np.asarray(f['s1_reward_seen']))
    plt.plot(np.asarray(f['s2_reward_seen']))
    plt.plot(np.asarray(f['s3_reward_seen']))

    plt.plot(np.asarray(f['rs1_reward_seen']))
    plt.plot(np.asarray(f['rs2_reward_seen']))
    plt.plot(np.asarray(f['rs3_reward_seen']))

    plt.xlabel('Number of episodes')
    plt.ylabel('Sum of the rewards after seeing the victim for the first time')
    plt.legend(['Rescuer 1', 'Rescuer 2',
                'Scout 1', 'Scout 2', 'Scout 3',
                'Rescuer/Scout 1', 'Rescuer/Scout 2', 'Rescuer/Scout 3'])

    plt.figure('steps')
    plt.plot(np.asarray(f['r1_steps']))
    plt.plot(np.asarray(f['r2_steps']))

    plt.plot(np.asarray(f['s1_steps']))
    plt.plot(np.asarray(f['s2_steps']))
    plt.plot(np.asarray(f['s3_steps']))

    plt.plot(np.asarray(f['rs1_steps']))
    plt.plot(np.asarray(f['rs2_steps']))
    plt.plot(np.asarray(f['rs3_steps']))

    plt.xlabel('Number of episodes')
    plt.ylabel('Number of steps to finish the rescue mission')
    plt.legend(['Rescuer 1', 'Rescuer 2',
                'Scout 1', 'Scout 2', 'Scout 3',
                'Rescuer/Scout 1', 'Rescuer/Scout 2', 'Rescuer/Scout 3'])

    plt.figure('steps_seen')
    plt.plot(np.asarray(f['r1_steps_seen']))
    plt.plot(np.asarray(f['r2_steps_seen']))

    plt.plot(np.asarray(f['s1_steps_seen']))
    plt.plot(np.asarray(f['s2_steps_seen']))
    plt.plot(np.asarray(f['s3_steps_seen']))

    plt.plot(np.asarray(f['rs1_steps_seen']))
    plt.plot(np.asarray(f['rs2_steps_seen']))
    plt.plot(np.asarray(f['rs3_steps_seen']))

    plt.xlabel('Number of episodes')
    plt.ylabel('Number of steps from seeing the victim for the first time to the end')
    plt.legend(['Rescuer 1', 'Rescuer 2',
                'Scout 1', 'Scout 2', 'Scout 3',
                'Rescuer/Scout 1', 'Rescuer/Scout 2', 'Rescuer/Scout 3'])
    plt.show()
