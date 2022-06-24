from gridworld_ma import animate
import numpy as np
import matplotlib.pyplot as plt
import h5py

rescuers_VFD = [2, 2, 2, 3]  # Hunter's visual field depth
scouts_VFD = [2, 3, 4]  # Visual field depth for each scout


filename = 'Tan1993_multi_rescuers_with_multi_learning_scout.hdf5'

with h5py.File(filename, 'r') as f:
    T_rescuers = f['T_rescuers']
    T_scouts = f['T_scouts']
    T_victim = f['T_victim']

    rewards = f['rewards']
    steps = f['steps']
    see_rewards = f['see_rewards']
    see_steps = f['see_steps']
    plt.plot(rewards[0])
    plt.show()
    # animate(T_rescuers, T_scouts, T_victim, rescuers_VFD, scouts_VFD, wait_time=.5, have_scout=True)