import h5py
import numpy as np
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt


def reward(true_reward, arm, sigma=1.0):
    '''
    draws rewards for each play
    :param true_reward: true reward
    :param arm: the arm that we played
    :param sigma: variance
    '''
    return np.random.normal(true_reward[arm], sigma)


def generate_P(A, kappa=0.02):
    '''
    generate the P matrix for updating estimates
    :param A: adjacency matrix
    :param kappa: parameter
    '''
    L = laplacian(A, normed=False)
    d_max = np.max(np.diag(L))
    num_agents = np.shape(A)[0]
    return np.identity(num_agents) - ((kappa / d_max) * L)


def update_est(P, st, nt, zt, rt):
    '''
    updating the estimate for rewaed and number of plays
    :param P: P matrix
    :param st: reward estimation
    :param nt: number of plays estimations
    :param zt: current number of plays
    :param rt: current reward
    :return: updated s and n
    '''
    num_arms = nt.shape[1]
    for i in range(num_arms):
        nt[:, i] = P @ (nt[:, i] + zt[:, i])
        st[:, i] = P @ (st[:, i] + rt[:, i])
    return nt, st


def play(A,T, MAB, Noise):
    """
    :param A: Adjacncy Matrix
    :param T: Time
    :return:
    n_t, s_t, I, imme_regret
    """
    # input
    N = 10  # arms
    M = np.shape(A)[0]  # agents

    gamma = 1.9
    eta = 0.1
    sigma_g = 0.1
    mu_true = 0.0
    sigma_true = 1.0
    zeta_t = np.zeros((M, N))
    rew_t = np.zeros((M, N))
    imme_regret = np.zeros((M, T))
    I = np.zeros((M, T))
    TrueReward = MAB
    P = generate_P(Adj)
    # P=p
    bestArm = np.argmax(TrueReward)
    best_reward = TrueReward[bestArm]

    G = 1 - (eta ** 2 / 16)
    s_t = np.zeros((M, N))
    n_t = np.zeros_like(s_t)
    Q = np.zeros_like(s_t)

    for t in range(T):
        if t < N:

            for k in range(M):
                I[k, t] = t
                rew_t[k, t] = reward(TrueReward, t)
                n_t[k, t] += 1
                s_t[k, t] = rew_t[k, t]
                #imme_regret[k, t] = best_reward - rew_t[k, t]
                imme_regret[k, t] = best_reward - TrueReward[t]

        else:
            for k in range(M):
                for i in range(N):
                    Q[k, i] = s_t[k, i] / n_t[k, i] + sigma_g * np.sqrt(
                        (2 * gamma / G) * ((n_t[k, i] + f(t - 1)) / (M * n_t[k, i])) * (np.log(t - 1) / n_t[k, i]))

                action = np.argmax(Q[k, :])
                I[k, t] = action

                rew_t[k, action] = reward(TrueReward, action)

                ## Monish
                if k == 4:
                    if Noise == 1: rew_t[k, action] *= -1
                    if Noise == 2: rew_t[k, action] = np.random.normal(-1 * rew_t[k, action], 1.0)
                    ######
                zeta_t[k, action] += 1
                imme_regret[k, t] = best_reward - TrueReward[action]
                #imme_regret[k, t] = best_reward - rew_t[k, action]

        n_t, s_t= update_est(P, s_t, n_t, zeta_t, rew_t)

    return n_t, s_t, I, imme_regret, P


if __name__ == '__main__':
    # Adj = np.array([[0, 1, 1, 0, 1],
    #                 [1, 0, 0, 1, 1],
    #                 [1, 0, 0, 1, 0],
    #                 [0, 1, 1, 0, 0],
    #                 [1, 1, 0, 0, 0]])
    Adj = np.array([[0, 1, 1, 0, 1, 0],
                    [1, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0]])
    T = 1000  ### more than 1000
    f = lambda x: np.sqrt(np.log(x))
    MABS = h5py.File('MAB10k.hdf5', 'r')
    TR = MABS['mabs']

    numRuns = 10000
    regret = np.zeros((6, T, numRuns))
    p= generate_P(Adj, kappa=0.02)
    print(p)
    Noise=2
    for run in range(numRuns):
        mab = TR[run,:]
        n_out, s_out, I_out, imm_regret, p_out = play(Adj, T, mab, Noise)
        regret[:, :, run] = imm_regret
        avg_regret = np.mean(regret, axis=2)
        print(run)

    plt.figure('house connect to A2 and A5 Flipped reward 10k arm=10 kappa= 0.02 gamma=1.9 June9 10k runs')
    data = h5py.File('HA2A5F10k.hdf5', "w")
    data.create_dataset('regret', data=imm_regret)
    data.create_dataset('avg_regret', data=avg_regret)
    data.create_dataset('P Matrix', data=p_out)
    data.create_dataset('estimated s', data=s_out)
    data.create_dataset('estimated n', data=n_out)
    data.create_dataset('Matrix I', data=I_out)

    plt.rcParams.update({'font.size': 22})

    plt.plot(np.cumsum(avg_regret[0, :]), label='agent1', color='red')
    plt.plot(np.cumsum(avg_regret[1, :]), label='agent2', color='blue')

    plt.plot(np.cumsum(avg_regret[2, :]), label='agent3', color='green')
    plt.plot(np.cumsum(avg_regret[3, :]), label='agent4', color='darkviolet')
    plt.plot(np.cumsum(avg_regret[4, :]), label='agent5', color='#F97306')
    plt.plot(np.cumsum(avg_regret[5, :]), label='agent6', color='#00FFFF')
            #
    plt.xlabel('time horizons')
    plt.ylabel('expected cumulative regret')
    plt.legend()
    plt.show()