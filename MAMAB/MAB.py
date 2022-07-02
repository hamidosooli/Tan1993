import h5py
import numpy as np
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt


def true_rew(n_arms, mu_true, sigma_true):
    '''
    generate true rewards for the MAB
    :param mu_true: true mean
    :param sigma_true: true sigma
    '''
    return np.random.normal(mu_true, sigma_true, size=(n_arms))

def genetrate_mab(mu, var, num_mab, num_arms):
    Mu= np.ones(num_arms) *mu
    cov= np.diag(np.ones(num_arms)*var)
    mabs= np.random.multivariate_normal(Mu, cov, num_mab)
    MABFile= h5py.File('MAB5k.hdf5', "w")
    MABFile.create_dataset('mabs', data=mabs)


if __name__ == '__main__':
    genetrate_mab(mu=0, var=1, num_mab=5000, num_arms=10)
