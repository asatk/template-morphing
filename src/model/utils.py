"""
Some helper functions

"""
from random import sample
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import functional as F
import sys
from scipy.stats import multivariate_normal
import os
from numpy import linalg as LA
from scipy import linalg
import json

from defs import *

################################################################################
# Progress Bar
class SimpleProgressBar():
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100:
            print('')


################################################################################
# torch dataset from numpy array
class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None):
        super(custom_dataset, self).__init__()

        self.data = data
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.n_samples = self.data.shape[0]

    def __getitem__(self, index):

        x = self.data[index]
        if self.labels is not None:
            y = self.labels[index]
        else:
            y = -1
        return x, y

    def __len__(self):
        return self.n_samples

################################################################################
# Plot training loss
def PlotLoss(loss, filename):
    x_axis = np.arange(start = 1, stop = len(loss)+1)
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_axis, np.array(loss))
    plt.legend()
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3)
    plt.title('Loss')
    plt.savefig(filename)

################################################################################
# Creat sampler for a mixture of Gaussian distributions
def sampler_CircleGaussian(n_samp_per_gaussian, angle_grid, radius, sigma = 0.05, dim = 2):
    '''

    n_samp_per_gaussian: how many samples will be draw from each Gaussian
    angle_grid: raw angles
    sigma: a fixed standard deviation
    dim: dimension of a component

    '''

    cov = np.diag(np.repeat(sigma**2, dim)) #covariance matrix; firxed for each component
    n_gaussians = len(angle_grid)
    means = np.zeros((n_gaussians, dim))
    for i in range(n_gaussians):
        angle = angle_grid[i]
        mean_curr = np.array([radius*np.sin(angle), radius*np.cos(angle)])
        means[i] = mean_curr

        if i == 0:
            samples = np.random.multivariate_normal(mean_curr, cov, size=n_samp_per_gaussian)
            angles = np.ones(n_samp_per_gaussian) * angle
        else:
            samples = np.concatenate((samples, np.random.multivariate_normal(mean_curr, cov, size=n_samp_per_gaussian)), axis=0)
            angles = np.concatenate((angles, np.ones(n_samp_per_gaussian) * angle), axis=0)

    assert len(samples) == n_samp_per_gaussian*n_gaussians
    assert len(angles) == n_samp_per_gaussian*n_gaussians
    assert samples.shape[1] == dim

    return samples, angles, means

def sampler_ROOT(axis: str='phi', const_mass: float=500., samples_are_data: bool=True, n_samples: int=10) -> tuple:
    '''
    returns indices
    '''
    axis_idx = 0 if axis == 'phi' else 1
    const_idx = 1 if axis == 'phi' else 0

    samples = np.empty((0,2),dtype=float)
    sample_mass_labels = np.empty((0,),dtype=float)
    masses = np.empty((0,2),dtype=float)

    phi_axis = np.linspace(phi_min,phi_max,phi_bins+1)
    omega_axis = np.linspace(omega_min,omega_max,omega_bins+1)
    # phi_temp, omega_temp = np.meshgrid(phi_axis, omega_axis)
    # mass_grid = np.array((phi_temp.ravel(), omega_temp.ravel())).T
    # mass_grid = np.reshape(mass_grid,(phi_bins+1,omega_bins+1,2), order='F')
    # np.set_printoptions(threshold=np.inf)
    # print("mass grid",mass_grid)
    # np.set_printoptions(threshold=1000)
    # print(mass_grid[300,200])
    # exit()

    with open("data_mapping.json",'r') as json_file:
        file_mapping: dict[str,list[int]] = json.load(json_file)
        for f in file_mapping:
            # print(f)
            mass_label = np.array(file_mapping[f])
            
            #check that a file has the desired mass along the desired constant mass axis
            if mass_label[const_idx] != const_mass:
                continue

            masses = np.append(masses,[mass_label],axis=0)
            
            data: np.ndarray = np.load(PROJECT_DIR+"/out/npy/"+f)

            if data_are_samples:
                samples_f = np.nonzero(data)
            else:
                data_flat = data.flatten()
                # print(np.sum(samples_flat))
                samples_f_flat = np.random.choice(a=data_flat.size,p=data_flat,size=n_samples)
                samples_f = np.unravel_index(samples_f_flat, data.shape)

            # samples_f = np.transpose(samples_f)
            # samples_f = mass_grid[samples_f]
            samples_f = np.array([phi_axis[samples_f[0]],omega_axis[samples_f[1]]]).T
            samples = np.append(samples,samples_f,axis=0)
            sample_mass_labels = np.concatenate((sample_mass_labels,np.ones(len(samples_f)) * mass_label[axis_idx]))

    return samples, sample_mass_labels, masses

################################################################################
# Plot samples in a 2-D coordinate
def ScatterPoints(tar_samples, prop_samples, filename, plot_real_samples = False, fig_size=5, point_size=None):
    # tar_samples and prop_samples are 2-D array: n_samples by num_features
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    plt.figure(figsize=(fig_size, fig_size), facecolor='w')
    plt.grid(b=True)
    plt.scatter(tar_samples[:, 0], tar_samples[:, 1], c='blue', edgecolor='none', alpha=0.5, s=point_size)
    if not os.path.isfile(filename[0:-4]+'_realsamples.pdf') and plot_real_samples:
        plt.savefig(filename[0:-4]+'_realsamples.png')
    plt.scatter(prop_samples[:, 0], prop_samples[:, 1], c='g', edgecolor='none', s=point_size)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

################################################################################
# Compute 2-Wasserstein distance
def two_wasserstein(mu1, mu2, cov1, cov2, eps=1e-10):

    mean_diff = mu1 - mu2
    # mean_diff = LA.norm( mu1 - mu2 )
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)#square root of a matrix
    covmean = covmean.real

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    #2-Wasserstein distance
    output = mean_diff.dot(mean_diff) + np.trace(cov1 + cov2 - 2*covmean)

    return output
