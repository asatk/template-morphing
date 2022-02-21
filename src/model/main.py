'''

2D-Gaussian Simulation

'''

print("\n==================================================================================================")

import argparse
import gc
from unittest.mock import NonCallableMagicMock
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import random
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import timeit

sys.path.append(os.getcwd())
from defs import *

from opts import parse_opts
args = parse_opts()
wd = args.root_path
os.chdir(wd)

from utils import *
from models import *
from Train_CcGAN_OG import *


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################

#--------------------------------
# system
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCPU = 8

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#--------------------------------
# Extra Data Generation Settings
phi_bins_eval = phi_bins_plot = phi_bins_train = phi_bins
omega_bins_eval = omega_bins_plot = omega_bins_train = omega_bins
phi_bins_all = phi_bins * 100
omega_bins_all = omega_bins * 100

n_dists = args.n_dists
n_features = 2  #2-D
# n_dists_eval = args.n_dists_eval
n_samples_train = args.n_samples_train
axis = args.axis
const_mass = args.const_mass

#rectangle grid of training data labels:
phi_labels_train = np.linspace(phi_min,phi_max,phi_bins_train+1,endpoint=True)
omega_labels_train = np.linspace(omega_min,omega_max,omega_bins_train,endpoint=True)

# masses for evaluation
phi_labels = np.linspace(phi_min,phi_max,phi_bins_all*100+1,endpoint= True)
phi_labels = np.setdiff1d(phi_labels,phi_labels_train)
omega_labels = np.linspace(omega_min,omega_max,omega_bins_all*100+1,endpoint= True)
omega_labels = np.setdiff1d(omega_labels,omega_labels_train)

phi_labels_eval = np.zeros(phi_bins_eval)
omega_labels_eval = np.zeros(omega_bins_eval)

for i in range(phi_bins_eval):
    quantile_i = (i+1)/phi_bins_eval
    phi_labels_eval[i] = np.quantile(phi_labels, quantile_i, interpolation='nearest')
for i in range(omega_bins_eval):
    quantile_i = (i+1)/omega_bins_eval
    omega_labels_eval[i] = np.quantile(omega_labels, quantile_i, interpolation='nearest')

# masses for plotting
phi_labels = np.linspace(phi_min,phi_max,phi_bins_all*100+1,endpoint= True)
phi_labels = np.setdiff1d(phi_labels,phi_labels_train)
omega_labels = np.linspace(omega_min,omega_max,omega_bins_all*100+1,endpoint= True)
omega_labels = np.setdiff1d(omega_labels,omega_labels_train)

phi_labels_plot = np.zeros(phi_bins_plot)
omega_labels_plot = np.zeros(omega_bins_plot)

for i in range(phi_bins_plot):
    quantile_i = (i+1)/phi_bins_plot
    phi_labels_plot[i] = np.quantile(phi_labels, quantile_i, interpolation='nearest')
for i in range(omega_bins_plot):
    quantile_i = (i+1)/omega_bins_plot
    omega_labels_plot[i] = np.quantile(omega_labels, quantile_i, interpolation='nearest')

# standard deviation of each Gaussian
sigma_gaussian = args.sigma_gaussian
### threshold to determine high quality samples
quality_threshold = sigma_gaussian*4 #good samples are within 5 standard deviation
print("Quality threshold is {}".format(quality_threshold))

label_min = phi_min if axis == 'phi' else omega_min
label_max = phi_max if axis == 'phi' else omega_max
# mass_grid = np.meshgrid(phi_labels,omega_labels)

#-------------------------------
# Plot Settings
plot_in_train = True
fig_size=7
point_size = 25

#-------------------------------
# output folders
save_models_folder = PROJECT_DIR + '/out/saved_models/'
os.makedirs(save_models_folder,exist_ok=True)
save_images_folder = PROJECT_DIR + '/out/saved_images/'
os.makedirs(save_images_folder,exist_ok=True)

#######################################################################################
'''                               Start Experiment                                 '''
#######################################################################################
#---------------------------------
# sampler for target distribution
def generate_data():
    mass_grid = phi_labels_train if axis == 'phi' else omega_bins_train
    # load sampled files/dists
    # return sampler_ROOT(const_mass=const_mass,axis=axis,samples_are_data=False,n_samples=2000)
    return sampler_GridGaussian(mass_grid, axis=axis, const_mass=const_mass, n_samples=20, phi_sigma = (phi_max-phi_min)/100, omega_sigma = (omega_max-omega_min)/100)

prop_recovered_modes = np.zeros(args.nsim) # num of recovered modes diveded by num of modes
prop_good_samples = np.zeros(args.nsim) # num of good fake samples diveded by num of all fake samples
avg_two_w_dist = np.zeros(args.nsim)

print("\n Begin The Experiment; Start Training {} >>>".format(args.GAN))
start = timeit.default_timer()
for nSim in range(args.nsim):
    print("Round %s" % (nSim))
    np.random.seed(nSim) #set seed for current simulation

    ###############################################################################
    # Data generation and dataloaders
    ###############################################################################
    mass_labels_train = phi_labels_train if axis == 'phi' else omega_labels_train
    mass_labels_plot = phi_labels_plot if axis == 'phi' else omega_labels_plot
    samples_train, samples_mass_labels_train, masses_train = generate_data() #this angles_train is not normalized; normalize if args.GAN is not cGAN.
    n_dists = len(masses_train)
    samples_plot_in_train, _, _ = generate_data()

    print("samples train:",samples_train)
    print("samples mass labels train", samples_mass_labels_train)
    print("masses train", masses_train)


    # print("samples",samples_train)
    # print("sample mass labels",samples_mass_labels_train)
    # for i in range(len(samples_train)):
    #     print("sample:",samples_train[i],"\tlabel:",samples_mass_labels_train[i])
    # print("masses train",masses_train)

    # plot training samples and their theoretical means
    filename_tmp = save_images_folder + 'samples_train_with_means_nSim_' + str(nSim) + '.png'
    # if not os.path.isfile(filename_tmp):
    plt.switch_backend('agg')
    mpl.style.use('seaborn')

    # plt.ion()

    # fig, ax = plt.subplots()

    # fig = plt.figure(figsize=(fig_size, fig_size), facecolor='w')
    # fig, ax = plt.subplot(111)

    fig = plt.figure(1, figsize=(fig_size, fig_size), facecolor='w')
    # ax = plt.subplot(111,xlim=(phi_min, phi_max),ylim=(omega_min, omega_max))
    
    # ax = fig.add_axes(xlim=(phi_min, phi_max),ylim=(omega_min, omega_max))

    # plt.axes()
    # ax = fig.add_axes(xlim=(phi_min,phi_max),ylim=(omega_min,omega_max))
    # fig.grid(b=True)
    # ax.
    plt.grid(b=True)
    # plt.axes(xlim=(phi_min, phi_max),ylim=(omega_min, omega_max))
    plt.xlim((phi_min, phi_max))
    plt.ylim((omega_min, omega_max))
    plt.scatter(samples_train[:,0], samples_train[:,1], c='blue', edgecolor='none', alpha=0.5, s=point_size, label="Real samples")
    plt.scatter(masses_train[:,0], masses_train[:,1], c='red', edgecolor='none', alpha=1, s=point_size, label="Masses")
    # plt.sca(plt.axes(xlim=(phi_min, phi_max),ylim=(omega_min, omega_max)))
    # plt.axes(xlim=(phi_min, phi_max),ylim=(omega_min, omega_max))
    # plt.axes.get_xaxis().set_xlim((phi_min, phi_max))
    # plt.axes.get_yaxis().set_ylim((omega_min, omega_max))
    plt.legend(loc=1)
    
    
    plt.savefig(filename_tmp)

    # exit()

    if args.GAN == 'CcGAN':
        #normalize
        samples_mass_labels_train = (samples_mass_labels_train - label_min)/(label_max-label_min)

        # rule-of-thumb for the bandwidth selection
        if args.kernel_sigma<0:
            std_angles_train = np.std(samples_mass_labels_train)
            # print(len(samples_mass_labels_train))
            args.kernel_sigma = 1.06*std_angles_train*(len(samples_mass_labels_train))**(-1/5)
            print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")

        if args.kappa < 0:
            kappa_base = np.abs(args.kappa)/n_dists

            if args.threshold_type=="hard":
                args.kappa = kappa_base
            else:
                args.kappa = 1/kappa_base**2

    ###############################################################################
    # Train a GAN model
    ###############################################################################
    print("{}/{}, {}, Sigma is {}, Kappa is {}".format(nSim+1, args.nsim, args.threshold_type, args.kernel_sigma, args.kappa))

    if args.GAN == 'CcGAN':
        save_GANimages_InTrain_folder = PROJECT_DIR + '/out/saved_images/{}_{}_{}_{}_nSim_{}_InTrain'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa, nSim)
    os.makedirs(save_GANimages_InTrain_folder,exist_ok=True)

    #----------------------------------------------
    # Continuous cGAN
    if args.GAN == "CcGAN":
        Filename_GAN = save_models_folder + '/ckpt_{}_niters_{}_seed_{}_{}_{}_{}_nSim_{}.pth'.format(args.GAN, args.niters_gan, args.seed, args.threshold_type, args.kernel_sigma, args.kappa, nSim)

        if not os.path.isfile(Filename_GAN):
            netG = cont_cond_generator(ngpu=NGPU, nz=args.dim_gan, out_dim=n_features, label_min=label_min, label_max=label_max, const_mass=const_mass,axis=axis)
            netD = cont_cond_discriminator(ngpu=NGPU, input_dim = n_features, label_min=label_min, label_max=label_max, const_mass=const_mass,axis=axis)

            # Start training
            netG, netD = train_CcGAN(args.kernel_sigma, args.kappa, samples_train, samples_mass_labels_train, netG, netD, save_images_folder=save_GANimages_InTrain_folder, save_models_folder = save_models_folder, plot_in_train=plot_in_train, samples_tar_eval = samples_plot_in_train, mass_labels_eval = mass_labels_plot, label_min=label_min, label_max=label_max, fig_size=fig_size, point_size=point_size)

            # store model
            torch.save({
                'netG_state_dict': netG.state_dict(),
            }, Filename_GAN)
        else:
            print("Loading pre-trained generator >>>")
            checkpoint = torch.load(Filename_GAN)
            netG = cont_cond_generator(ngpu=NGPU, nz=args.dim_gan, out_dim=n_features, label_min=label_min, label_max=label_max, const_mass=const_mass,axis=axis).to(device)
            netG.load_state_dict(checkpoint['netG_state_dict'])

        def fn_sampleGAN_given_label(nfake, label, batch_size):
            fake_samples, _ = SampCcGAN_given_label(netG, label, path=None, NFAKE = nfake, batch_size = batch_size)
            return fake_samples

    ###############################################################################
    # Evaluation
    ###############################################################################
    if args.eval:
        print("\n Start evaluation >>>")

        mass_labels_eval = phi_labels_eval if args.axis == 'phi' else omega_labels_eval
        n_samples_eval = args.n_samples_eval

        # percentage of high quality and recovered modes
        for i_mass in range(len(mass_labels_eval)):
            mass_label = mass_labels_eval[i_mass]

            if axis == 'phi':
                masses = np.transpose(np.array((phi_labels_eval,np.ones(len(mass_labels_eval)) * const_mass)))
            else:
                masses = np.transpose(np.array((np.ones(len(mass_labels_eval)) * const_mass,omega_labels_eval)))

            fake_samples_curr = fn_sampleGAN_given_label(
                    n_samples_eval,
                    (mass_label - label_min)/(label_max - label_min),
                    batch_size=n_samples_eval)
                    
            mass_curr_repeat = np.repeat(masses[i].reshape(1,n_features), n_samples_eval, axis=0)

            #l2 distance between a fake sample and its mean
            l2_dis_fake_samples_curr = np.sqrt(np.sum((fake_samples_curr-mass_curr_repeat)**2, axis=1))
            
            if i_mass == 0:
                l2_dis_fake_samples = l2_dis_fake_samples_curr
            else:
                l2_dis_fake_samples = np.concatenate((l2_dis_fake_samples, l2_dis_fake_samples_curr))

            # whether this mode is recovered?
            if sum(l2_dis_fake_samples_curr<=quality_threshold)>0:
                prop_recovered_modes[nSim] += 1
        #end for i_ang
        prop_recovered_modes[nSim] = (prop_recovered_modes[nSim]/len(mass_labels_eval))*100
        prop_good_samples[nSim] = sum(l2_dis_fake_samples<=quality_threshold)/len(l2_dis_fake_samples)*100 #proportion of good fake samples


        # 2-Wasserstein Distance
        real_cov = np.eye(n_features)*sigma_gaussian**2 #covraiance matrix for each Gaussian
        # real_cov = np.eye(n_features)
        # real_cov = np.cov
        for i_mass in tqdm(range(len(mass_labels_eval))):
            mass_curr = mass_labels_eval[i_mass]
            # the mean for current Gaussian (angle)
            if axis == 'phi':
                real_mass_curr = np.array((mass_curr,const_mass))
            else:
                real_mass_curr = np.array((const_mass,mass_curr))
            # sample from trained GAN
            fake_samples_curr = fn_sampleGAN_given_label(
                    n_samples_eval,
                    (mass_curr - label_min)/(label_max - label_min),
                    batch_size=n_samples_eval)
            # the sample mass and sample cov of fake samples with current label
            fake_mass_curr = np.mean(fake_samples_curr, axis = 0)
            fake_cov_curr = np.cov(fake_samples_curr.transpose())

            # 2-W distance for current label
            two_w_dist_curr = two_wasserstein(real_mass_curr, fake_mass_curr, real_cov, fake_cov_curr, eps=1e-20)

            if i_mass == 0:
                two_w_dist_all = [two_w_dist_curr]
            else:
                two_w_dist_all.append(two_w_dist_curr)
        # end for i_ang
        avg_two_w_dist[nSim] = sum(two_w_dist_all)/len(two_w_dist_all) #average over all evaluation angles

        ### visualize fake samples
        if args.GAN == "CcGAN":
            filename_tmp = save_images_folder + '{}_real_fake_samples_{}_sigma_{}_kappa_{}_nSim_{}.png'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa, nSim)

        # n_dists_plot = args.n_dists_plot
        n_samples_plot = args.n_samples_plot

        fake_samples = np.zeros((n_dists*n_samples_plot, n_features))
        for i_tmp in range(n_dists):
            mass_label = mass_labels_plot[i_tmp]
            fake_samples_curr = fn_sampleGAN_given_label(
                    n_samples_plot,
                    (mass_label - label_min)/(label_max - label_min),
                    batch_size=n_samples_plot)
            if i_tmp == 0:
                fake_samples = fake_samples_curr
            else:
                fake_samples = np.concatenate((fake_samples, fake_samples_curr), axis=0)

        real_samples_plot, _, _ = generate_data()

        plt.switch_backend('agg')
        mpl.style.use('seaborn')
        plt.figure(figsize=(fig_size, fig_size), facecolor='w')
        plt.grid(b=True)
        plt.xlim((phi_min, phi_max))
        plt.ylim=((omega_min, omega_max))
        plt.scatter(real_samples_plot[:, 0], real_samples_plot[:, 1], c='blue', edgecolor='none', alpha=0.5, s=point_size, label="Real samples")
        plt.scatter(fake_samples[:, 0], fake_samples[:, 1], c='green', edgecolor='none', alpha=1, s=point_size, label="Fake samples")
        plt.legend(loc=1)
        plt.savefig(filename_tmp)

# for nSim
stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))
print("\n {}, Sigma is {}, Kappa is {}".format(args.threshold_type, args.kernel_sigma, args.kappa))
print("\n Prop. of good quality samples>>>\n")
print(prop_good_samples)
print("\n Prop. good samples over %d Sims: %.1f (%.1f)" % (args.nsim, np.mean(prop_good_samples), np.std(prop_good_samples)))
print("\n Prop. of recovered modes>>>\n")
print(prop_recovered_modes)
print("\n Prop. recovered modes over %d Sims: %.1f (%.1f)" % (args.nsim, np.mean(prop_recovered_modes), np.std(prop_recovered_modes)))
print("\r 2-Wasserstein Distance: %.2e (%.2e)"% (np.mean(avg_two_w_dist), np.std(avg_two_w_dist)))
print(avg_two_w_dist)
print("\n===================================================================================================")